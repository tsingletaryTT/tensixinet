# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""A chip worker: a subprocess that holds a leased chip and runs the TT-Lang kernels on request.

The package never imports `ttnn` or `ttl` (importing `ttnn` opens the device), so the kernels run in `kernels/worker.py`
and this module talks to it. The worker is started under `gozer run`, which gives it a lease for as long as it lives and
releases the chip when it exits. `host_command` starts the same worker in its host mode, which runs each kernel's formula
in numpy with no chip, no lease and no TT-Lang, for tests and for hearing the wiring without hardware.

Messages are length-prefixed `.npz` files on the worker's standard input and output. A request has `op`, `n` and the
packed input arrays `a0`, `a1`, ...; a reply has `out` (the first `n` samples) or `error`. The worker sends one `ready`
message when the chip is open and every kernel has compiled.
"""
from __future__ import annotations

import io
import os
import re
import select
import signal
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import BinaryIO, Dict, List, Optional, Sequence

import numpy as np

# The kernels the worker runs, and how many packed input arrays each takes (see `tensixinet/kernels/packing.py`).
CHIP_OPS: Dict[str, int] = {"ttshock": 3, "ttshimmer": 5, "ttgong": 16, "ttavdog": 10, "ttmix": 4, "ttcoco": 8}
DEFAULT_CHUNK_TILES = 448  # a chunk is this many 1024-sample tiles, 10.4 s at 44.1 kHz: the longest live segment fits in one
WORKER = Path(__file__).resolve().parents[2] / "kernels" / "pool.py"  # starts one kernels/worker.py per chip
CHIP_START_TIMEOUT_S = 300.0  # a chip takes a while to open, and each kernel compiles on first use
HOST_START_TIMEOUT_S = 30.0
RUN_TIMEOUT_S = 120.0
# Closing a chip worker ends `gozer run`, which resets the chips and re-initialises the boards before it releases the lease:
# 44 s on this box. Killing it part-way leaves the lease for gozer to reap as "owning process is dead", so wait for it.
CHIP_CLOSE_TIMEOUT_S = 180.0
HOST_CLOSE_TIMEOUT_S = 30.0


class ChipError(RuntimeError):
    """The worker could not start, died, took too long, or refused a request."""


def write_frame(stream: BinaryIO, arrays: Dict[str, np.ndarray]) -> None:
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    payload = buffer.getvalue()
    stream.write(struct.pack(">Q", len(payload)) + payload)
    stream.flush()


def _read_exact(stream: BinaryIO, count: int, timeout: Optional[float]) -> Optional[bytes]:
    """`count` bytes, or None if the stream ends first. Raises TimeoutError when `timeout` seconds pass with nothing to read."""
    chunks: List[bytes] = []
    remaining = count
    while remaining:
        if timeout is not None and not select.select([stream], [], [], timeout)[0]:
            raise TimeoutError
        chunk = os.read(stream.fileno(), min(remaining, 1 << 20))
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def read_frame(stream: BinaryIO, timeout: Optional[float] = None) -> Optional[Dict[str, np.ndarray]]:
    """The next message as a dict of arrays, or None when the other end has closed."""
    header = _read_exact(stream, 8, timeout)
    if header is None:
        return None
    payload = _read_exact(stream, struct.unpack(">Q", header)[0], timeout)
    if payload is None:
        return None
    with np.load(io.BytesIO(payload), allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def host_command(chunk_tiles: Optional[int] = None, devices: int = 1) -> List[str]:
    """Start the worker in host mode: each kernel's formula in numpy, no chip, with `devices` pretend devices."""
    cmd = [sys.executable, str(WORKER), "--host", f"--devices={devices}"]
    return cmd + (["--chunk-tiles", str(chunk_tiles)] if chunk_tiles else [])


CHIPS_PER_BOARD = 2  # a p300c board carries two chips; they are one tt-metal cluster and a lease covers the whole board


def board_of(chip: int):
    """(the board, and the chip's place on it) for chip number `chip` of a lease, which lists a board's chips together."""
    return chip // CHIPS_PER_BOARD, chip % CHIPS_PER_BOARD


def boards_needed(chips: int) -> int:
    return -(-chips // CHIPS_PER_BOARD)


def board_visible_devices(visible: Optional[str], board: int, chips: int = 1) -> str:
    """The `TT_VISIBLE_DEVICES` for the worker of board `board`: that board's two chips and no others. `gozer run` sets the
    variable to every chip it leased. A worker that saw all of them tried to claim all of them, and workers on four chips
    waited on each other's locks ("CHIP_IN_USE_0_PCIe"); showing a worker only one chip of a two-chip board made tt-metal call
    the cluster CUSTOM and refuse to start. One worker per board, each seeing its own board, runs in parallel."""
    leased = [c.strip() for c in (visible or "").split(",") if c.strip()]
    if not leased:
        raise ChipError("TT_VISIBLE_DEVICES is not set: a chip worker must run under `gozer run`, which sets it to the leased chips")
    mine = leased[board * CHIPS_PER_BOARD:(board + 1) * CHIPS_PER_BOARD]
    if not mine:
        raise ChipError(f"the run needs board {board} (for {chips} chips) and the lease holds only {len(leased)} chips ({visible})")
    return ",".join(mine)


def chip_command(chips: int = 1) -> List[str]:
    """Start the worker on `chips` leased chips, with one device open per chip. `gozer run` takes the lease, queues (exit 10)
    when the chips are busy, and resets and releases exactly these chips when the worker exits. The leases are by board
    (two chips), so one chip takes a board, and four take both boards."""
    return ["gozer", "run", "--chips", str(chips), "--who", "claude:tensixinet", "--reason",
            f"live soundtrack on the TT-Lang kernels, {chips} chip{'' if chips == 1 else 's'}",
            "--", "python3", str(WORKER), f"--devices={chips}"]


class ChipWorker:
    """A running worker. Use it as a context manager, or call `start` and `close`."""

    def __init__(self, command: Sequence[str], start_timeout: Optional[float] = None, run_timeout: float = RUN_TIMEOUT_S,
                 close_timeout: Optional[float] = None, devices: int = 1):
        self.command = list(command)
        self.devices = devices  # how many chips the worker holds open; a request names one of them
        self.close_timeout = close_timeout if close_timeout is not None else (
            HOST_CLOSE_TIMEOUT_S if "--host" in self.command else CHIP_CLOSE_TIMEOUT_S)
        self.start_timeout = start_timeout if start_timeout is not None else (
            HOST_START_TIMEOUT_S if "--host" in self.command else CHIP_START_TIMEOUT_S)
        self.run_timeout = run_timeout
        self.process: Optional[subprocess.Popen] = None
        self._stderr = None

    def __enter__(self) -> "ChipWorker":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def start(self) -> None:
        if self.process is not None:
            return
        self._stderr = tempfile.TemporaryFile()
        try:
            # Its own process group, so the whole tree (gozer, the pool, the workers) can be ended together if it must be
            self.process = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._stderr,
                                            start_new_session=True)
        except OSError as e:
            raise ChipError(f"could not start the chip worker ({self.command[0]}): {e}") from e
        try:
            ready = read_frame(self.process.stdout, self.start_timeout)
        except TimeoutError:
            self.close(force=True)  # it is stuck, so closing its input would change nothing: end the whole tree now
            raise ChipError(f"the chip worker was not ready after {self.start_timeout:.0f} s") from None
        if ready is None or "ready" not in ready:
            code = self.process.poll()
            detail = self._stderr_tail()
            self.close()
            hint = ""
            if code == 10:  # gozer's exit when it queued the request instead of granting it
                hint = " (gozer queued the request: the chips are busy)"
                ticket = re.search(r"ticket (\w+)", detail)
                if ticket:  # a ticket left in the queue would hold up everyone behind it
                    self.cancel(ticket.group(1))
                    hint += f"; cancelled ticket {ticket.group(1)}"
            raise ChipError(f"the chip worker exited before it was ready (exit {code}){hint}: {detail}")

    @staticmethod
    def _end_tree(process: subprocess.Popen) -> None:
        """The worker would not exit in time: ask the whole process group to stop, then kill what is left, so no worker is left
        holding a chip (gozer alone, killed, leaves its children running)."""
        for sig, grace in ((signal.SIGTERM, 15), (signal.SIGKILL, 15)):
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                break
            try:
                process.wait(timeout=grace)
                break
            except subprocess.TimeoutExpired:
                continue
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass

    def cancel(self, ticket: str) -> None:
        """Drop a gozer queue ticket this worker's own request left behind."""
        try:
            subprocess.run(["gozer", "cancel", ticket], capture_output=True, timeout=30)
        except (OSError, subprocess.SubprocessError):
            pass

    def _stderr_tail(self, lines: int = 6) -> str:
        if self._stderr is None:
            return ""
        self._stderr.seek(0)
        return " | ".join(self._stderr.read().decode(errors="replace").strip().splitlines()[-lines:])

    def run(self, op: str, inputs: Sequence[np.ndarray], n: int, device: int = 0) -> np.ndarray:
        """Run kernel `op` on the packed `inputs` on the worker's chip number `device`, and return its first `n` samples as
        float64."""
        if not 0 <= device < self.devices:
            raise ChipError(f"no device {device}; this worker holds {self.devices}")
        if op not in CHIP_OPS:
            raise ChipError(f"unknown kernel {op}; the kernels are {', '.join(CHIP_OPS)}")
        if len(inputs) != CHIP_OPS[op]:
            raise ChipError(f"{op} takes {CHIP_OPS[op]} inputs, got {len(inputs)}")
        if self.process is None:
            self.start()
        request = {"op": np.array(op), "n": np.array(n), "device": np.array(device)}
        request.update({f"a{i}": np.ascontiguousarray(a, dtype=np.float32) for i, a in enumerate(inputs)})
        try:
            write_frame(self.process.stdin, request)
            reply = read_frame(self.process.stdout, self.run_timeout)
        except (BrokenPipeError, OSError) as e:
            raise ChipError(f"the chip worker is gone: {e}; {self._stderr_tail()}") from e
        except TimeoutError:
            raise ChipError(f"the chip worker did not answer {op} within {self.run_timeout:.0f} s") from None
        if reply is None:
            raise ChipError(f"the chip worker exited (exit {self.process.poll()}): {self._stderr_tail()}")
        if "error" in reply:
            raise ChipError(f"the chip worker refused {op}: {reply['error']}")
        return np.asarray(reply["out"], dtype=np.float64)

    def close(self, force: bool = False) -> None:
        """End the worker (closing its input ends it) and wait for it, so a lease is released before this returns. Under
        gozer that takes the chips' reset (about 40 s); the wait is `close_timeout`, and only after it is the process killed."""
        process, self.process = self.process, None
        if process is None:
            return
        try:
            process.stdin.close()
        except OSError:
            pass
        if force:
            self._end_tree(process)
        else:
            try:
                process.wait(timeout=self.close_timeout)
            except subprocess.TimeoutExpired:
                self._end_tree(process)
        for stream in (process.stdout,):
            try:
                stream.close()
            except OSError:
                pass
        if self._stderr is not None:
            self._stderr.close()
            self._stderr = None
