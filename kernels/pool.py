#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The chip pool: one worker process per leased board, behind one pipe.

`tensixinet/media/chip.py` starts this under a single `gozer run` lease. A board carries two chips that tt-metal treats as one
cluster, and on this box they cannot be worked at once from one process or from two, so the pool starts one `kernels/worker.py` per
board (each shown only its own board's chips, running in parallel with the other boards) and waits until every one is ready. It
forwards each request to the worker of the board the named chip is on, with the chip's place on that board (0 or 1), and returns
the reply. A worker keeps one chip open and switches to the other when asked, which takes about 0.1 s. When the pool's input
closes it ends the workers and waits for them, so `gozer run` only resets and releases the chips after every device is closed.

    gozer run --chips 4 --who ... --reason ... -- python3 kernels/pool.py --devices=4
    python3 kernels/pool.py --host --devices=4        # numpy workers, no chip

Each worker is told to exit if this process dies, so a killed pool cannot leave a worker holding a chip.
"""
import argparse
import ctypes
import os
import signal
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from tensixinet.media.chip import (  # noqa: E402
    DEFAULT_CHUNK_TILES, ChipError, board_of, board_visible_devices, boards_needed, read_frame, write_frame,
)

WORKER = Path(__file__).resolve().parent / "worker.py"
PR_SET_PDEATHSIG = 1


def die_with_parent() -> None:
    """Runs in each worker just before it starts: SIGTERM it when this pool dies (Linux)."""
    try:
        ctypes.CDLL("libc.so.6").prctl(PR_SET_PDEATHSIG, signal.SIGTERM)
    except OSError:
        pass


def start_workers(chips: int, host: bool, chunk_tiles: int):
    """One worker per board of the `chips` chips. On a chip, each is shown only its own board of the lease."""
    workers = []
    for board in range(boards_needed(chips)):
        env = dict(os.environ)
        if not host:
            env["TT_VISIBLE_DEVICES"] = board_visible_devices(os.environ.get("TT_VISIBLE_DEVICES"), board, chips)
        cmd = [sys.executable, str(WORKER), f"--chunk-tiles={chunk_tiles}"] + (["--host"] if host else [])
        workers.append(subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env, preexec_fn=die_with_parent))
    return workers


def error_frame(message: str):
    return {"error": np.array(message)}


def main() -> int:
    parser = argparse.ArgumentParser(description="One chip worker per leased chip, behind one pipe.")
    parser.add_argument("--host", action="store_true", help="numpy workers instead of chips")
    parser.add_argument("--devices", type=int, default=1, help="how many chips, device 0 up")
    parser.add_argument("--chunk-tiles", type=int, default=DEFAULT_CHUNK_TILES)
    args = parser.parse_args()

    proto_in = os.fdopen(os.dup(0), "rb")
    proto_out = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)
    sys.stdout = sys.stderr

    try:
        workers = start_workers(args.devices, args.host, args.chunk_tiles)
    except ChipError as e:
        print(f"pool: {e}", file=sys.stderr, flush=True)
        return 2
    try:
        for index, worker in enumerate(workers):
            ready = read_frame(worker.stdout)
            if ready is None or "ready" not in ready:
                print(f"pool: the worker for board {index} exited before it was ready (exit {worker.poll()})", file=sys.stderr, flush=True)
                return 2
        write_frame(proto_out, {"ready": np.array(1)})
        while True:
            request = read_frame(proto_in)
            if request is None:
                break
            op = str(request["op"])
            chip = int(request["device"]) if "device" in request else 0
            if not 0 <= chip < args.devices:
                write_frame(proto_out, error_frame(f"ValueError: device {chip} is not open (this pool opened {args.devices})"))
                continue
            board, place = board_of(chip)
            worker, index = workers[board], board
            request = dict(request, device=np.array(place))  # the worker knows its board's chips as 0 and 1
            try:
                write_frame(worker.stdin, request)
                reply = read_frame(worker.stdout)
            except OSError:
                reply = None
            if reply is None:
                write_frame(proto_out, error_frame(f"RuntimeError: the worker for board {index} (chip {chip}) exited (exit {worker.poll()}) while running {op}"))
                continue
            write_frame(proto_out, reply)
    finally:
        for worker in workers:
            try:
                worker.stdin.close()
            except OSError:
                pass
        for worker in workers:
            try:
                worker.wait(timeout=120)
            except subprocess.TimeoutExpired:
                worker.kill()
    return 0


if __name__ == "__main__":
    sys.exit(main())
