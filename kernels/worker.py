#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The chip worker: runs the TT-Lang kernels on one board's two chips, on request.

`kernels/pool.py` starts one of these for each board, under the one `gozer run` lease, and sends it packed kernel inputs on
standard input; it replies with the kernel's output on standard output. Keeping the kernels compiled is what lets live mode use
the chips every few seconds. Each request is run in fixed-size chunks (the last padded with zeros and trimmed), so every kernel
compiles for exactly one shape.

A request names one of the board's two chips (`device` 0 or 1). The worker keeps one chip open and closes it to open the other
when asked, about 0.1 s. Two chips of a board cannot be used at once: with both open in one process the first kernel launch
hung on this box, and a second process waited for the first to close the board. Different boards run in parallel.

    TT_VISIBLE_DEVICES=<the board's two chips> python3 kernels/worker.py     # under the lease
    python3 kernels/worker.py --host                                           # numpy, no chip

Standard output carries only the protocol: the real stdout is moved aside and fd 1 is pointed at standard error, so the
chip runtime's own messages cannot corrupt it.
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from tensixinet.kernels import packing as pk  # noqa: E402
from tensixinet.media.chip import CHIP_OPS, DEFAULT_CHUNK_TILES, read_frame, write_frame  # noqa: E402

ROWS_PER_TILE = pk.TILE  # a packed array has 32 columns, so a 1024-sample tile is 32 rows

HOST_OPS = {
    "ttshock": lambda tiles, n: pk.shock_on_host(*tiles, n),
    "ttshimmer": lambda tiles, n: pk.shimmer_on_host(*tiles, n),
    "ttgong": pk.gong_on_host,
    "ttavdog": pk.avdog_on_host,
    "ttmix": pk.mix_on_host,
    "ttcoco": pk.coco_on_host,
}


def run_chunked(compute, inputs, n: int, chunk_rows: int) -> np.ndarray:
    """Run `compute(chunk_inputs, chunk_samples)` over every chunk of the packed `inputs` and join the first `n` samples.
    Every chunk has exactly `chunk_rows` rows: the last is padded with zeros, which every kernel turns into silence."""
    rows = inputs[0].shape[0]
    pieces = []
    for start in range(0, rows, chunk_rows):
        chunk = []
        for a in inputs:
            part = np.asarray(a[start:start + chunk_rows], dtype=np.float32)
            if part.shape[0] < chunk_rows:
                part = np.concatenate([part, np.zeros((chunk_rows - part.shape[0], pk.TILE), dtype=np.float32)])
            chunk.append(part)
        pieces.append(np.asarray(compute(chunk, chunk_rows * pk.TILE), dtype=np.float64))
    return np.concatenate(pieces)[:n] if pieces else np.zeros(0)


class Board:
    """The board's two chips, one open at a time."""

    def __init__(self):
        import ttnn
        self.ttnn, self.open_index, self.device = ttnn, None, None

    def use(self, index: int):
        if index != self.open_index:
            self.close()
            self.device = self.ttnn.open_device(device_id=index)
            self.open_index = index
        return self.device

    def close(self):
        if self.device is not None:
            self.ttnn.close_device(self.device)
            self.device, self.open_index = None, None


def chip_compute(board):
    """`compute(op, chunk_inputs, samples, index)` that runs the kernels on the board's chip number `index`."""
    from _common import from_device, to_device
    from ttavdog import ttavdog_op
    from ttcoco import ttcoco_op
    from ttgong import ttgong_op
    from ttmix import ttmix_op
    from ttshimmer import ttshimmer_op
    from ttshock import ttshock_op
    ops = {"ttshock": ttshock_op, "ttshimmer": ttshimmer_op, "ttgong": ttgong_op, "ttavdog": ttavdog_op, "ttmix": ttmix_op,
           "ttcoco": ttcoco_op}

    def compute(op, chunk, samples, index):
        device = board.use(index)
        tensors = [to_device(a, device) for a in chunk]
        y = to_device(np.zeros_like(chunk[0]), device)
        ops[op](*tensors, y)
        return from_device(y, samples)

    return compute


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the TT-Lang kernels for the live soundtrack.")
    parser.add_argument("--host", action="store_true", help="compute each kernel's formula in numpy instead of on a chip")
    parser.add_argument("--chunk-tiles", type=int, default=DEFAULT_CHUNK_TILES, help="1024-sample tiles per chunk")
    args = parser.parse_args()
    chunk_rows = args.chunk_tiles * ROWS_PER_TILE

    proto_in = os.fdopen(os.dup(0), "rb")
    proto_out = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)
    sys.stdout = sys.stderr

    board = None
    if args.host:
        def compute(op, chunk, samples, index):
            return HOST_OPS[op](chunk, samples)
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from _common import lease_guard
        refused = lease_guard()
        if refused:
            return refused
        import ttnn
        board = Board()
        compute = chip_compute(board)
        for index in range(2):  # compile every kernel on both chips now, so the first segment is not the one that waits
            for op, arity in CHIP_OPS.items():
                warm = [np.zeros((chunk_rows, pk.TILE), dtype=np.float32) for _ in range(arity)]
                compute(op, warm, chunk_rows * pk.TILE, index)
                print(f"compiled {op} on chip {index}", file=sys.stderr, flush=True)
    try:
        write_frame(proto_out, {"ready": np.array(1)})
        while True:
            request = read_frame(proto_in)
            if request is None:
                break
            try:
                op, n = str(request["op"]), int(request["n"])
                index = int(request["device"]) if "device" in request else 0
                if op not in CHIP_OPS:
                    raise ValueError(f"unknown kernel {op}")
                if index not in (0, 1):
                    raise ValueError(f"a board has chips 0 and 1, not {index}")
                inputs = [request[f"a{i}"] for i in range(CHIP_OPS[op])]
                out = run_chunked(lambda chunk, samples: compute(op, chunk, samples, index), inputs, n, chunk_rows)
                write_frame(proto_out, {"out": out})
            except Exception as e:  # reported to the client, which decides what to do; the worker stays up
                write_frame(proto_out, {"error": np.array(f"{type(e).__name__}: {e}")})
    finally:
        if board is not None:
            board.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
