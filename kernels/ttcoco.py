#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""ttcoco: the eight-tap delay mix as a TT-Lang kernel.

The device (`tensixinet.music.ttcoco.ttcoco`) builds each of its eight taps on the host: a truncated geometric
series of delayed, decaying, bit-crunched copies of the ttquantussy's tone (`echo_tap`), one tap per GDDR channel.
That part is sequential and data-dependent (how many repeats a tap needs depends on how fast its own feedback
decays), so it stays on the host, the same way the gong's ring-down and the AVDog's undulation do.

What the chip does is the final mix: `tanh(a0 + a1 + ... + a7)`, each tap already scaled by the device's own
headroom term (`tensixinet.kernels.packing.coco_inputs`). This is the same shape of problem as `ttmix.py`, with
eight inputs instead of four, so it is built the way `ttgong.py` builds its five-partial sum: in stages, each
stage's running total passed to the next through its own dataflow buffer, because a kernel has four working
registers and eight live inputs would not fit in one expression.

Run it with `tt-lang-sim kernels/ttcoco.py`, or on a chip under `gozer run` (see README.md).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dataclasses  # noqa: E402

import ttl  # noqa: E402
import ttnn  # noqa: E402

from _common import TILE, lease_guard, parse_args, run_check  # noqa: E402
from tensixinet.kernels.packing import COCO_TAPS, coco_inputs  # noqa: E402
from tensixinet.music.devices import quantussy_tone  # noqa: E402
from tensixinet.music.ttcoco import echo_tap, tap_bits, tap_delay_ms, tap_feedback, ttcoco  # noqa: E402
from tensixinet.telemetry import FALLBACK  # noqa: E402


@ttl.operation(grid=(1, 1))
def ttcoco_op(
    a0: ttnn.Tensor, a1: ttnn.Tensor, a2: ttnn.Tensor, a3: ttnn.Tensor,
    a4: ttnn.Tensor, a5: ttnn.Tensor, a6: ttnn.Tensor, a7: ttnn.Tensor,
    y: ttnn.Tensor,
):
    """y = tanh(a0 + a1 + ... + a7), one tile at a time, built in four two-at-a-time stages."""
    rows = a0.shape[0] // TILE
    cols = a0.shape[1] // TILE

    a0_dfb = ttl.make_dataflow_buffer_like(a0, shape=(1, 1), block_count=2)
    a1_dfb = ttl.make_dataflow_buffer_like(a1, shape=(1, 1), block_count=2)
    a2_dfb = ttl.make_dataflow_buffer_like(a2, shape=(1, 1), block_count=2)
    a3_dfb = ttl.make_dataflow_buffer_like(a3, shape=(1, 1), block_count=2)
    a4_dfb = ttl.make_dataflow_buffer_like(a4, shape=(1, 1), block_count=2)
    a5_dfb = ttl.make_dataflow_buffer_like(a5, shape=(1, 1), block_count=2)
    a6_dfb = ttl.make_dataflow_buffer_like(a6, shape=(1, 1), block_count=2)
    a7_dfb = ttl.make_dataflow_buffer_like(a7, shape=(1, 1), block_count=2)
    s_dfb = ttl.make_dataflow_buffer_like(y, shape=(1, 1), block_count=2)
    y_dfb = ttl.make_dataflow_buffer_like(y, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        for _ in range(rows):
            for _ in range(cols):
                with a0_dfb.wait() as a0_blk, a1_dfb.wait() as a1_blk, s_dfb.reserve() as s_blk:
                    s_blk.store(a0_blk + a1_blk)
                with a2_dfb.wait() as a2_blk, a3_dfb.wait() as a3_blk, s_dfb.wait() as s_prev, s_dfb.reserve() as s_blk:
                    s_blk.store(s_prev + a2_blk + a3_blk)
                with a4_dfb.wait() as a4_blk, a5_dfb.wait() as a5_blk, s_dfb.wait() as s_prev, s_dfb.reserve() as s_blk:
                    s_blk.store(s_prev + a4_blk + a5_blk)
                with (
                    a6_dfb.wait() as a6_blk,
                    a7_dfb.wait() as a7_blk,
                    s_dfb.wait() as s_prev,
                    y_dfb.reserve() as y_blk,
                ):
                    y_blk.store(ttl.math.tanh(s_prev + a6_blk + a7_blk))

    @ttl.datamovement()
    def read():
        for row in range(rows):
            for col in range(cols):
                with (
                    a0_dfb.reserve() as a0_blk,
                    a1_dfb.reserve() as a1_blk,
                    a2_dfb.reserve() as a2_blk,
                    a3_dfb.reserve() as a3_blk,
                    a4_dfb.reserve() as a4_blk,
                    a5_dfb.reserve() as a5_blk,
                    a6_dfb.reserve() as a6_blk,
                    a7_dfb.reserve() as a7_blk,
                ):
                    tx_a0 = ttl.copy(a0[row, col], a0_blk)
                    tx_a1 = ttl.copy(a1[row, col], a1_blk)
                    tx_a2 = ttl.copy(a2[row, col], a2_blk)
                    tx_a3 = ttl.copy(a3[row, col], a3_blk)
                    tx_a4 = ttl.copy(a4[row, col], a4_blk)
                    tx_a5 = ttl.copy(a5[row, col], a5_blk)
                    tx_a6 = ttl.copy(a6[row, col], a6_blk)
                    tx_a7 = ttl.copy(a7[row, col], a7_blk)
                    tx_a0.wait()
                    tx_a1.wait()
                    tx_a2.wait()
                    tx_a3.wait()
                    tx_a4.wait()
                    tx_a5.wait()
                    tx_a6.wait()
                    tx_a7.wait()

    @ttl.datamovement()
    def write():
        for row in range(rows):
            for col in range(cols):
                with y_dfb.wait() as y_blk:
                    tx = ttl.copy(y_blk, y[row, col])
                    tx.wait()


def main() -> int:
    args = parse_args(default_samples=8000 * 3, default_atol=1e-3)  # the chip's tanh is within a few e-4 of numpy's
    refused = lease_guard()
    if refused:
        return refused
    sample_rate = 8000
    t = dataclasses.replace(FALLBACK, gddr_temps=(28.0, 55.0, 70.0, 40.0, 90.0, 33.0, 60.0, 48.0), power=95.0)
    duration = args.samples / sample_rate
    tone = quantussy_tone(t, duration, sample_rate, 0)
    feedback = tap_feedback(t)
    taps = []
    for ch in range(COCO_TAPS):
        delay_ms = tap_delay_ms(t.gddr_temps[ch])
        bits = tap_bits(delay_ms)
        delay_samples = int(round(delay_ms / 1000.0 * sample_rate))
        taps.append(echo_tap(tone, delay_samples, feedback, bits))
    expected = ttcoco(t, duration, sample_rate, 0)
    return run_check("ttcoco", ttcoco_op, coco_inputs(taps, args.samples), expected, args, sample_rate)


if __name__ == "__main__":
    sys.exit(main())
