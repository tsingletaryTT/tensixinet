#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""ttchoir: the choir's voiced part as a TT-Lang kernel.

The device (`tensixinet.music.ttchoir.ttchoir`) sings by additive formant synthesis: at every sample the voice is a sum over up
to 40 harmonics of the harmonic's level (read from a table of 200 frames a second) times `sin((k + 1) * phase)`. Both choirs, the
old vowel choir and the syllable choir, end in that sum (`ttchoir.harmonic_sum`). It is the part that is all `sin`, so it is
the part the chip does.

The host keeps everything sequential or table-driven, through the device's own code (`ttchoir.choir_voice` with a `sines` that
calls this kernel): the syllable plan from the triggers, the frame tables, the pitch and its phase, the consonants' noise, the
howl and the crush. For each group of eight harmonics the host (`tensixinet.kernels.packing.choir_groups`) reduces each
harmonic's phase to one turn in float64 and interpolates its level to samples. The chip computes `sum_j a_j * sin(p_j)` for
the eight, and the host adds the groups. Forty harmonics are five runs.

The sum is built in four stages of two harmonics, each stage's running total passed on through its own dataflow buffer: a
kernel has four working registers. Each phase buffer feeds only `sin` and each level buffer only a multiply, because one buffer
cannot feed both the vector and the matrix unit on a chip.

    tt-lang-sim kernels/ttchoir.py                       # the vowel choir
    tt-lang-sim kernels/ttchoir.py --voice syllable      # the syllable choir (consonants, a coda, drift)

or on a chip under `gozer run` (see README.md).
"""
import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import ttl  # noqa: E402
import ttnn  # noqa: E402

from _common import TILE, from_device, lease_guard, parse_args, report, timed, to_device  # noqa: E402
from tensixinet.kernels.packing import choir_groups  # noqa: E402
from tensixinet.music.patch import DEFAULT_PATCH  # noqa: E402
from tensixinet.music.ttchoir import choir_voice, ttchoir  # noqa: E402
from tensixinet.telemetry import FALLBACK  # noqa: E402


@ttl.operation(grid=(1, 1))
def ttchoir_op(
    p0: ttnn.Tensor, p1: ttnn.Tensor, p2: ttnn.Tensor, p3: ttnn.Tensor, p4: ttnn.Tensor, p5: ttnn.Tensor, p6: ttnn.Tensor, p7: ttnn.Tensor,
    a0: ttnn.Tensor, a1: ttnn.Tensor, a2: ttnn.Tensor, a3: ttnn.Tensor, a4: ttnn.Tensor, a5: ttnn.Tensor, a6: ttnn.Tensor, a7: ttnn.Tensor,
    y: ttnn.Tensor,
):
    """y = sum over eight harmonics of a_j * sin(p_j), one tile at a time, two harmonics per stage."""
    rows = p0.shape[0] // TILE
    cols = p0.shape[1] // TILE

    p0_dfb = ttl.make_dataflow_buffer_like(p0, shape=(1, 1), block_count=2)
    p1_dfb = ttl.make_dataflow_buffer_like(p1, shape=(1, 1), block_count=2)
    p2_dfb = ttl.make_dataflow_buffer_like(p2, shape=(1, 1), block_count=2)
    p3_dfb = ttl.make_dataflow_buffer_like(p3, shape=(1, 1), block_count=2)
    p4_dfb = ttl.make_dataflow_buffer_like(p4, shape=(1, 1), block_count=2)
    p5_dfb = ttl.make_dataflow_buffer_like(p5, shape=(1, 1), block_count=2)
    p6_dfb = ttl.make_dataflow_buffer_like(p6, shape=(1, 1), block_count=2)
    p7_dfb = ttl.make_dataflow_buffer_like(p7, shape=(1, 1), block_count=2)
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
                with p0_dfb.wait() as p0_blk, p1_dfb.wait() as p1_blk, a0_dfb.wait() as a0_blk, a1_dfb.wait() as a1_blk, s_dfb.reserve() as s_blk:
                    s_blk.store(a0_blk * ttl.math.sin(p0_blk) + a1_blk * ttl.math.sin(p1_blk))
                with (
                    p2_dfb.wait() as p2_blk, p3_dfb.wait() as p3_blk, a2_dfb.wait() as a2_blk, a3_dfb.wait() as a3_blk,
                    s_dfb.wait() as s_prev, s_dfb.reserve() as s_blk,
                ):
                    s_blk.store(s_prev + a2_blk * ttl.math.sin(p2_blk) + a3_blk * ttl.math.sin(p3_blk))
                with (
                    p4_dfb.wait() as p4_blk, p5_dfb.wait() as p5_blk, a4_dfb.wait() as a4_blk, a5_dfb.wait() as a5_blk,
                    s_dfb.wait() as s_prev, s_dfb.reserve() as s_blk,
                ):
                    s_blk.store(s_prev + a4_blk * ttl.math.sin(p4_blk) + a5_blk * ttl.math.sin(p5_blk))
                with (
                    p6_dfb.wait() as p6_blk, p7_dfb.wait() as p7_blk, a6_dfb.wait() as a6_blk, a7_dfb.wait() as a7_blk,
                    s_dfb.wait() as s_prev, y_dfb.reserve() as y_blk,
                ):
                    y_blk.store(s_prev + a6_blk * ttl.math.sin(p6_blk) + a7_blk * ttl.math.sin(p7_blk))

    @ttl.datamovement()
    def read():
        for row in range(rows):
            for col in range(cols):
                with (
                    p0_dfb.reserve() as p0_blk,
                    p1_dfb.reserve() as p1_blk,
                    p2_dfb.reserve() as p2_blk,
                    p3_dfb.reserve() as p3_blk,
                    p4_dfb.reserve() as p4_blk,
                    p5_dfb.reserve() as p5_blk,
                    p6_dfb.reserve() as p6_blk,
                    p7_dfb.reserve() as p7_blk,
                    a0_dfb.reserve() as a0_blk,
                    a1_dfb.reserve() as a1_blk,
                    a2_dfb.reserve() as a2_blk,
                    a3_dfb.reserve() as a3_blk,
                    a4_dfb.reserve() as a4_blk,
                    a5_dfb.reserve() as a5_blk,
                    a6_dfb.reserve() as a6_blk,
                    a7_dfb.reserve() as a7_blk,
                ):
                    tx_p0 = ttl.copy(p0[row, col], p0_blk)
                    tx_p1 = ttl.copy(p1[row, col], p1_blk)
                    tx_p2 = ttl.copy(p2[row, col], p2_blk)
                    tx_p3 = ttl.copy(p3[row, col], p3_blk)
                    tx_p4 = ttl.copy(p4[row, col], p4_blk)
                    tx_p5 = ttl.copy(p5[row, col], p5_blk)
                    tx_p6 = ttl.copy(p6[row, col], p6_blk)
                    tx_p7 = ttl.copy(p7[row, col], p7_blk)
                    tx_a0 = ttl.copy(a0[row, col], a0_blk)
                    tx_a1 = ttl.copy(a1[row, col], a1_blk)
                    tx_a2 = ttl.copy(a2[row, col], a2_blk)
                    tx_a3 = ttl.copy(a3[row, col], a3_blk)
                    tx_a4 = ttl.copy(a4[row, col], a4_blk)
                    tx_a5 = ttl.copy(a5[row, col], a5_blk)
                    tx_a6 = ttl.copy(a6[row, col], a6_blk)
                    tx_a7 = ttl.copy(a7[row, col], a7_blk)
                    tx_p0.wait()
                    tx_p1.wait()
                    tx_p2.wait()
                    tx_p3.wait()
                    tx_p4.wait()
                    tx_p5.wait()
                    tx_p6.wait()
                    tx_p7.wait()
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


# The singer the check uses: its gong follows a roll (mask 0x3002) so the syllable choir has triggers in a few seconds, and its
# memory channels differ so the vowels change.
SINGER = dataclasses.replace(FALLBACK, tensix_mask=0x3002, ddr_status=(1, 1, 1, 1), asic_temperature=55.0,
                             gddr_temps=(30.0, 44.0, 52.0, 38.0, 61.0, 35.0, 47.0, 41.0))
SYLLABLES = dataclasses.replace(DEFAULT_PATCH, choir_onset="gong", choir_coda="gong", choir_drift=0.5)


def _voice_option(parser) -> None:
    parser.add_argument("--voice", choices=("vowel", "syllable"), default="vowel",
                        help="check the old vowel choir (default) or the syllable choir")


def main() -> int:
    # The tolerance: the sum of up to forty `a * sin` terms, each level under 1 and all of them summing to at most 1, so the
    # chip's multiplies (about 1e-3 relative in the other kernels) and float32 adds stay well under 2e-3 after the choir's level.
    args = parse_args(default_samples=16000 * 3, default_atol=2e-3, extra=_voice_option)
    refused = lease_guard()
    if refused:
        return refused
    sample_rate = 16000
    patch = SYLLABLES if args.voice == "syllable" else DEFAULT_PATCH
    name = "ttchoir_syllable" if args.voice == "syllable" else "ttchoir"
    duration = args.samples / sample_rate
    expected = ttchoir(SINGER, duration, sample_rate, 2, patch)

    device = ttnn.open_device(device_id=args.device)
    timing = {"ms": 0.0}
    try:
        def on_chip(sample_t, frame_t, harmonics, phase, ks):
            n = len(sample_t)
            out = np.zeros(n)
            for inputs in choir_groups(sample_t, frame_t, harmonics, phase, ks):  # eight harmonics a run
                tensors = [to_device(x, device) for x in inputs]
                y = to_device(np.zeros_like(inputs[0]), device)
                ttchoir_op(*tensors, y)  # the first call of a shape compiles the kernel; the second call is timed
                timing["ms"] += timed(ttchoir_op, *tensors, y)
                out += from_device(y, n)
            return out
        got = choir_voice(SINGER, duration, sample_rate, 2, patch, sines=on_chip)
    finally:
        ttnn.close_device(device)
    return report(name, got, expected, args, sample_rate, timing["ms"])


if __name__ == "__main__":
    sys.exit(main())
