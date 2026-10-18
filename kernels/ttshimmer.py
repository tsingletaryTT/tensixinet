#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""ttshimmer: the ultrasonic shimmer as a TT-Lang kernel.

The device (`tensixinet.music.devices.ultrasonic_shimmer`) is two close high tones whose level snaps up on every gong
strike: `(sin p1 + sin p2) * (a0 + a1 * exp(e))`. The host (`tensixinet.kernels.packing.shimmer_inputs`) reduces
each tone's phase to one turn in float64, works out the snap's exponent and the two level constants. The chip does
four transcendental calls per sample, two `sin`, one `exp`, and the arithmetic.

Run it with `tt-lang-sim kernels/ttshimmer.py`, or on a chip under `gozer run` (see README.md).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dataclasses  # noqa: E402

import ttl  # noqa: E402
import ttnn  # noqa: E402

from _common import TILE, lease_guard, parse_args, run_check  # noqa: E402
from tensixinet.kernels.packing import shimmer_inputs  # noqa: E402
from tensixinet.music.devices import seconds_since, shimmer_parameters, strike_times, ultrasonic_shimmer  # noqa: E402
from tensixinet.music.patch import DEFAULT_PATCH  # noqa: E402
from tensixinet.telemetry import FALLBACK  # noqa: E402


@ttl.operation(grid=(1, 1))
def ttshimmer_op(p1: ttnn.Tensor, p2: ttnn.Tensor, e: ttnn.Tensor, a0: ttnn.Tensor, a1: ttnn.Tensor, y: ttnn.Tensor):
    """y = (sin p1 + sin p2) * (a0 + a1 * exp(e)), one tile at a time."""
    rows = p1.shape[0] // TILE
    cols = p1.shape[1] // TILE

    p1_dfb = ttl.make_dataflow_buffer_like(p1, shape=(1, 1), block_count=2)
    p2_dfb = ttl.make_dataflow_buffer_like(p2, shape=(1, 1), block_count=2)
    e_dfb = ttl.make_dataflow_buffer_like(e, shape=(1, 1), block_count=2)
    a0_dfb = ttl.make_dataflow_buffer_like(a0, shape=(1, 1), block_count=2)
    a1_dfb = ttl.make_dataflow_buffer_like(a1, shape=(1, 1), block_count=2)
    y_dfb = ttl.make_dataflow_buffer_like(y, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        for _ in range(rows):
            for _ in range(cols):
                with (
                    p1_dfb.wait() as p1_blk,
                    p2_dfb.wait() as p2_blk,
                    e_dfb.wait() as e_blk,
                    a0_dfb.wait() as a0_blk,
                    a1_dfb.wait() as a1_blk,
                    y_dfb.reserve() as y_blk,
                ):
                    y_blk.store((ttl.math.sin(p1_blk) + ttl.math.sin(p2_blk)) * (a0_blk + a1_blk * ttl.math.exp(e_blk)))

    @ttl.datamovement()
    def read():
        for row in range(rows):
            for col in range(cols):
                with (
                    p1_dfb.reserve() as p1_blk,
                    p2_dfb.reserve() as p2_blk,
                    e_dfb.reserve() as e_blk,
                    a0_dfb.reserve() as a0_blk,
                    a1_dfb.reserve() as a1_blk,
                ):
                    tx_p1 = ttl.copy(p1[row, col], p1_blk)
                    tx_p2 = ttl.copy(p2[row, col], p2_blk)
                    tx_e = ttl.copy(e[row, col], e_blk)
                    tx_a0 = ttl.copy(a0[row, col], a0_blk)
                    tx_a1 = ttl.copy(a1[row, col], a1_blk)
                    tx_p1.wait()
                    tx_p2.wait()
                    tx_e.wait()
                    tx_a0.wait()
                    tx_a1.wait()

    @ttl.datamovement()
    def write():
        for row in range(rows):
            for col in range(cols):
                with y_dfb.wait() as y_blk:
                    tx = ttl.copy(y_blk, y[row, col])
                    tx.wait()


def main() -> int:
    args = parse_args(default_samples=44100, default_atol=2e-3)
    refused = lease_guard()
    if refused:
        return refused
    sample_rate = 44100
    t = dataclasses.replace(FALLBACK, asic_temperature=72.0, gddr_temps=(30.0, 41.0) * 4, power=90.0, heartbeat=5)
    duration = args.samples / sample_rate
    f1, f2, level = shimmer_parameters(t, sample_rate, DEFAULT_PATCH)
    since = seconds_since(strike_times(t, max(duration, 12.0)), args.samples, sample_rate)
    expected = ultrasonic_shimmer(t, duration, sample_rate, 0)
    return run_check("ttshimmer", ttshimmer_op, shimmer_inputs(since, f1, f2, level, args.samples, sample_rate), expected, args.atol, args.device)


if __name__ == "__main__":
    sys.exit(main())
