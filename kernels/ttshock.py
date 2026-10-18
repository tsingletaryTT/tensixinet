#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""ttshock: the drone's shock wave as a TT-Lang kernel.

The device (`tensixinet.music.devices.shock_wave`) is a damped swing that reverses on every edge, rectified because
the output is mono: `level = |sin(2 pi rate tau)| * exp(-tau / decay)` for the `tau` seconds since the latest strike,
and 0 before the first one. Every sample is independent, so it is an elementwise kernel. The host works out each
sample's phase and decay exponent in float64 (`tensixinet.kernels.packing`), and the chip does the per-sample
`sin`, `abs`, `exp` and the mask.

Run it in the simulator, which needs no chip and no lease (from an environment that has `tt-lang-sim`):

    tt-lang-sim kernels/ttshock.py

or on a chip, which needs a lease. `gozer run` sets TT_VISIBLE_DEVICES, and this script refuses to open a real
device without it:

    gozer run --chips 1 --who "claude:tensixinet" --reason "ttshock kernel check" -- python3 kernels/ttshock.py

It prints one JSON line (the worst sample difference from the numpy device and whether it is within tolerance) and
exits 0 when it is.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import ttl  # noqa: E402
import ttnn  # noqa: E402

from _common import TILE, lease_guard, parse_args, run_check  # noqa: E402
from tensixinet.kernels.packing import shock_inputs  # noqa: E402
from tensixinet.music.devices import seconds_since, shock_wave  # noqa: E402


@ttl.operation(grid=(1, 1))
def ttshock_op(x: ttnn.Tensor, neg_e: ttnn.Tensor, live: ttnn.Tensor, y: ttnn.Tensor):
    """y = live * |sin(x)| * exp(neg_e), one tile at a time."""
    rows = x.shape[0] // TILE
    cols = x.shape[1] // TILE

    x_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=2)
    e_dfb = ttl.make_dataflow_buffer_like(neg_e, shape=(1, 1), block_count=2)
    live_dfb = ttl.make_dataflow_buffer_like(live, shape=(1, 1), block_count=2)
    y_dfb = ttl.make_dataflow_buffer_like(y, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        for _ in range(rows):
            for _ in range(cols):
                with (
                    x_dfb.wait() as x_blk,
                    e_dfb.wait() as e_blk,
                    live_dfb.wait() as live_blk,
                    y_dfb.reserve() as y_blk,
                ):
                    y_blk.store(live_blk * ttl.math.abs(ttl.math.sin(x_blk)) * ttl.math.exp(e_blk))

    @ttl.datamovement()
    def read():
        for row in range(rows):
            for col in range(cols):
                with x_dfb.reserve() as x_blk, e_dfb.reserve() as e_blk, live_dfb.reserve() as live_blk:
                    tx_x = ttl.copy(x[row, col], x_blk)
                    tx_e = ttl.copy(neg_e[row, col], e_blk)
                    tx_live = ttl.copy(live[row, col], live_blk)
                    tx_x.wait()
                    tx_e.wait()
                    tx_live.wait()

    @ttl.datamovement()
    def write():
        for row in range(rows):
            for col in range(cols):
                with y_dfb.wait() as y_blk:
                    tx = ttl.copy(y_blk, y[row, col])
                    tx.wait()


def main() -> int:
    args = parse_args(default_samples=8000 * 4, default_atol=2e-4)
    refused = lease_guard()
    if refused:
        return refused
    # Strikes every 1.1 s, so the ripple restarts a few times; nothing is live before the first.
    sample_rate = 8000
    strikes = np.arange(0.4, args.samples / sample_rate, 1.1)
    since = seconds_since(strikes, args.samples, sample_rate)
    rate_hz, decay_s = 3.0, 0.8
    return run_check("ttshock", ttshock_op, shock_inputs(since, rate_hz, decay_s), shock_wave(since, rate_hz, decay_s),
                     args, sample_rate)


if __name__ == "__main__":
    sys.exit(main())
