#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""ttavdog: the AVDog's two voices as a TT-Lang kernel.

The device (`tensixinet.music.ttavdog`) is a dogtone, a triangle that skews toward a saw, under the two halves of a ringing
filter's undulation: the white voice on the up half and the red voice, with the opposite slope, on the down half. The host
(`tensixinet.kernels.packing.avdog_inputs`) prepares the phase, the slopes, the undulation's phase and decay exponent, and four
constants that carry the level and the red gain (block maths takes tensors, not Python numbers). The chip computes
`relu(u) * (min(x ip, ox iq) aw2 - aw) + relu(-u) * (min(x iq, ox ip) ar2 - ar)` with `u = sin(pu) * exp(e)`. It uses `min`,
`relu` and `neg` as well as `sin` and `exp`, so it exercises more of the chip's maths than the earlier kernels.

Run it with `tt-lang-sim kernels/ttavdog.py`, or on a chip under `gozer run` (see README.md).
"""
import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import ttl  # noqa: E402
import ttnn  # noqa: E402

from _common import TILE, lease_guard, parse_args, run_check  # noqa: E402
from tensixinet.kernels.packing import avdog_inputs  # noqa: E402
from tensixinet.music.devices import seconds_since, strike_times  # noqa: E402
from tensixinet.music.ttavdog import avdog_parameters, ttavdog  # noqa: E402
from tensixinet.telemetry import FALLBACK  # noqa: E402


@ttl.operation(grid=(1, 1))
def ttavdog_op(
    x: ttnn.Tensor, ox: ttnn.Tensor, ip: ttnn.Tensor, iq: ttnn.Tensor, pu: ttnn.Tensor, e: ttnn.Tensor,
    aw2: ttnn.Tensor, aw: ttnn.Tensor, ar2: ttnn.Tensor, ar: ttnn.Tensor,
    y: ttnn.Tensor,
):
    """y = the white voice under the up half of the undulation plus the red voice under the down half, one tile at a time."""
    rows = x.shape[0] // TILE
    cols = x.shape[1] // TILE

    x_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=2)
    ox_dfb = ttl.make_dataflow_buffer_like(ox, shape=(1, 1), block_count=2)
    ip_dfb = ttl.make_dataflow_buffer_like(ip, shape=(1, 1), block_count=2)
    iq_dfb = ttl.make_dataflow_buffer_like(iq, shape=(1, 1), block_count=2)
    pu_dfb = ttl.make_dataflow_buffer_like(pu, shape=(1, 1), block_count=2)
    e_dfb = ttl.make_dataflow_buffer_like(e, shape=(1, 1), block_count=2)
    aw2_dfb = ttl.make_dataflow_buffer_like(aw2, shape=(1, 1), block_count=2)
    aw_dfb = ttl.make_dataflow_buffer_like(aw, shape=(1, 1), block_count=2)
    ar2_dfb = ttl.make_dataflow_buffer_like(ar2, shape=(1, 1), block_count=2)
    ar_dfb = ttl.make_dataflow_buffer_like(ar, shape=(1, 1), block_count=2)
    y_dfb = ttl.make_dataflow_buffer_like(y, shape=(1, 1), block_count=2)
    u_dfb = ttl.make_dataflow_buffer_like(y, shape=(1, 1), block_count=2)
    white_raw_dfb = ttl.make_dataflow_buffer_like(y, shape=(1, 1), block_count=2)
    red_raw_dfb = ttl.make_dataflow_buffer_like(y, shape=(1, 1), block_count=2)
    white_dfb = ttl.make_dataflow_buffer_like(y, shape=(1, 1), block_count=2)
    red_dfb = ttl.make_dataflow_buffer_like(y, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        for _ in range(rows):
            for _ in range(cols):
                # The chip has four working registers, so each part is made in its own stage and passed on in a buffer.
                with pu_dfb.wait() as pu_blk, e_dfb.wait() as e_blk, u_dfb.reserve() as u_blk:
                    u_blk.store(ttl.math.sin(pu_blk) * ttl.math.exp(e_blk))
                with (
                    x_dfb.wait() as x_blk,
                    ox_dfb.wait() as ox_blk,
                    ip_dfb.wait() as ip_blk,
                    iq_dfb.wait() as iq_blk,
                    white_raw_dfb.reserve() as white_raw_blk,
                    red_raw_dfb.reserve() as red_raw_blk,
                ):
                    white_raw_blk.store(ttl.math.min(x_blk * ip_blk, ox_blk * iq_blk))
                    red_raw_blk.store(ttl.math.min(x_blk * iq_blk, ox_blk * ip_blk))
                with aw2_dfb.wait() as aw2_blk, aw_dfb.wait() as aw_blk, white_raw_dfb.wait() as white_raw_blk, white_dfb.reserve() as white_blk:
                    white_blk.store(white_raw_blk * aw2_blk - aw_blk)
                with ar2_dfb.wait() as ar2_blk, ar_dfb.wait() as ar_blk, red_raw_dfb.wait() as red_raw_blk, red_dfb.reserve() as red_blk:
                    red_blk.store(red_raw_blk * ar2_blk - ar_blk)
                with u_dfb.wait() as u_blk, white_dfb.wait() as white_blk, red_dfb.wait() as red_blk, y_dfb.reserve() as y_blk:
                    y_blk.store(ttl.math.relu(u_blk) * white_blk + ttl.math.relu(ttl.math.neg(u_blk)) * red_blk)

    @ttl.datamovement()
    def read():
        for row in range(rows):
            for col in range(cols):
                with (
                    x_dfb.reserve() as x_blk,
                    ox_dfb.reserve() as ox_blk,
                    ip_dfb.reserve() as ip_blk,
                    iq_dfb.reserve() as iq_blk,
                    pu_dfb.reserve() as pu_blk,
                    e_dfb.reserve() as e_blk,
                    aw2_dfb.reserve() as aw2_blk,
                    aw_dfb.reserve() as aw_blk,
                    ar2_dfb.reserve() as ar2_blk,
                    ar_dfb.reserve() as ar_blk,
                ):
                    tx_x = ttl.copy(x[row, col], x_blk)
                    tx_ox = ttl.copy(ox[row, col], ox_blk)
                    tx_ip = ttl.copy(ip[row, col], ip_blk)
                    tx_iq = ttl.copy(iq[row, col], iq_blk)
                    tx_pu = ttl.copy(pu[row, col], pu_blk)
                    tx_e = ttl.copy(e[row, col], e_blk)
                    tx_aw2 = ttl.copy(aw2[row, col], aw2_blk)
                    tx_aw = ttl.copy(aw[row, col], aw_blk)
                    tx_ar2 = ttl.copy(ar2[row, col], ar2_blk)
                    tx_ar = ttl.copy(ar[row, col], ar_blk)
                    tx_x.wait()
                    tx_ox.wait()
                    tx_ip.wait()
                    tx_iq.wait()
                    tx_pu.wait()
                    tx_e.wait()
                    tx_aw2.wait()
                    tx_aw.wait()
                    tx_ar2.wait()
                    tx_ar.wait()

    @ttl.datamovement()
    def write():
        for row in range(rows):
            for col in range(cols):
                with y_dfb.wait() as y_blk:
                    tx = ttl.copy(y_blk, y[row, col])
                    tx.wait()


def main() -> int:
    args = parse_args(default_samples=8000 * 10, default_atol=5e-3)  # on a chip the multiplies carry about 10 bits (worst 2.3e-3); the simulator is exact
    refused = lease_guard()
    if refused:
        return refused
    sample_rate = 8000
    # Close to its throttle point, so the wave is skewed and the two voices differ in slope.
    t = dataclasses.replace(FALLBACK, asic_temperature=60.0, power=50.0, thm_limit_throttle=65.0)
    duration = args.samples / sample_rate
    pitch, shape, rate_hz, decay_s = avdog_parameters(t)
    since = seconds_since(strike_times(t, duration), args.samples, sample_rate)
    expected = ttavdog(t, duration, sample_rate, 0)
    return run_check("ttavdog", ttavdog_op, avdog_inputs(since, rate_hz, decay_s, pitch, shape, args.samples, sample_rate), expected,
                     args, sample_rate)


if __name__ == "__main__":
    sys.exit(main())
