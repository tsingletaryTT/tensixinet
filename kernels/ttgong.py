#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""ttgong: one ringing gong strike as a TT-Lang kernel.

The device (`tensixinet.music.devices.gong`) rings up to five inharmonic partials after every strike, each doubled by a
slightly detuned copy, each dying at its own rate, with the pitch sagging a little as the strike rings. For one strike that is
`att * sum_p exp(e_p) * (sin a_p + sin(a_p + b))`: the detuned copy of a partial runs at a phase that is the partial's plus
the same beat phase `b` for every partial. The host (`tensixinet.kernels.packing.gong_inputs`) works out the
phases (with the closed form of the pitch sag), the decay exponents with each amplitude folded in, and the attack ramp.
The chip does ten `sin`, five `exp` and the arithmetic per sample, which makes this the heaviest kernel so far.

Strikes overlap while they ring, so the kernel runs once for each of up to three overlapping strikes and the host adds the
three outputs and applies `0.6 * tanh`, as the device does.

Run it with `tt-lang-sim kernels/ttgong.py`, or on a chip under `gozer run` (see README.md).
"""
import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import ttl  # noqa: E402
import ttnn  # noqa: E402

from _common import TILE, from_device, lease_guard, parse_args, to_device  # noqa: E402
from tensixinet.kernels.packing import GONG_OVERLAP, gong_combine, gong_inputs, strike_ages, within_tolerance  # noqa: E402
from tensixinet.music.devices import gong, gong_parameters, strike_times  # noqa: E402
from tensixinet.music.patch import DEFAULT_PATCH, GONG_TUNINGS  # noqa: E402
from tensixinet.telemetry import FALLBACK  # noqa: E402


@ttl.operation(grid=(1, 1))
def ttgong_op(
    a0: ttnn.Tensor, a1: ttnn.Tensor, a2: ttnn.Tensor, a3: ttnn.Tensor, a4: ttnn.Tensor,
    b: ttnn.Tensor,
    e0: ttnn.Tensor, e1: ttnn.Tensor, e2: ttnn.Tensor, e3: ttnn.Tensor, e4: ttnn.Tensor,
    att: ttnn.Tensor,
    y: ttnn.Tensor,
):
    """y = att * sum over five partials of exp(e) * (sin a + sin(a + b)), one tile at a time."""
    rows = a0.shape[0] // TILE
    cols = a0.shape[1] // TILE

    a0_dfb = ttl.make_dataflow_buffer_like(a0, shape=(1, 1), block_count=2)
    a1_dfb = ttl.make_dataflow_buffer_like(a1, shape=(1, 1), block_count=2)
    a2_dfb = ttl.make_dataflow_buffer_like(a2, shape=(1, 1), block_count=2)
    a3_dfb = ttl.make_dataflow_buffer_like(a3, shape=(1, 1), block_count=2)
    a4_dfb = ttl.make_dataflow_buffer_like(a4, shape=(1, 1), block_count=2)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1, 1), block_count=2)
    e0_dfb = ttl.make_dataflow_buffer_like(e0, shape=(1, 1), block_count=2)
    e1_dfb = ttl.make_dataflow_buffer_like(e1, shape=(1, 1), block_count=2)
    e2_dfb = ttl.make_dataflow_buffer_like(e2, shape=(1, 1), block_count=2)
    e3_dfb = ttl.make_dataflow_buffer_like(e3, shape=(1, 1), block_count=2)
    e4_dfb = ttl.make_dataflow_buffer_like(e4, shape=(1, 1), block_count=2)
    att_dfb = ttl.make_dataflow_buffer_like(att, shape=(1, 1), block_count=2)
    y_dfb = ttl.make_dataflow_buffer_like(y, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        for _ in range(rows):
            for _ in range(cols):
                with (
                    a0_dfb.wait() as a0_blk,
                    a1_dfb.wait() as a1_blk,
                    a2_dfb.wait() as a2_blk,
                    a3_dfb.wait() as a3_blk,
                    a4_dfb.wait() as a4_blk,
                    b_dfb.wait() as b_blk,
                    e0_dfb.wait() as e0_blk,
                    e1_dfb.wait() as e1_blk,
                    e2_dfb.wait() as e2_blk,
                    e3_dfb.wait() as e3_blk,
                    e4_dfb.wait() as e4_blk,
                    att_dfb.wait() as att_blk,
                    y_dfb.reserve() as y_blk,
                ):
                    # each partial and its detuned copy: sin(a) + sin(a + b)
                    p0 = ttl.math.exp(e0_blk) * (ttl.math.sin(a0_blk) + ttl.math.sin(a0_blk + b_blk))
                    p1 = ttl.math.exp(e1_blk) * (ttl.math.sin(a1_blk) + ttl.math.sin(a1_blk + b_blk))
                    p2 = ttl.math.exp(e2_blk) * (ttl.math.sin(a2_blk) + ttl.math.sin(a2_blk + b_blk))
                    p3 = ttl.math.exp(e3_blk) * (ttl.math.sin(a3_blk) + ttl.math.sin(a3_blk + b_blk))
                    p4 = ttl.math.exp(e4_blk) * (ttl.math.sin(a4_blk) + ttl.math.sin(a4_blk + b_blk))
                    y_blk.store(att_blk * (p0 + p1 + p2 + p3 + p4))

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
                    b_dfb.reserve() as b_blk,
                    e0_dfb.reserve() as e0_blk,
                    e1_dfb.reserve() as e1_blk,
                    e2_dfb.reserve() as e2_blk,
                    e3_dfb.reserve() as e3_blk,
                    e4_dfb.reserve() as e4_blk,
                    att_dfb.reserve() as att_blk,
                ):
                    tx_a0 = ttl.copy(a0[row, col], a0_blk)
                    tx_a1 = ttl.copy(a1[row, col], a1_blk)
                    tx_a2 = ttl.copy(a2[row, col], a2_blk)
                    tx_a3 = ttl.copy(a3[row, col], a3_blk)
                    tx_a4 = ttl.copy(a4[row, col], a4_blk)
                    tx_b = ttl.copy(b[row, col], b_blk)
                    tx_e0 = ttl.copy(e0[row, col], e0_blk)
                    tx_e1 = ttl.copy(e1[row, col], e1_blk)
                    tx_e2 = ttl.copy(e2[row, col], e2_blk)
                    tx_e3 = ttl.copy(e3[row, col], e3_blk)
                    tx_e4 = ttl.copy(e4[row, col], e4_blk)
                    tx_att = ttl.copy(att[row, col], att_blk)
                    tx_a0.wait()
                    tx_a1.wait()
                    tx_a2.wait()
                    tx_a3.wait()
                    tx_a4.wait()
                    tx_b.wait()
                    tx_e0.wait()
                    tx_e1.wait()
                    tx_e2.wait()
                    tx_e3.wait()
                    tx_e4.wait()
                    tx_att.wait()

    @ttl.datamovement()
    def write():
        for row in range(rows):
            for col in range(cols):
                with y_dfb.wait() as y_blk:
                    tx = ttl.copy(y_blk, y[row, col])
                    tx.wait()


def main() -> int:
    args = parse_args(default_samples=8000 * 9, default_atol=2e-4)
    refused = lease_guard()
    if refused:
        return refused
    sample_rate = 8000
    t = dataclasses.replace(FALLBACK, aiclk=1350.0, asic_temperature=60.0, heartbeat=3)
    duration = args.samples / sample_rate
    n = args.samples
    # The tam-tam tuning has five partials, so every slot of the kernel is used.
    partials, amps = GONG_TUNINGS["tamtam"]
    patch = dataclasses.replace(DEFAULT_PATCH, gong_partials=partials, gong_amps=amps)
    strike = gong_parameters(t, sample_rate, patch)
    strikes = strike_times(t, duration)
    expected = gong(t, duration, sample_rate, 0, patch)

    device = ttnn.open_device(device_id=args.device)
    try:
        runs = []
        for back in range(GONG_OVERLAP):  # one run for each strike that can still be ringing
            inputs = gong_inputs(strike_ages(strikes, n, sample_rate, strike.length, back), strike, sample_rate)
            tensors = [to_device(x, device) for x in inputs]
            y = to_device(np.zeros_like(inputs[0]), device)
            ttgong_op(*tensors, y)
            runs.append(from_device(y, n))
    finally:
        ttnn.close_device(device)
    got = gong_combine(runs)
    ok, worst = within_tolerance(got, expected, args.atol)
    import json
    print(json.dumps({"kernel": "ttgong", "samples": n, "worst": worst, "atol": args.atol, "ok": ok, "simulator": "sim" in getattr(ttnn, "__name__", "")}))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
