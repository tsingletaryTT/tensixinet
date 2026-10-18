#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""ttmix: the chips' mixer as a TT-Lang kernel.

The device (`tensixinet.music.devices.current_mix`) mixes by current: each signal is weighted by its share of the
total current and the sum goes through a saturating `tanh`, so piling on sources compresses and never clips. The host
(`tensixinet.kernels.packing.mix_inputs`) multiplies each signal by its weight, already divided by `sqrt(chips)`.
The chip adds four signals and applies `tanh`. Up to four chips for now; a fifth is refused on the host.

Run it with `tt-lang-sim kernels/ttmix.py`, or on a chip under `gozer run` (see README.md).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ttl  # noqa: E402
import ttnn  # noqa: E402

from _common import TILE, lease_guard, parse_args, run_check  # noqa: E402
from tensixinet.kernels.packing import mix_inputs  # noqa: E402
from tensixinet.music.chorus import DEVICES  # noqa: E402
from tensixinet.music.devices import current_mix  # noqa: E402
from tensixinet.telemetry.mock import mock_telemetry  # noqa: E402


@ttl.operation(grid=(1, 1))
def ttmix_op(a: ttnn.Tensor, b: ttnn.Tensor, c: ttnn.Tensor, d: ttnn.Tensor, y: ttnn.Tensor):
    """y = tanh(a + b + c + d), one tile at a time."""
    rows = a.shape[0] // TILE
    cols = a.shape[1] // TILE

    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1, 1), block_count=2)
    c_dfb = ttl.make_dataflow_buffer_like(c, shape=(1, 1), block_count=2)
    d_dfb = ttl.make_dataflow_buffer_like(d, shape=(1, 1), block_count=2)
    y_dfb = ttl.make_dataflow_buffer_like(y, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        for _ in range(rows):
            for _ in range(cols):
                with (
                    a_dfb.wait() as a_blk,
                    b_dfb.wait() as b_blk,
                    c_dfb.wait() as c_blk,
                    d_dfb.wait() as d_blk,
                    y_dfb.reserve() as y_blk,
                ):
                    y_blk.store(ttl.math.tanh(a_blk + b_blk + c_blk + d_blk))

    @ttl.datamovement()
    def read():
        for row in range(rows):
            for col in range(cols):
                with a_dfb.reserve() as a_blk, b_dfb.reserve() as b_blk, c_dfb.reserve() as c_blk, d_dfb.reserve() as d_blk:
                    tx_a = ttl.copy(a[row, col], a_blk)
                    tx_b = ttl.copy(b[row, col], b_blk)
                    tx_c = ttl.copy(c[row, col], c_blk)
                    tx_d = ttl.copy(d[row, col], d_blk)
                    tx_a.wait()
                    tx_b.wait()
                    tx_c.wait()
                    tx_d.wait()

    @ttl.datamovement()
    def write():
        for row in range(rows):
            for col in range(cols):
                with y_dfb.wait() as y_blk:
                    tx = ttl.copy(y_blk, y[row, col])
                    tx.wait()


def main() -> int:
    args = parse_args(default_samples=8000 * 3, default_atol=1e-3)  # the chip's tanh is within 3e-4 of numpy's; the simulator's is exact
    refused = lease_guard()
    if refused:
        return refused
    sample_rate = 8000
    chips = mock_telemetry(4)
    duration = args.samples / sample_rate
    # Four different devices stand in for four chips' signals.
    signals = [DEVICES[d](chips[c], duration, sample_rate, d)[: args.samples] for c, d in enumerate((0, 1, 5, 6))]
    currents = [t.current for t in chips]
    return run_check("ttmix", ttmix_op, mix_inputs(signals, currents, args.samples), current_mix(signals, currents)[: args.samples],
                     args, sample_rate)


if __name__ == "__main__":
    sys.exit(main())
