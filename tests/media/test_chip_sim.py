# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The kernels as the chip worker runs them in the TT-Lang simulator, through the pool, with no chip and no lease.

Run only when TT_LANG_SIM names a `tt-lang-sim` from an environment with no real `ttnn` (see kernels/README.md). The simulator's
arithmetic is the numpy formulas' to about 1e-6, so a kernel's output is held to the host path's with a little room."""
import os

import numpy as np
import pytest

from tensixinet.media.chip import CHIP_OPS, ChipWorker, simulator_command
from tests.media.test_chip import EXPECTED, N, _inputs

SIM = os.environ.get("TT_LANG_SIM")
pytestmark = [pytest.mark.sim, pytest.mark.skipif(not SIM, reason="set TT_LANG_SIM to a tt-lang-sim with no real ttnn")]


@pytest.fixture(scope="module")
def worker():
    with ChipWorker(simulator_command(4, exe=SIM, chunk_tiles=8), devices=4) as w:
        yield w


@pytest.mark.parametrize("op", sorted(CHIP_OPS))
def test_every_kernel_run_in_the_simulator_matches_the_hosts_formula(worker, op):
    inputs = _inputs(op)
    want = np.asarray(EXPECTED[op](inputs, N), dtype=np.float64)
    got = worker.run(op, inputs, N, device=0)
    assert got.shape == (N,) and np.max(np.abs(got - want)) < 1e-4


def test_the_simulated_chips_on_both_boards_answer(worker):
    inputs = _inputs("ttmix")
    outs = [worker.run("ttmix", inputs, N, device=d) for d in range(4)]
    assert all(np.allclose(outs[0], o) for o in outs[1:])
