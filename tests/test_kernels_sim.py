# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The kernels in `kernels/`, run in the TT-Lang simulator.

The simulator needs no chip and no lease, but the `tt-lang-sim` that comes with the full TT-Lang install can reach
the real `ttnn`. So these run only when TT_LANG_SIM names a `tt-lang-sim` from an environment that has no `ttnn` at
all (`pip install tt-lang-sim` into a fresh venv). The tests start it as a subprocess and never import `ttl`.
"""
import json
import os
import subprocess
from pathlib import Path

import pytest

SIM = os.environ.get("TT_LANG_SIM")
KERNELS = Path(__file__).parent.parent / "kernels"

pytestmark = [pytest.mark.sim, pytest.mark.skipif(not SIM, reason="set TT_LANG_SIM to a tt-lang-sim with no real ttnn")]


def run_kernel(name, *args):
    env = {k: v for k, v in os.environ.items() if not k.startswith("TT_VISIBLE")}
    done = subprocess.run([SIM, str(KERNELS / name), *args], capture_output=True, text=True, env=env, timeout=300)
    last = [l for l in done.stdout.splitlines() if l.startswith("{")]
    return done.returncode, (json.loads(last[-1]) if last else None), done


KERNELS_UNDER_TEST = [
    # (script, samples it checks by default, the worst difference the simulator may show)
    ("ttshock.py", 32000, 1e-5),
    ("ttshimmer.py", 44100, 1e-5),
    ("ttmix.py", 24000, 1e-6),
    ("ttgong.py", 72000, 1e-5),
]


@pytest.mark.parametrize("script,samples,worst", KERNELS_UNDER_TEST)
def test_each_kernel_matches_the_numpy_device_it_ports(script, samples, worst):
    code, result, done = run_kernel(script)
    assert code == 0 and result is not None, done.stderr[-600:]
    assert result["ok"] and result["simulator"] and result["samples"] == samples and result["worst"] < worst


@pytest.mark.parametrize("script", [k[0] for k in KERNELS_UNDER_TEST])
def test_a_length_that_is_not_a_whole_number_of_tiles_still_matches(script):
    code, result, done = run_kernel(script, "--samples", "5000")
    assert code == 0 and result["ok"] and result["samples"] == 5000, done.stderr[-600:]


@pytest.mark.parametrize("script", [k[0] for k in KERNELS_UNDER_TEST])
def test_a_tolerance_the_kernel_cannot_meet_fails_the_run(script):
    code, result, _ = run_kernel(script, "--atol", "1e-12")
    assert code == 1 and result["ok"] is False


def test_every_kernel_script_is_under_test():
    scripts = {p.name for p in KERNELS.glob("tt*.py")}
    assert scripts == {k[0] for k in KERNELS_UNDER_TEST}
