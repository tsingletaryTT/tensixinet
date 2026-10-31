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
    ("ttavdog.py", 80000, 1e-5),
    ("ttcoco.py", 24000, 1e-5),
    ("ttchoir.py", 48000, 1e-5),
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


@pytest.mark.parametrize("script,sample_rate", [("ttshock.py", 8000), ("ttshimmer.py", 44100), ("ttmix.py", 8000), ("ttgong.py", 8000), ("ttavdog.py", 8000),
                                                       ("ttchoir.py", 16000)])
def test_save_writes_what_the_kernel_computed_next_to_the_numpy_reference(tmp_path, script, sample_rate):
    import numpy as np
    import soundfile as sf
    code, result, done = run_kernel(script, "--save", str(tmp_path))
    name = script[:-3]
    assert code == 0, done.stderr[-600:]
    chip, rate = sf.read(str(tmp_path / f"{name}_kernel.wav"))
    ref, rate2 = sf.read(str(tmp_path / f"{name}_numpy.wav"))
    assert rate == rate2 == sample_rate == result["sample_rate"]
    assert len(chip) == len(ref) == result["samples"]
    assert np.abs(chip - ref).max() < 1e-3 + 1e-4  # 16-bit WAV rounding on top of the kernel's own difference
    assert np.abs(chip).max() > 0.01  # the file has sound in it
    assert result["run_ms"] > 0


def test_a_run_without_save_leaves_no_files(tmp_path):
    code, result, _ = run_kernel("ttshock.py")
    assert code == 0 and not list(Path.cwd().glob("ttshock_*.wav"))


def test_the_choir_kernel_also_matches_the_syllable_choir_and_names_the_run_apart():
    code, result, done = run_kernel("ttchoir.py", "--voice", "syllable")
    assert code == 0 and result is not None, done.stderr[-600:]
    assert result["kernel"] == "ttchoir_syllable" and result["ok"] and result["worst"] < 1e-5


def test_a_broken_choir_sum_fails_the_check(tmp_path):
    """The check is not on silence or on a sum that ignores the chip: the kernel's own output is what is compared."""
    broken = tmp_path / "ttchoir.py"
    src = (KERNELS / "ttchoir.py").read_text()
    assert src.count("ttl.math.sin(p7_blk)") == 1
    broken.write_text(src.replace("ttl.math.sin(p7_blk)", "ttl.math.cos(p7_blk)").replace(
        "Path(__file__).resolve().parent", f"Path({str(KERNELS)!r})"))
    env = {k: v for k, v in os.environ.items() if not k.startswith("TT_VISIBLE")}
    done = subprocess.run([SIM, str(broken)], capture_output=True, text=True, env=env, timeout=300)
    last = [l for l in done.stdout.splitlines() if l.startswith("{")]
    assert done.returncode == 1 and last and json.loads(last[-1])["ok"] is False, done.stderr[-600:]
