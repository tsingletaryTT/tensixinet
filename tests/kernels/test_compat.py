# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The compatibility runtime root (`kernels/compat/make_runtime_root.sh`): a tree of symlinks to tt-metal with two headers replaced, so TT-Lang's
generated kernels compile against a tt-metal that removed the names they call. These tests build it from a small fake tt-metal; that the kernels
then compile and run is checked on a chip (`kernels/README.md`)."""
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "kernels" / "compat" / "make_runtime_root.sh"
COMPUTE = Path("tt_metal/hw/inc/api/compute")


@pytest.fixture()
def fake_metal(tmp_path):
    src = tmp_path / "tt-metal"
    (src / COMPUTE).mkdir(parents=True)
    (src / COMPUTE / "eltwise_binary.h").write_text("// the real header\n")
    (src / COMPUTE / "common.h").write_text("// another header\n")
    (src / "tt_metal" / "hw" / "ckernels").mkdir(parents=True)
    (src / "tools").mkdir()
    (src / "tools" / "scripts").mkdir()
    (src / "README.md").write_text("top level\n")
    return src


def build(src, out):
    return subprocess.run(["bash", str(SCRIPT), str(out)], env=dict(os.environ, TT_METAL_HOME=str(src)), capture_output=True, text=True)


def test_the_binary_header_is_replaced_by_one_that_includes_the_real_one_and_adds_the_removed_names(fake_metal, tmp_path):
    out = tmp_path / "root"
    assert build(fake_metal, out).returncode == 0
    wrapper = out / COMPUTE / "eltwise_binary.h"
    text = wrapper.read_text()
    assert wrapper.is_file() and not wrapper.is_symlink()
    assert f'#include "{fake_metal / COMPUTE / "eltwise_binary.h"}"' in text  # the real header, by its full path (the wrapper hides its own name)
    for name in ("binary_op_init_common", "add_tiles_init", "sub_tiles_init", "mul_tiles_init"):
        assert f"{name}(" in text
    assert "compute_kernel_hw_startup(icb0, icb1, ocb)" in text
    for op in ("ELWADD", "ELWSUB", "ELWMUL"):
        assert f"EltwiseBinaryType::{op}" in text
    assert "namespace ckernel {" in text  # the generated kernels call these unqualified from inside ckernel


def test_every_other_file_stays_a_link_to_the_real_tree(fake_metal, tmp_path):
    out = tmp_path / "root"
    assert build(fake_metal, out).returncode == 0
    assert (out / COMPUTE / "common.h").is_symlink() and (out / COMPUTE / "common.h").resolve() == (fake_metal / COMPUTE / "common.h").resolve()
    assert (out / "README.md").is_symlink()
    assert (out / "tools" / "scripts").is_symlink()
    assert (out / "tt_metal" / "hw" / "ckernels").is_symlink()


def test_the_old_profiler_header_path_exists_and_points_at_the_new_one(fake_metal, tmp_path):
    out = tmp_path / "root"
    assert build(fake_metal, out).returncode == 0  # the fake tree has no tools/profiler, as tt-metal no longer does
    header = out / "tools" / "profiler" / "kernel_profiler.hpp"
    assert header.is_file() and '#include "api/debug/kernel_profiler.hpp"' in header.read_text()


def test_building_again_replaces_the_old_root_and_leaves_the_real_tree_alone(fake_metal, tmp_path):
    out = tmp_path / "root"
    assert build(fake_metal, out).returncode == 0 and build(fake_metal, out).returncode == 0
    assert (fake_metal / COMPUTE / "eltwise_binary.h").read_text() == "// the real header\n"
    (out / "stale.txt").write_text("left from an older root\n")
    assert build(fake_metal, out).returncode == 0 and not (out / "stale.txt").exists()
    assert not (fake_metal / "tools" / "profiler").exists()


def test_without_a_tt_metal_it_says_where_it_looked_and_writes_nothing(tmp_path):
    out = tmp_path / "root"
    r = build(tmp_path / "nowhere", out)
    assert r.returncode != 0 and "no tt-metal at" in r.stderr and "TT_METAL_HOME" in r.stderr and not out.exists()
