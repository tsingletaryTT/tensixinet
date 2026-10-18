# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Repo rules that must hold for every module, checked by reading the source."""
import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "tensixinet"
# Importing any of these opens a Tenstorrent device (ttnn, ttl, ttml) or needs a sound card.
FORBIDDEN = {"ttnn", "ttl", "ttml", "torch", "sounddevice"}
SOURCES = sorted(PACKAGE.rglob("*.py")) + sorted((ROOT / "tests").rglob("*.py"))


def imported_roots(path):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node.module.split(".")[0]


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: str(p.relative_to(ROOT)))
def test_no_module_can_open_a_device_by_importing(path):
    if path == Path(__file__).resolve():
        pytest.skip("this file names the forbidden modules")
    assert not FORBIDDEN & set(imported_roots(path))


def test_every_package_directory_has_an_init():
    dirs = [d for d in PACKAGE.rglob("*") if d.is_dir() and d.name != "__pycache__"]
    assert dirs and all((d / "__init__.py").exists() for d in dirs)


def test_every_module_has_a_test_in_the_matching_folder():
    missing = []
    for mod in PACKAGE.rglob("*.py"):
        if mod.name in ("__init__.py", "__main__.py"):
            continue
        rel = mod.relative_to(PACKAGE)
        candidates = [ROOT / "tests" / rel.parent / f"test_{mod.stem}.py"]
        if not any(c.exists() for c in candidates):
            missing.append(str(rel))
    assert not missing, f"no tests/<same folder>/test_<name>.py for: {missing}"


# --- licensing -------------------------------------------------------------------

SPDX = (
    "# SPDX-License-Identifier: Apache-2.0",
    "# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.",
)
LICENSED = sorted(
    list(PACKAGE.rglob("*.py")) + list((ROOT / "tests").rglob("*.py")) + list((ROOT / "legacy").glob("*.py"))
    + list((ROOT / "demo").glob("*.sh"))
)


@pytest.mark.parametrize("path", LICENSED, ids=lambda p: str(p.relative_to(ROOT)))
def test_every_source_file_carries_the_spdx_header(path):
    lines = path.read_text().splitlines()
    if lines and lines[0].startswith("#!"):
        lines = lines[1:]  # the header goes right after a shebang
    assert tuple(lines[:2]) == SPDX, f"{path.relative_to(ROOT)} needs the two SPDX lines at the top"


def test_the_license_files_are_present_and_are_apache_2():
    text = (ROOT / "LICENSE").read_text()
    assert "Apache License" in text and "Version 2.0, January 2004" in text
    assert (ROOT / "LICENSE_understanding.txt").read_text().strip()
