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
