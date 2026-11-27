# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The `--view-map` file: which saved patch goes with which tt-toplike view.

A view map is a small JSON object, for example ``{"defrag": "defrag"}``. The
values are saved patch names (`--library/patches/`) or paths; they are resolved
and loaded here, before any audio starts, so a typo is a command-line error and
not a surprise halfway through a run. Unknown views are left alone.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from .music.library import resolve_patch
from .music.patch import Patch, load_patch


def load_view_map(path, root) -> Dict[str, Patch]:
    """Read the map at `path`, resolving patch names under `root`. Raises `ValueError` with a readable reason."""
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, ValueError) as e:  # JSONDecodeError is a ValueError
        raise ValueError(f"cannot read view map {path}: {e}") from e
    if not isinstance(data, dict):
        raise ValueError(f"view map {path} must be a JSON object of view name to saved patch name")
    patches: Dict[str, Patch] = {}
    for view, name in data.items():
        if not isinstance(view, str):
            raise ValueError(f"view map {path}: view names must be strings, got {view!r}")
        if not isinstance(name, str):
            raise ValueError(f"view map {path}: view {view!r} must name a saved patch, got {name!r}")
        try:
            patches[view] = load_patch(resolve_patch(name, root=root))
        except ValueError as e:
            raise ValueError(f"view map {path}: {e}") from e
    return patches
