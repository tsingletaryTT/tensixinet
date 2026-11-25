# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""A folder of patches and styles a listener can write, save and replay.

The layout is plain and deliberately small: `<library>/patches/<name>.json` holds the exact patch of a run, and
`<library>/styles/<name>.json` holds a style (see `styles.py`). The CLI's `--patch NAME` and `--style NAME` look here when a bare
name is given, and the TUI's `p` and `s` keys save here. Every file is ordinary JSON: rename it, copy it, or edit it by hand.

Names are file stems. A name with spaces becomes dashes, and anything that could escape the library is refused. Saving the same
name twice adds `-2`, `-3` and so on rather than overwriting a take you liked.
"""
from __future__ import annotations

import dataclasses
import re
import time
from pathlib import Path
from typing import Dict, Optional

from ..music.patch import Patch, save_patch
from ..music.styles import Style, load_style, save_style

DEFAULT_LIBRARY = "library"  # relative to where the program is run, like logs/ and audio/
PATCH_FOLDER = "patches"
STYLE_FOLDER = "styles"


def _stem(name: str) -> str:
    """A safe file stem: letters, numbers, dashes and underscores only."""
    if not isinstance(name, str) or name in ("", ".", "..") or "/" in name or "\\" in name:
        raise ValueError(f"invalid library name {name!r}: use a file name without a path")
    stem = re.sub(r"[^A-Za-z0-9_-]+", "-", name).strip("-")
    if not stem:
        raise ValueError(f"invalid library name {name!r}: use at least one letter or number")
    return stem


def _folder(root, kind: str) -> Path:
    return Path(root) / (PATCH_FOLDER if kind == "patch" else STYLE_FOLDER)


def patch_path(name: str, root=DEFAULT_LIBRARY) -> Path:
    """Where a named patch is kept."""
    return _folder(root, "patch") / f"{_stem(name)}.json"


def style_path(name: str, root=DEFAULT_LIBRARY) -> Path:
    """Where a named style is kept."""
    return _folder(root, "style") / f"{_stem(name)}.json"


def _next_path(path: Path) -> Path:
    """The first free path: `x.json`, then `x-2.json`, `x-3.json`, ... Nothing is overwritten."""
    if not path.exists():
        return path
    n = 2
    while path.with_name(f"{path.stem}-{n}{path.suffix}").exists():
        n += 1
    return path.with_name(f"{path.stem}-{n}{path.suffix}")


def save_named_patch(patch: Patch, name: Optional[str] = None, root=DEFAULT_LIBRARY,
                     stamp: Optional[str] = None, clock=time.strftime) -> Path:
    """Save `patch` in the library and return its path. With no name it gets a time stamp, so holding `p` makes a history."""
    if name is None:
        name = f"patch-{stamp or clock('%Y%m%d-%H%M%S')}"
    path = _next_path(patch_path(name, root))
    path.parent.mkdir(parents=True, exist_ok=True)
    save_patch(patch, path)
    return path


def save_named_style(style: Style, name: str, root=DEFAULT_LIBRARY) -> Path:
    """Save `style` in the library under `name` and return its path."""
    stem = _stem(name)
    path = _next_path(style_path(stem, root))
    save_style(dataclasses.replace(style, name=stem), path)  # a renamed file renames the style
    return path


def _file_order(path: Path):
    """Sort file stems so `x` comes before `x-2`; plain string order would put the dash first."""
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path.stem)]


def library_styles(root=DEFAULT_LIBRARY) -> Dict[str, Style]:
    """Every `*.json` style in the library, keyed by file name (not by the name inside the file)."""
    folder = _folder(root, "style")
    styles: Dict[str, Style] = {}
    if not folder.is_dir():
        return styles
    for path in sorted(folder.glob("*.json"), key=_file_order):
        styles[path.stem] = dataclasses.replace(load_style(path), name=path.stem)
    return styles


def resolve_patch(name_or_path: str, root=DEFAULT_LIBRARY) -> Path:
    """A path that exists is used as it is; otherwise the name is looked up in the library's patches folder."""
    path = Path(name_or_path)
    if path.is_file():
        return path
    saved = patch_path(name_or_path, root)
    if saved.is_file():
        return saved
    raise ValueError(f"cannot find patch {name_or_path!r} at that path or as {saved}; saved patches live in {_folder(root, 'patch')}")
