# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The app's form, as plain functions: what the choices are and how a set of choices becomes a session. No UI library, so it is tested alone."""
from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

from ..media.chip import ChipWorker, find_simulator, simulator_command
from ..music.patch import Patch, load_patch
from ..music.styles import STYLES
from .session import Session, make_patch, mock_sample, real_sample

SOURCES = ("synthetic chips", "the real chips (tt-smi, read-only)")
CPU = "CPU only"
SIMULATED = "TT-Lang kernels in the simulator (no chip)"
DRAW = "draw from the seed and the style"
PLAIN = "plain draw"


def source_choices(which: Callable = shutil.which) -> List[str]:
    """The sources this machine can offer: the real chips need `tt-smi`."""
    return list(SOURCES) if which("tt-smi") else [SOURCES[0]]


def kernel_choices(find: Callable = find_simulator) -> List[str]:
    """Where the devices that have TT-Lang kernels run: on the CPU, or (when `tt-lang-sim` is there) through the simulator."""
    return [CPU, SIMULATED] if find() else [CPU]


def default_worker(chips: int) -> ChipWorker:
    return ChipWorker(simulator_command(chips), devices=chips)


def style_choices() -> List[str]:
    return [PLAIN] + list(STYLES)


def patch_choices(dirs: Sequence[Path]) -> List[str]:
    """The seeded draw, then every `*.patch.json` in `dirs` by file name."""
    names = sorted({f.name for d in dirs if Path(d).is_dir() for f in Path(d).glob("*.patch.json")})
    return [DRAW] + names


def new_session(
    source: str,
    style: str,
    seed: int,
    chips: int,
    load_s: float,
    patch_name: str,
    patch_dirs: Sequence[Path],
    sleep: Callable[[float], None],
    stereo: bool = True,
    evict: bool = True,
    cables: bool = True,
    real: Callable = real_sample,
    clock: Callable[[], float] = time.monotonic,
    kernels: str = CPU,
    worker_factory: Callable = default_worker,
    lead_s: float = 0.0,
) -> Tuple[Session, Patch]:
    """A session and its patch from the form's values. `source` is one of SOURCES; synthetic chips cycle through load every `load_s`
    seconds when that is above 0 (so an eviction can be seen); `kernels` SIMULATED runs the devices that have TT-Lang kernels through a worker
    under the TT-Lang simulator, one pretend chip for each chip in the first reading; `lead_s` is the session's lead (see `Session`); a saved patch is used as it is and the style and the seed are ignored."""
    if source == SOURCES[0]:
        sample = mock_sample(int(chips), seed=0, load_period=load_s if load_s and load_s > 0 else None, clock=clock)
        label = f"{int(chips)} synthetic chips"
    elif source == SOURCES[1]:
        sample = real()
        label = "the real chips"
    else:
        raise ValueError(f"unknown source {source!r}; the sources are {', '.join(SOURCES)}")
    first = sample()
    if not first:
        raise ValueError("no chips in the first reading, so there is nothing to play")
    if patch_name and patch_name != DRAW:
        found = [Path(d) / patch_name for d in patch_dirs if (Path(d) / patch_name).is_file()]
        if not found:
            raise ValueError(f"no saved patch named {patch_name!r}")
        patch = load_patch(found[0])
    else:
        patch = make_patch("" if style in ("", PLAIN) else style, int(seed), first)
    worker = worker_factory(len(first)) if kernels == SIMULATED else None  # started by the session's thread, closed when it ends
    return Session(sample, patch, source=label, stereo=stereo, evict=evict, cables=cables, sleep=sleep, worker=worker, lead_s=lead_s), patch
