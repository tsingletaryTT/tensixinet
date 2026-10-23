# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""What the Play tab plays: one device, a module (a named group of devices), or the whole system.

A scope is a string the page can hold in a dropdown: `"system"`, `"module:<name>"` or `"device:<name>"`. `resolve_scope` turns it into the
`only` list that `render_chorus` takes (`None` for the system, which plays the whole chorus with its cables, coco bus, eviction and turn-taking).
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from ..music.chorus import assign_device, device_names, n_slots, slot_chip

SYSTEM = "the whole system"

MODULES = {
    "rhythm section": ("ttrollz", "gong", "ttavdog", "ttsteam"),
    "drones and air": ("ttdrone", "ultrasonic_shimmer", "ttchoir"),
    "melody": ("lead", "ttquantus", "ttrax"),
    "delay and echo": ("ttquantus", "ttcoco"),
}


def scope_choices() -> List[Tuple[str, str]]:
    """(label, value) for a dropdown: the system, then the modules, then each device."""
    return ([(SYSTEM, "system")]
            + [(f"module: {name}", f"module:{name}") for name in MODULES]
            + [(f"device: {name}", f"device:{name}") for name in device_names()])


def resolve_scope(value: str) -> Optional[List[str]]:
    """The devices a scope plays, as `render_chorus(only=...)` takes them: None for the system. `ValueError` for anything else."""
    if value == "system":
        return None
    kind, _, name = value.partition(":")
    if kind == "module" and name in MODULES:
        return list(MODULES[name])
    if kind == "device" and name in device_names():
        return [name]
    raise ValueError(f"unknown scope {value!r}; use 'system', 'module:<name>' ({', '.join(MODULES)}) or 'device:<name>' ({', '.join(device_names())})")


def heard_on(only: Optional[Sequence[str]], n_chips: int) -> List[str]:
    """The devices of a scope that are dealt to at least one of `n_chips` chips, in the scope's order (all of them, for the system).

    `n_slots` is never fewer than the number of devices, so today every device is dealt even on one chip. This asks the dealing itself, so it
    stays right if that changes."""
    if n_chips < 1:
        raise ValueError("at least one chip is needed")
    dealt = {assign_device(s).__name__ for s in range(n_slots(n_chips)) if slot_chip(s, n_chips) < n_chips}
    wanted = device_names() if only is None else list(only)
    return [name for name in wanted if name in dealt]
