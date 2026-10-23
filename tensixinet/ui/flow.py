# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""What is playing right now, as plain data: the chips, the devices dealt to them, the cables between the devices and where each voice sits.

`build_flow` takes what the chorus used for one segment (the readings, the patch, who had the floor) and works out the same
things the chorus works out, from the same functions, so the picture cannot drift from the sound. It draws nothing (see chart.py)
and imports no UI library.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from ..music.cables import draw_cables, strength
from ..music.chorus import assign_device, device_names, n_slots, slot_chip, slot_weights
from ..music.evict import EVICT_SECONDS, FILL_SECONDS
from ..music.mapping import chip_chaos, chip_pressure
from ..music.patch import Patch
from ..music.space import chip_azimuth, device_space
from ..telemetry import Telemetry


@dataclass(frozen=True)
class ChipView:
    index: int
    temp_c: float
    power_w: float
    pressure: float  # 0 to 1: how close the chip is to its throttle point or power limit
    chaos: float  # 0 to 1: pressure plus how fast the readings move
    azimuth: float  # the chip's home in the stereo field, -1 (left) to +1 (right)
    phase: str  # running, evicting, idle or filling (see music/evict.py)


@dataclass(frozen=True)
class DeviceView:
    name: str
    slot: int
    chip: int
    soloist: bool  # has the floor this segment
    share: float  # its turn weight over all the weights
    azimuth: float
    depth: float  # 0 near to 1 far
    heard: bool = True  # false for a device the Play tab is not playing (its scope leaves it out): drawn faint


@dataclass(frozen=True)
class CableView:
    source: str
    target: str
    kind: str  # fm, crunch, inflate or fold
    depth: float
    push: float  # how hard it pushes now: its depth, scaled by the chaos of the chip that plays the source


@dataclass(frozen=True)
class FlowState:
    segment: int
    seconds: float
    chips: Tuple[ChipView, ...]
    devices: Tuple[DeviceView, ...]
    cables: Tuple[CableView, ...]
    soloist_device: str  # "" when nobody has the floor (a scope that plays some devices at full level)
    soloist_chip: int  # -1 then
    patch: dict


def build_flow(
    chips: Sequence[Telemetry],
    patch: Patch,
    segment: int,
    soloist_slot: Optional[int],
    seconds: float,
    phases: Dict[int, str] = None,
    cables: bool = True,
    heard: Optional[Sequence[str]] = None,
) -> FlowState:
    """The state of one segment. `phases` maps a chip index to a phase other than "running"; `soloist_slot` is None when nobody has the floor;
    `heard` names the devices that sound (None: all of them)."""
    n = len(chips)
    phases = phases or {}
    chip_views = tuple(
        ChipView(i, t.asic_temperature, t.power, chip_pressure(t), chip_chaos(t), chip_azimuth(i, n, t), phases.get(i, "running"))
        for i, t in enumerate(chips)
    )
    weights = slot_weights(chips, patch)
    total = sum(weights)
    devices: List[DeviceView] = []
    for slot in range(n_slots(n)):
        name = assign_device(slot).__name__
        chip = slot_chip(slot, n)
        offset, _, depth = device_space(name)
        azimuth = max(-1.0, min(1.0, chip_views[chip].azimuth + offset))
        devices.append(DeviceView(name, slot, chip, slot == soloist_slot, weights[slot] / total, azimuth, depth, heard is None or name in heard))
    chip_of = {d.name: d.chip for d in devices}
    wiring = draw_cables(patch.seed, device_names()) if cables else []
    cable_views = tuple(
        CableView(w.source, w.target, w.kind, w.depth, strength(w, chip_views[chip_of[w.source]].chaos)) for w in wiring
    )
    solo = next((d for d in devices if d.slot == soloist_slot), None)
    summary = {
        "seed": patch.seed, "style": patch.style, "scale": patch.scale_name, "gong": patch.gong_tuning, "rax_bars": patch.rax_bars,
        "routes": tuple(patch.steam_routes), "bars_per_turn": patch.bars_per_turn, "floor": patch.floor, "mode": patch.mode,
    }
    return FlowState(segment, seconds, chip_views, tuple(devices), cable_views, solo.name if solo else "", solo.chip if solo else -1, summary)


def phases_at(events: Sequence[Tuple[float, str, int]], now: float, n_chips: int) -> Dict[int, str]:
    """Each chip's phase at `now`, from the log's (time, "evict" or "fill", chip) events. A chip that has had no event is running and
    left out. An eviction dissolves for EVICT_SECONDS and then the chip is idle; a fill lasts FILL_SECONDS and then it is running."""
    phases: Dict[int, str] = {}
    for chip in range(n_chips):
        last = [(t, kind) for t, kind, c in events if c == chip and t <= now]
        if not last:
            continue
        t, kind = max(last, key=lambda e: e[0])
        if kind == "evict":
            phases[chip] = "evicting" if now - t < EVICT_SECONDS else "idle"
        elif now - t < FILL_SECONDS:
            phases[chip] = "filling"
    return phases
