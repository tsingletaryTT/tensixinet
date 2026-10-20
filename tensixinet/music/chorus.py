# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""A chorus of Ciat-Lonbarde-style devices, dealt out across any number of chips.

Every device is a slot. There are always at least len(DEVICES) slots, dealt to the
chips in turn (slot % chips), so a 1-chip box plays the whole ensemble and a
chip-rich box plays one device per chip. Players trade solos: each turn one slot
is featured and the others drop to a lower level, and the chips doing the most
work get the floor more often.
"""
from __future__ import annotations

import math
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from .cables import Cable, apply_cables, draw_cables
from .devices import ttquantus, current_mix, gong, ttdrone, ttrollz, ttrax, ultrasonic_shimmer
from .events import error_events
from .lead import lead
from .ttavdog import ttavdog
from .ttchoir import ttchoir
from .ttcoco import ttcoco
from .ttsteam import ttsteam
from .mapping import aiclk_to_bpm, chip_chaos, chip_pressure, power_to_volume, turn_weight
from .patch import DEFAULT_PATCH, Patch
from .ttrolzer import SEGMENT_SEED_STRIDE  # noqa: F401  (the chorus uses it; devices read the segment back out of their seed)
from ..telemetry import Telemetry
from .turns import soloists as plan_soloists, turn_gains

DEVICES = (ttrollz, ttrax, ttquantus, ttdrone, lead, gong, ultrasonic_shimmer, ttsteam, ttavdog, ttchoir, ttcoco)  # new devices go on the end, so slot numbers do not move


def n_slots(n_chips: int) -> int:
    return max(n_chips, len(DEVICES))


def assign_device(slot: int) -> Callable:
    return DEVICES[slot % len(DEVICES)]


def slot_chip(slot: int, n_chips: int) -> int:
    return slot % n_chips


def device_names() -> List[str]:
    return [d.__name__ for d in DEVICES]


def chip_devices(chip_index: int, n_chips: int) -> List[Callable]:
    """The devices this chip plays."""
    return [assign_device(s) for s in range(n_slots(n_chips)) if slot_chip(s, n_chips) == chip_index]


def turn_seconds(telemetry: Sequence[Telemetry], patch: Patch = DEFAULT_PATCH) -> float:
    """A turn lasts `patch.bars_per_turn` bars at the average chip tempo."""
    mean_bpm = sum(aiclk_to_bpm(t.aiclk) for t in telemetry) / len(telemetry)
    return patch.bars_per_turn * 4 * 60.0 / mean_bpm


def emphasis_of(patch: Patch, device_name: str) -> float:
    """The patch's factor for a device (a style's balance of the band), 1 when it names none."""
    return dict(patch.emphasis).get(device_name, 1.0)


def slot_weights(telemetry: Sequence[Telemetry], patch: Patch = DEFAULT_PATCH) -> List[float]:
    """How often each slot solos. A slot's weight is `mapping.turn_weight` for its chip: the power it
    draws, raised by a heavier board load, by pressure near a limit and by processes using it, so
    busier chips solo more often. The patch's emphasis scales a device's share (a style's balance)."""
    n = len(telemetry)
    mean_board = sum(t.board_power for t in telemetry) / n
    chip_weight = [turn_weight(t.power, t.board_power, mean_board, chip_pressure(t), t.busy) for t in telemetry]
    return [chip_weight[slot_chip(s, n)] * emphasis_of(patch, assign_device(s).__name__) for s in range(n_slots(n))]


def plan_turns(telemetry: Sequence[Telemetry], duration: float, patch: Patch = DEFAULT_PATCH) -> Tuple[List[int], float]:
    """(soloist slot for each turn, turn length in seconds)."""
    turn_s = turn_seconds(telemetry, patch)
    return plan_soloists(slot_weights(telemetry, patch), max(1, math.ceil(duration / turn_s))), turn_s




def render_chorus(
    telemetry: Sequence[Telemetry],
    duration: float,
    sample_rate: int = 44100,
    patch: Patch = DEFAULT_PATCH,
    segment: int = 0,
    soloists: Optional[Sequence[int]] = None,
    only: Optional[Sequence[str]] = None,
    cables: Optional[Sequence[Cable]] = None,
    backend=None,
) -> np.ndarray:
    """Render every device, give the soloist the floor, add each chip's event sounds, scale by power, mix by current.

    Live mode renders one segment at a time. `segment` moves every seed on by a fixed stride, so the
    segments are not copies of each other, and `soloists` names the slot (or slots, one per turn) that has the
    floor instead of planning it from the readings.

    `only` names the devices to play, for auditioning: the others are not rendered, and with no soloist the named
    devices play at full level with no turn-taking, so one device is heard as it is.

    `cables` are the patch cables between devices (`cables.py`). Left as None they are drawn from the patch seed, which
    live mode keeps for the whole run, so the wiring does not change from segment to segment. An empty list leaves the
    devices unpatched. The cables act on the voices as the devices made them, before the turn gains, so a device drives
    its cables whether or not it has the floor.

    `backend` (a `tensixinet.kernels.backend.KernelBackend`) runs the devices that have TT-Lang kernels, and the mixer, on a
    chip; the other devices stay on the CPU. Left as None everything is numpy."""
    if not telemetry:
        raise ValueError("render_chorus needs at least one chip's telemetry")
    if only is not None:
        if not list(only):
            raise ValueError("only needs at least one device name")
        unknown = [name for name in only if name not in device_names()]
        if unknown:
            raise ValueError(f"unknown device {', '.join(unknown)}; the devices are {', '.join(device_names())}")
    n = len(telemetry)
    n_samples = int(duration * sample_rate)
    planned, turn_s = plan_turns(telemetry, duration, patch)
    solo_slots = planned if soloists is None else list(soloists)
    seed_base = segment * SEGMENT_SEED_STRIDE
    gains = turn_gains(solo_slots, n_slots(n), n_samples, sample_rate, turn_s, floor=patch.floor)
    if only is not None:
        gains = np.ones_like(gains)
    per_chip = [np.zeros(n_samples) for _ in range(n)]
    slots = range(n_slots(n))
    def device_for(slot: int) -> Callable:
        """The device this slot plays: the chip's own kernels' version when a backend has one, else the numpy device."""
        device = assign_device(slot)
        on_chip = backend.device_for(device.__name__, slot_chip(slot, n)) if backend else None
        return on_chip or device

    voices: List[Optional[np.ndarray]] = [
        None if only is not None and assign_device(s).__name__ not in only
        else device_for(s)(telemetry[slot_chip(s, n)], duration, sample_rate, seed=seed_base + s, patch=patch)
        for s in slots
    ]
    wiring = draw_cables(patch.seed, device_names()) if cables is None else list(cables)
    voices = apply_cables(
        voices, [assign_device(s).__name__ for s in slots], wiring,
        [chip_chaos(telemetry[slot_chip(s, n)]) for s in slots], sample_rate,
    )
    for s in slots:
        if voices[s] is not None:
            balance = 1.0 if only is not None else emphasis_of(patch, assign_device(s).__name__)  # auditioning plays a device as it is
            per_chip[slot_chip(s, n)] += balance * gains[s] * voices[s]
    signals = []
    for chip, t in enumerate(telemetry):
        voices = len(chip_devices(chip, n))
        events = error_events(t, duration, sample_rate, seed=seed_base + 1000 + chip)  # not part of the turns
        signals.append((per_chip[chip] / math.sqrt(voices) + events) * power_to_volume(t.power))
    mix = backend.mix if backend else current_mix
    return mix(signals, [t.current for t in telemetry]).astype(np.float32)
