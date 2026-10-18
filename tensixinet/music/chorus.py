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

from .devices import ttquantus, current_mix, gong, ttdrone, ttrollz, ttsidrax, ultrasonic_shimmer
from .events import error_events
from .lead import lead
from .ttavdog import ttavdog
from .ttchoir import ttchoir
from .ttsteam import ttsteam
from .mapping import aiclk_to_bpm, chip_pressure, power_to_volume, turn_weight
from .patch import DEFAULT_PATCH, Patch
from ..telemetry import Telemetry
from .turns import soloists as plan_soloists, turn_gains

DEVICES = (ttrollz, ttsidrax, ttquantus, ttdrone, lead, gong, ultrasonic_shimmer, ttsteam, ttavdog, ttchoir)  # new devices go on the end, so slot numbers do not move


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


def slot_weights(telemetry: Sequence[Telemetry]) -> List[float]:
    """How often each slot solos. A slot's weight is `mapping.turn_weight` for its chip: the power it
    draws, raised by a heavier board load, by pressure near a limit and by processes using it, so
    busier chips solo more often."""
    n = len(telemetry)
    mean_board = sum(t.board_power for t in telemetry) / n
    chip_weight = [turn_weight(t.power, t.board_power, mean_board, chip_pressure(t), t.busy) for t in telemetry]
    return [chip_weight[slot_chip(s, n)] for s in range(n_slots(n))]


def plan_turns(telemetry: Sequence[Telemetry], duration: float, patch: Patch = DEFAULT_PATCH) -> Tuple[List[int], float]:
    """(soloist slot for each turn, turn length in seconds)."""
    turn_s = turn_seconds(telemetry, patch)
    return plan_soloists(slot_weights(telemetry), max(1, math.ceil(duration / turn_s))), turn_s


SEGMENT_SEED_STRIDE = 10_007  # a prime, so no device seed of one segment lands on another segment's


def render_chorus(
    telemetry: Sequence[Telemetry],
    duration: float,
    sample_rate: int = 44100,
    patch: Patch = DEFAULT_PATCH,
    segment: int = 0,
    soloists: Optional[Sequence[int]] = None,
    only: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """Render every device, give the soloist the floor, add each chip's event sounds, scale by power, mix by current.

    Live mode renders one segment at a time. `segment` moves every seed on by a fixed stride, so the
    segments are not copies of each other, and `soloists` names the slot (or slots, one per turn) that has the
    floor instead of planning it from the readings.

    `only` names the devices to play, for auditioning: the others are not rendered, and with no soloist the named
    devices play at full level with no turn-taking, so one device is heard as it is."""
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
    for s in range(n_slots(n)):
        if only is not None and assign_device(s).__name__ not in only:
            continue
        chip = slot_chip(s, n)
        per_chip[chip] += gains[s] * assign_device(s)(telemetry[chip], duration, sample_rate, seed=seed_base + s, patch=patch)
    signals = []
    for chip, t in enumerate(telemetry):
        voices = len(chip_devices(chip, n))
        events = error_events(t, duration, sample_rate, seed=seed_base + 1000 + chip)  # not part of the turns
        signals.append((per_chip[chip] / math.sqrt(voices) + events) * power_to_volume(t.power))
    return current_mix(signals, [t.current for t in telemetry]).astype(np.float32)
