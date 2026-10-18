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
from typing import Callable, List, Sequence, Tuple

import numpy as np

from .devices import cocoquantus, current_mix, gong, plumbutter_drone, rollz, sidrax, ultrasonic_shimmer
from .lead import lead
from .mapping import aiclk_to_bpm, power_to_volume
from .patch import DEFAULT_PATCH, Patch
from ..telemetry import Telemetry
from .turns import soloists, turn_gains

DEVICES = (rollz, sidrax, cocoquantus, plumbutter_drone, lead, gong, ultrasonic_shimmer)


def n_slots(n_chips: int) -> int:
    return max(n_chips, len(DEVICES))


def assign_device(slot: int) -> Callable:
    return DEVICES[slot % len(DEVICES)]


def slot_chip(slot: int, n_chips: int) -> int:
    return slot % n_chips


def chip_devices(chip_index: int, n_chips: int) -> List[Callable]:
    """The devices this chip plays."""
    return [assign_device(s) for s in range(n_slots(n_chips)) if slot_chip(s, n_chips) == chip_index]


def plan_turns(telemetry: Sequence[Telemetry], duration: float, patch: Patch = DEFAULT_PATCH) -> Tuple[List[int], float]:
    """(soloist slot for each turn, turn length in seconds).

    A turn lasts `patch.bars_per_turn` bars at the average chip tempo. A slot's weight is
    the power its chip draws, so busier chips solo more often.
    """
    n = len(telemetry)
    mean_bpm = sum(aiclk_to_bpm(t.aiclk) for t in telemetry) / n
    turn_s = patch.bars_per_turn * 4 * 60.0 / mean_bpm
    weights = [telemetry[slot_chip(s, n)].power for s in range(n_slots(n))]
    return soloists(weights, max(1, math.ceil(duration / turn_s))), turn_s


def render_chorus(
    telemetry: Sequence[Telemetry], duration: float, sample_rate: int = 44100, patch: Patch = DEFAULT_PATCH
) -> np.ndarray:
    """Render every device, give the soloist the floor, scale each chip by power, mix by current."""
    if not telemetry:
        raise ValueError("render_chorus needs at least one chip's telemetry")
    n = len(telemetry)
    n_samples = int(duration * sample_rate)
    solo_slots, turn_s = plan_turns(telemetry, duration, patch)
    gains = turn_gains(solo_slots, n_slots(n), n_samples, sample_rate, turn_s, floor=patch.floor)
    per_chip = [np.zeros(n_samples) for _ in range(n)]
    for s in range(n_slots(n)):
        chip = slot_chip(s, n)
        per_chip[chip] += gains[s] * assign_device(s)(telemetry[chip], duration, sample_rate, seed=s, patch=patch)
    signals = []
    for chip, t in enumerate(telemetry):
        voices = len(chip_devices(chip, n))
        signals.append(per_chip[chip] / math.sqrt(voices) * power_to_volume(t.power))
    return current_mix(signals, [t.current for t in telemetry]).astype(np.float32)
