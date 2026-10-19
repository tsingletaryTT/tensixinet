# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The shift register of ttsteam (Man With The Red Steam), apart from the snare it plays.

The register passes a pulse up a stack of steps on every clock. ttsteam uses the steps that fire to sound a
snare. Other devices read the same register: its hits can set off a gong or ping the AVDog's filter, accent
the ttrollz clock, and its steps can each hold a note, as the real register's stages hold a voltage. The
routes that patch it into other devices are listed in `patch.STEAM_ROUTES`.

The clock is the chip's eighth notes, the stack is as long as the chip has Tensix columns, and a step
fires where its column is harvested (its mask bit is 0). A chip with every column enabled never fires.
This module imports nothing from the devices, so a device can read the register without a circular import.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from .mapping import aiclk_to_bpm
from .patch import DEFAULT_PATCH, Patch
from ..telemetry import Telemetry

STEPS = 14  # Tensix columns
VALUE_STREAM = 0x57EA  # keeps the held notes' random stream apart from every other draw made from the patch seed


def step_pattern(tensix_mask: int) -> Tuple[bool, ...]:
    """Which of the 14 steps fire: those whose Tensix column is harvested (its mask bit is 0)."""
    return tuple(not (tensix_mask >> i) & 1 for i in range(STEPS))


def step_seconds(t: Telemetry) -> float:
    """One step of the register: an eighth note at the chip's tempo."""
    return 30.0 / aiclk_to_bpm(t.aiclk)


def step_times(t: Telemetry, duration: float) -> np.ndarray:
    """Onsets (s) of every step that fires before `duration`, the register looping round its 14 steps."""
    step_s = step_seconds(t)
    pattern = step_pattern(t.tensix_mask)
    return np.array([k * step_s for k in range(int(duration / step_s) + 1) if pattern[k % STEPS] and k * step_s < duration])


def step_values(seed: int) -> np.ndarray:
    """The value each step holds, in [0, 1): sampled noise, one per step, fixed for a patch so the register's lap repeats."""
    return np.random.default_rng([seed, VALUE_STREAM, 1]).random(STEPS)


def held_semitones(t: Telemetry, n: int, sample_rate: int, patch: Patch = DEFAULT_PATCH) -> np.ndarray:
    """For each of `n` samples, the semitone offset the register holds: its step's value taken as a note of the patch's
    scale, held until the next step. The register clocks every step, whether or not the step fires."""
    step = (np.arange(n) * (1.0 / sample_rate) // step_seconds(t)).astype(int) % STEPS
    scale = np.asarray(patch.scale)
    notes = scale[np.minimum((step_values(patch.seed) * len(scale)).astype(int), len(scale) - 1)]
    return notes[step]
