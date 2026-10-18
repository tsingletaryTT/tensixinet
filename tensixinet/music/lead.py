# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The lead voice: a short motif, shifted or inverted when the system's readings say so.

Shift and invert variation in the manner of the Cocoquantus, driven by
telemetry. There is no random generator. Each phrase (one bar) reads the next GDDR channel and compares it
with the average channel: a hotter channel shifts the motif up, a cooler one
inverts it, and a reading near the average repeats it. Flat readings loop.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np

from .mapping import aiclk_to_bpm, chip_pressure, personality, temp_to_pitch_hz
from .patch import DEFAULT_PATCH, SCALES, Patch

PENTATONIC = SCALES["major_pentatonic"]
MOTIF: Tuple[int, ...] = DEFAULT_PATCH.motif  # scale degrees, one per eighth note; a patch brings its own
REPEAT_BAND = 1.0  # deg C: closer to the average than this repeats the phrase
MAX_SHIFT = 4  # scale degrees


@dataclass(frozen=True)
class Transform:
    kind: str  # "repeat", "shift" or "invert"
    amount: int  # scale degrees for a shift; 0 otherwise


def degree_to_semitones(degree: int, scale: Sequence[int] = PENTATONIC) -> int:
    octave, step = divmod(degree, len(scale))
    return 12 * octave + scale[step]


def phrase_transform(delta_c: float) -> Transform:
    """What a channel reading `delta_c` degrees from the average does to a phrase."""
    if abs(delta_c) < REPEAT_BAND:
        return Transform("repeat", 0)
    if delta_c > 0:
        return Transform("shift", min(MAX_SHIFT, int(round(delta_c))))
    return Transform("invert", 0)


def apply_transform(motif: Sequence[int], tr: Transform) -> Tuple[int, ...]:
    if tr.kind == "shift":
        return tuple(d + tr.amount for d in motif)
    if tr.kind == "invert":
        root = motif[0]
        return tuple(2 * root - d for d in motif)
    return tuple(motif)


def lead_transforms(gddr_temps: Sequence[float], n_phrases: int) -> list:
    """One transform per phrase; phrase k reads channel k mod the channel count."""
    mean = sum(gddr_temps) / len(gddr_temps)
    return [phrase_transform(gddr_temps[k % len(gddr_temps)] - mean) for k in range(n_phrases)]


BRIGHTNESS = 0.6  # level of the octave overtone at full pressure


def identity_offset(identity: int) -> int:
    """0, 1 or 2 scale degrees up, from a chip's identity, so chips with different harvesting
    start their phrases on different notes."""
    return identity % 3


def _note(freq: float, n: int, sample_rate: int, brightness: float = 0.0) -> np.ndarray:
    """One plucked note that has fully ended inside its slot, so bars never bleed into each other.

    `brightness` (0-1) adds the octave overtone, while it stays under Nyquist, and the sum is scaled
    so a bright note is never louder than a plain one."""
    k = np.arange(n) / sample_rate
    slot = n / sample_rate
    env = np.exp(-k / (0.35 * slot)) * np.minimum(1.0, (slot - k) / (0.1 * slot))
    tone = np.sin(2 * math.pi * freq * k)
    if 2 * freq < 0.45 * sample_rate:
        tone = tone + BRIGHTNESS * brightness * np.sin(4 * math.pi * freq * k)
    return tone / (1 + BRIGHTNESS * brightness) * env


def lead(t, duration: float, sample_rate: int, seed: int, patch: Patch = DEFAULT_PATCH) -> np.ndarray:
    n_total = int(duration * sample_rate)
    bpm = aiclk_to_bpm(t.aiclk)
    eighth = int(round(30.0 / bpm * sample_rate))
    bar = eighth * len(patch.motif)
    base = temp_to_pitch_hz(t.asic_temperature)
    transforms = lead_transforms(t.gddr_temps, n_total // bar + 1)
    offset = identity_offset(personality(t))
    brightness = chip_pressure(t)
    out = np.zeros(n_total)
    for phrase, tr in enumerate(transforms):
        for i, degree in enumerate(apply_transform(tuple(d + offset for d in patch.motif), tr)):
            start = phrase * bar + i * eighth
            if start >= n_total:
                break
            freq = min(base * 2 ** (degree_to_semitones(degree, patch.scale) / 12), 0.45 * sample_rate)
            seg = _note(freq, eighth, sample_rate, brightness)[: n_total - start]
            out[start:start + len(seg)] += 0.8 * seg
    return out.astype(np.float32)
