"""Multi-voice synthesis for any number of chips.

Each chip owns `voices_per_chip` voices and drives them with its own telemetry.
Nothing here assumes a chip count: 1 chip, 4 chips and 32 chips take the same
code path. Pure numpy; no device access.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from ..telemetry import Telemetry

# Semitone offsets of a C major triad; voice k takes the next chord tone upward.
_TRIAD = (0, 4, 7)
C2_HZ = 65.41
PEAK = 0.7  # normalised output peak


@dataclass(frozen=True)
class Voice:
    chip_id: int
    base_freq: float  # Hz
    lfo_rate: float  # Hz, slow pitch wobble


def build_voices(
    n_chips: int,
    voices_per_chip: int = 4,
    f_low: float = C2_HZ,
    f_high: float = 4186.0,
) -> List[Voice]:
    """Lay out `n_chips * voices_per_chip` voices, chip by chip.

    Voices climb a C major arpeggio from `f_low`. A note that would pass
    `f_high` is folded down whole octaves, so any system size stays in band.
    """
    if n_chips < 1:
        raise ValueError(f"n_chips must be >= 1, got {n_chips}")
    voices = []
    for k in range(n_chips * voices_per_chip):
        semitones = 12 * (k // len(_TRIAD)) + _TRIAD[k % len(_TRIAD)]
        freq = f_low * 2 ** (semitones / 12)
        while freq > f_high:
            freq /= 2
        voices.append(
            Voice(
                chip_id=k // voices_per_chip,
                base_freq=freq,
                lfo_rate=0.1 + 0.1 * (k % 5),  # 0.1-0.5 Hz
            )
        )
    return voices


def render(
    voices: Sequence[Voice],
    telemetry: Sequence[Telemetry],
    duration: float,
    sample_rate: int = 44100,
    drift: float = 0.1,
) -> np.ndarray:
    """Render `duration` seconds. `telemetry[i]` drives the voices of chip i.

    Pitch is shifted by `drift * temp_norm` where temp_norm maps 25-100 C to 0-1.
    Phase is the running integral of frequency, so a changing frequency does not
    smear the tone the way `sin(2*pi*f(t)*t)` does.
    """
    for v in voices:
        if not 0 <= v.chip_id < len(telemetry):
            raise ValueError(
                f"voice targets chip {v.chip_id} but only {len(telemetry)} telemetry readings given"
            )
    n = int(duration * sample_rate)
    t = np.arange(n) / sample_rate
    envelope = np.exp(-t / 30.0) * (1 + 0.1 * np.sin(2 * math.pi * 0.05 * t))
    audio = np.zeros(n)
    nyquist = sample_rate / 2
    for v in voices:
        temp_norm = (telemetry[v.chip_id].asic_temperature - 25.0) / 75.0
        freq = v.base_freq * (1 + drift * temp_norm) * (1 + 0.02 * np.sin(2 * math.pi * v.lfo_rate * t))
        phase = 2 * math.pi * np.cumsum(freq) / sample_rate
        wave = 0.5 * np.sin(phase) + 0.1 * np.sin(0.5 * phase)
        # Upper partials only while they stay below Nyquist, otherwise they alias.
        wave += np.where(2 * freq < nyquist, 0.2 * np.sin(2 * phase), 0.0)
        wave += np.where(1.5 * freq < nyquist, 0.05 * np.sin(1.5 * phase), 0.0)
        audio += wave * envelope
    peak = np.max(np.abs(audio)) if n else 0.0
    if peak > 0:
        audio = audio / peak * PEAK
    return audio.astype(np.float32)
