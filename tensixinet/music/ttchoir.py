# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""ttchoir: a chip as a singer, for a little humanity in the mix.

A voice is a source and a filter. The source is a buzz at the singer's pitch with its harmonics falling away
(a glottal pulse). The filter is the vocal tract, a few resonances called formants. Moving the jaw, tongue and lips
moves the formants, and that is what makes one vowel sound like another: "oo" has a low first formant and a low second,
"ah" a high first and a middle second, "ee" a low first and a very high second.

This is the additive form of formant synthesis. At each moment the sound is a sum of the source's harmonics, each as loud as
the tract lets it be at its own frequency. That form suits the chips: every harmonic is a `sin` of a multiple of one phase.

What a chip sings:

* the pitch is the chip's pitch, snapped to the patch's scale, so a choir of chips sings in tune with itself;
* the vowel changes once a bar, and each bar reads the next GDDR channel: a channel hotter than the others sings toward
  "ee", a cooler one toward "oo", and equal temperatures sing "ah";
* the voice type (a vocal tract 0%, 8%, 17% or 25% shorter, from bass toward soprano) comes from the chip's identity;
* pressure flattens the glottal slope, so a chip near a limit sings with more effort and a brighter tone;
* each syllable swells in and dips between, as a singer breathes, and the pitch has a gentle vibrato.

Not modelled: consonants, nasals, words, breath noise, and the tract's zeros. Those are the next steps.
"""
from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import numpy as np

from .mapping import aiclk_to_bpm, chip_pressure, personality, temp_to_pitch_hz
from .patch import DEFAULT_PATCH, Patch
from ..telemetry import Telemetry

# Formants F1, F2, F3 in Hz for a male voice (Peterson and Barney), per vowel.
VOWELS = {
    "a": (730, 1090, 2440),
    "e": (530, 1840, 2480),
    "i": (270, 2290, 3010),
    "o": (570, 840, 2410),
    "u": (300, 870, 2240),
}
UPPER_FORMANTS = (3400.0, 4400.0)  # F4 and F5 barely move between vowels
BANDWIDTHS = (80.0, 100.0, 120.0, 160.0, 200.0)  # Hz, widening up the tract
VOWEL_ORDER = "uoaei"  # from the darkest (closed, back) to the brightest (closed, front), with "ah" in the middle
VOICE_SCALES = (1.0, 1.08, 1.17, 1.25)  # the tract is shorter, so every formant is higher

SYLLABLE_BEATS = 4  # a syllable is a bar
GLIDE_S = 0.12  # the formants move between vowels over this long
FRAME_HZ = 200  # the tract is updated this often and the harmonic levels are interpolated between frames
MAX_HARMONICS = 40
TILT_CALM, TILT_STRAINED = 1.2, 0.6  # the glottal source falls as 1 / k ** tilt; pressure flattens it
VIBRATO_HZ = 5.5
VIBRATO_SEMITONES = (0.25, 0.6)  # calm and strained
DRIFT_SEMITONES = 0.04  # the slow random wander of the pitch
BREATH_DIP = 0.7  # how far the level falls between syllables
BREATH_S = 0.08
FADE_IN_S = 0.1
CHOIR_LEVEL = 0.5
C3_HZ = 130.8128


def formant_response(freqs: np.ndarray, formants: Sequence[float]) -> np.ndarray:
    """The vocal tract's gain at each frequency: a product of two-pole resonances, each 1 well below its formant and
    `formant / bandwidth` at it."""
    f = np.asarray(freqs, dtype=np.float64)
    gain = np.ones_like(f)
    for centre, bandwidth in zip(formants, BANDWIDTHS):
        r = f / centre
        q = centre / bandwidth
        gain = gain / np.sqrt((1 - r * r) ** 2 + (r / q) ** 2)
    return gain


def harmonic_amplitudes(f0: float, formants: Sequence[float], pressure: float, sample_rate: int) -> np.ndarray:
    """The level of each harmonic of a `f0` buzz after the tract, from harmonic 1 up, summing to 1. Harmonics stay below
    0.44 of the sample rate, with room for the vibrato."""
    count = max(1, min(MAX_HARMONICS, int(0.44 * sample_rate / (f0 * 1.05))))
    k = np.arange(1, count + 1)
    tilt = TILT_CALM + (TILT_STRAINED - TILT_CALM) * float(np.clip(pressure, 0.0, 1.0))
    amps = k ** (-tilt) * formant_response(k * f0, formants)
    return amps / amps.sum()


def choir_pitch(temp_c: float, scale: Sequence[int]) -> float:
    """The chip's pitch snapped to the nearest note of `scale` (semitones above C3, over three octaves)."""
    semis = 12 * math.log2(temp_to_pitch_hz(temp_c) / C3_HZ)
    notes = [12 * octave + degree for octave in range(3) for degree in scale]
    return C3_HZ * 2 ** (min(notes, key=lambda note: abs(note - semis)) / 12)


def voice_scale(identity: int) -> float:
    return VOICE_SCALES[identity % len(VOICE_SCALES)]


def vowel_sequence(gddr_temps: Sequence[float], count: int, offset: int) -> List[str]:
    """The vowel of each syllable, from `offset`. Each syllable reads the next GDDR channel: hotter than the average
    channel sings toward "ee", cooler toward "oo". Equal temperatures sing "ah"."""
    temps = list(gddr_temps)
    mean = sum(temps) / len(temps)
    half_spread = (max(temps) - min(temps)) / 2
    out = []
    for j in range(count):
        c = temps[(offset + j) % len(temps)]
        position = 2.0 if half_spread < 0.25 else 2 + 2 * (c - mean) / half_spread
        out.append(VOWEL_ORDER[int(np.clip(round(position), 0, len(VOWEL_ORDER) - 1))])
    return out


def pitch_modulation(sample_t: np.ndarray, pressure: float, seed: int) -> np.ndarray:
    """How far the pitch is from the note, in semitones at each time: a vibrato (deeper under pressure) and a slow random
    drift, so the voice is not a machine's."""
    rng = np.random.default_rng(seed)
    if len(sample_t) == 0:
        return np.zeros(0)
    knots = np.arange(0.0, float(sample_t[-1]) + 0.2, 0.2)
    wander = np.convolve(np.pad(rng.standard_normal(len(knots)), 1, mode="edge"), np.ones(3) / 3, mode="valid")  # three-point average
    drift = np.interp(sample_t, knots, wander) * DRIFT_SEMITONES
    depth = VIBRATO_SEMITONES[0] + (VIBRATO_SEMITONES[1] - VIBRATO_SEMITONES[0]) * float(np.clip(pressure, 0.0, 1.0))
    return depth * np.sin(2 * math.pi * VIBRATO_HZ * sample_t) + drift


def ttchoir(t: Telemetry, duration: float, sample_rate: int, seed: int, patch: Patch = DEFAULT_PATCH) -> np.ndarray:
    n = int(duration * sample_rate)
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    syllable_s = SYLLABLE_BEATS * 60.0 / aiclk_to_bpm(t.aiclk)
    identity = personality(t)
    pressure = chip_pressure(t)
    f0 = choir_pitch(t.asic_temperature, patch.scale)
    voice = voice_scale(identity)

    # The tract, frame by frame: each syllable's vowel sets the target formants and the formants glide between them.
    frames = max(2, int(duration * FRAME_HZ) + 2)
    frame_t = np.arange(frames) / FRAME_HZ
    vowels = vowel_sequence(t.gddr_temps, int(duration / syllable_s) + 2, identity % len(t.gddr_temps))
    syllable_of_frame = (frame_t / syllable_s).astype(int)
    targets = np.array([list(VOWELS[vowels[s]]) + list(UPPER_FORMANTS) for s in syllable_of_frame], dtype=np.float64) * voice
    width = max(1, int(GLIDE_S * FRAME_HZ))
    padded = np.pad(targets, ((width // 2, width - width // 2 - 1), (0, 0)), mode="edge")
    smooth = np.stack([np.convolve(padded[:, i], np.ones(width) / width, mode="valid") for i in range(targets.shape[1])], axis=1)

    harmonics = np.array([harmonic_amplitudes(f0, smooth[f], pressure, sample_rate) for f in range(frames)])
    sample_t = np.arange(n) / sample_rate

    semitones = pitch_modulation(sample_t, pressure, seed)
    phase = 2 * math.pi * np.cumsum(f0 * 2 ** (semitones / 12)) / sample_rate

    out = np.zeros(n)
    for k in range(harmonics.shape[1]):
        out += np.interp(sample_t, frame_t, harmonics[:, k]) * np.sin((k + 1) * phase)

    # A syllable swells in and dips where it meets the next, and the whole voice fades in.
    distance = np.abs(sample_t - np.round(sample_t / syllable_s) * syllable_s)
    envelope = (1 - BREATH_DIP * np.exp(-((distance / BREATH_S) ** 2))) * np.minimum(1.0, sample_t / FADE_IN_S)
    return (CHOIR_LEVEL * envelope * out).astype(np.float32)
