# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Space: where a voice sits and how it moves, so the band is spread across a room and not stacked in the middle.

Stereo in the Ciat-Lonbarde instruments is more than left and right: the Plumbutter's AVDogs put their white and red voices on the
two channels and swing between them, and a patch can send one sound across the field while another stays put. Here a voice has
three coordinates:

* **azimuth**, from -1 (hard left) to +1 (hard right), panned with an equal-power law so a voice keeps its loudness as it moves;
* **width**, from 0 (a point) to 1 (wide): a decorrelated copy is added to one ear and subtracted from the other;
* **depth**, from 0 (near) to 1 (far): farther is quieter, darker, and carries an early reflection.

Some things are fixed. Each chip has a home in the field, spread left to right by its index and nudged by its identity, and each device
has its own offset from home and its own width and depth (the air of the shimmer is wide and far, the lead is near and narrow). Some
things travel, and the signals decide how much and how fast: a chip under pressure or changing quickly orbits widely while a calm chip
barely moves, a faster clock orbits faster, the soloist comes forward and the others fall back, and a few devices pan from their own
internals (`spatial.py`). Nothing here imports a device, so any module can use it.
"""
from __future__ import annotations

import math
import zlib
from typing import Tuple, Union

import numpy as np

from .mapping import aiclk_to_bpm, personality
from ..telemetry import Telemetry

Curve = Union[float, np.ndarray]

SPREAD = 0.75  # the outermost chips sit this far from the middle
IDENTITY_NUDGE = 0.04  # a chip's identity moves its place by up to this
ORBIT_CALM = 0.04  # how far (in azimuth) a calm chip orbits
ORBIT_CHAOS = 0.30  # and how much farther a chip at full chaos does
ORBIT_HZ_PER_BPM = 0.05 / 60.0  # one orbit takes 20 beats
DEPTH_DARKEN_HZ = 2500.0  # a voice at full depth is rolled off here
DEPTH_LEVEL = 0.55  # and is this much quieter
REFLECTION_S, REFLECTION_LEVEL = 0.023, 0.35  # the early reflection a far voice carries
WIDTH_DELAY_S = 0.011  # the decorrelated copy that makes a voice wide

# device -> (azimuth offset from the chip's home, width, depth). The air is wide and far; the lead and the snare are near.
DEVICE_SPACE = {
    "lead": (0.15, 0.15, 0.0),
    "ttrollz": (-0.15, 0.4, 0.1),
    "ttsteam": (-0.30, 0.2, 0.1),
    "ttquantus": (0.25, 0.5, 0.2),
    "ttavdog": (0.0, 0.7, 0.3),
    "ttrax": (0.0, 0.8, 0.3),
    "ttchoir": (0.0, 0.4, 0.35),
    "ttcoco": (0.0, 1.0, 0.5),
    "ttdrone": (0.0, 1.0, 0.6),
    "gong": (-0.10, 0.5, 0.7),
    "ultrasonic_shimmer": (0.0, 1.0, 0.9),
}
UNKNOWN_SPACE = (0.0, 0.5, 0.5)


def device_space(name: str) -> Tuple[float, float, float]:
    """(azimuth offset, width, depth) for a device; a device that is not in the table sits in the middle at a middling distance."""
    return DEVICE_SPACE.get(name, UNKNOWN_SPACE)


def chip_azimuth(index: int, n_chips: int, t: Telemetry) -> float:
    """Chip `index`'s home: the chips are spread from left to right in order, and one chip sits in the middle. The chip's identity
    nudges it a little, never far enough to pass its neighbour."""
    base = 0.0 if n_chips <= 1 else -SPREAD + 2 * SPREAD * index / (n_chips - 1)
    return base + ((personality(t) % 10007) / 10007 * 2 - 1) * IDENTITY_NUDGE


def pan_gains(azimuth: Union[float, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """(left gain, right gain) for an azimuth from -1 to +1 (past the edge is the edge), with the power constant."""
    theta = (np.clip(np.asarray(azimuth, dtype=np.float64), -1.0, 1.0) + 1.0) * math.pi / 4
    return np.cos(theta), np.sin(theta)


def smooth(x: np.ndarray, sample_rate: int, ms: float) -> np.ndarray:
    """`x` averaged over `ms` milliseconds, as long as `x`, with its ends held (a jump becomes a short ramp)."""
    x = np.asarray(x, dtype=np.float64)
    width = max(1, int(sample_rate * ms / 1000.0)) | 1
    if len(x) == 0 or width == 1:
        return x.copy()
    half = width // 2
    padded = np.concatenate([np.full(half, x[0]), x, np.full(half, x[-1])])
    sums = np.concatenate([[0.0], np.cumsum(padded)])
    return (sums[width:] - sums[:-width]) / width


def motion_azimuth(t: Telemetry, name: str, chip: int, n: int, sample_rate: int, chaos: float) -> np.ndarray:
    """How far a device wanders from its place, in azimuth, for each of `n` samples. A calm chip (`chaos` 0) barely moves and a chip
    at full chaos orbits widely; a faster clock orbits faster; each chip and each device orbits out of step with the others."""
    rng = np.random.default_rng([personality(t), chip, zlib.crc32(name.encode())])
    phase1, phase2 = rng.uniform(0, 2 * math.pi, size=2)
    amplitude = ORBIT_CALM + ORBIT_CHAOS * min(1.0, max(0.0, chaos))
    hz = aiclk_to_bpm(t.aiclk) * ORBIT_HZ_PER_BPM
    k = np.arange(n) / sample_rate
    return amplitude * (np.sin(2 * math.pi * hz * k + phase1) + 0.35 * np.sin(2 * math.pi * hz * 0.37 * k + phase2))


def _curve(value: Curve, n: int, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        return np.full(n, float(array))
    if len(array) != n:
        raise ValueError(f"{label} has {len(array)} values for a voice of {n} samples")
    return array


def _lowpass(x: np.ndarray, cutoff_hz: float, sample_rate: int) -> np.ndarray:
    """Two one-pole low-passes in a row (12 dB an octave), each a short convolution so it stays vectorised."""
    alpha = 1.0 - np.exp(-2 * np.pi * cutoff_hz / sample_rate)
    length = min(64, int(6 / alpha) + 1)
    kernel = alpha * (1.0 - alpha) ** np.arange(length)
    return np.convolve(np.convolve(x, kernel)[: len(x)], kernel)[: len(x)]


def _delay(x: np.ndarray, seconds: float, sample_rate: int) -> np.ndarray:
    shift = max(1, int(seconds * sample_rate))
    out = np.zeros_like(x)
    if shift < len(x):
        out[shift:] = x[:-shift]
    return out


def spatialize(mono: np.ndarray, azimuth: Curve, width: Curve, depth: Curve, sample_rate: int) -> np.ndarray:
    """A mono voice placed in the field: float32, shape (n, 2). `azimuth`, `width` and `depth` are numbers or curves with one value per
    sample. A centred, narrow, near voice is the same in both ears at 1/sqrt(2) of the mono level, so its power is the mono voice's."""
    x = np.asarray(mono, dtype=np.float64)
    n = len(x)
    if n == 0:
        return np.zeros((0, 2), dtype=np.float32)
    az, w, d = _curve(azimuth, n, "azimuth"), np.clip(_curve(width, n, "width"), 0, 1), np.clip(_curve(depth, n, "depth"), 0, 1)
    dark = _lowpass(x, min(DEPTH_DARKEN_HZ, 0.15 * sample_rate), sample_rate)  # 15% of the rate at most, so it darkens at low rates too
    body = ((1.0 - d) * x + d * dark) * (1.0 - DEPTH_LEVEL * d)  # farther is darker and quieter
    reflection = _delay(dark, REFLECTION_S, sample_rate) * (REFLECTION_LEVEL * d)  # and carries a reflection, the same in both ears
    side = _delay(body, WIDTH_DELAY_S, sample_rate)
    left_gain, right_gain = pan_gains(az)
    keep = 1.0 / np.sqrt(1.0 + 0.5 * w * w)  # the added side does not make a wide voice louder
    left = (left_gain * body + 0.5 * w * side) * keep + reflection * left_gain
    right = (right_gain * body - 0.5 * w * side) * keep + reflection * right_gain
    return np.stack([left, right], axis=1).astype(np.float32)
