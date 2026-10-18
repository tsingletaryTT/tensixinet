# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Event sounds for the rare readings: memory errors and thermal trips.

A healthy chip reports zero for all three and this layer is silent. Each kind has its own
sound so a listener can tell them apart:

* correctable GDDR errors: short high ticks, one per error, up to 40
* uncorrectable GDDR errors: a low breath of filtered noise, one per error, up to 3
* thermal trips: a falling sweep from 2 kHz to 200 Hz, one per trip, up to 3

Events are not part of the turn-taking. They sound whoever has the floor.
"""
from __future__ import annotations

import math

import numpy as np

from ..telemetry import Telemetry

# The Telemetry fields this layer reads. tests/music/test_events.py checks that nothing else matters.
EVENT_INPUTS = frozenset({"corr_errs", "uncorr_errs", "thermal_trips"})

MAX_TICKS = 40
MAX_BREATHS = 3
MAX_SWEEPS = 3
TICK_HZ = 3500.0
SWEEP_HZ = (2000.0, 200.0)
SWEEP_S = 1.2
BREATH_S = 0.9


def _tick(sample_rate: int) -> np.ndarray:
    k = np.arange(int(0.025 * sample_rate)) / sample_rate
    return 0.5 * np.sin(2 * math.pi * TICK_HZ * k) * np.exp(-k / 0.005)


def _breath(sample_rate: int, rng: np.random.Generator) -> np.ndarray:
    n = int(BREATH_S * sample_rate)
    noise = rng.standard_normal(n)
    width = max(2, sample_rate // 400)  # a moving average keeps only the low end of the noise
    low = np.convolve(noise, np.ones(width) / width, mode="same")
    low /= np.max(np.abs(low)) or 1.0
    return 0.35 * low * np.sin(math.pi * np.arange(n) / n) ** 2


def _sweep(sample_rate: int) -> np.ndarray:
    k = np.arange(int(SWEEP_S * sample_rate)) / sample_rate
    hi, lo = SWEEP_HZ
    freq = hi * (lo / hi) ** (k / SWEEP_S)  # exponential, so equal steps in pitch
    phase = 2 * math.pi * np.cumsum(freq) / sample_rate
    return 0.4 * np.sin(phase) * np.minimum(1.0, (SWEEP_S - k) / 0.1) * np.minimum(1.0, k / 0.005)


def _place(out: np.ndarray, sound: np.ndarray, at_s: float, sample_rate: int) -> None:
    start = max(0, int(at_s * sample_rate))  # a sound centred near the start begins at the start
    if start >= len(out):
        return
    seg = sound[: len(out) - start]
    out[start:start + len(seg)] += seg


def error_events(t: Telemetry, duration: float, sample_rate: int, seed: int) -> np.ndarray:
    """Float32 audio, `duration` long, bounded by 1. All zeros when the chip reports no errors."""
    n = int(duration * sample_rate)
    out = np.zeros(n)
    rng = np.random.default_rng(seed)
    ticks = min(max(t.corr_errs, 0), MAX_TICKS)
    for i in range(ticks):
        jitter = rng.uniform(-0.3, 0.3)
        _place(out, _tick(sample_rate), duration * (i + 0.5 + jitter) / ticks, sample_rate)
    breaths = min(max(t.uncorr_errs, 0), MAX_BREATHS)
    for i in range(breaths):
        _place(out, _breath(sample_rate, rng), duration * (i + 1) / (breaths + 1) - BREATH_S / 2, sample_rate)
    sweeps = min(max(t.thermal_trips, 0), MAX_SWEEPS)
    for i in range(sweeps):
        _place(out, _sweep(sample_rate), duration * (i + 0.5) / sweeps - SWEEP_S / 2, sample_rate)
    return np.tanh(out).astype(np.float32)
