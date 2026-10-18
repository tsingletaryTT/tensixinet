# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""ttsteam, after Man With The Red Steam in the Plumbutter guide: a shift register that steps the Snare.

The register passes a pulse up its stack on every clock, and each step can fire a module. Here the
clock is the chip's eighth notes and the stack is as long as the chip has Tensix columns, 14. A step
fires where a column is harvested, so a chip with every column enabled is silent and each real chip has
its own rhythm of holes. The Snare is quantum noise (a reverse-biased transistor leaking) under an
envelope whose decay runs from 0.75 s down to 75 ms, plus dust: a continuous crackle.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from .devices import seconds_since
from .mapping import aiclk_to_bpm, chip_pressure
from .patch import DEFAULT_PATCH, Patch
from ..telemetry import Telemetry

STEPS = 14  # Tensix columns
SNARE_DECAY_S = (0.75, 0.075)  # the guide's decay range: open at no pressure, fully tight at full pressure
SNARE_LEVEL = 0.6
ATTACK_S = 0.002
DUST_PER_PCIE_STEP = 15.0  # crackle pops a second for each step of PCIe usage
DUST_LEVEL = 0.25
SNARE_CUTOFF_HZ = 4000.0  # white noise is rolled off here so the snare does not hiss
MAX_PCIE_USAGE = 6


def step_pattern(tensix_mask: int) -> Tuple[bool, ...]:
    """Which of the 14 steps fire: those whose Tensix column is harvested (its mask bit is 0)."""
    return tuple(not (tensix_mask >> i) & 1 for i in range(STEPS))


def snare_decay(pressure: float) -> float:
    """Seconds the snare takes to fall to a third: 0.75 at no pressure down to 0.075 at full pressure."""
    slow, fast = SNARE_DECAY_S
    return slow + (fast - slow) * min(1.0, max(0.0, pressure))


def dust_per_second(pcie_usage: int) -> float:
    """Dust pops a second: PCIe traffic is data moving, so more of it is a denser crackle."""
    return DUST_PER_PCIE_STEP * min(MAX_PCIE_USAGE, max(0, int(pcie_usage)))


def lowpass(x: np.ndarray, cutoff_hz: float, sample_rate: int) -> np.ndarray:
    """Two one-pole low-passes in a row (12 dB an octave), each a short convolution so it stays vectorised."""
    alpha = 1.0 - np.exp(-2 * np.pi * cutoff_hz / sample_rate)
    length = min(64, int(6 / alpha) + 1)
    kernel = alpha * (1.0 - alpha) ** np.arange(length)
    once = np.convolve(x, kernel)[: len(x)]
    return np.convolve(once, kernel)[: len(x)]


def ttsteam(t: Telemetry, duration: float, sample_rate: int, seed: int, patch: Patch = DEFAULT_PATCH) -> np.ndarray:
    """A noise snare on the steps the Tensix holes pick, with a bed of dust."""
    n = int(duration * sample_rate)
    rng = np.random.default_rng(seed)
    step_s = 30.0 / aiclk_to_bpm(t.aiclk)
    pattern = step_pattern(t.tensix_mask)
    hits = np.array([k * step_s for k in range(int(duration / step_s) + 1) if pattern[k % STEPS] and k * step_s < duration])
    since = seconds_since(hits, n, sample_rate)
    envelope = np.minimum(1.0, since / ATTACK_S) * np.exp(-since / snare_decay(chip_pressure(t)))
    snare = SNARE_LEVEL * envelope * lowpass(rng.uniform(-1.0, 1.0, n), SNARE_CUTOFF_HZ, sample_rate)
    dust = DUST_LEVEL * (rng.random(n) < dust_per_second(t.pcie_usage) / sample_rate) * rng.uniform(-1.0, 1.0, n)
    return (snare + dust).astype(np.float32)
