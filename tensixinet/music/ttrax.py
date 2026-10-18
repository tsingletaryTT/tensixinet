# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""ttrax, after the Sidrax Guide: barres, each an oscillator that bounces between two bounds.

The real instruments come in three sizes: Tetrax (4 bars), Sidrax (7, the guide this module was built from)
and Quatrax (14). `patch.rax_bars` picks which; everything here generalises from the Sidrax guide's own figures
to any bar count. Tetrax and Quatrax have no guide here, so their behaviour is this module's own extension of
the Sidrax pattern, not a transcription.

An oscillator is a ball bouncing between two walls. Its frequency is the bounce divided by the distance
between the walls, so narrowing the window raises the pitch and a window that closes makes the oscillator
ride its lower wall. Unpatched, the walls are fixed voltages. Patched, they are other barres' oscillators,
which move, so the pitch and level swell and sag with the window. The chaos knob cross-modulates the
oscillators, and at the top it is noise.

Playing a barre is two gestures. The press opens one stereo channel and the release the other. In the
guide's Pressure Bound patch a press plays the tuned tone and the release switches the bounds on, so a
tone is followed by a decaying glitch. Here the gestures come from the ttrolzer: barre b is pressed every
four steps of ttrolzer barre b, so the seven barres keep different tempos.
"""
from __future__ import annotations

import math
from typing import Sequence, Tuple

import numpy as np

from .lead import degree_to_semitones
from .mapping import aiclk_to_bpm, chip_chaos, norm_temp, temp_to_pitch_hz
from .patch import DEFAULT_PATCH, Patch
from .ttrolzer import barre_bpm
from ..telemetry import Telemetry

N_BARRES = 7

# Which barres bound each barre (guide, figure 1), counting barres from 0. Each pair is (lower, upper).
# "3" and "5" are the two bounds switches; "35" is both on.
BOUNDS_TABLE = {
    "3": ((1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 0), (0, 1)),
    "5": ((3, 6), (0, 4), (1, 5), (2, 6), (3, 0), (4, 1), (5, 2)),
    "35": ((4, 5), (5, 6), (0, 6), (0, 1), (1, 2), (2, 3), (3, 4)),
}

FALLBACK_ARCCLK = 800.0  # MHz: the ARC clock at which the first barre bounces at the chip's pitch
MAX_SPEEDUP = 8.0  # a closing window raises the pitch by at most this much, and the oscillator never passes Nyquist
MIN_SPEED = 0.25  # a very wide window slows the oscillator by at most this much
CHAOS_INDEX = 2.0  # at full chaos the leader oscillator swings the frequency by +-200%
GESTURE_STEPS = 4  # a barre is pressed every four steps of its ttrolzer barre
GESTURE_DECAY = 0.15  # each channel's envelope falls with this time constant, in gesture periods
LEVEL = 0.5  # drive into the output's saturation: seven barres sum, and tanh keeps the sum below 1


def glitch_bounds(n_bars: int, barre: int) -> Tuple[int, int]:
    """Which two barres' oscillators bound this barre's release glitch: the next barre and the one after.

    For seven bars this reproduces the Sidrax guide's own Figure 1 ("3" bounds, `BOUNDS_TABLE["3"]`) exactly.
    Tetrax (4 bars) and Quatrax (14) have no guide here, so the same neighbour pattern is extended to them."""
    return ((barre + 1) % n_bars, (barre + 2) % n_bars)


def bounce_wave(freq_hz: float, lower: float, upper: float, n: int, sample_rate: int) -> np.ndarray:
    """Triangle wave that reflects between `lower` and `upper`."""
    ph = 2 * math.pi * freq_hz * np.arange(n) / sample_rate
    unit = (2 / math.pi) * np.arcsin(np.sin(ph))  # -1..1
    return lower + (unit + 1) / 2 * (upper - lower)


def barre_bounds(gddr_temps: Sequence[float], barre: int) -> Tuple[float, float]:
    """(lower, upper) fixed bounds of a barre: its own GDDR channel sets the upper wall and the next
    channel the lower, so a hot channel pushes a wall further out and a lopsided memory swings lopsided."""
    n = len(gddr_temps)
    return (
        -(0.2 + 0.7 * norm_temp(gddr_temps[(barre + 1) % n])),
        0.2 + 0.7 * norm_temp(gddr_temps[barre % n]),
    )


def barre_frequency(base_hz: float, barre: int, scale: Sequence[int]) -> float:
    """The seven barres are tuned to the first seven degrees of the patch's scale above `base_hz`."""
    return base_hz * 2 ** (degree_to_semitones(barre, scale) / 12)


def windowed_bounce(
    freq_hz: float,
    lower: np.ndarray,
    upper: np.ndarray,
    width: float,
    sample_rate: int,
    chaos: float = 0.0,
    leader: np.ndarray = None,
) -> np.ndarray:
    """An oscillator bouncing between a moving `lower` and `upper`, never outside them.

    `freq_hz` is its pitch when the window is `width` wide. Its frequency goes as bounce over bounds,
    so a narrower window is faster, between MIN_SPEED and MAX_SPEED times. Where the upper wall is below
    the lower the oscillator sits on the lower wall. `chaos` (0 to 1) swings the frequency with `leader`,
    another oscillator between -1 and 1."""
    gap = upper - lower
    speed = np.clip(width / np.maximum(gap, 1e-12), MIN_SPEED, MAX_SPEEDUP)
    freq = freq_hz * speed
    if chaos and leader is not None:
        freq = freq * (1 + CHAOS_INDEX * chaos * leader)
    freq = np.clip(freq, -0.45 * sample_rate, 0.45 * sample_rate)
    phase = np.cumsum(freq) / sample_rate
    unit = 1 - np.abs(2 * (phase % 1.0) - 1)  # 0 to 1 and back, starting on the lower wall
    return lower + unit * np.maximum(gap, 0.0)


def gesture_envelopes(bpm: float, barre: int, duration: float, sample_rate: int) -> Tuple[np.ndarray, np.ndarray]:
    """(press, release) envelopes of one barre: each peaks at 1 on its gesture and decays. The barre is pressed at
    the start of every period and released half a period later."""
    n = int(duration * sample_rate)
    period = GESTURE_STEPS * 60.0 / barre_bpm(bpm, barre)
    half = period / 2
    tau = (np.arange(n) / sample_rate) % period
    decay = GESTURE_DECAY * period
    press = np.where(tau < half, np.exp(-tau / decay), 0.0)
    release = np.where(tau >= half, np.exp(-(tau - half) / decay), 0.0)
    return press, release


def ttrax(t: Telemetry, duration: float, sample_rate: int, seed: int, patch: Patch = DEFAULT_PATCH) -> np.ndarray:
    """`patch.rax_bars` barres tuned to the patch's scale. Each plays a tone on its press and a glitch on its release.

    The tone bounces between fixed walls set by the memory temperatures. The glitch bounces between two other
    barres' oscillators. Pressure is the chaos knob."""
    n_bars = patch.rax_bars
    n = int(duration * sample_rate)
    bpm = aiclk_to_bpm(t.aiclk)
    base = temp_to_pitch_hz(t.asic_temperature) * t.arcclk / FALLBACK_ARCCLK
    chaos = chip_chaos(t)
    nyquist_limit = 0.45 * sample_rate
    freqs = [min(barre_frequency(base, b, patch.scale), nyquist_limit) for b in range(n_bars)]
    bounds = [barre_bounds(t.gddr_temps, b) for b in range(n_bars)]
    widths = [hi - lo for lo, hi in bounds]
    # Each barre's own oscillator between its fixed walls: what the other barres see as a wall.
    walls = [
        windowed_bounce(f, np.full(n, lo), np.full(n, hi), w, sample_rate)
        for f, (lo, hi), w in zip(freqs, bounds, widths)
    ]
    total = np.zeros(n)
    for b in range(n_bars):
        lo, hi = bounds[b]
        leader = walls[(b + 2) % n_bars] / 0.9
        press, release = gesture_envelopes(bpm, b, duration, sample_rate)
        tone = windowed_bounce(freqs[b], np.full(n, lo), np.full(n, hi), widths[b], sample_rate, chaos, leader)
        lower_barre, upper_barre = glitch_bounds(n_bars, b)
        glitch = windowed_bounce(freqs[b], walls[lower_barre], walls[upper_barre], widths[b], sample_rate, chaos, leader)
        total += press * tone + release * glitch
    return np.tanh(LEVEL * total).astype(np.float32)
