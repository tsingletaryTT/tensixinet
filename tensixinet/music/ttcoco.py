# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""ttcoco, after the Cocoquantus's own delay chips: the Quantussy's tone through many parallel echoes.

The real Cocoquantus pairs its five-petal Quantussy with a pair of crunchy 8-bit delay chips: discrete echo
ICs, each a buffer of a few tens of kilobits clocked by a VCO, whose length trades directly against its
resolution (a shorter delay holds more bits; a longer one holds fewer, because the same silicon serves
both). There are two of them because each delay line costs a separate piece of hardware.

A Tensix chip has no such limit. Where the real instrument affords two delay lines, this one runs eight,
one per GDDR channel, each independently timed and independently crunched, genuinely in parallel: here a
delay line costs a core's worth of work, not a chip's worth of board space. This is the one voice built
explicitly around that difference, and the first candidate for a TT-Lang kernel that uses more than one core
(see `kernels/README.md`).

Every tap takes the ttquantussy's own tone (`devices.quantussy_tone`, the same notes ttquantus plays) and
echoes it: a geometric series of delayed, decaying, independently re-crunched copies, truncated once a
repeat falls below hearing. This is a simplification of a real feedback loop through a lo-fi buffer (the
real chip's own output re-enters and degrades further on every pass; here every repeat of one tap is
crunched to that tap's own bit depth, not to a depth that keeps falling). Flip, skip, looping the buffer
itself, the Dolby switches and the iron cross are not modelled.
"""
from __future__ import annotations

import numpy as np

from .devices import quantussy_tone
from .mapping import chip_pressure, norm_temp
from .patch import DEFAULT_PATCH, Patch
from ..telemetry import Telemetry

N_TAPS = 8  # one per GDDR channel: four times the real instrument's two delay chips
DELAY_RANGE_MS = (40.0, 900.0)  # a hot channel's tap echoes fast; a cool one, slow
BITS_RANGE = (4.0, 8.0)  # the real chips' trade: the longest delay holds the fewest bits
FEEDBACK_MAX = 0.8  # feedback never reaches 1, or the echoes would never die
ECHO_FLOOR = 1.0 / 400.0  # an echo this much quieter than the first repeat is inaudible; the series stops there
LEVEL = 0.5  # headroom before tanh, with the taps' own 1/sqrt(N_TAPS) mix


def crunch_bits(x: np.ndarray, bits: float) -> np.ndarray:
    """Round `x` (-1..1) to `bits` signed levels, as the real chip's buffer does on every sample."""
    levels = 2 ** (bits - 1) - 1
    return np.round(x * levels) / levels


def shift_right(x: np.ndarray, n: int) -> np.ndarray:
    """`x` delayed by `n` samples, with silence at the start. `n` past the end of `x` is all silence."""
    if n <= 0:
        return x.copy()
    if n >= len(x):
        return np.zeros_like(x)
    out = np.zeros_like(x)
    out[n:] = x[: len(x) - n]
    return out


def echo_tap(x: np.ndarray, delay_samples: int, feedback: float, bits: float) -> np.ndarray:
    """One tap: the dry signal plus a truncated geometric series of copies, each one delay further back,
    `feedback` times quieter, and crunched to `bits` on its own (each repeat is a fresh pass through the
    buffer at this tap's bit depth, not a chain of ever-coarser passes)."""
    delay_samples = max(1, int(round(delay_samples)))
    out = np.zeros_like(x)
    gain = 1.0
    k = 0
    while gain > ECHO_FLOOR and k * delay_samples < len(x):
        out += gain * crunch_bits(shift_right(x, k * delay_samples), bits)
        gain *= feedback
        k += 1
    return out


def tap_delay_ms(gddr_temp_c: float) -> float:
    """A hotter channel's tap echoes faster: `DELAY_RANGE_MS` is (fast, slow)."""
    fast, slow = DELAY_RANGE_MS
    return slow - (slow - fast) * norm_temp(gddr_temp_c)


def tap_bits(delay_ms: float) -> float:
    """The buffer's own trade: a longer delay holds fewer bits. `delay_ms` is expected to fall in `DELAY_RANGE_MS`."""
    lo, hi = DELAY_RANGE_MS
    share = (delay_ms - lo) / (hi - lo) if hi > lo else 0.0
    bits_lo, bits_hi = BITS_RANGE
    return bits_hi - (bits_hi - bits_lo) * share


def tap_feedback(t: Telemetry) -> float:
    """How hard every tap feeds back: pressure, scaled to the feedback ceiling."""
    return FEEDBACK_MAX * chip_pressure(t)


def coco_taps(t: Telemetry, duration: float, sample_rate: int, seed: int, patch: Patch = DEFAULT_PATCH) -> list:
    """The eight delay taps, one per GDDR channel, each an echo of the ttquantussy's own tone. The device adds them; the
    kernel's host side hands them to the chip, which adds them and applies `tanh`."""
    tone = quantussy_tone(t, duration, sample_rate, seed, patch)
    feedback = tap_feedback(t)
    taps = []
    for ch in range(N_TAPS):
        delay_ms = tap_delay_ms(t.gddr_temps[ch % len(t.gddr_temps)])
        delay_samples = int(round(delay_ms / 1000.0 * sample_rate))
        taps.append(echo_tap(tone, delay_samples, feedback, tap_bits(delay_ms)))
    return taps


def ttcoco(t: Telemetry, duration: float, sample_rate: int, seed: int, patch: Patch = DEFAULT_PATCH) -> np.ndarray:
    """Eight parallel delay taps, one per GDDR channel, echoing the ttquantussy's own tone."""
    total = sum(coco_taps(t, duration, sample_rate, seed, patch))
    return np.tanh(LEVEL / np.sqrt(N_TAPS) * total).astype(np.float32)
