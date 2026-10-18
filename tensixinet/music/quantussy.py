# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The Cocoquantus's Quantussy, from the Cocoquantus tasting guide.

Five petals, each a triangle oscillator. Every petal also has a castle: a sample-and-hold that, at each
peak and valley of its own petal's triangle, samples the triangle of another petal (its leader) and holds
the value. By default the leader is two positions counter-clockwise. Two does not divide five, so the
chain of leaders spirals through every petal. The castle is not random. Petals whose frequencies have
low prime ratios make a repeating pattern, and petals that are slightly off make that pattern drift.

The Cocoquantus's delay buffers store audio in 8 bits and take their sample rate from the clock, so a
slower clock holds each sample longer and aliases.
"""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

N_PETALS = 5
DEFAULT_LEADER_OFFSET = 2
# The castle mod select cycles through these on the hardware; a chip's identity picks one.
LEADER_OFFSETS = (2, 3, 4, 1)
# Low prime frequency ratios, so the castles make a pattern you can hear repeat. The seed nudges each by up to 3%.
BASE_RATIOS = (1.0, 3 / 2, 5 / 4, 4 / 3, 5 / 3)
RATIO_NUDGE = 0.03
BUFFER_LEVELS = 127  # 8 bits, signed


def triangle(freq_hz: float, phase: float, t: np.ndarray) -> np.ndarray:
    """A triangle from 0 up to 1 and back at `freq_hz`, `phase` cycles in at t = 0."""
    cycle = (freq_hz * np.asarray(t) + phase) % 1.0
    return 1.0 - np.abs(2.0 * cycle - 1.0)


def castle_leader(petal: int, offset: int = DEFAULT_LEADER_OFFSET) -> int:
    """The petal whose triangle this petal's castle samples: `offset` positions counter-clockwise."""
    return (petal - offset) % N_PETALS


def castle_offset(identity: int) -> int:
    """How many places counter-clockwise this chip's castles look for their leader: one of 2, 3, 4 and 1."""
    return LEADER_OFFSETS[identity % len(LEADER_OFFSETS)]


def castle_output(
    petal: int, freqs: Sequence[float], phases: Sequence[float], t: np.ndarray, offset: int = DEFAULT_LEADER_OFFSET
) -> np.ndarray:
    """The castle of `petal` at times `t`: its leader's triangle as it was at the petal's last peak or valley."""
    turn = np.floor(np.asarray(t) * 2 * freqs[petal] + 1e-9)  # which half cycle of this petal we are in
    sample_time = turn / (2 * freqs[petal])
    leader = castle_leader(petal, offset)
    return triangle(freqs[leader], phases[leader], sample_time)


def petal_ratios(seed: int) -> Tuple[float, ...]:
    """Frequency ratios of the five petals: low primes, each nudged by the seed. Petal 0 sets the rate and is exact."""
    nudge = np.random.default_rng(seed).uniform(-RATIO_NUDGE, RATIO_NUDGE, N_PETALS)
    return (BASE_RATIOS[0],) + tuple(float(r * (1 + x)) for r, x in zip(BASE_RATIOS[1:], nudge[1:]))


def petal_phases(seed: int) -> Tuple[float, ...]:
    """Where in its cycle each petal's triangle starts, from the seed."""
    return tuple(float(x) for x in np.random.default_rng(seed + 1).random(N_PETALS))


def crunch(x: np.ndarray, rate: float) -> np.ndarray:
    """What the coco's buffer does to audio: round to 8 bits and hold each sample for 1/rate samples, where
    `rate` (0 to 1) is the clock as a share of the sample rate."""
    hold = max(1, int(round(1.0 / rate)))
    held = np.asarray(x)[(np.arange(len(x)) // hold) * hold]
    return np.round(held * BUFFER_LEVELS) / BUFFER_LEVELS
