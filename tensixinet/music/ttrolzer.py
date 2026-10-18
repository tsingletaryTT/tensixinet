# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""ttrolzer's rolls, after the Rolzer in the Plumbutter & Rolzer Dance Party Guide.

A roll is a ring of transistor stages with a capacitor each. The seven barres use the E6 capacitor
values 1 to 10 uF, which sets their tempos. Capacitors are good to 20%, so no two steps last exactly as
long. An even roll has a self-reinforcing cycle that swings between long and short steps and stays
steady. An odd roll cannot return to its starting state in one cycle, so its steps are unevenly spread and
it wanders, like a double pendulum.
"""
from __future__ import annotations

import math

import numpy as np

# Beats per minute of the barres (guide, Rolzer): 1, 1.5, 2.2, 3.3, 4.7, 6.8 and 10 microfarads.
# Neighbours run at about 2:3 and every other barre at about 4:9.
BARRE_BPM = (192, 122, 84, 56, 39, 27, 18)
REFERENCE_BARRE = 2  # the barre that plays at the chip's own tempo (moderato)
# Which barre and which roll each lane plays. DDR lanes take the middle barres, ethernet lanes the slow
# ones, and the two sets of rolls differ so that no lane repeats another's geometry on the same barre.
DDR_LANE_BARRES = (1, 2, 3, 4)
ETH_LANE_BARRES = (3, 4, 5, 6)
ETH_LANE_SIDES = (6, 5, 4, 3)
TOLERANCE = 0.2  # E6 capacitors are good to 20%
SWING = 0.25  # an even roll's long and short steps differ by this much of the mean step
TROT = 0.4  # how unevenly an odd roll spreads its steps over a cycle
EVEN_STABILITY = 0.5  # an even roll holds its tolerance to this share, because its cycle reinforces itself
ODD_INSTABILITY = 2.5  # an odd roll magnifies its tolerance this much


def barre_bpm(chip_bpm: float, barre: int) -> float:
    """Tempo of a ttrolzer barre when the chip's own tempo is that of the reference barre."""
    return chip_bpm * BARRE_BPM[barre] / BARRE_BPM[REFERENCE_BARRE]


def roll_pulse_times(sides: int, bpm: float, duration: float, seed: int, tolerance: float = TOLERANCE) -> np.ndarray:
    """Onsets (s) of a roll of `sides` stages with a mean step of 60/bpm, starting at 0.

    Even rolls alternate long and short steps and keep their tolerance small. Odd rolls spread their steps
    unevenly over each cycle and magnify the tolerance, so they trot and wander. `tolerance` 0 gives a
    perfect roll, the same for every seed."""
    rng = np.random.default_rng(seed)
    base = 60.0 / bpm
    even = sides % 2 == 0
    spread = tolerance * (EVEN_STABILITY if even else ODD_INSTABILITY)
    times, t, step = [], 0.0, 0
    while t < duration:
        times.append(t)
        if even:
            shape = 1 + SWING * (1 if step % 2 == 0 else -1)
        else:
            shape = 1 + TROT * math.cos(2 * math.pi * (step % sides) / sides)
        t += base * shape * (1 + rng.uniform(-spread, spread))
        step += 1
    return np.array(times)
