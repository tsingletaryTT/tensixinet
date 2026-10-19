# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""ttrolzer's rolls, after the Rolzer in the Plumbutter & Rolzer Dance Party Guide.

A roll is a ring of transistor stages with a capacitor each. The seven barres use the E6 capacitor
values 1 to 10 uF, which sets their tempos. Capacitors are good to 20%, so no two steps last exactly as
long. An even roll has a self-reinforcing cycle that swings between long and short steps and stays
steady. An odd roll cannot return to its starting state in one cycle, so its steps are unevenly spread and
it wanders, like a double pendulum.

`roll_pulse_times` makes one independent roll and is what ttrax uses. `roll_ensemble` is the fuller model that ttrollz
uses: its rolls keep their capacitors from step to step (so an even roll trots in the same way every cycle), the
capacitors drift slowly, an odd roll's pattern slides round its cycle, and rolls are coupled the way the guide says
Browns couple them. A Brown is a Sandrode, a jack that is input and output at once, so patching one roll's Brown to
another's weaves their circuits together. Here a link makes every pulse of the source pull the target's next step
earlier, which syncs an even target to a faster source "mostly" and knocks it off its pattern when the source is odd.
Grounding a Brown (a DDR status of 0) removes the roll before it gets here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

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
    """Tempo of a ttrolzer barre when the chip's own tempo is that of the reference barre.

    The guide's own table has seven entries; a barre index past that (Quatrax's fourteen) wraps,
    so the same seven tempos repeat rather than index past the table."""
    return chip_bpm * BARRE_BPM[barre % len(BARRE_BPM)] / BARRE_BPM[REFERENCE_BARRE]


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


# --- the coupled model ---------------------------------------------------------------------------------------------

EVEN_SWING = {2: 0.25, 4: 0.25, 6: 0.05}  # long and short steps of an even roll; the hexagon swings far less ...
HEX_TRIPLET = 0.26  # ... because it carries two threes interleaved, which gives it a triplet feel
SLIP = 0.12  # how far (in cycles) an odd roll's pattern slides round, as a random walk each cycle
JITTER = 0.01  # a little loose timing on every step, on top of the capacitors
DRIFT_RANGE = (0.005, 0.05)  # standard deviation of a capacitor's drift each step, cool to hot
DRIFT_TEMP_RANGE_C = (30.0, 90.0)  # the chip temperatures between which drift grows from the first to the second
DRIFT_REVERT = 0.1  # a drifting capacitor is pulled back toward its value by this share each step
MIN_STEP = 0.15  # no step is pulled shorter than this share of the barre's step, however hard it is pulled
LINK_PULL_RANGE = (0.05, 0.3)  # the share of its remaining step a target loses when its source fires
MIN_LINKS, MAX_LINKS = 2, 4
DOWNBEAT_GAIN = 0.6  # ttrollz adds this much of a blip, an octave up, on the first step of a roll's cycle
LINK_STREAM = 0xB20
CAP_STREAM = 0xCA9


@dataclass(frozen=True)
class Roll:
    sides: int
    bpm: float  # the barre's tempo: one step lasts 60 / bpm on average


@dataclass(frozen=True)
class Link:
    source: int  # index of the roll whose Brown spikes
    target: int  # index of the roll whose Brown receives them
    pull: float  # share of the target's remaining step lost on every source pulse


def roll_drift(asic_temperature: float) -> float:
    """How far a capacitor's value drifts each step: a hotter chip drifts more, from DRIFT_RANGE's low to its high."""
    lo_t, hi_t = DRIFT_TEMP_RANGE_C
    lo, hi = DRIFT_RANGE
    return lo + (hi - lo) * min(1.0, max(0.0, (asic_temperature - lo_t) / (hi_t - lo_t)))


def draw_links(n_rolls: int, patch_seed: int, identity: int, pressure: float) -> List[Link]:
    """Which Browns are patched to which: two to four distinct pairs, drawn from the patch and the chip's identity
    so a chip is wired the same way for a whole run. Pressure strengthens every link and moves none."""
    if n_rolls < 2:
        return []
    rng = np.random.default_rng([patch_seed, identity, LINK_STREAM])
    count = min(int(rng.integers(MIN_LINKS, MAX_LINKS + 1)), n_rolls * (n_rolls - 1))
    pairs: List[Tuple[int, int]] = []
    while len(pairs) < count:
        a, b = int(rng.integers(n_rolls)), int(rng.integers(n_rolls))
        if a != b and (a, b) not in pairs:
            pairs.append((a, b))
    lo, hi = LINK_PULL_RANGE
    scales = rng.uniform(0.5, 1.0, size=count)
    pressure = min(1.0, max(0.0, pressure))
    return [Link(a, b, round(lo + (hi - lo) * float(sc) * (0.2 + 0.8 * pressure), 4)) for (a, b), sc in zip(pairs, scales)]


class _RollState:
    """One roll as it runs: its capacitors, their drift, and where its pattern has slid to."""

    def __init__(self, roll: Roll, rng: np.random.Generator, cap_rng: np.random.Generator, tolerance: float, drift: float, jitter: float):
        n = roll.sides
        self.n, self.base, self.rng, self.drift, self.jitter = n, 60.0 / roll.bpm, rng, drift, jitter
        self.even = n % 2 == 0
        spread = tolerance * (EVEN_STABILITY if self.even else 1.0)
        caps = 1 + spread * cap_rng.uniform(-1.0, 1.0, size=n)
        self.caps = caps / caps.mean()  # the barre's capacitors fix its tempo, so mismatch moves steps about without moving the mean
        self.walk = np.zeros(n)
        self.slip = 0.0
        self.step = 0

    def step_length(self) -> float:
        """The time from this step to the next, then move on one step."""
        k, n = self.step, self.n
        i = k % n
        if self.even:
            shape = 1 + EVEN_SWING.get(n, 0.25) * (1 if k % 2 == 0 else -1)
            if n == 6:
                shape += HEX_TRIPLET * math.cos(2 * math.pi * i / 3)
        else:
            if i == 0 and k > 0:
                self.slip += SLIP * float(self.rng.normal())  # the cycle does not come back round in the same phase
            shape = 1 + TROT * math.cos(2 * math.pi * (i / n + self.slip))
        self.walk[i] = (1 - DRIFT_REVERT) * self.walk[i] + self.drift * float(self.rng.normal())
        loose = 1 + self.jitter * float(self.rng.normal())
        self.step += 1
        return max(MIN_STEP, shape * self.caps[i] * (1 + self.walk[i]) * loose) * self.base


def roll_ensemble(
    rolls: Sequence[Roll],
    links: Sequence[Link],
    duration: float,
    seed: int,
    tolerance: float = TOLERANCE,
    drift: float = DRIFT_RANGE[0],
    jitter: float = JITTER,
    cap_seed: Optional[int] = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Run the rolls together for `duration` seconds. Returns, for each roll, (the times of its pulses, which of those
    pulses begin one of its cycles). Every roll starts at 0. A roll with no link into it runs as it would alone.

    `seed` decides the drift, the slide of odd rolls and the loose timing. `cap_seed` (the seed when left out) decides
    each roll's capacitor mismatch, which is a property of the circuit: give a chip's rolls the same `cap_seed` and they
    trot the same way however the `seed` changes."""
    for link in links:
        for index in (link.source, link.target):
            if not 0 <= index < len(rolls):
                raise ValueError(f"link {link} names roll {index}, and there are only {len(rolls)} rolls")
        if link.source == link.target:
            raise ValueError(f"link {link} joins a roll to itself")
    cap_seed = seed if cap_seed is None else cap_seed
    states = [
        _RollState(r, np.random.default_rng([seed, i]), np.random.default_rng([cap_seed, i, CAP_STREAM]), tolerance, drift, jitter)
        for i, r in enumerate(rolls)
    ]
    into = [[l for l in links if l.source == i] for i in range(len(rolls))]
    due = np.zeros(len(rolls))
    last = np.full(len(rolls), -np.inf)
    out: List[Tuple[List[float], List[bool]]] = [([], []) for _ in rolls]
    while len(rolls):
        r = int(np.argmin(due))
        now = float(due[r])
        if now >= duration:
            break
        out[r][0].append(now)
        out[r][1].append(states[r].step % states[r].n == 0)
        last[r] = now
        due[r] = now + states[r].step_length()
        for link in into[r]:  # this roll's spike reaches the Browns patched to it
            remaining = due[link.target] - now
            if remaining > 0:
                floor = last[link.target] + MIN_STEP * states[link.target].base
                due[link.target] = max(now + remaining * (1 - link.pull), floor)
    return [(np.array(t), np.array(d, dtype=bool)) for t, d in out]
