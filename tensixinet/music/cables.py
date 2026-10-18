# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Patch cables: one device's sound bends another's, in the manner of the purple jacks.

The Ciat-Lonbarde guides describe jacks that reach into the innards of a module, so that one
instrument's signal degrades, mutates or inflates another's. The devices here are rendered on
their own and then mixed, so a cable acts on a finished voice: the source voice drives, the target
voice is changed, and the result goes on to the turn gains and the mix.

Four kinds of cable:

* `fm`: the target is read through a delay that follows the source waveform itself. It mutates.
* `crunch`: the target is rounded to coarse steps, coarser while the source is loud. It degrades.
* `inflate`: the target is amplified while the source sounds, with a soft ceiling. It inflates.
* `fold`: the target is folded back on itself, deeper while the source is loud. It adds overtones.

Which cables exist comes from the patch seed, so a saved patch replays with the same wiring. How
hard a cable pushes comes from the chip that plays the source device: a chip under pressure or
changing fast drives its cables harder, so load on one chip changes how other devices sound. The
choir is always wired as both a source and a target.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

ANCHOR = "ttchoir"  # always has a cable out and a cable in
KINDS = ("fm", "crunch", "inflate", "fold")
MIN_CABLES, MAX_CABLES = 5, 8
MIN_DEPTH, MAX_DEPTH = 0.3, 1.0

QUIET_PUSH = 0.25  # share of its depth a cable pushes when the source chip is calm
ENVELOPE_S = 0.02  # smoothing of the loudness that drives a cable
FM_MAX_DELAY_S = 0.003  # swing of the delay at full strength
CRUNCH_MAX_STEP = 0.5  # step size at full strength and full source loudness
INFLATE_MAX = 2.0  # extra gain at full strength and full source loudness
FOLD_MAX = 4.0  # extra fold drive at full strength and full source loudness
KNEE = 0.9  # level where inflate starts to bend and fold starts to reflect


@dataclass(frozen=True)
class Cable:
    source: str  # device whose voice drives the cable
    target: str  # device whose voice is changed
    kind: str
    depth: float  # 0 to 1: how much of the effect a fully chaotic source chip gives


def draw_cables(seed: int, names: Sequence[str]) -> List[Cable]:
    """Five to eight cables between different devices, one cable at most per ordered pair, every kind
    in use, and (when the choir is among the devices) one cable out of it and one into it."""
    names = list(dict.fromkeys(names))
    if len(names) < 2:
        raise ValueError("cables need at least two devices")
    rng = np.random.default_rng([seed, 0xCAB1E])
    count = int(rng.integers(MIN_CABLES, MAX_CABLES + 1))
    kind_offset = int(rng.integers(len(KINDS)))
    wanted: List[tuple] = []
    if ANCHOR in names:
        others = [n for n in names if n != ANCHOR]
        wanted.append((ANCHOR, others[int(rng.integers(len(others)))]))
        wanted.append((others[int(rng.integers(len(others)))], ANCHOR))
    pairs = {(s, t) for s in names for t in names if s != t}
    count = min(count, len(pairs))
    while len(wanted) < count:
        pair = (names[int(rng.integers(len(names)))], names[int(rng.integers(len(names)))])
        if pair[0] != pair[1] and pair not in wanted:
            wanted.append(pair)
    return [
        Cable(s, t, KINDS[(kind_offset + i) % len(KINDS)], round(float(rng.uniform(MIN_DEPTH, MAX_DEPTH)), 3))
        for i, (s, t) in enumerate(wanted)
    ]


def strength(cable: Cable, chaos: float) -> float:
    """How hard a cable pushes: its depth, a quarter of it from a calm chip and all of it from a chaotic one."""
    return cable.depth * (QUIET_PUSH + (1.0 - QUIET_PUSH) * chaos)


def envelope(x: np.ndarray, sample_rate: int) -> np.ndarray:
    """Loudness over time, smoothed over 20 ms and scaled so its peak is 1 (all zero for silence)."""
    n = max(1, int(ENVELOPE_S * sample_rate))
    rect = np.abs(x.astype(np.float64))
    padded = np.concatenate([np.zeros(n), rect])
    sums = np.cumsum(padded)
    smooth = (sums[n:] - sums[:-n]) / n
    peak = float(smooth.max()) if len(smooth) else 0.0
    return smooth / peak if peak > 1e-12 else np.zeros_like(smooth)


def _fm(target: np.ndarray, source: np.ndarray, push: float, sample_rate: int) -> np.ndarray:
    peak = float(np.max(np.abs(source))) if len(source) else 0.0
    if peak <= 1e-12:
        return target
    index = np.arange(len(target), dtype=np.float64)
    delay = push * FM_MAX_DELAY_S * sample_rate * source / peak
    return np.interp(index + delay, index, target)


def _crunch(target: np.ndarray, env: np.ndarray, push: float) -> np.ndarray:
    step = push * CRUNCH_MAX_STEP * env
    coarse = step > 1e-4
    out = target.copy()
    out[coarse] = np.round(target[coarse] / step[coarse]) * step[coarse]
    return out


def _inflate(target: np.ndarray, env: np.ndarray, push: float) -> np.ndarray:
    """Gain with a soft ceiling: a level up to KNEE is passed as it is, and anything above bends toward 1."""
    y = target * (1.0 + INFLATE_MAX * push * env)
    over = np.maximum(np.abs(y) - KNEE, 0.0)
    return np.sign(y) * (np.minimum(np.abs(y), KNEE) + (1.0 - KNEE) * np.tanh(over / (1.0 - KNEE)))


def _fold(target: np.ndarray, env: np.ndarray, push: float) -> np.ndarray:
    """Drive the target and reflect whatever passes +-KNEE back inside, so a level inside the knee is unchanged."""
    y = target * (1.0 + FOLD_MAX * push * env)
    return KNEE - np.abs((y + KNEE) % (4.0 * KNEE) - 2.0 * KNEE)


def apply_cables(
    voices: Sequence[Optional[np.ndarray]],
    names: Sequence[str],
    cables: Sequence[Cable],
    chaos: Sequence[float],
    sample_rate: int,
) -> List[Optional[np.ndarray]]:
    """Run each cable in order. `voices`, `names` and `chaos` are lists with one entry per slot (a voice is
    None when that slot was not rendered). A cable is driven by the first slot that plays its source device,
    as that slot was rendered, so the order of cables never changes what drives them, and it changes every slot
    that plays its target device, each after the cables before it. A cable whose source or target is not
    there is skipped."""
    if not len(voices) == len(names) == len(chaos):
        raise ValueError("voices, names and chaos need one entry per slot")
    out = list(voices)
    for cable in cables:
        driver = next((i for i, n in enumerate(names) if n == cable.source), None)
        if driver is None or voices[driver] is None:
            continue
        push = strength(cable, chaos[driver])
        source = voices[driver].astype(np.float64)
        env = envelope(source, sample_rate)
        for slot, name in enumerate(names):
            if name != cable.target or out[slot] is None:
                continue
            target = out[slot].astype(np.float64)
            if cable.kind == "fm":
                changed = _fm(target, source, push, sample_rate)
            elif cable.kind == "crunch":
                changed = _crunch(target, env, push)
            elif cable.kind == "inflate":
                changed = _inflate(target, env, push)
            else:
                changed = _fold(target, env, push)
            out[slot] = changed.astype(voices[slot].dtype)
    return out
