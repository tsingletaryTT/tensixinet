# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Getting audio onto a chip and back: packing into tiles, and the split of each device's maths.

A Tenstorrent chip works on 32 x 32 tiles, so a stream of samples is padded to a whole number of tiles and laid out
as a matrix with 32 columns. The kernels do the per-sample transcendental work (`sin`, `exp`). The host does the
cheap multiplications that turn a reading into each sample's argument, in float64, so the chip is given arguments
that are already small and exact. Everything is float32 on the way in: a bfloat16 phase of 40 radians would be
off by a quarter of a radian.

This module is plain numpy. The kernels themselves live in `kernels/` at the top of the repository, which imports
`ttl` and so is only run under a lease or in the simulator.
"""
from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import numpy as np

from ..music.ttavdog import AVDOG_LEVEL, RED_GAIN, SHAPE_LIMIT
from ..music.devices import (
    GONG_ATTACK_S,
    GONG_SAG_DEPTH,
    GONG_SAG_S,
    SNAP_DECAY_S,
    SNAP_DEPTH,
    GongStrike,
)

TILE = 32
TILE_SAMPLES = TILE * TILE
SNAP_FLOOR = -80.0  # exp(-80) is 1.8e-35: nothing, and finite, for a sample with no strike behind it
MIX_INPUTS = 4  # the mixer kernel takes this many signals; unused ones are silence
GONG_PARTIAL_SLOTS = 5  # the most partials any gong tuning has; a tuning with fewer leaves the rest silent
GONG_OVERLAP = 3  # a strike rings for 2.5 intervals and strikes are at least one interval apart, so up to three ring at once


def padded_length(n: int) -> int:
    """`n` samples rounded up to whole tiles, and never less than one tile."""
    return max(1, math.ceil(n / TILE_SAMPLES)) * TILE_SAMPLES


def pack(samples: np.ndarray) -> np.ndarray:
    """A float32 matrix with 32 columns and a multiple of 32 rows, the samples in order, zero padded."""
    flat = np.zeros(padded_length(len(samples)), dtype=np.float32)
    flat[: len(samples)] = samples
    return flat.reshape(-1, TILE)


def unpack(tiles: np.ndarray, n: int) -> np.ndarray:
    """The first `n` samples of a packed matrix, as float64."""
    if tiles.ndim != 2 or tiles.shape[1] != TILE or tiles.shape[0] % TILE:
        raise ValueError(f"expected whole 32 x 32 tiles, got shape {tiles.shape}")
    return np.asarray(tiles, dtype=np.float64).ravel()[:n]


def shock_inputs(since: np.ndarray, rate_hz: float, decay_s: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """What the chip needs for the AVDog-style shock wave: the phase `2 pi rate tau`, the decay exponent `-tau / decay`
    and a mask that is 1 where a strike has happened. Before the first strike `since` is infinite: its tau is 0 and
    its mask is 0, so nothing infinite ever reaches the chip."""
    since = np.asarray(since, dtype=np.float64)
    live = np.isfinite(since)
    tau = np.where(live, since, 0.0)
    return pack(2 * math.pi * rate_hz * tau), pack(-tau / decay_s), pack(live.astype(np.float64))


def shock_on_host(x: np.ndarray, neg_e: np.ndarray, live: np.ndarray, n: int) -> np.ndarray:
    """The chip's formula, `live * |sin(x)| * exp(neg_e)`, in numpy. It checks that the split between host and
    chip gives the device's own answer before a kernel is involved."""
    return unpack(live * np.abs(np.sin(x)) * np.exp(neg_e), n)


def within_tolerance(got: np.ndarray, expected: np.ndarray, atol: float) -> Tuple[bool, float]:
    """(whether every sample is within `atol`, the largest difference). Different lengths never match."""
    if len(got) != len(expected):
        return False, float("inf")
    worst = float(np.max(np.abs(np.asarray(got) - np.asarray(expected)))) if len(got) else 0.0
    return worst <= atol, worst


def shimmer_inputs(
    since: np.ndarray, f1_hz: float, f2_hz: float, level: float, n: int, sample_rate: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """What the chip needs for the ultrasonic shimmer: both tone phases (each reduced to one turn in float64 first, so
    float32 holds them exactly enough), the snap's decay exponent (`SNAP_FLOOR` where no strike has happened), and the
    two constants `a0` and `a1`, the level with and without the snap, already halved for the two tones.
    The chip computes `(sin p1 + sin p2) * (a0 + a1 * exp(e))`."""
    k = np.arange(n) / sample_rate
    since = np.asarray(since, dtype=np.float64)
    e = np.where(np.isfinite(since), -np.where(np.isfinite(since), since, 0.0) / SNAP_DECAY_S, SNAP_FLOOR)
    half = 0.5 * level
    return (
        pack(np.mod(2 * math.pi * f1_hz * k, 2 * math.pi)),
        pack(np.mod(2 * math.pi * f2_hz * k, 2 * math.pi)),
        pack(e),
        pack(np.full(n, half * (1 - SNAP_DEPTH))),
        pack(np.full(n, half * SNAP_DEPTH)),
    )


def shimmer_on_host(p1, p2, e, a0, a1, n: int) -> np.ndarray:
    """The chip's formula for the shimmer, in float32 numpy."""
    return unpack((np.sin(p1) + np.sin(p2)) * (a0 + a1 * np.exp(e)), n)


def mix_inputs(signals: Sequence[np.ndarray], currents: Sequence[float], n: int) -> List[np.ndarray]:
    """The chips' signals ready for the mixer kernel: each one already multiplied by its share of the current and by
    `1 / sqrt(chips)` (the weights of `devices.current_mix`), packed, and padded with silence to `MIX_INPUTS`.
    The chip adds them and applies `tanh`."""
    if len(signals) > MIX_INPUTS:
        raise ValueError(f"the mixer kernel takes {MIX_INPUTS} signals, got {len(signals)}")
    cur = np.maximum(np.asarray(currents, dtype=float), 0.0)
    total = cur.sum()
    weights = np.ones(len(cur)) if total == 0 else cur / total * len(cur)
    scaled = [pack(w / math.sqrt(len(cur)) * np.asarray(sig, dtype=np.float64)[:n]) for w, sig in zip(weights, signals)]
    scaled += [pack(np.zeros(n)) for _ in range(MIX_INPUTS - len(scaled))]
    return scaled


def mix_on_host(tiles: Sequence[np.ndarray], n: int) -> np.ndarray:
    """The mixer kernel's formula, `tanh(sum)`, in float32 numpy."""
    return unpack(np.tanh(sum(tiles)), n)


def strike_ages(strikes: np.ndarray, n: int, sample_rate: int, length: int, back: int = 0) -> np.ndarray:
    """For each of `n` samples, how many samples ago the `back`-th latest strike began (0 is the latest), placed the way
    `devices.gong` places it (at `int(onset * sample_rate)`). -1 where there is no such strike or it has rung out."""
    starts = np.array([int(onset * sample_rate) for onset in strikes], dtype=np.int64)
    i = np.arange(n)
    idx = np.searchsorted(starts, i, side="right") - 1 - back
    age = i - starts[np.maximum(idx, 0)] if len(starts) else np.full(n, -1)
    return np.where((idx >= 0) & (age < length) & (age >= 0), age, -1)


def gong_inputs(ages: np.ndarray, strike: GongStrike, sample_rate: int) -> List[np.ndarray]:
    """What the chip needs for one ringing strike, twelve packed tiles: each partial's phase (five slots), the beat's phase,
    each partial's decay exponent with its amplitude folded in (`SNAP_FLOOR` for an empty slot or a dead sample) and the
    attack ramp, which is 0 for a dead sample. The strike's pitch sags as it rings, and the device sums the sag sample by
    sample, so the phase uses the closed form of that sum. The chip computes
    `att * sum_p exp(e_p) * (sin a_p + sin(a_p + b))`: each partial plus its detuned copy, whose phase is the partial's plus `b`."""
    ages = np.asarray(ages)
    live = ages >= 0
    j = np.where(live, ages, 0).astype(np.float64)
    k = j / sample_rate
    c = 1.0 / (GONG_SAG_S * sample_rate)
    ramp_sum = (1 - np.exp(-(j + 1) * c)) / (1 - np.exp(-c))  # sum of exp(-m c) for m = 0..j
    sag_sum = (j + 1) - GONG_SAG_DEPTH * ((j + 1) - ramp_sum)  # sum of the sag over samples 0..j
    unit = 2 * math.pi / sample_rate * sag_sum
    silent = np.full(len(ages), SNAP_FLOOR)
    phases, exponents = [], []
    for slot in range(GONG_PARTIAL_SLOTS):
        if slot < len(strike.partials):
            freq, decay, amp = strike.partials[slot]
            phases.append(pack(np.where(live, np.mod(freq * unit, 2 * math.pi), 0.0)))
            exponents.append(pack(np.where(live, -k / decay + math.log(0.5 * amp), SNAP_FLOOR)))
        else:
            phases.append(pack(np.zeros(len(ages))))
            exponents.append(pack(silent))
    beat = pack(np.where(live, np.mod(strike.beat_hz * unit, 2 * math.pi), 0.0))
    attack = pack(np.where(live, np.minimum(1.0, k / GONG_ATTACK_S), 0.0))
    return [*phases, beat, *exponents, attack]


def gong_on_host(tiles: Sequence[np.ndarray], n: int) -> np.ndarray:
    """The chip's formula for one strike, in float32 numpy."""
    a, b, e, att = tiles[:GONG_PARTIAL_SLOTS], tiles[GONG_PARTIAL_SLOTS], tiles[GONG_PARTIAL_SLOTS + 1 : 2 * GONG_PARTIAL_SLOTS + 1], tiles[-1]
    total = sum(np.exp(e[p]) * (np.sin(a[p]) + np.sin(a[p] + b)) for p in range(GONG_PARTIAL_SLOTS))
    return unpack(att * total, n)


def gong_combine(runs: Sequence[np.ndarray]) -> np.ndarray:
    """The device's last step over the overlapping strikes: add them, `0.6 * tanh`."""
    return 0.6 * np.tanh(sum(runs))


def avdog_inputs(
    since: np.ndarray, rate_hz: float, decay_s: float, pitch_hz: float, shape: float, n: int, sample_rate: int
) -> List[np.ndarray]:
    """What the chip needs for the AVDog voices, ten packed tiles: the dogtone's phase in cycles `x` and `1 - x`, the two
    slopes `ip = 1 / peak` and `iq = 1 / (1 - peak)` that skew the triangle, the undulation's phase and decay exponent, and
    four constants that carry the level and the red gain, because block maths takes no Python numbers: `aw2 = 2 level`,
    `aw = level`, and the same two times the red gain. The chip computes
    `relu(u) * (min(x ip, ox iq) aw2 - aw) + relu(-u) * (min(x iq, ox ip) ar2 - ar)` with `u = sin(pu) exp(e)`: the white
    voice under the up half of the undulation, the red under the down half with the slopes swapped."""
    since = np.asarray(since, dtype=np.float64)
    live = np.isfinite(since)
    tau = np.where(live, since, 0.0)
    x = np.mod(pitch_hz * np.arange(n) / sample_rate, 1.0)
    peak = (1 + float(np.clip(shape, -SHAPE_LIMIT, SHAPE_LIMIT))) / 2
    const = lambda value: pack(np.full(n, value))
    return [
        pack(x),
        pack(1.0 - x),
        const(1.0 / peak),
        const(1.0 / (1.0 - peak)),
        pack(np.mod(2 * math.pi * rate_hz * tau, 2 * math.pi)),
        pack(np.where(live, -tau / decay_s, SNAP_FLOOR)),
        const(2 * AVDOG_LEVEL),
        const(AVDOG_LEVEL),
        const(2 * AVDOG_LEVEL * RED_GAIN),
        const(AVDOG_LEVEL * RED_GAIN),
    ]


def avdog_on_host(tiles: Sequence[np.ndarray], n: int) -> np.ndarray:
    """The chip's formula for the AVDog voices, in float32 numpy."""
    x, ox, ip, iq, pu, e, aw2, aw, ar2, ar = tiles
    u = np.sin(pu) * np.exp(e)
    white = np.maximum(u, 0) * (np.minimum(x * ip, ox * iq) * aw2 - aw)
    red = np.maximum(-u, 0) * (np.minimum(x * iq, ox * ip) * ar2 - ar)
    return unpack(white + red, n)

