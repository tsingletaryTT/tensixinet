# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Evict: what happens to the band when a chip's model unloads.

tt-toplike's Defrag view draws a chip's memory as a map of blocks that fill as weights arrive and dissolve when the model goes. Its EVICT
animation is set off by one thing (src/animation/defrag.rs): a chip that has been running, has *seen a load* (power above 1.5 times its
learned rest, or above 30% of its TDP when no rest has been seen) and has fallen back to about rest (at most 1.2 times, and never below 8 W)
has unloaded its model. The "seen a load" gate is what stops a chip that merely sits idle from evicting in a loop. The blocks then dissolve
right to left, each channel at its own prime-multiplied rate, and when load returns they fill left to right.

The same trigger moves the band the same way:

* **evicting** - the chip's voices dissolve right to left across the stereo field (each one fading and drifting to the left edge and back in
  the depth, at its own prime-multiplied rate and a step behind the one to its right), and a noise sweep falls in pitch with them;
* **idle** - the chip is silent and almost never has the floor;
* **filling** - when load returns the voices come back left to right, a sweep rises, and the chip has the floor less often than a settled one;
* **running** - nothing changes. A box that only ever idles never leaves this phase.

`EvictTracker` follows each chip's readings and hands the chorus one `ChipDefrag` per chip.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from .space import _lowpass
from ..telemetry import Telemetry

LOAD_SURGE_FACTOR = 1.5  # power this many times the learned rest counts as a load (tt-toplike's LOAD_SURGE_FACTOR)
RUN_TDP_FRACTION = 0.30  # or this share of the board's TDP limit when no rest has been seen (tt-toplike's RUN_TDP_FRACTION)
EVICT_MARGIN = 1.2  # a chip is back at rest when power is at most this many times its rest ...
POWER_IDLE_W = 8.0  # ... and never counts as unloaded above this many watts of floor
EMA_ALPHA = 0.5  # how far a reading moves the smoothed power when it comes EMA_FULL_STEP_S or more after the one before (live mode: 2 s or more)
EMA_FULL_STEP_S = 1.0  # a closer reading moves it by 1 - 0.5 ** gap instead, so the smoothing takes the same seconds however often chips are read
RAN_LONG_ENOUGH_S = 10.0  # a chip must have been running this long before it can evict
STABLE_BAND = 0.10  # rest is learned only from readings within this share of the smoothed power, not from the edge of a fall
EVICT_SECONDS = 8.0  # the dissolve of the slowest voice
FILL_SECONDS = 8.0  # and the fill
STAGGER_S = 0.35  # each voice starts this long after the one to its right (and fills this long after the one to its left)
DRAIN_PRIMES = (2, 3, 5, 7, 11, 13, 17, 19)  # drain multipliers by voice, as the Defrag view gives each channel
SWEEP_S = 3.0
SWEEP_LEVEL = 0.4
SWEEP_TOP_HZ, SWEEP_BOTTOM_HZ = 7000.0, 200.0
SWEEP_LAYERS = 24

PRESENCE = {"running": 1.0, "filling": 0.6, "evicting": 0.25, "idle": 0.05}  # a chip's share of the turns, by phase


def shows_load(power_ema: float, idle_power: float, tdp_limit: Optional[float]) -> bool:
    """Whether power says the chip is doing work: against its own learned rest, or (only while none has been seen, `idle_power` 0) against its TDP.
    Once a rest is known the TDP no longer counts: a chip that rests near 30% of its limit (this box's rest is 36 to 40 W against 37.5 W) would
    otherwise cross the line at rest, count each crossing as a load and evict every time it fell back."""
    if idle_power > 0.0:
        return power_ema > idle_power * LOAD_SURGE_FACTOR
    return bool(tdp_limit and tdp_limit > 0.0 and power_ema > tdp_limit * RUN_TDP_FRACTION)


def evict_eligible(running_for_s: float, power_ema: float, idle_power: float, saw_load: bool) -> bool:
    """tt-toplike's rule: it has run a while, it has seen a load, and it has fallen back to about rest."""
    return running_for_s >= RAN_LONG_ENOUGH_S and saw_load and power_ema <= max(idle_power * EVICT_MARGIN, POWER_IDLE_W)


def presence_weight(phase: str) -> float:
    """How much of the band a chip is in a phase: its share of the turns (it also fades from the mix, by `drain_progress`)."""
    return PRESENCE[phase]


@dataclass(frozen=True)
class ChipDefrag:
    phase: str = "running"  # running, evicting, idle or filling
    elapsed: float = 0.0  # seconds since the phase began, at the start of the segment being rendered


@dataclass(frozen=True)
class EvictEvent:
    kind: str  # "evict" or "fill"
    chip: int
    what: str
    why: str


class _Chip:
    def __init__(self, now: float):
        self.phase, self.since, self.running_since = "running", now, now
        self.ema: Optional[float] = None
        self.idle = 0.0  # learned rest; 0 means none seen yet
        self.saw_load = False
        self.peak = 0.0
        self.last_reading: Optional[float] = None  # when the previous reading came, to weigh the next one by the gap


class EvictTracker:
    """Follows every chip's power reading by reading and decides when each one evicts and when it fills again."""

    def __init__(self) -> None:
        self._chips: List[_Chip] = []
        self._events: List[EvictEvent] = []

    def take_events(self) -> List[EvictEvent]:
        """The events since the last call, once."""
        events, self._events = self._events, []
        return events

    def update(self, readings: Sequence[Telemetry], now: float) -> List[ChipDefrag]:
        if len(self._chips) != len(readings):  # a different number of chips: start again
            self._chips = [_Chip(now) for _ in readings]
        return [self._step(i, c, r, now) for i, (c, r) in enumerate(zip(self._chips, readings))]

    def _step(self, index: int, c: _Chip, r: Telemetry, now: float) -> ChipDefrag:
        p = float(r.power)
        if c.ema is None or c.last_reading is None:
            c.ema = p
        else:
            gap = now - c.last_reading
            alpha = EMA_ALPHA if gap >= EMA_FULL_STEP_S else 1.0 - 0.5 ** max(gap, 0.0)
            c.ema = (1.0 - alpha) * c.ema + alpha * p
        c.last_reading = now
        tdp = r.tdp_limit if r.tdp_limit and r.tdp_limit > 0 else None
        by_tdp = bool(tdp and c.ema > tdp * RUN_TDP_FRACTION)
        if not by_tdp and abs(p - c.ema) <= STABLE_BAND * max(c.ema, 1.0):  # a steady reading at rest: the baseline is a running minimum
            c.idle = c.ema if c.idle == 0.0 else min(c.idle, c.ema)
        load = shows_load(c.ema, c.idle, tdp)
        if load:
            c.saw_load = True
            c.peak = max(c.peak, p)
        if c.phase == "running":
            if evict_eligible(now - c.running_since, c.ema, c.idle, c.saw_load):
                c.phase, c.since, c.saw_load = "evicting", now, False
                self._events.append(EvictEvent(
                    "evict", index,
                    f"chip {index} evicted: power fell back to {c.ema:.0f} W (rest {c.idle:.0f} W) after a load that peaked at {c.peak:.0f} W",
                    f"its model unloaded, as tt-toplike's Defrag view reads it: the chip had run for {now - c.running_since:.0f} s, seen a load "
                    f"above {LOAD_SURGE_FACTOR:g} times its rest and fallen back to at most {EVICT_MARGIN:g} times it. Its voices now dissolve "
                    "right to left across the stereo field at staggered prime-multiplied rates, a sweep falls in pitch with them, and the chip "
                    "sits out of the rotation until load returns",
                ))
                c.peak = 0.0
        elif c.phase == "evicting":
            if load:  # the load came back before the dissolve was done: turn round from where it had got to
                done = min(1.0, (now - c.since) / EVICT_SECONDS)
                c.phase, c.since = "filling", now - FILL_SECONDS * (1.0 - done)
                self._events.append(self._fill_event(index, c))
            elif now - c.since >= EVICT_SECONDS:
                c.phase, c.since = "idle", now
        elif c.phase == "idle":
            if load:
                c.phase, c.since = "filling", now
                self._events.append(self._fill_event(index, c))
        elif c.phase == "filling":
            if now - c.since >= FILL_SECONDS:
                c.phase, c.since, c.running_since = "running", now, now
        return ChipDefrag(c.phase, max(0.0, now - c.since))

    @staticmethod
    def _fill_event(index: int, c: _Chip) -> EvictEvent:
        return EvictEvent(
            "fill", index,
            f"chip {index} loaded again: power rose to {c.ema:.0f} W (rest {c.idle:.0f} W)",
            "a load is back, so its voices fill in again left to right as the Defrag view's blocks do when weights arrive, a sweep rises, and "
            "the chip takes its turns again as it settles",
        )


def drain_progress(phase: str, elapsed: float, n: int, sample_rate: int, rank: int, count: int, prime_index: int) -> np.ndarray:
    """How much of a voice has dissolved, 0 (whole) to 1 (gone), for each of `n` samples. `elapsed` is the seconds into the phase at the first
    sample, `rank` the voice's place from the right among the chip's `count` voices (0 is the rightmost), `prime_index` picks its drain multiplier.

    Evicting: the rightmost voice starts first and each one after it a step later, each at its prime's speed. Filling: the leftmost starts first.
    Running is 0 throughout and idle is 1."""
    if phase == "running":
        return np.zeros(n)
    if phase == "idle":
        return np.ones(n)
    speed = DRAIN_PRIMES[prime_index % len(DRAIN_PRIMES)] / DRAIN_PRIMES[0]
    t = elapsed + np.arange(n) / sample_rate
    if phase == "evicting":
        return np.clip((t - rank * STAGGER_S) * speed / EVICT_SECONDS, 0.0, 1.0)
    return 1.0 - np.clip((t - (count - 1 - rank) * STAGGER_S) * speed / FILL_SECONDS, 0.0, 1.0)


def _sweep_layers(noise: np.ndarray, sample_rate: int) -> np.ndarray:
    cutoffs = np.geomspace(SWEEP_TOP_HZ, SWEEP_BOTTOM_HZ, SWEEP_LAYERS)
    return np.stack([_lowpass(noise, min(c, 0.45 * sample_rate), sample_rate) for c in cutoffs])


def evict_sweep(duration: float, sample_rate: int, seed: int) -> np.ndarray:
    """A noise sweep that falls from bright to dark over `SWEEP_S` seconds and then stops: the sound of blocks draining away. Silent after."""
    n = int(duration * sample_rate)
    m = min(n, int(SWEEP_S * sample_rate))
    noise = np.random.default_rng(seed).uniform(-1.0, 1.0, m)
    layers = _sweep_layers(noise, sample_rate)
    position = np.linspace(0.0, SWEEP_LAYERS - 1.0, m)  # which filtered copy each sample takes, falling in cutoff
    lo = np.floor(position).astype(int)
    hi = np.minimum(lo + 1, SWEEP_LAYERS - 1)
    frac = position - lo
    k = np.arange(m)
    body = (1.0 - frac) * layers[lo, k] + frac * layers[hi, k]
    envelope = np.sin(np.pi * np.linspace(0.0, 1.0, m)) ** 0.5  # a swell and a fade, so it neither clicks in nor stops short
    out = np.zeros(n)
    out[:m] = SWEEP_LEVEL * envelope * body / max(1e-9, float(np.abs(body).max()))
    return out


def fill_sweep(duration: float, sample_rate: int, seed: int) -> np.ndarray:
    """The evict sweep run backwards, rising from dark to bright: the sound of blocks filling in. Silent after its length."""
    n = int(duration * sample_rate)
    m = min(n, int(SWEEP_S * sample_rate))
    out = np.zeros(n)
    out[:m] = evict_sweep(SWEEP_S, sample_rate, seed)[:m][::-1]
    return out


def phase_sweep(phase: str, elapsed: float, duration: float, sample_rate: int, seed: int) -> Optional[np.ndarray]:
    """The part of a phase's sweep that falls inside this render, or None when the sweep has finished or the phase has none. `elapsed` is the
    seconds into the phase at the start of the render."""
    if phase not in ("evicting", "filling") or elapsed >= SWEEP_S:
        return None
    full = (evict_sweep if phase == "evicting" else fill_sweep)(SWEEP_S, sample_rate, seed)
    n = int(duration * sample_rate)
    part = full[int(elapsed * sample_rate):][:n]
    return np.concatenate([part, np.zeros(n - len(part))])
