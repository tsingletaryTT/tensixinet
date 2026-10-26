# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Live mode's two pure pieces: what changed between two readings, and joining segments without a click.

A live run plays the chorus one segment (one turn) at a time. Each segment is rendered from a fresh reading,
so it starts its own clocks from zero and its devices begin their phases afresh. `Stitcher` overlaps each
segment's tail with the next segment's start and cross-fades them, so the join is smooth. The cost: a gong
that is still ringing when its segment ends fades out over the crossfade.
"""
from __future__ import annotations

import dataclasses
import math
from typing import List, Optional, Sequence

import numpy as np

from ..telemetry import Telemetry

XFADE_S = 0.4  # how long the end of one segment and the start of the next overlap
RATE_WINDOW_S = 2.0  # a rate is measured over at least this long, because readings jitter by a few tenths of a degree


def advance(previous: Optional[Sequence[Telemetry]], current: Sequence[Telemetry], seconds: float) -> List[Telemetry]:
    """`current`, with what moved since `previous` filled in.

    Temperature and power get a rate per second. The three event counters (memory errors, thermal trips)
    count up for the life of the chip, so they become what is new since the last reading, and a segment
    plays only the new events, and an old error does not tick in every segment. The first reading, and a reading with a
    different number of chips (the chips can no longer be paired), stand alone with their totals as reported.
    """
    if previous is None or len(previous) != len(current):
        return list(current)
    out = []
    for before, now in zip(previous, current):
        moving = seconds > 0
        out.append(dataclasses.replace(
            now,
            temp_rate=(now.asic_temperature - before.asic_temperature) / seconds if moving else 0.0,
            power_rate=(now.power - before.power) / seconds if moving else 0.0,
            corr_errs=max(0, now.corr_errs - before.corr_errs),
            uncorr_errs=max(0, now.uncorr_errs - before.uncorr_errs),
            thermal_trips=max(0, now.thermal_trips - before.thermal_trips),
        ))
    return out


class Tracker:
    """Turns a stream of readings into readings that carry rates and new errors.

    New errors are counted from the previous reading. A rate is measured from a baseline reading that is at
    least `window_s` old, because the loop can take two readings a fraction of a second apart and a 0.3 degree
    wobble over 0.3 s is not a heating chip. Readings that come sooner than that carry a rate of 0 and the
    baseline waits for them. Once a rate has been measured, the latest reading becomes the new baseline. A
    different number of chips starts the tracker over.

    With `hold`, a reading inside the window carries the last rate measured instead of 0. Fast mode reads the chips
    many times a window, and without it most of its readings would say the chips are still."""

    def __init__(self, window_s: float = RATE_WINDOW_S, hold: bool = False):
        self.window_s = window_s
        self.hold = hold
        self._previous: Optional[List[Telemetry]] = None
        self._baseline: List[Telemetry] = []
        self._baseline_at = 0.0
        self._rates: Optional[List[tuple]] = None  # with hold: (temp_rate, power_rate) for each chip, as last measured

    def update(self, readings: Sequence[Telemetry], now: float) -> List[Telemetry]:
        readings = list(readings)
        if self._previous is None or len(self._previous) != len(readings):
            self._previous, self._baseline, self._baseline_at = readings, readings, now
            self._rates = None
            return readings
        out = advance(self._previous, readings, 0.0)  # new errors since the last reading; no rates yet
        elapsed = now - self._baseline_at
        if elapsed >= self.window_s:
            rated = advance(self._baseline, readings, elapsed)
            out = [dataclasses.replace(o, temp_rate=r.temp_rate, power_rate=r.power_rate) for o, r in zip(out, rated)]
            self._baseline, self._baseline_at = readings, now
            self._rates = [(r.temp_rate, r.power_rate) for r in rated]
        elif self.hold and self._rates is not None:
            out = [dataclasses.replace(o, temp_rate=tr, power_rate=pr) for o, (tr, pr) in zip(out, self._rates)]
        self._previous = readings
        return out


class Stitcher:
    """Joins segments end to end. Each pushed segment carries `xfade_s` extra seconds at its end.

    The extra seconds are kept back, faded out, and added to the start of the next segment, which fades in
    with the matching curve (sine in, cosine out: the two have equal power, so the join keeps its level
    when the two sides are unrelated, as consecutive segments are)."""

    def __init__(self, sample_rate: int, xfade_s: float = XFADE_S):
        self.n = max(1, int(xfade_s * sample_rate))
        k = (np.arange(self.n) + 0.5) / self.n
        self._fade_in = np.sin(0.5 * math.pi * k)
        self._fade_out = np.cos(0.5 * math.pi * k)
        self._tail: Optional[np.ndarray] = None
        self._channels = 1  # 1 for mono segments, 2 for stereo ones, as last pushed

    def push(self, segment: np.ndarray) -> np.ndarray:
        """The audio that is ready to play: the segment less its tail. The tail waits for the next push. A segment is mono, shape (n,),
        or stereo, shape (n, 2); both ears take the same crossfade."""
        if len(segment) <= 2 * self.n:
            raise ValueError(f"a segment needs more than two crossfades of audio ({2 * self.n} samples), got {len(segment)}")
        audio = segment.astype(np.float64)  # a copy, so the caller's array is left alone
        fade_in, fade_out = (self._fade_in[:, None], self._fade_out[:, None]) if audio.ndim == 2 else (self._fade_in, self._fade_out)
        self._channels = audio.shape[1] if audio.ndim == 2 else 1
        if self._tail is not None:
            audio[: self.n] = audio[: self.n] * fade_in + self._tail
        self._tail = audio[-self.n:] * fade_out
        return audio[: -self.n].astype(segment.dtype)

    def flush(self) -> np.ndarray:
        """The last segment's faded-out tail, once. Empty when there is none."""
        tail, self._tail = self._tail, None
        if tail is None:
            return np.zeros((0, self._channels) if self._channels > 1 else 0, dtype=np.float32)
        return tail.astype(np.float32)
