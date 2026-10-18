# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Live mode: read the chips every turn, render that turn, and play it while the next one is made.

A turn is the unit. Each turn the chips are read afresh, the readings are compared with the last ones
(temperature and power get a rate, error counters become what is new), the soloist is drawn from the
chips' current weights, and one segment of the chorus is rendered and cross-faded onto the end of the
last. A loudness gain follows the music as it goes.

The loop stays about one segment ahead of the sound. A second thread hands audio to the sink, whose
writes block at the speed of the music (a player's pipe fills, or `NullSink` sleeps). The loop reads the
chips only once a segment has finished playing, so the reading behind each segment is at most one
segment old when that segment starts.
"""
from __future__ import annotations

import dataclasses
import math
import queue
import threading
import time
from typing import Callable, List, Optional, Sequence

import numpy as np

from .music.chorus import assign_device, chip_devices, n_slots, render_chorus, slot_chip, slot_weights, turn_seconds
from .music.eq import Equaliser
from .music.live import XFADE_S, Stitcher, Tracker
from .music.loudness import LiveGain
from .music.mapping import chip_chaos, chip_pressure, rate_activity
from .music.patch import DEFAULT_PATCH, Patch
from .music.turns import pick_soloist
from .narrate.explain import explain_events
from .narrate.runlog import RunLog
from .media.player import PlayerError
from .run import describe_readings, log_eq, log_only
from .telemetry import Telemetry

MIN_SEGMENT_S = 2.0  # a segment is at least this long: the crossfade is a fifth of it
MAX_SEGMENT_S = 10.0  # and at most this long: the chips are read once per segment
MOTION_LOGGED = 0.1  # a chip whose readings move at a tenth of the full rate or more gets a log line
LEAD_IN_SEGMENTS = 2  # how many segments the loop may render before the first one has finished playing


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def segment_seconds(turn_s: float, remaining: Optional[float]) -> float:
    """How long the next segment is: a turn, held between the minimum and the maximum. The last segment
    takes up whatever would be left over if it were any shorter than the minimum."""
    seconds = min(max(turn_s, MIN_SEGMENT_S), MAX_SEGMENT_S)
    if remaining is not None and remaining - seconds < MIN_SEGMENT_S:
        seconds = max(remaining, MIN_SEGMENT_S)
    return seconds


def run_live(
    sample: Callable[[], Sequence[Telemetry]],
    sink,
    log: RunLog,
    sample_rate: int = 44100,
    duration: Optional[float] = 60.0,
    source: str = "the telemetry source",
    patch: Patch = DEFAULT_PATCH,
    patch_source: str = "the built-in default patch",
    normalise: bool = True,
    clock: Callable[[], float] = time.monotonic,
    eq: bool = True,
    only: Optional[Sequence[str]] = None,
) -> int:
    """Play the chips live for `duration` seconds (until Ctrl-C when `None`). Returns the number of segments.

    `sample` returns the current readings, an empty list when they could not be read. `sink` takes the
    audio (see media/player.py). The last segment rings out for `XFADE_S` more seconds, so the sound
    lasts the duration plus that. Raises `ValueError` when the very first reading is empty and
    `PlayerError` when the player stops early."""
    audio_queue: "queue.Queue[Optional[np.ndarray]]" = queue.Queue()
    room = threading.Semaphore(LEAD_IN_SEGMENTS)
    failures: List[BaseException] = []

    def play() -> None:
        while True:
            chunk = audio_queue.get()
            if chunk is None:
                return
            try:
                if not failures:
                    sink.write(chunk)
            except BaseException as e:  # reported by the loop, which then stops
                failures.append(e)
            finally:
                room.release()

    player = threading.Thread(target=play, name="tensixinet-player", daemon=True)
    player.start()

    stitcher = Stitcher(sample_rate, XFADE_S)
    equaliser = Equaliser(sample_rate) if eq else None
    gain = LiveGain() if normalise else None
    tracker = Tracker()
    last_good: Optional[Sequence[Telemetry]] = None
    soloist: Optional[int] = None
    played = 0.0
    segment = 0
    try:
        while duration is None or played < duration - 1e-9:
            room.acquire()
            if failures:
                break
            readings = sample()
            now = clock()
            if not readings:
                if last_good is None:
                    log.event("discover", "no chips found, nothing to play",
                              f"{source} returned zero devices; refusing to invent any")
                    raise ValueError("no chips in telemetry")
                chips = [dataclasses.replace(t, temp_rate=0.0, power_rate=0.0, corr_errs=0, uncorr_errs=0, thermal_trips=0)
                         for t in last_good]
                log.event("live", f"segment {segment}: no chips in this reading",
                          "the last readings that worked are played again with no movement and no new errors",
                          segment=segment)
            else:
                if last_good is not None and len(last_good) != len(readings):
                    log.event("live", f"the reading changed from {_plural(len(last_good), 'chip')} to {len(readings)}",
                              "chips are paired by position, so with a different number the movement since the last reading "
                              "cannot be told; the music follows the new number",
                              segment=segment)
                chips = tracker.update(readings, now)
                last_good = readings

            turn_s = turn_seconds(chips, patch)
            remaining = None if duration is None else duration - played
            seconds = segment_seconds(turn_s, remaining)
            if segment == 0:
                describe_readings(chips, seconds, log, source, patch, patch_source, turns=False)
                log_only(log, only)
                log_eq(log, eq)
            else:
                _log_changes(log, chips, segment, patch)
            weights = slot_weights(chips)
            soloist = pick_soloist(weights, segment, soloist)
            audio = render_chorus(chips, seconds + XFADE_S, sample_rate, patch, segment, [soloist], only=only)
            ready = _finish(stitcher.push(audio), equaliser)
            level = None
            if gain is not None:
                ready, level = gain.apply(ready)
            else:
                ready = np.clip(ready, -1.0, 1.0)  # the EQ's lift can pass full scale when nothing sets the level
            played += len(ready) / sample_rate
            chip = slot_chip(soloist, len(chips))
            name = assign_device(soloist).__name__
            log.event(
                "live",
                f"segment {segment}: {seconds:.1f} s, {name} on chip {chip} has the floor"
                + (f", gain x{level:.2f}" if level is not None else ""),
                f"the soloist is drawn from the chips' weights as they are now (chip {chip} holds "
                f"{weights[soloist] * _slots_on_chip(soloist, len(chips)) / sum(weights):.0%} of them) and nobody solos twice running; "
                "this segment is rendered from one fresh reading and crossfaded onto the last",
                segment=segment, chip=chip, device=name, seconds=round(seconds, 3),
            )
            audio_queue.put(ready)
            segment += 1
        if not failures:
            tail = _finish_tail(stitcher.flush(), equaliser, gain)
            room.acquire()
            audio_queue.put(tail)
    except KeyboardInterrupt:
        log.event("live", "stopped by Ctrl-C", "the segments already made are played to the end")
        audio_queue.put(_finish_tail(stitcher.flush(), equaliser, gain))
    finally:
        audio_queue.put(None)
        player.join()
        try:
            sink.close()
        except PlayerError as e:
            failures.append(e)
    if failures:
        error = failures[0]
        log.event("player", f"the audio output stopped: {error}", "a live run ends when it has nowhere to play")
        raise error
    log.event("done", f"played {played:.1f} s in {_plural(segment, 'segment')}",
              "the chips were read once per segment; the last segment's tail rang out after the duration")
    return segment


def _finish(audio: np.ndarray, equaliser: Optional[Equaliser]) -> np.ndarray:
    return audio if equaliser is None else equaliser.process(audio)


def _finish_tail(tail: np.ndarray, equaliser: Optional[Equaliser], gain: Optional[LiveGain]) -> np.ndarray:
    """The last segment's faded-out tail, through the EQ and at the gain the music ended on. The EQ's own
    ring-out is not played: the tail has faded to nothing by then."""
    tail = _finish(tail, equaliser)
    if gain is not None and gain.gain is not None:
        tail = (tail * gain.gain).astype(np.float32)
    return np.clip(tail, -1.0, 1.0)


def _slots_on_chip(slot: int, n_chips: int) -> int:
    chip = slot_chip(slot, n_chips)
    return sum(1 for s in range(n_slots(n_chips)) if slot_chip(s, n_chips) == chip)


def _log_changes(log: RunLog, chips: Sequence[Telemetry], segment: int, patch: Patch) -> None:
    """What moved since the last reading: chips whose temperature or power is changing fast, and new errors."""
    n = len(chips)
    for i, t in enumerate(chips):
        activity = rate_activity(t.temp_rate, t.power_rate)
        if activity >= MOTION_LOGGED:
            plays_ttsidrax = any(d.__name__ == "ttsidrax" for d in chip_devices(i, n))
            why = (
                f"this chip plays ttsidrax, so its chaos knob is {chip_chaos(t):.0%}: pressure {chip_pressure(t):.2f} plus "
                f"{activity:.2f} for the faster of the two readings, which moves at {activity:.0%} of the full rate"
                if plays_ttsidrax else
                "this chip does not play ttsidrax, the only device whose sound follows how fast the readings move, "
                "so the movement is logged and only its pressure reaches the sound"
            )
            log.event("motion", f"chip {i}: ASIC {t.temp_rate:+.1f} C/s, power {t.power_rate:+.0f} W/s", why,
                      chip=i, segment=segment)
        if t.corr_errs or t.uncorr_errs or t.thermal_trips:
            d = explain_events(i, t)
            log.event("events", d.what.replace("reports", "reports a further", 1), d.why, chip=i, segment=segment)
