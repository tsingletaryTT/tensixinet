# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The knobs of the Play tab: slider values and events that stand in for chip readings.

The instruments are played by the readings, so a performer's hands are the readings. `Knobs` holds five values per chip (temperature, power,
aiclk, GDDR spread, voltage) and a queue of events (a memory error, a thermal trip, a load burst). `to_telemetry` turns them into one
`Telemetry` per chip, once per block, and passes the list through a `Tracker`, so moving a slider fast gives the chip a temperature or power
rate (which moves ttrax's chaos) exactly as a real reading that changes fast does.

Everything the knobs do not set comes from `mock_telemetry`, generated once for the largest chip count so a chip keeps its masks (its identity
in the music) whatever the chip count is.
"""
from __future__ import annotations

import dataclasses
import threading
from typing import Dict, List, Optional, Tuple

from ..music.consonants import CONSONANTS
from ..music.live import Tracker
from ..music.ttchoir import VOWEL_ORDER
from ..telemetry import Telemetry, mock_telemetry

MAX_CHIPS = 4

# name: (low, high, default). Temperature and aiclk cover what mapping.py maps to pitch and tempo; power reaches past the TDP limit so
# pressure can be pushed to its top.
KNOB_RANGES: Dict[str, Tuple[float, float, float]] = {
    "temperature": (20.0, 105.0, 50.0),  # deg C, the ASIC
    "power": (5.0, 150.0, 30.0),  # W
    "aiclk": (200.0, 1350.0, 800.0),  # MHz
    "gddr_spread": (0.0, 30.0, 8.0),  # deg C between the coolest and the hottest of the eight GDDR channels
    "voltage": (0.65, 0.95, 0.75),  # V
}
EVENTS = ("memory_error", "thermal_trip", "load_burst")
LOAD_BURST_W = 70.0  # a burst adds this much power to its chip ...
LOAD_BURST_S = 4.0  # ... for this long, from the block that takes the event
# What the choir's two pickers offer: "" leaves that part of the next syllables to the readings.
CHOIR_ONSETS: Tuple[str, ...] = ("",) + tuple(CONSONANTS)
CHOIR_VOWELS: Tuple[str, ...] = ("",) + tuple(VOWEL_ORDER)
GDDR_TOP_C = 40.0  # the hottest GDDR channel; the rest sit below it by the spread

_BASE: Optional[List[Telemetry]] = None


def _base() -> List[Telemetry]:
    """The readings the knobs do not set (masks, heartbeat, DDR status and so on), for the largest chip count, made once."""
    global _BASE
    if _BASE is None:
        _BASE = mock_telemetry(MAX_CHIPS, seed=0)
    return _BASE


def _check_chips(n_chips: int) -> None:
    if not 1 <= n_chips <= MAX_CHIPS:
        raise ValueError(f"the Play tab plays 1 to {MAX_CHIPS} chips, got {n_chips}")


def _defaults() -> Dict[str, float]:
    return {name: default for name, (_, _, default) in KNOB_RANGES.items()}


class Knobs:
    """Thread-safe store of each chip's slider values, pending events, running error counts and load bursts.

    It remembers the values of all `MAX_CHIPS` chips and plays the first `n_chips` of them (`resize` changes that number), so a chip the page
    hides still has the value its slider shows when it comes back.

    The page's handlers write it from Gradio's threads and the render thread reads it once per block, so every method takes the lock."""

    def __init__(self, n_chips: int):
        _check_chips(n_chips)
        self._lock = threading.Lock()
        self._n = n_chips
        self._values: List[Dict[str, float]] = [_defaults() for _ in range(MAX_CHIPS)]
        self._pending: List[Tuple[int, str]] = []
        self._counts: List[Dict[str, int]] = [{"corr_errs": 0, "thermal_trips": 0} for _ in range(MAX_CHIPS)]
        self._burst_until: List[Optional[float]] = [None] * MAX_CHIPS
        self._syllable: Tuple[str, str] = ("", "")  # the choir's picked onset and vowel; "" leaves that part to the readings

    def set(self, chip: int, name: str, value: float) -> None:
        """Set one knob, held inside its range. An unknown chip or knob, and NaN, are ignored. A chip that is not being played keeps its value."""
        if name not in KNOB_RANGES or value != value:
            return
        low, high, _ = KNOB_RANGES[name]
        with self._lock:
            if 0 <= chip < MAX_CHIPS:
                self._values[chip][name] = min(high, max(low, float(value)))

    def get(self, chip: int, name: str) -> float:
        with self._lock:
            return self._values[chip][name] if 0 <= chip < MAX_CHIPS else KNOB_RANGES[name][2]

    def values(self) -> List[Dict[str, float]]:
        """A copy of the knob values of the chips that are being played."""
        with self._lock:
            return [dict(v) for v in self._values[: self._n]]

    def set_syllable(self, onset: str, vowel: str) -> None:
        """Pick the choir's onset consonant and vowel (either may be "" for the readings' own choice). A pair with an unknown value is ignored
        whole, so a half-bad request never leaves half a pick."""
        if onset not in CHOIR_ONSETS or vowel not in CHOIR_VOWELS:
            return
        with self._lock:
            self._syllable = (onset, vowel)

    def syllable(self) -> Tuple[str, str]:
        """The choir's picked (onset, vowel); it holds until it is changed."""
        with self._lock:
            return self._syllable

    def trigger(self, chip: int, event: str) -> None:
        """Queue an event for the next block. An unknown chip or event is ignored."""
        with self._lock:
            if event in EVENTS and 0 <= chip < self._n:
                self._pending.append((chip, event))

    def take_events(self) -> List[Tuple[int, str]]:
        """The pending events in the order they were triggered; the queue is emptied."""
        with self._lock:
            taken, self._pending = self._pending, []
            return taken

    def resize(self, n_chips: int) -> None:
        """Play `n_chips` chips. Every chip keeps its values; events for chips that are no longer played are dropped."""
        _check_chips(n_chips)
        with self._lock:
            self._n = n_chips
            self._pending = [(c, e) for c, e in self._pending if c < n_chips]

    def advance(self, now: float):
        """Take the pending events and apply them, in one step: errors and trips are added to the chip's running counts, a load burst runs
        until `now + LOAD_BURST_S`. Returns (values, counts, bursting) for every chip: a copy of the knob values, of the running counts, and
        whether a burst is on at `now`."""
        with self._lock:
            events, self._pending = self._pending, []
            for chip, event in events:
                if event == "memory_error":
                    self._counts[chip]["corr_errs"] += 1
                elif event == "thermal_trip":
                    self._counts[chip]["thermal_trips"] += 1
                else:
                    self._burst_until[chip] = now + LOAD_BURST_S
            bursting = [until is not None and now < until for until in self._burst_until]
            return [dict(v) for v in self._values[: self._n]], [dict(c) for c in self._counts[: self._n]], bursting[: self._n]


def gddr_temps(spread: float) -> Tuple[float, ...]:
    """Eight channel temperatures from `GDDR_TOP_C - spread` up to `GDDR_TOP_C`, evenly spaced, so max minus min is `spread`."""
    return tuple(round(GDDR_TOP_C - spread * (7 - i) / 7, 3) for i in range(8))


def to_telemetry(knobs: Knobs, tracker: Tracker, now: float) -> List[Telemetry]:
    """One `Telemetry` per chip from the knobs, with the events taken. The running error counts go through `tracker`, which turns them into
    what is new since the last block, so an error is heard in one block and a slider's movement becomes a temperature and power rate."""
    values, counts, bursting = knobs.advance(now)
    readings = []
    for chip, (v, c, burst) in enumerate(zip(values, counts, bursting)):
        base = _base()[chip]
        power = v["power"] + (LOAD_BURST_W if burst else 0.0)
        readings.append(dataclasses.replace(
            base,
            asic_temperature=v["temperature"],
            aiclk=v["aiclk"],
            power=power,
            current=round(power * 1.3, 3),  # mock_telemetry's ratio: the chorus mixes the chips by current
            voltage=v["voltage"],
            gddr_temps=gddr_temps(v["gddr_spread"]),
            board_power=round(base.board_power - base.power + power, 3),
            corr_errs=c["corr_errs"],
            uncorr_errs=0,
            thermal_trips=c["thermal_trips"],
        ))
    return tracker.update(readings, now)
