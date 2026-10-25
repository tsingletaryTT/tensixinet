# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The events a render is made of, as plain data, and the pitch helpers MIDI needs.

A `Score` lists what the devices played: pitched `Note`s, `Trigger`s (a hit with no pitch,
such as a gong strike or a roll pulse) and `Control`s (a named value that moves, such as a
chip's pressure). Times are seconds from the start of the segment the score describes.
Sinks (a MIDI file, a live port, later CV) read a score and never touch the audio.
"""
import math
from dataclasses import dataclass, replace
from typing import Tuple, Union

# MIDI note 69 is A4 at 440 Hz; twelve semitones to the octave.
A4_HZ = 440.0
A4_NOTE = 69


@dataclass(frozen=True)
class Note:
    """A pitched note. `velocity` is 0 to 1. `bend` is (seconds from the note's start,
    semitones from `pitch_hz`) pairs, empty for a steady note."""
    t: float
    dur: float
    pitch_hz: float
    velocity: float
    source: str
    chip: int
    bend: Tuple[Tuple[float, float], ...] = ()


@dataclass(frozen=True)
class Trigger:
    """An unpitched hit. `strength` is 0 to 1; `lane` tells apart several triggers of one source."""
    t: float
    strength: float
    source: str
    chip: int
    lane: int = 0


@dataclass(frozen=True)
class Control:
    """A named control value (0 to 1) at one moment, for one chip."""
    t: float
    name: str
    value: float
    chip: int


Event = Union[Note, Trigger, Control]


@dataclass(frozen=True)
class Score:
    """Every event of a segment, with the segment's length and tempo."""
    events: Tuple[Event, ...]
    duration: float
    bpm: float

    def _of(self, kind) -> tuple:
        # sorted() is stable, so events at the same time keep the order they were given in.
        return tuple(sorted((e for e in self.events if isinstance(e, kind)), key=lambda e: e.t))

    def notes(self) -> Tuple[Note, ...]:
        return self._of(Note)

    def triggers(self) -> Tuple[Trigger, ...]:
        return self._of(Trigger)

    def controls(self) -> Tuple[Control, ...]:
        return self._of(Control)

    def shifted(self, dt: float) -> "Score":
        """The same score `dt` seconds later (every event's `t` plus `dt`)."""
        return replace(self, events=tuple(replace(e, t=e.t + dt) for e in self.events))


def hz_to_midi(hz: float) -> float:
    """Fractional MIDI note number of a frequency (A4 = 440 Hz = 69.0)."""
    return A4_NOTE + 12.0 * math.log2(hz / A4_HZ)


def note_and_bend(hz: float) -> Tuple[int, float]:
    """The nearest integer MIDI note and what is left over, in semitones, within [-0.5, 0.5]."""
    exact = hz_to_midi(hz)
    note = int(round(exact))
    return note, exact - note
