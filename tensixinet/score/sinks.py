# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Where a render's `Score` goes: a Standard MIDI File, or several places at once."""
from typing import List, Protocol, Tuple

from .events import Score
from .midi import Message, to_messages
from .routing import DEFAULT_ROUTES, Routes
from .smf import write_midi_file


class ScoreSink(Protocol):
    """Takes each render's score and the time (seconds from the start of the run) it begins at."""

    def add(self, score: Score, start_s: float) -> None: ...

    def close(self) -> None: ...


class TeeScoreSink:
    """Hands every score to each sink in turn."""

    def __init__(self, *sinks) -> None:
        self.sinks = sinks

    def add(self, score: Score, start_s: float) -> None:
        for sink in self.sinks:
            sink.add(score, start_s)

    def close(self) -> None:
        for sink in self.sinks:
            sink.close()


class MidiFileSink:
    """Keeps every score's messages, shifted to its start, and writes one file on `close()`.
    The tempo in the file is the first score's. With no scores the file holds only the tempo track."""

    DEFAULT_BPM = 120.0

    def __init__(self, path, routes: Routes = DEFAULT_ROUTES) -> None:
        self.path = path
        self.routes = routes
        self.bpm = None
        self.messages: List[Message] = []
        self.notes = self.triggers = self.controls = 0

    def add(self, score: Score, start_s: float) -> None:
        if self.bpm is None:
            self.bpm = score.bpm
        self.notes += len(score.notes())
        self.triggers += len(score.triggers())
        self.controls += len(score.controls())
        self.messages.extend((t + start_s, data, chip) for t, data, chip in to_messages(score, self.routes))

    def summary(self) -> Tuple[int, int, int]:
        """How many notes, triggers and controls have been added."""
        return self.notes, self.triggers, self.controls

    def close(self) -> None:
        write_midi_file(self.path, self.messages, self.bpm or self.DEFAULT_BPM)
