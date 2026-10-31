# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The playhead: the TUI shows the segment the listener is hearing, and the log lines that were made with it.

The live loop renders up to two segments ahead of the sound (`live.LEAD_IN_SEGMENTS`), and fast mode keeps a lead. A picture
drawn from the newest render would run ahead of the music by that much. So the loop hands each chunk of audio to the playhead
just before it is queued for the sink (`hand_over`), with the state it shows, and the playhead keeps a list of (offset, state):
the offset is where the chunk starts in the audio. Every log record is tagged with the audio handed over at the moment it was
logged (`note`), so a record logged while a segment is being made carries that segment's offset.

The listener's time is `clock() - t0 - buffer_s`. `t0` is the clock at the first write to the sink (`TimedSink` calls
`started`). For a real player that is the moment the first chunk goes into the player's pipe. The first sample is heard later
than that. `buffer_s` is the player's own buffer, when it is known (fast mode asks the player for one). In plain live mode the
player is started without a latency request, so its buffer is not known and `buffer_s` is 0. The picture can then lead the sound
by the player's start-up time and by whatever it buffers. These amounts have not been measured. With `--player none`
(`NullSink`) the timing is exact, because that sink takes the audio at the speed of the music.

`shown` gives the newest state whose offset the listener has reached, and the records whose tag the listener has reached.
For the first `buffer_s` seconds after the first write the listener's time is below zero. The opening lines (tagged 0) stay
shown through that time, and the first state waits for it to end.

The loop's calls (`hand_over`, `note`, and the sink's `started`) take the lock, append and return. All the drawing happens on the
TUI's own thread, which calls `shown` and gets back tuples it can keep: nothing the TUI does can hold up the audio.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Optional, Tuple

from ..ui.flow import FlowState

KEEP_RECORDS = 400  # the newest log records kept for the log pane; older ones are dropped as new ones arrive


@dataclass(frozen=True)
class Shown:
    """What the TUI draws now: the state of the segment being heard (None before any audio is heard), the log records the listener
    has reached, oldest first, and the listener's time in seconds (None before the first write)."""

    state: Optional[FlowState]
    records: Tuple[dict, ...]
    heard_s: Optional[float]


class Playhead:
    def __init__(self, clock: Callable[[], float] = time.monotonic, buffer_s: float = 0.0, keep: int = KEEP_RECORDS):
        self._clock = clock
        self._buffer_s = buffer_s  # the player's buffer when known; 0 when it is not, so the picture can then lead the sound
        self._lock = threading.Lock()
        self._t0: Optional[float] = None  # the clock at the first write to the sink, moved on after a gap
        self._delivered = 0.0  # seconds of audio handed over so far: where the next chunk starts
        self._states: Deque[Tuple[float, Optional[FlowState]]] = deque()  # (offset, state), oldest first
        self._records: Deque[Tuple[float, dict]] = deque(maxlen=keep)  # (offset when logged, record), oldest first

    # -- the loop's side: take the lock, append, return ----------------------------------------------------------------------------

    def hand_over(self, seconds: float, state: Optional[FlowState]) -> None:
        """A chunk of `seconds` that shows `state` is about to be queued for the sink. It starts where the audio handed over so
        far ends. If the listener has already run past that point, nothing was queued for a while (a gap): the listener's time
        starts again from the audio handed over, the same rule `fast.run_fast` uses."""
        with self._lock:
            if self._t0 is not None:
                now = self._clock()
                # A chunk queued less than buffer_s after the listener's clock passed `delivered` is no real gap (the player still
                # held audio), but t0 still moves, by less than buffer_s. This is the same rule `run_fast` uses.
                if now - self._t0 > self._delivered:
                    self._t0 = now - self._delivered
            self._states.append((self._delivered, state))
            self._delivered += seconds
            self._prune(self._heard())  # a drawing thread that is stuck never calls `shown`, so the list is kept short here too

    def set_buffer(self, seconds: float) -> None:
        """The player's buffer, once it is known (fast mode learns it when the sink is made): the listener hears audio this much after
        it is written."""
        with self._lock:
            self._buffer_s = seconds

    def note(self, record: dict) -> None:
        """`RunLog`'s `on_event`. The record is shown once the listener reaches the audio handed over at this moment, which is the
        start of the segment being made."""
        with self._lock:
            self._records.append((self._delivered, record))

    def started(self) -> None:
        """The sink is writing its first chunk: the listener's time starts now. Later calls change nothing."""
        with self._lock:
            if self._t0 is None:
                self._t0 = self._clock()

    # -- the TUI's side ------------------------------------------------------------------------------------------------------------

    def heard_s(self) -> Optional[float]:
        """How far into the audio the listener is, in seconds, or None before the first write."""
        with self._lock:
            return self._heard()

    def _heard(self) -> Optional[float]:
        if self._t0 is None:
            return None
        return self._clock() - self._t0 - self._buffer_s

    def shown(self) -> Shown:
        """The state of the newest chunk the listener has reached, and the records the listener has reached. Records logged
        before any audio was handed over (the run's opening lines) are shown at once. States the listener has passed are dropped."""
        with self._lock:
            heard = self._heard()
            reached = 0.0 if heard is None else max(heard, 0.0)  # for the records: the opening lines stay shown while the buffer fills
            state = None
            if heard is not None:
                self._prune(heard)
                if self._states and self._states[0][0] <= heard:
                    state = self._states[0][1]
            records = tuple(record for tag, record in self._records if tag <= reached)
            return Shown(state, records, heard)

    def _prune(self, heard: Optional[float]) -> None:
        """Drop the states the listener has passed: a later chunk has been reached, so the one before it is over. Called with the lock held."""
        if heard is None:
            return
        while len(self._states) > 1 and self._states[1][0] <= heard:
            self._states.popleft()

    def last_records(self, n: int) -> Tuple[dict, ...]:
        """The newest `n` records, oldest first, whether or not the listener has reached them. The TUI prints these when the
        run ends, after the sound has stopped, so a record logged ahead of the listener (an error at the end) is included."""
        with self._lock:
            return tuple(record for _, record in self._records)[-n:] if n > 0 else ()

    def wrap(self, sink) -> "TimedSink":
        """`sink`, with its first write starting the listener's time."""
        return TimedSink(sink, self)


class TimedSink:
    """A sink that tells the playhead when the first audio is written and then passes everything to the sink inside it.

    Every name it does not define (`command`, `sinks` and the rest) is looked up on the inner sink, so code that asks a sink which
    player it runs (`fast.player_in_use`) gets the same answer through the wrapper."""

    def __init__(self, inner, playhead: Playhead):
        self._inner = inner
        self._playhead = playhead

    def write(self, audio) -> None:
        self._playhead.started()
        self._inner.write(audio)

    def close(self) -> None:
        self._inner.close()

    def __getattr__(self, name: str):
        if name in ("_inner", "_playhead"):  # not set yet (a copy being built): never look these up on the inner sink
            raise AttributeError(name)
        return getattr(self._inner, name)
