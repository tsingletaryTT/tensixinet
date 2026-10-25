# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Turn a `Score` into timed MIDI messages: `(seconds, raw bytes, chip)`, sorted by time."""
from typing import List, Tuple

from .events import Control, Note, Score, Trigger, note_and_bend
from .routing import DEFAULT_ROUTES, Routes

Message = Tuple[float, bytes, int]

# Order of messages that share a time: a note ending frees its number before anything
# starts, a bend lands before the note it belongs to, and controllers go last.
_OFF, _BEND, _ON, _CC, _LATE_OFF = 0, 1, 2, 3, 4

BEND_CENTRE = 8192
BEND_MAX = 8191


def _bend_bytes(channel: int, semitones: float, bend_range: float) -> bytes:
    """A 14-bit pitch-bend message. `bend_range` semitones is the full +-8191 swing; larger
    bends are clamped to the ends."""
    value = BEND_CENTRE + round(semitones / bend_range * BEND_MAX)
    value = max(BEND_CENTRE - BEND_MAX, min(BEND_CENTRE + BEND_MAX, value))
    return bytes([0xE0 | channel, value & 0x7F, value >> 7])


def _velocity(x: float) -> int:
    return max(1, min(127, round(127 * x)))


def _into_range(note: int) -> int:
    """Move a note number by whole octaves until it is 0 to 127."""
    while note < 0:
        note += 12
    while note > 127:
        note -= 12
    return note


def to_messages(score: Score, routes: Routes = DEFAULT_ROUTES) -> List[Message]:
    # Each entry is (time, class, order, message, chip); sorting on the first three keeps
    # the rules in one place and is stable for equal keys.
    items = []

    def add(t, cls, data, chip):
        items.append((t, cls, len(items), data, chip))

    for ev in score.events:
        if isinstance(ev, Note):
            if ev.source not in routes.note_channels:
                raise KeyError(f"no MIDI channel for note source {ev.source!r}")
            ch = routes.note_channels[ev.source]
            raw, remainder = note_and_bend(ev.pitch_hz)
            num = _into_range(raw)
            end = max(ev.t, min(ev.t + ev.dur, score.duration))
            add(ev.t, _BEND, _bend_bytes(ch, remainder, routes.bend_range), ev.chip)
            add(ev.t, _ON, bytes([0x90 | ch, num, _velocity(ev.velocity)]), ev.chip)
            for dt, semis in ev.bend:
                if ev.t + dt <= end:
                    add(ev.t + dt, _BEND, _bend_bytes(ch, remainder + semis, routes.bend_range), ev.chip)
            add(end, _OFF if end > ev.t else _LATE_OFF, bytes([0x80 | ch, num, 0]), ev.chip)
        elif isinstance(ev, Trigger):
            if ev.source == "ttrollz":
                # The roll clock is lane -1; the other lanes cycle through the lane notes.
                if ev.lane < 0:
                    num = routes.trigger_notes["ttrollz_clock"]
                else:
                    num = routes.roll_lane_notes[ev.lane % len(routes.roll_lane_notes)]
            elif ev.source in routes.trigger_notes:
                num = routes.trigger_notes[ev.source]
            else:
                raise KeyError(f"no MIDI note for trigger source {ev.source!r}")
            ch = routes.trigger_channel
            end = max(ev.t, min(ev.t + routes.pulse_s, score.duration))
            add(ev.t, _ON, bytes([0x90 | ch, num, _velocity(ev.strength)]), ev.chip)
            add(end, _OFF if end > ev.t else _LATE_OFF, bytes([0x80 | ch, num, 0]), ev.chip)
    # A controller goes out only when its 0 to 127 value changes, tracked per chip and name.
    last = {}
    for ev in score.controls():
        if ev.name not in routes.control_ccs:
            raise KeyError(f"no MIDI controller for control {ev.name!r}")
        value = max(0, min(127, round(127 * ev.value)))
        if last.get((ev.chip, ev.name)) == value:
            continue
        last[(ev.chip, ev.name)] = value
        add(ev.t, _CC, bytes([0xB0 | routes.control_channel, routes.control_ccs[ev.name], value]), ev.chip)
    items.sort(key=lambda i: i[:3])
    return [(t, data, chip) for t, _, _, data, chip in items]
