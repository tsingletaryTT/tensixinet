# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Tests for the score event types and the pitch helpers."""
import dataclasses
import math

import pytest

from tensixinet.score.events import (
    Control, Note, Score, Trigger, hz_to_midi, note_and_bend,
)


def _note(t, **kw):
    base = dict(t=t, dur=0.5, pitch_hz=440.0, velocity=0.8, source="lead", chip=0)
    base.update(kw)
    return Note(**base)


def test_hz_to_midi_anchors():
    assert hz_to_midi(440.0) == pytest.approx(69.0)
    assert hz_to_midi(880.0) == pytest.approx(81.0)
    assert hz_to_midi(220.0) == pytest.approx(57.0)


def test_note_and_bend_on_pitch():
    note, rem = note_and_bend(466.16)
    assert note == 70
    assert abs(rem) < 0.01


def test_note_and_bend_quarter_tone_adds_back():
    hz = 452.89
    note, rem = note_and_bend(hz)
    assert note in (69, 70)
    assert abs(abs(rem) - 0.5) < 0.01
    assert note + rem == pytest.approx(hz_to_midi(hz))


def test_note_and_bend_keeps_sign_below_the_note():
    # a little flat of A4: nearest note 69, remainder negative
    note, rem = note_and_bend(440.0 * 2 ** (-0.3 / 12))
    assert note == 69
    assert rem == pytest.approx(-0.3)
    note, rem = note_and_bend(440.0 * 2 ** (0.3 / 12))
    assert note == 69
    assert rem == pytest.approx(0.3)


def test_remainder_always_within_half_a_semitone():
    for hz in (30.0, 100.0, 333.3, 1234.5, 5000.0):
        note, rem = note_and_bend(hz)
        assert isinstance(note, int)
        assert -0.5 <= rem <= 0.5


def test_events_are_frozen():
    for ev in (_note(0.0), Trigger(0.0, 1.0, "gong", 0), Control(0.0, "x", 0.5, 0)):
        with pytest.raises(dataclasses.FrozenInstanceError):
            ev.t = 1.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        Score((), 1.0, 120.0).bpm = 1.0


def test_defaults():
    assert _note(0.0).bend == ()
    assert Trigger(0.0, 1.0, "gong", 0).lane == 0


def _mixed():
    return Score(
        events=(
            _note(2.0), Trigger(1.5, 0.9, "gong", 1, lane=2), Control(0.5, "pressure", 0.3, 0),
            _note(0.25, source="other"), Trigger(0.1, 0.5, "ttrollz", 0), Control(3.0, "chaos", 0.1, 1),
        ),
        duration=4.0, bpm=96.0,
    )


def test_filters_return_only_their_type_in_time_order():
    s = _mixed()
    assert [e.t for e in s.notes()] == [0.25, 2.0]
    assert [e.t for e in s.triggers()] == [0.1, 1.5]
    assert [e.t for e in s.controls()] == [0.5, 3.0]
    assert all(isinstance(e, Note) for e in s.notes())
    assert all(isinstance(e, Trigger) for e in s.triggers())
    assert all(isinstance(e, Control) for e in s.controls())
    assert isinstance(s.notes(), tuple)
    assert isinstance(s.triggers(), tuple)
    assert isinstance(s.controls(), tuple)


def test_filters_are_stable_for_equal_times():
    a, b = _note(1.0, source="a"), _note(1.0, source="b")
    assert [n.source for n in Score((a, b), 2.0, 120.0).notes()] == ["a", "b"]
    assert [n.source for n in Score((b, a), 2.0, 120.0).notes()] == ["b", "a"]


def test_shifted_moves_every_event_and_keeps_the_rest():
    s = _mixed()
    d = s.shifted(10.0)
    assert d.duration == s.duration and d.bpm == s.bpm
    assert len(d.events) == len(s.events)
    for old, new in zip(s.events, d.events):
        assert type(new) is type(old)
        assert new.t == pytest.approx(old.t + 10.0)
        assert dataclasses.replace(new, t=old.t) == old
    assert any(isinstance(e, Control) for e in d.events)
    assert s.events[0].t == 2.0  # original untouched


def test_shifted_keeps_bend_curve():
    n = _note(1.0, bend=((0.0, 0.0), (0.2, 0.5)))
    assert Score((n,), 2.0, 120.0).shifted(1.0).notes()[0].bend == ((0.0, 0.0), (0.2, 0.5))
