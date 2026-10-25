# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Tests for turning a Score into timed MIDI bytes."""
import dataclasses
import math

import pytest

from tensixinet.score.events import Control, Note, Score, Trigger
from tensixinet.score.midi import to_messages
from tensixinet.score.routing import DEFAULT_ROUTES


def score(*events, duration=10.0):
    return Score(events=tuple(events), duration=duration, bpm=120.0)


def hz(midi):
    return 440.0 * 2 ** ((midi - 69) / 12)


def bend_value(msg):
    """The 14-bit value in a pitch-bend message."""
    return msg[1] | (msg[2] << 7)


def test_lead_note_at_440_gives_centre_bend_on_and_off():
    out = to_messages(score(Note(1.0, 0.5, 440.0, 0.5, "lead", 2)))
    v = round(127 * 0.5)
    assert out == [
        (1.0, bytes([0xE0, 0x00, 0x40]), 2),
        (1.0, bytes([0x90, 69, v]), 2),
        (1.5, bytes([0x80, 69, 0]), 2),
    ]


def test_second_source_uses_its_own_channel():
    out = to_messages(score(Note(0.0, 1.0, 440.0, 1.0, "ttquantus", 0)))
    assert out[0][1][0] == 0xE1 and out[1][1][0] == 0x91 and out[2][1][0] == 0x81


def test_velocity_is_scaled_and_never_zero():
    out = to_messages(score(Note(0.0, 1.0, 440.0, 1.0, "lead", 0),
                            Note(2.0, 1.0, 440.0, 0.0, "lead", 0)))
    ons = [m for m in out if m[1][0] == 0x90]
    assert ons[0][1][2] == 127 and ons[1][1][2] == 1


def test_a_quarter_tone_sharp_sets_a_bend_of_that_size():
    # 69.4 semitones: note 69, remainder +0.4, range 2 semitones -> 0.2 of the half-wheel.
    out = to_messages(score(Note(0.0, 1.0, hz(69.4), 1.0, "lead", 0)))
    assert bend_value(out[0][1]) == 8192 + round(0.4 / 2.0 * 8191)
    assert out[1][1][1] == 69


def test_flat_remainder_bends_below_centre():
    out = to_messages(score(Note(0.0, 1.0, hz(68.6), 1.0, "lead", 0)))
    # nearest note is 69, remainder -0.4
    assert out[1][1][1] == 69
    assert bend_value(out[0][1]) == 8192 - round(0.4 / 2.0 * 8191)


def test_bend_range_scales_the_bend():
    wide = dataclasses.replace(DEFAULT_ROUTES, bend_range=12.0)
    out = to_messages(score(Note(0.0, 1.0, hz(69.4), 1.0, "lead", 0)), wide)
    assert bend_value(out[0][1]) == 8192 + round(0.4 / 12.0 * 8191)


def test_bend_points_follow_the_note_and_add_the_remainder():
    n = Note(1.0, 2.0, hz(69.4), 1.0, "lead", 0, bend=((0.5, 0.6), (1.0, -1.0)))
    out = to_messages(score(n))
    bends = [m for m in out if m[1][0] == 0xE0]
    assert [m[0] for m in bends] == [1.0, 1.5, 2.0]
    assert bend_value(bends[1][1]) == 8192 + round(1.0 / 2.0 * 8191)
    assert bend_value(bends[2][1]) == 8192 + round(-0.6 / 2.0 * 8191)


def test_bend_beyond_the_range_is_clamped_both_ways():
    n = Note(0.0, 2.0, 440.0, 1.0, "lead", 0, bend=((0.5, 5.0), (1.0, -5.0)))
    bends = [m for m in to_messages(score(n)) if m[1][0] == 0xE0]
    assert bytes(bends[1][1]) == bytes([0xE0, 0x7F, 0x7F])
    assert bend_value(bends[2][1]) == 1


def test_offs_sort_before_ons_and_bends_at_one_time():
    a = Note(0.0, 1.0, 440.0, 1.0, "lead", 0)
    b = Note(1.0, 1.0, 440.0, 1.0, "lead", 0)
    out = to_messages(score(b, a))
    at_one = [m[1][0] for m in out if m[0] == 1.0]
    assert at_one == [0x80, 0xE0, 0x90]


def test_input_order_is_kept_for_equal_messages():
    a = Note(0.0, 1.0, hz(60), 1.0, "lead", 0)
    b = Note(0.0, 1.0, hz(64), 1.0, "ttquantus", 1)
    out = to_messages(score(a, b))
    ons = [m[1][1] for m in out if m[1][0] & 0xF0 == 0x90]
    assert ons == [60, 64]


def test_note_past_the_end_is_cut_at_duration():
    out = to_messages(score(Note(8.0, 5.0, 440.0, 1.0, "lead", 0), duration=10.0))
    assert out[-1] == (10.0, bytes([0x80, 69, 0]), 0)


def test_bend_points_after_the_cut_are_dropped():
    n = Note(8.0, 5.0, 440.0, 1.0, "lead", 0, bend=((1.0, 0.5), (4.0, 0.5)))
    bends = [m for m in to_messages(score(n, duration=10.0)) if m[1][0] == 0xE0]
    assert [m[0] for m in bends] == [8.0, 9.0]


def test_a_note_starting_at_the_end_still_gets_its_off_after_its_on():
    out = to_messages(score(Note(10.0, 1.0, 440.0, 1.0, "lead", 0), duration=10.0))
    kinds = [m[1][0] for m in out]
    assert kinds.index(0x90) < kinds.index(0x80)


@pytest.mark.parametrize("midi,expected", [(130, 118), (140, 116), (-3, 9), (-14, 10)])
def test_notes_outside_the_range_move_by_octaves(midi, expected):
    out = to_messages(score(Note(0.0, 1.0, hz(midi), 1.0, "lead", 0)))
    on = [m for m in out if m[1][0] == 0x90][0][1][1]
    assert 0 <= on <= 127 and on % 12 == midi % 12
    assert on == expected


def test_note_range_edges_are_untouched():
    out = to_messages(score(Note(0.0, 1.0, hz(127), 1.0, "lead", 0),
                            Note(1.0, 1.0, hz(0), 1.0, "lead", 0)))
    assert [m[1][1] for m in out if m[1][0] == 0x90] == [127, 0]


def test_unknown_note_source_raises_naming_it():
    with pytest.raises(KeyError) as e:
        to_messages(score(Note(0.0, 1.0, 440.0, 1.0, "ttfuture", 0)))
    assert "ttfuture" in str(e.value)


def test_empty_score_gives_no_messages():
    assert to_messages(score()) == []


def test_gong_trigger_is_a_drum_note_with_an_off_pulse_later():
    out = to_messages(score(Trigger(2.0, 0.5, "gong", 3)))
    assert out == [(2.0, bytes([0x99, 49, 64]), 3), (2.05, bytes([0x89, 49, 0]), 3)]


def test_trigger_velocity_is_scaled_and_never_zero():
    out = to_messages(score(Trigger(0.0, 1.0, "gong", 0), Trigger(1.0, 0.0, "gong", 0)))
    assert [m[1][2] for m in out if m[1][0] == 0x99] == [127, 1]


def test_pulse_length_comes_from_the_routes():
    r = dataclasses.replace(DEFAULT_ROUTES, pulse_s=0.2)
    out = to_messages(score(Trigger(1.0, 1.0, "ttsteam", 0)), r)
    assert out[1][0] == pytest.approx(1.2) and out[1][1][1] == 38


def test_roll_lanes_cycle_through_the_lane_notes_and_the_clock_has_its_own():
    lanes = DEFAULT_ROUTES.roll_lane_notes
    evs = [Trigger(0.0, 1.0, "ttrollz", 0, lane=-1)] + [
        Trigger(float(i), 1.0, "ttrollz", 0, lane=i) for i in range(12)]
    ons = [m[1][1] for m in to_messages(score(*evs, duration=20.0)) if m[1][0] == 0x99]
    assert ons[0] == 36
    assert ons[1:] == [lanes[i % len(lanes)] for i in range(12)]


def test_unknown_trigger_source_raises_naming_it():
    with pytest.raises(KeyError) as e:
        to_messages(score(Trigger(0.0, 1.0, "ttfuture", 0)))
    assert "ttfuture" in str(e.value)


def test_trigger_off_for_the_same_note_comes_before_the_next_on_at_that_time():
    out = to_messages(score(Trigger(0.0, 1.0, "gong", 0), Trigger(0.05, 1.0, "gong", 0)))
    at = [m[1][0] for m in out if m[0] == pytest.approx(0.05)]
    assert at == [0x89, 0x99]


def test_trigger_off_is_cut_at_the_end_and_after_its_on():
    out = to_messages(score(Trigger(9.99, 1.0, "gong", 0), Trigger(10.0, 1.0, "gong", 0),
                            duration=10.0))
    assert max(m[0] for m in out) == 10.0
    last_two = [m[1][0] for m in out if m[0] == 10.0]
    assert last_two == [0x89, 0x99, 0x89]


def test_a_control_is_one_cc_on_the_control_channel():
    out = to_messages(score(Control(0.5, "pressure", 0.5, 2)))
    assert out == [(0.5, bytes([0xBF, 22, 64]), 2)]


def test_each_control_name_has_its_cc():
    names = ["temperature", "power", "pressure", "chaos"]
    out = to_messages(score(*[Control(0.0, n, 1.0, 0) for n in names]))
    assert [m[1][1] for m in out] == [20, 21, 22, 23]


def test_equal_integer_values_send_one_cc():
    out = to_messages(score(Control(0.0, "power", 0.500, 0), Control(1.0, "power", 0.501, 0),
                            Control(2.0, "power", 0.9, 0), Control(3.0, "power", 0.9, 0)))
    assert [(m[0], m[1][2]) for m in out] == [(0.0, 64), (2.0, 114)]


def test_a_value_that_returns_sends_again():
    out = to_messages(score(Control(0.0, "power", 0.0, 0), Control(1.0, "power", 1.0, 0),
                            Control(2.0, "power", 0.0, 0)))
    assert [m[1][2] for m in out] == [0, 127, 0]


def test_controls_are_tracked_per_chip_and_name():
    out = to_messages(score(Control(0.0, "power", 0.5, 0), Control(0.0, "power", 0.5, 1),
                            Control(0.0, "chaos", 0.5, 0)))
    assert len(out) == 3


def test_unknown_control_raises_naming_it():
    with pytest.raises(KeyError) as e:
        to_messages(score(Control(0.0, "mystery", 0.5, 0)))
    assert "mystery" in str(e.value)


def test_control_values_outside_zero_to_one_are_clamped():
    out = to_messages(score(Control(0.0, "power", 2.0, 0), Control(1.0, "power", -1.0, 0)))
    assert [m[1][2] for m in out] == [127, 0]


def test_messages_are_sorted_by_time_across_kinds():
    out = to_messages(score(Control(3.0, "power", 1.0, 0), Trigger(1.0, 1.0, "gong", 0),
                            Note(2.0, 0.5, 440.0, 1.0, "lead", 0)))
    times = [m[0] for m in out]
    assert times == sorted(times)


def test_a_bend_point_exactly_at_the_cut_is_kept():
    n = Note(8.0, 5.0, 440.0, 1.0, "lead", 0, bend=((2.0, 0.5),))
    bends = [m for m in to_messages(score(n, duration=10.0)) if m[1][0] == 0xE0]
    assert [m[0] for m in bends] == [8.0, 10.0]


def test_velocity_above_one_is_clamped_to_127():
    out = to_messages(score(Note(0.0, 1.0, 440.0, 3.0, "lead", 0), Trigger(0.0, 3.0, "gong", 0)))
    assert [m[1][2] for m in out if m[1][0] & 0xF0 == 0x90] == [127, 127]


def test_trigger_channel_comes_from_the_routes():
    r = dataclasses.replace(DEFAULT_ROUTES, trigger_channel=4)
    out = to_messages(score(Trigger(0.0, 1.0, "gong", 0)), r)
    assert [m[1][0] for m in out] == [0x94, 0x84]


def test_controls_given_out_of_time_order_are_deduplicated_in_time_order():
    out = to_messages(score(Control(2.0, "power", 0.5, 0), Control(1.0, "power", 0.5, 0),
                            Control(0.0, "power", 0.9, 0)))
    assert [(m[0], m[1][2]) for m in out] == [(0.0, 114), (1.0, 64)]


def _wire(out, status):
    return [(m[0], m[1][0] & 0xF0, m[1][1], m[2]) for m in out if m[1][0] & 0xF0 == status]


def test_same_pitch_overlap_across_chips_retriggers_and_drops_the_stale_off():
    a = Note(0.0, 2.0, 440.0, 1.0, "lead", 0)
    b = Note(1.0, 2.0, 440.0, 1.0, "lead", 1)
    out = to_messages(score(a, b))
    seq = [(m[0], m[1][0] & 0xF0, m[2]) for m in out if m[1][0] & 0xF0 in (0x80, 0x90)]
    assert seq == [(0.0, 0x90, 0), (1.0, 0x80, 0), (1.0, 0x90, 1), (3.0, 0x80, 1)]


def test_overlap_gives_exactly_one_off_per_sounding_note_and_none_mid_note():
    a = Note(0.0, 3.0, 440.0, 1.0, "lead", 0)
    b = Note(1.0, 3.0, 440.0, 1.0, "lead", 1)
    c = Note(2.0, 3.0, 440.0, 1.0, "lead", 2)
    out = to_messages(score(a, b, c))
    offs = _wire(out, 0x80)
    assert [o[0] for o in offs] == [1.0, 2.0, 5.0]
    assert len(_wire(out, 0x90)) == 3


def test_overlap_of_different_numbers_is_left_alone():
    a = Note(0.0, 2.0, hz(60), 1.0, "lead", 0)
    b = Note(1.0, 2.0, hz(62), 1.0, "lead", 1)
    out = to_messages(score(a, b))
    assert [o[0] for o in _wire(out, 0x80)] == [2.0, 3.0]


def test_same_number_on_different_channels_is_not_a_retrigger():
    a = Note(0.0, 2.0, 440.0, 1.0, "lead", 0)
    b = Note(1.0, 2.0, 440.0, 1.0, "ttquantus", 1)
    out = to_messages(score(a, b))
    assert [o[0] for o in _wire(out, 0x80)] == [2.0, 3.0]


def test_a_repeated_degree_one_ulp_apart_still_plays_the_second_note():
    assert 0.1 + 0.2 != 0.3  # the end of the first note is one ulp after the second's start
    a = Note(0.1, 0.2, 440.0, 1.0, "lead", 0)
    b = Note(0.3, 0.2, 440.0, 1.0, "lead", 0)
    out = to_messages(score(a, b))
    seq = [m[1][0] & 0xF0 for m in out if m[1][0] & 0xF0 in (0x80, 0x90)]
    assert seq == [0x90, 0x80, 0x90, 0x80]


def test_message_times_are_rounded_to_the_microsecond():
    out = to_messages(score(Note(0.1, 0.2, 440.0, 1.0, "lead", 0)))
    assert out[-1][0] == 0.3


def test_trigger_off_one_ulp_after_the_next_trigger_on_still_comes_first():
    r = dataclasses.replace(DEFAULT_ROUTES, pulse_s=0.2)
    out = to_messages(score(Trigger(0.1, 1.0, "gong", 0), Trigger(0.3, 1.0, "gong", 0)), r)
    seq = [m[1][0] & 0xF0 for m in out]
    assert seq == [0x90, 0x80, 0x90, 0x80]


def test_cc_comes_after_off_bend_and_on_at_one_time():
    out = to_messages(score(Note(0.0, 1.0, 440.0, 1.0, "lead", 0),
                            Control(1.0, "power", 1.0, 0),
                            Note(1.0, 1.0, 440.0, 1.0, "lead", 0)))
    assert [m[1][0] & 0xF0 for m in out if m[0] == 1.0] == [0x80, 0xE0, 0x90, 0xB0]


def test_times_keep_microsecond_resolution():
    out = to_messages(score(Note(0.123456, 0.000001, 440.0, 1.0, "lead", 0)))
    assert [m[0] for m in out] == [0.123456, 0.123456, 0.123457]


def test_retrigger_off_carries_the_chip_of_the_note_it_ends():
    a = Note(0.0, 2.0, 440.0, 1.0, "lead", 2)
    b = Note(1.0, 2.0, 440.0, 1.0, "lead", 3)
    out = to_messages(score(a, b))
    assert _wire(out, 0x80)[0] == (1.0, 0x80, 69, 2)


def test_a_bend_point_one_ulp_past_the_end_is_kept_at_the_end():
    n = Note(0.1, 0.2, 440.0, 1.0, "lead", 0, bend=((0.2, 0.5),))
    assert 0.1 + 0.2 > 0.3
    bends = [m for m in to_messages(score(n)) if m[1][0] == 0xE0]
    assert [m[0] for m in bends] == [0.1, 0.3]
