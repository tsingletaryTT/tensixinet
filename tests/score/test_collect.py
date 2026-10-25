# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Tests for the per-device event extractors: each must agree with the function the audio reads."""
import dataclasses

import numpy as np
import pytest

from tensixinet.music import devices
from tensixinet.music.lead import lead_notes
from tensixinet.music.mapping import chip_chaos, chip_pressure, norm_temp, temp_to_pitch_hz
from tensixinet.music.patch import DEFAULT_PATCH
from tensixinet.music.ttrolzer import SEGMENT_SEED_STRIDE
from tensixinet.music.ttsteam import steam_hits
from tensixinet.score import collect
from tensixinet.score.events import Control, Note, Trigger
from tensixinet.telemetry import FALLBACK

SR = 8000
DUR = 12.0
CHIP = 2
# a seed as the chorus builds it: segment 3 of the run, slot 5
SEED = 3 * SEGMENT_SEED_STRIDE + 5


def tel(**kw):
    return dataclasses.replace(FALLBACK, **kw)


T = tel(tensix_mask=0x37F7, asic_temperature=55.0, aiclk=900.0, gddr_temps=(30., 34., 29., 38., 31., 27., 36., 30.))
MOVING = dataclasses.replace(DEFAULT_PATCH, quantus_mod=0.8)


# --- lead ---
def test_lead_events_are_the_lead_notes():
    patch = dataclasses.replace(DEFAULT_PATCH, lead_rests=0.4)
    ev = collect.lead_events(T, CHIP, DUR, SR, SEED, patch)
    ref = lead_notes(T, DUR, SR, SEED, patch)
    assert len(ev) == len(ref) > 0
    for e, (s, d, hz) in zip(ev, ref):
        assert isinstance(e, Note)
        assert (e.t, e.dur, e.pitch_hz, e.velocity, e.source, e.chip, e.bend) == (s, d, hz, 1.0, "lead", CHIP, ())


# --- gong ---
def test_gong_events_are_strike_times_of_the_segment_in_the_seed():
    ev = collect.gong_events(T, CHIP, DUR, SR, SEED, DEFAULT_PATCH)
    ref = devices.strike_times(T, DUR, DEFAULT_PATCH, 3)
    assert len(ev) == len(ref) > 0
    assert [e.t for e in ev] == list(ref)
    assert all(isinstance(e, Trigger) and e.source == "gong" and e.chip == CHIP and e.strength == 1.0 for e in ev)


def test_gong_events_follow_the_segment_in_the_seed_not_segment_zero():
    # the roll-linked gong's strikes depend on the segment, so a wrong segment gives other times for some chip
    patch = dataclasses.replace(DEFAULT_PATCH, seed=1)
    found = False
    for mask in (0x3002, 0x37F7, 0x2D5A, 0x3FFF, 0x1234):
        t = tel(tensix_mask=mask, eth_mask=0x7FF, ddr_status=(1, 1, 1, 1))
        a = [e.t for e in collect.gong_events(t, 0, 30.0, SR, 3 * SEGMENT_SEED_STRIDE, patch)]
        b = [e.t for e in collect.gong_events(t, 0, 30.0, SR, 0, patch)]
        assert a == list(devices.strike_times(t, 30.0, patch, 3))
        found = found or a != b
    assert found


# --- rolls ---
def test_roll_events_match_the_pulse_sets_and_cycle_starts_are_stronger():
    patch = dataclasses.replace(DEFAULT_PATCH, roll_rests=0.3)
    t = tel(ddr_status=(1, 1, 1, 1), eth_mask=0x7FF)
    ev = collect.roll_events(t, CHIP, DUR, SR, SEED, patch)
    sets = devices.ttrollz_pulse_sets(t, DUR, SEED, patch, 3)
    assert len(ev) == sum(len(times) for _, _, times, _ in sets)
    k = 0
    for _, lane, times, flags in sets:
        for time, flag in zip(times, flags):
            e = ev[k]
            k += 1
            assert (e.t, e.lane, e.source, e.chip) == (time, lane, "ttrollz", CHIP)
            assert e.strength == (1.0 if flag else 0.6)
    assert {-1} < {e.lane for e in ev}  # the clock and at least one lane
    assert {e.strength for e in ev if e.lane >= 0} == {1.0, 0.6}
    assert {e.strength for e in ev if e.lane == -1} == {0.6}


# --- steam ---
def test_steam_events_are_the_hits_the_snare_plays():
    patch = dataclasses.replace(DEFAULT_PATCH, snare=0.5)
    ev = collect.steam_events(T, CHIP, DUR, SR, SEED, patch)
    ref = steam_hits(T, DUR, SEED, patch)
    assert len(ev) == len(ref) > 0
    assert [e.t for e in ev] == list(ref)
    assert all(e.source == "ttsteam" and e.chip == CHIP and e.strength == 1.0 for e in ev)


def test_a_smaller_snare_share_plays_fewer_steam_hits_and_none_at_zero():
    full = collect.steam_events(T, CHIP, DUR, SR, SEED, DEFAULT_PATCH)
    half = collect.steam_events(T, CHIP, DUR, SR, SEED, dataclasses.replace(DEFAULT_PATCH, snare=0.3))
    none = collect.steam_events(T, CHIP, DUR, SR, SEED, dataclasses.replace(DEFAULT_PATCH, snare=0.0))
    assert len(half) < len(full) and none == []


# --- quantus ---
def test_quantus_notes_start_a_note_length_apart_and_each_pitch_is_its_first_samples_semitones():
    ev = collect.quantus_events(T, CHIP, DUR, SR, SEED, DEFAULT_PATCH)
    semis = devices.ttquantus_steps(T, DUR, SR, SEED, DEFAULT_PATCH)
    note_len = 1.0 / devices._note_rate(T)
    assert len(ev) == int(np.ceil(DUR / note_len - 1e-9))
    for k, e in enumerate(ev):
        assert isinstance(e, Note) and e.source == "ttquantus" and e.chip == CHIP and e.velocity == 1.0
        assert e.t == pytest.approx(k * note_len, abs=1.0 / SR)
        first = semis[int(round(e.t * SR))]
        assert e.pitch_hz == pytest.approx(temp_to_pitch_hz(T.asic_temperature) * 2 ** (first / 12))
        assert e.bend == ()


def test_consecutive_notes_of_one_pitch_stay_separate_notes():
    ev = collect.quantus_events(T, CHIP, DUR, SR, SEED, DEFAULT_PATCH)
    pitches = [e.pitch_hz for e in ev]
    assert any(a == b for a, b in zip(pitches, pitches[1:]))
    assert all(a.t + a.dur <= b.t + 1e-9 for a, b in zip(ev, ev[1:]))


def test_rests_leave_out_notes_and_the_gate_is_read_in_the_middle_of_each_note():
    none = collect.quantus_events(T, CHIP, DUR, SR, SEED, DEFAULT_PATCH)
    patch = dataclasses.replace(DEFAULT_PATCH, rests=0.9)
    some = collect.quantus_events(T, CHIP, DUR, SR, SEED, patch)
    assert len(some) < len(none)
    gate = devices.ttquantus_gate(T, DUR, SR, SEED, patch)
    for e in some:
        assert gate[int((e.t + e.dur / 2) * SR)] > 0.5
    shut = {round(e.t, 3) for e in none} - {round(e.t, 3) for e in some}
    note_len = 1.0 / devices._note_rate(T)
    for s in shut:
        assert gate[min(len(gate) - 1, int((s + note_len / 2) * SR))] <= 0.5


def test_a_gate_value_of_exactly_one_half_does_not_play(monkeypatch):
    monkeypatch.setattr(collect.devices, "ttquantus_gate", lambda *a, **k: np.full(int(DUR * SR), 0.5))
    assert collect.quantus_events(T, CHIP, DUR, SR, SEED, DEFAULT_PATCH) == []


def test_bend_is_empty_without_pitch_movement_and_present_with_it():
    assert all(e.bend == () for e in collect.quantus_events(T, CHIP, DUR, SR, SEED, DEFAULT_PATCH))
    ev = collect.quantus_events(T, CHIP, DUR, SR, SEED, MOVING)
    assert ev and all(e.bend for e in ev)


def test_bend_is_the_per_sample_semitones_every_20_ms_relative_to_the_first_sample():
    ev = collect.quantus_events(T, CHIP, DUR, SR, SEED, MOVING)
    semis = devices.ttquantus_steps(T, DUR, SR, SEED, MOVING)
    for e in ev[:20]:
        i0 = int(round(e.t * SR))
        for tau, delta in e.bend:
            assert 0.0 <= tau < e.dur
            assert delta == pytest.approx(semis[min(len(semis) - 1, i0 + int(round(tau * SR)))] - semis[i0])
        taus = [tau for tau, _ in e.bend]
        assert taus[0] == 0.0 and taus[1] == pytest.approx(0.02, abs=1.0 / SR)
        assert e.bend[0][1] == 0.0
    # vibrato and glide make at least one note move
    assert any(abs(d) > 0.05 for e in ev for _, d in e.bend)


def test_a_chip_with_everything_resting_has_no_quantus_notes():
    patch = dataclasses.replace(DEFAULT_PATCH, rests=0.999999)
    assert len(collect.quantus_events(T, CHIP, DUR, SR, SEED, patch)) <= 1


# --- controls ---
def test_controls_are_four_values_in_range_at_time_zero():
    ev = collect.control_events(T, CHIP, 4.0)
    assert [c.name for c in ev] == ["temperature", "power", "pressure", "chaos"]
    assert all(isinstance(c, Control) and c.t == 0.0 and c.chip == CHIP and 0.0 <= c.value <= 1.0 for c in ev)
    by = {c.name: c.value for c in ev}
    assert by["temperature"] == pytest.approx(norm_temp(T.asic_temperature))
    assert by["power"] == pytest.approx(T.power / 200.0)
    assert by["pressure"] == pytest.approx(chip_pressure(T))
    assert by["chaos"] == pytest.approx(chip_chaos(T))


def test_controls_clamp_to_the_range():
    hot = {c.name: c.value for c in collect.control_events(tel(asic_temperature=500.0, power=900.0), 0, 1.0)}
    cold = {c.name: c.value for c in collect.control_events(tel(asic_temperature=-20.0, power=-5.0), 0, 1.0)}
    assert hot["temperature"] == 1.0 and hot["power"] == 1.0
    assert cold["temperature"] == 0.0 and cold["power"] == 0.0


def test_pressure_rises_as_temperature_rises_toward_the_throttle_point():
    vals = [{c.name: c.value for c in collect.control_events(tel(asic_temperature=x, power=10.0), 0, 1.0)}["pressure"] for x in (45.0, 65.0, 85.0)]
    assert vals[0] < vals[1] < vals[2]


def test_chaos_adds_the_rate_of_change_to_the_pressure():
    by = {c.name: c.value for c in collect.control_events(tel(temp_rate=0.5, asic_temperature=50.0, power=10.0), 0, 1.0)}
    assert by["chaos"] > by["pressure"] + 0.3
