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
def _mid(e, n):
    """The sample at the middle of a note, where the extractor reads the held step."""
    return min(n - 1, int(round((e.t + e.dur / 2) * SR)))


def test_quantus_notes_start_a_note_length_apart_and_each_pitch_is_the_held_step_at_its_middle():
    ev = collect.quantus_events(T, CHIP, DUR, SR, SEED, DEFAULT_PATCH)
    held = devices.ttquantus_held(T, DUR, SR, SEED, DEFAULT_PATCH)
    note_len = 1.0 / devices._note_rate(T)
    assert len(ev) == int(np.ceil(DUR / note_len - 1e-9))
    for k, e in enumerate(ev):
        assert isinstance(e, Note) and e.source == "ttquantus" and e.chip == CHIP and e.velocity == 1.0
        assert e.t == pytest.approx(k * note_len, abs=1.0 / SR)
        assert e.pitch_hz == pytest.approx(temp_to_pitch_hz(T.asic_temperature) * 2 ** (held[_mid(e, len(held))] / 12))
        assert e.bend == ()


def _patches():
    from tensixinet.music.patch import draw_patch
    return {"default": DEFAULT_PATCH, "moving": MOVING, "drawn": dataclasses.replace(draw_patch(5, [T]), quantus_mod=0.7)}


@pytest.mark.parametrize("name", ["default", "moving", "drawn"])
def test_each_notes_pitch_is_the_held_step_at_its_middle_and_matches_the_audio_there(name):
    patch = _patches()[name]
    ev = collect.quantus_events(T, CHIP, DUR, SR, SEED, patch)
    held = devices.ttquantus_held(T, DUR, SR, SEED, patch)
    steps = devices.ttquantus_steps(T, DUR, SR, SEED, patch)
    base = temp_to_pitch_hz(T.asic_temperature)
    assert len(ev) > 10
    near = 0
    for e in ev:
        m = _mid(e, len(held))
        note_semis = 12 * np.log2(e.pitch_hz / base)
        assert note_semis == pytest.approx(held[m], abs=1e-9)
        near += abs(note_semis - steps[m]) <= 0.5
    # the audio at the middle differs from the held step only by the vibrato (at most 0.4 semitone)
    assert near >= 0.95 * len(ev)


@pytest.mark.parametrize("name", ["moving", "drawn"])
def test_note_plus_bend_is_the_audio_semitones_wherever_the_bend_is_not_clamped(name):
    patch = _patches()[name]
    ev = collect.quantus_events(T, CHIP, DUR, SR, SEED, patch)
    steps = devices.ttquantus_steps(T, DUR, SR, SEED, patch)
    base = temp_to_pitch_hz(T.asic_temperature)
    checked = 0
    for e in ev:
        i0 = int(round(e.t * SR))
        note_semis = 12 * np.log2(e.pitch_hz / base)
        for tau, delta in e.bend:
            if abs(delta) <= 2.0:
                checked += 1
                assert note_semis + delta == pytest.approx(steps[min(len(steps) - 1, i0 + int(round(tau * SR)))], abs=0.01)
    assert checked > 100


def test_a_notes_bend_in_the_middle_of_a_note_is_small_so_it_does_not_clamp():
    ev = collect.quantus_events(T, CHIP, DUR, SR, SEED, MOVING)
    note_len = 1.0 / devices._note_rate(T)
    for e in ev:
        if e.dur < note_len - 1e-9:
            continue  # the last note is cut short by the render's end, which also clips the glide's window
        mid = min(e.bend, key=lambda b: abs(b[0] - e.dur / 2))
        assert abs(mid[1]) <= 0.5  # only the vibrato (up to 0.4 semitone) is left once the note sits on the held step


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


def test_bend_is_the_per_sample_semitones_every_20_ms_relative_to_the_held_step_at_the_notes_middle():
    ev = collect.quantus_events(T, CHIP, DUR, SR, SEED, MOVING)
    semis = devices.ttquantus_steps(T, DUR, SR, SEED, MOVING)
    held = devices.ttquantus_held(T, DUR, SR, SEED, MOVING)
    for e in ev[:20]:
        i0 = int(round(e.t * SR))
        ref = held[_mid(e, len(held))]
        for tau, delta in e.bend:
            assert 0.0 <= tau < e.dur
            assert delta == pytest.approx(semis[min(len(semis) - 1, i0 + int(round(tau * SR)))] - ref)
        taus = [tau for tau, _ in e.bend]
        assert taus[0] == 0.0 and taus[1] == pytest.approx(0.02, abs=1.0 / SR)
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


# --- chorus_score ---
from tensixinet.music import chorus  # noqa: E402
from tensixinet.music.evict import ChipDefrag  # noqa: E402
from tensixinet.music.patch import draw_patch  # noqa: E402
from tensixinet.telemetry.mock import mock_telemetry  # noqa: E402

CSR = 4000


def _box(n, seed=1):
    return mock_telemetry(n, seed=seed)


def _devices_in(score):
    return {e.source for e in score.events if not isinstance(e, Control)}


def test_the_dealt_devices_with_extractors_are_the_ones_present():
    for n in (1, 4):
        tels = _box(n)
        score = collect.chorus_score(tels, 20.0, DEFAULT_PATCH, sample_rate=CSR)
        dealt = {d.__name__ for c in range(n) for d in chorus.chip_devices(c, n)}
        assert _devices_in(score) <= dealt & set(collect.EXTRACTORS)
        assert {e.chip for e in score.controls()} == set(range(n))
        assert len(score.controls()) == 4 * n  # once per chip whatever devices it plays
        assert score.bpm == pytest.approx(__import__("tensixinet.music.mapping", fromlist=["x"]).aiclk_to_bpm(tels[0].aiclk))
    assert "lead" in _devices_in(collect.chorus_score(_box(1), 20.0, sample_rate=CSR))


def test_only_gong_gives_full_strength_gong_triggers_and_controls():
    tels = _box(4)
    patch = dataclasses.replace(DEFAULT_PATCH, emphasis=(("gong", 0.3),))
    score = collect.chorus_score(tels, 20.0, patch, only=["gong"], sample_rate=CSR)
    assert len(score.triggers()) > 0 and not score.notes()
    assert all(e.source == "gong" and e.strength == 1.0 for e in score.triggers())
    assert len(score.controls()) == 16


def test_a_soloist_is_louder_than_a_non_soloist():
    tels = _box(4)
    gong_slot = next(s for s in range(chorus.n_slots(4)) if chorus.assign_device(s).__name__ == "gong")
    other = next(s for s in range(chorus.n_slots(4)) if chorus.assign_device(s).__name__ == "gong" or s != gong_slot)
    patch = dataclasses.replace(DEFAULT_PATCH, floor=0.2)
    solo = collect.chorus_score(tels, 20.0, patch, soloists=[gong_slot], only=None, sample_rate=CSR)
    quiet = collect.chorus_score(tels, 20.0, patch, soloists=[(gong_slot + 1) % chorus.n_slots(4)], sample_rate=CSR)
    g = lambda sc: [e.strength for e in sc.triggers() if e.source == "gong"]
    assert g(solo) and g(quiet)
    assert max(g(solo)) > max(g(quiet))
    assert max(g(quiet)) < 0.5 and min(g(quiet)) >= collect.MIN_VELOCITY


def test_balance_scales_velocity():
    tels = _box(4)
    half = dataclasses.replace(DEFAULT_PATCH, emphasis=(("gong", 0.5),))
    gong_slot = next(s for s in range(chorus.n_slots(4)) if chorus.assign_device(s).__name__ == "gong")
    a = collect.chorus_score(tels, 20.0, DEFAULT_PATCH, soloists=[gong_slot], sample_rate=CSR)
    b = collect.chorus_score(tels, 20.0, half, soloists=[gong_slot], sample_rate=CSR)
    sa = sorted(e.strength for e in a.triggers() if e.source == "gong")
    sb = sorted(e.strength for e in b.triggers() if e.source == "gong")
    assert len(sa) == len(sb) > 0
    assert all(y == pytest.approx(max(collect.MIN_VELOCITY, x * 0.5)) for x, y in zip(sa, sb))


def test_an_idle_chip_has_no_notes_or_triggers_but_keeps_controls():
    tels = _box(4)
    base = collect.chorus_score(tels, 20.0, DEFAULT_PATCH, sample_rate=CSR)
    busy_chips = {e.chip for e in base.events if not isinstance(e, Control)}
    assert busy_chips
    idle = busy_chips.pop()
    defrag = [ChipDefrag("idle" if c == idle else "running") for c in range(4)]
    score = collect.chorus_score(tels, 20.0, DEFAULT_PATCH, defrag=defrag, sample_rate=CSR)
    assert idle not in {e.chip for e in score.events if not isinstance(e, Control)}
    assert {e.chip for e in score.controls()} == {0, 1, 2, 3}
    assert any(e.chip != idle for e in score.events if not isinstance(e, Control))


def test_an_evicting_chip_is_quieter_than_a_running_one():
    tels = _box(4)
    run = collect.chorus_score(tels, 20.0, DEFAULT_PATCH, sample_rate=CSR)
    chip = next(e.chip for e in run.triggers())
    defrag = [ChipDefrag("evicting", 0.0) if c == chip else ChipDefrag() for c in range(4)]
    ev = collect.chorus_score(tels, 20.0, DEFAULT_PATCH, defrag=defrag, sample_rate=CSR)
    total = lambda sc: sum(e.strength for e in sc.triggers() if e.chip == chip)
    assert total(ev) < total(run)


def test_a_fully_dissolved_chip_is_silent_in_the_score_as_in_the_audio():
    tels = _box(4)
    run = collect.chorus_score(tels, 20.0, DEFAULT_PATCH, sample_rate=CSR)
    chip = next(e.chip for e in run.triggers())
    # long into an eviction every voice of the chip has dissolved, but the phase is still "evicting", not "idle"
    defrag = [ChipDefrag("evicting", 1000.0) if c == chip else ChipDefrag() for c in range(4)]
    ev = collect.chorus_score(tels, 20.0, DEFAULT_PATCH, defrag=defrag, sample_rate=CSR)
    assert chip not in {e.chip for e in ev.events if not isinstance(e, Control)}
    assert any(e.chip != chip for e in ev.events if not isinstance(e, Control))
    assert {e.chip for e in ev.controls()} == {0, 1, 2, 3}


def test_an_event_is_dropped_exactly_when_its_voices_keep_factor_is_at_or_below_the_threshold():
    from tensixinet.music.evict import drain_progress
    tels = _box(4)
    run = collect.chorus_score(tels, 20.0, DEFAULT_PATCH, sample_rate=CSR)
    chip = next(e.chip for e in run.triggers())
    state = ChipDefrag("evicting", 0.0)
    defrag = [state if c == chip else ChipDefrag() for c in range(4)]
    ev = collect.chorus_score(tels, 20.0, DEFAULT_PATCH, defrag=defrag, sample_rate=CSR)
    n_samples = int(20.0 * CSR)
    # per source, the times at which the voice's keep factor is above the threshold (a chip has at most one slot of each source here)
    slots = {chorus.assign_device(s).__name__: s for s in range(chorus.n_slots(4)) if chorus.slot_chip(s, 4) == chip}
    kept_any = dropped_any = False
    for source in ("lead", "ttquantus", "gong", "ttrollz", "ttsteam"):
        if source not in slots:
            continue
        rank, count, prime = chorus.voice_order(slots[source], 4)
        keep = (1.0 - drain_progress("evicting", 0.0, n_samples, CSR, rank, count, prime)) ** 2
        before = [e for e in run.events if not isinstance(e, Control) and e.chip == chip and e.source == source]
        after = {round(e.t, 6) for e in ev.events if not isinstance(e, Control) and e.chip == chip and e.source == source}
        for e in before:
            alive = keep[min(max(int(e.t * CSR), 0), n_samples - 1)] > collect.DISSOLVED_KEEP
            kept_any |= alive
            dropped_any |= not alive
            assert (round(e.t, 6) in after) == alive, (source, e.t)
    assert kept_any and dropped_any, "the scenario must have events on both sides of the threshold"


def test_the_threshold_for_a_dissolved_voice_is_two_percent_and_the_velocity_floor_stays_five_percent():
    assert collect.DISSOLVED_KEEP == 0.02 and collect.MIN_VELOCITY == 0.05


def test_the_segment_moves_gong_and_roll_events():
    tels = _box(4)
    patch = draw_patch(7, tels)
    a = collect.chorus_score(tels, 20.0, patch, segment=0, sample_rate=CSR)
    b = collect.chorus_score(tels, 20.0, patch, segment=5, sample_rate=CSR)
    pick = lambda sc, src: [(e.t, e.chip) for e in sc.events if getattr(e, "source", None) == src]
    assert pick(a, "gong") and pick(a, "ttrollz")
    assert pick(a, "ttrollz") != pick(b, "ttrollz")
    moved = False  # a gong on its chip's clock keeps its strikes; one patched to a roll moves with the segment
    for box in range(1, 8):
        t2 = _box(4, seed=box)
        p2 = draw_patch(box, t2)
        x = collect.chorus_score(t2, 20.0, p2, segment=0, only=["gong"], sample_rate=CSR)
        y = collect.chorus_score(t2, 20.0, p2, segment=5, only=["gong"], sample_rate=CSR)
        moved = moved or pick(x, "gong") != pick(y, "gong")
    assert moved


def test_a_duration_shorter_than_a_note_drops_it():
    tels = _box(1)
    long = collect.chorus_score(tels, 20.0, DEFAULT_PATCH, only=["lead"], sample_rate=CSR)
    short = collect.chorus_score(tels, 1.0, DEFAULT_PATCH, only=["lead"], sample_rate=CSR)
    assert len(short.notes()) < len(long.notes())
    assert all(n.t < 1.0 for n in short.notes())


@pytest.mark.parametrize("segment", [3])
def test_wiring_gong_and_roll_events_use_the_seeds_the_audio_uses(segment):
    tels = _box(4, seed=2)
    patch = draw_patch(11, tels)
    dur = 20.0
    score = collect.chorus_score(tels, dur, patch, segment=segment, only=["gong", "ttrollz"], sample_rate=CSR)
    base = segment * SEGMENT_SEED_STRIDE
    checked = 0
    for s in range(chorus.n_slots(4)):
        name = chorus.assign_device(s).__name__
        chip = chorus.slot_chip(s, 4)
        if name == "gong":
            want = [float(x) for x in devices.strike_times(tels[chip], dur, patch, segment)]
            got = [e.t for e in score.triggers() if e.source == "gong" and e.chip == chip]
            assert got == want
            checked += bool(want)
        if name == "ttrollz":
            want = sorted(float(x) for _, _, times, _ in devices.ttrollz_pulse_sets(tels[chip], dur, base + s, patch, segment) for x in times)
            got = sorted(e.t for e in score.triggers() if e.source == "ttrollz" and e.chip == chip)
            assert got == want
            checked += bool(want)
    assert checked == 2


def test_velocity_is_clamped_to_the_floor_and_to_one():
    tels = _box(4)
    gong_slot = next(s for s in range(chorus.n_slots(4)) if chorus.assign_device(s).__name__ == "gong")
    loud = dataclasses.replace(DEFAULT_PATCH, emphasis=(("gong", 3.0),))
    sc = collect.chorus_score(tels, 20.0, loud, soloists=[gong_slot], sample_rate=CSR)
    assert max(e.strength for e in sc.triggers() if e.source == "gong") == 1.0
    tiny = dataclasses.replace(DEFAULT_PATCH, emphasis=(("gong", 0.001),))
    sc = collect.chorus_score(tels, 20.0, tiny, soloists=[gong_slot], sample_rate=CSR)
    assert {e.strength for e in sc.triggers() if e.source == "gong"} == {collect.MIN_VELOCITY}
    for factor, want in ((3.0, 1.0), (0.001, collect.MIN_VELOCITY)):  # notes are clamped too (a soloist in a one-chip box)
        patch = dataclasses.replace(DEFAULT_PATCH, emphasis=(("lead", factor),))
        lead_slot = next(s for s in range(chorus.n_slots(1)) if chorus.assign_device(s).__name__ == "lead")
        sc = collect.chorus_score(_box(1), 20.0, patch, soloists=[lead_slot], sample_rate=CSR)
        assert {n.velocity for n in sc.notes() if n.source == "lead"} == {want}
