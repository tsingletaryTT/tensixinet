# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The trigger sources the choir can follow: what each one fires on, and how hard."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from tensixinet.music.devices import (
    CHOIR_SOURCES, NO_STRENGTH, SOURCE_SLOTS, choir_roll, choir_triggers, ttquantus_onsets, ttrollz_pulse_times,
)
from tensixinet.music.devices import roll_pulses, strike_times
from tensixinet.music.lead import lead_notes
from tensixinet.music.mapping import aiclk_to_bpm
from tensixinet.music.patch import DEFAULT_PATCH
from tensixinet.music.register import step_times
from tensixinet.music.ttrolzer import SEGMENT_SEED_STRIDE
from tensixinet.score.collect import quantus_events
from tensixinet.telemetry import FALLBACK

SR = 8000
ROLLED = dataclasses.replace(FALLBACK, tensix_mask=0x3002, ddr_status=(1, 1, 1, 1))  # its gong follows a roll
HARVESTED = dataclasses.replace(FALLBACK, tensix_mask=0x37F7, ddr_status=(1, 1, 1, 1))  # its register fires


def test_the_sources_are_the_six_the_spec_names_and_the_borrowed_slots_match_the_chorus():
    assert CHOIR_SOURCES == ("gong", "rolls", "register", "quantus", "lead", "rax")
    from tensixinet.music.chorus import DEVICES
    names = [d.__name__ for d in DEVICES]
    assert SOURCE_SLOTS == {"quantus": names.index("ttquantus"), "lead": names.index("lead")}


def test_the_no_strength_is_one_half():
    assert NO_STRENGTH == 0.5


def test_gong_triggers_are_the_strikes_at_full_strength():
    times, strengths = choir_triggers("gong", ROLLED, 15.0, SR, 3)
    assert len(times) > 0
    assert np.array_equal(times, strike_times(ROLLED, 15.0, DEFAULT_PATCH, 3)) and (strengths == 1.0).all()


def test_register_and_roll_triggers_have_the_constant_strength_of_one_half():
    for source in ("register", "rolls"):
        _, s = choir_triggers(source, HARVESTED, 15.0, SR, 0)
        assert len(s) > 0 and (s == 0.5).all()


def test_register_triggers_ignore_the_snare_share():
    quiet = dataclasses.replace(DEFAULT_PATCH, snare=0.0)
    got = choir_triggers("register", HARVESTED, 15.0, SR, 0, quiet)[0]
    assert len(got) > 0
    assert np.array_equal(got, step_times(HARVESTED, 15.0))


def test_roll_triggers_are_the_whole_roll_of_the_followed_lane():
    lane = choir_roll(HARVESTED)
    assert lane is not None
    expected = roll_pulses(HARVESTED, 15.0, DEFAULT_PATCH, 2)[lane][0]
    assert np.array_equal(choir_triggers("rolls", HARVESTED, 15.0, SR, 2)[0], expected)


def test_the_followed_lane_depends_on_the_chips_identity():
    from tensixinet.music.devices import ttrollz_lanes
    from tensixinet.music.mapping import personality
    lanes = [l.index for l in ttrollz_lanes(HARVESTED)]
    assert len(lanes) > 1
    assert choir_roll(HARVESTED) == lanes[personality(HARVESTED) % len(lanes)]
    # a chip whose identity picks a different lane follows a different roll
    seen = set()
    for mask in range(0x37F0, 0x3800):
        t = dataclasses.replace(HARVESTED, tensix_mask=mask)
        seen.add(choir_roll(t))
    assert len(seen) > 1


def test_quantus_onsets_are_the_notes_the_score_says_play():
    p = dataclasses.replace(DEFAULT_PATCH, rests=0.3)
    seed = 2 * SEGMENT_SEED_STRIDE + 2
    expected = [n.t for n in quantus_events(FALLBACK, 0, 8.0, SR, seed, p)]
    assert len(expected) > 0
    assert np.allclose(ttquantus_onsets(FALLBACK, 8.0, SR, seed, p), expected)
    assert np.allclose(choir_triggers("quantus", FALLBACK, 8.0, SR, 2, p)[0], expected)


def test_lead_triggers_are_its_notes_and_a_phrases_first_note_is_strong():
    times, strengths = choir_triggers("lead", FALLBACK, 8.0, SR, 0)
    assert np.allclose(times, [n[0] for n in lead_notes(FALLBACK, 8.0, SR, SOURCE_SLOTS["lead"], DEFAULT_PATCH)])
    assert strengths[0] == 1.0 and set(strengths.tolist()) == {0.5, 1.0}


def test_the_lead_borrows_the_seed_of_the_segment():
    times, _ = choir_triggers("lead", FALLBACK, 8.0, SR, 2, dataclasses.replace(DEFAULT_PATCH, lead_rests=0.5))
    seed = 2 * SEGMENT_SEED_STRIDE + SOURCE_SLOTS["lead"]
    p = dataclasses.replace(DEFAULT_PATCH, lead_rests=0.5)
    assert np.allclose(times, [n[0] for n in lead_notes(FALLBACK, 8.0, SR, seed, p)])


def test_lead_strength_is_one_exactly_on_the_first_note_of_each_phrase():
    p = DEFAULT_PATCH
    times, strengths = choir_triggers("lead", FALLBACK, 20.0, SR, 0, p)
    eighth = round(30.0 / aiclk_to_bpm(FALLBACK.aiclk) * SR)
    bar = eighth * len(p.motif)
    first = np.array([round(t * SR) % bar == 0 for t in times])
    assert first.sum() >= 2 and (~first).any()
    assert (strengths[first] == 1.0).all() and (strengths[~first] == 0.5).all()


def test_rax_triggers_follow_the_patchs_bar_count():
    small = choir_triggers("rax", FALLBACK, 10.0, SR, 0, dataclasses.replace(DEFAULT_PATCH, rax_bars=4))[0]
    big = choir_triggers("rax", FALLBACK, 10.0, SR, 0, dataclasses.replace(DEFAULT_PATCH, rax_bars=14))[0]
    assert len(big) > len(small)


def test_rax_strength_is_one_on_a_press_and_one_half_on_a_release():
    from tensixinet.music.ttrax import gesture_times
    times, s = choir_triggers("rax", FALLBACK, 10.0, SR, 0)
    g_times, is_press = gesture_times(aiclk_to_bpm(FALLBACK.aiclk), DEFAULT_PATCH.rax_bars, 10.0)
    assert np.array_equal(times, g_times)
    assert (s[is_press] == 1.0).all() and (s[~is_press] == 0.5).all()


def test_a_chip_with_no_lane_up_falls_back_to_its_clock():
    bare = dataclasses.replace(FALLBACK, ddr_status=(0, 0, 0, 0), eth_mask=0)
    assert choir_roll(bare) is None
    got = choir_triggers("rolls", bare, 6.0, SR, 0)
    assert np.allclose(got[0], ttrollz_pulse_times(aiclk_to_bpm(bare.aiclk), 6.0))
    assert (got[1] == 0.5).all()


def test_the_segment_moves_the_roll_and_gong_triggers():
    assert not np.array_equal(choir_triggers("rolls", ROLLED, 15.0, SR, 0)[0], choir_triggers("rolls", ROLLED, 15.0, SR, 1)[0])


def test_every_source_gives_times_inside_the_render_and_matching_lengths():
    for source in CHOIR_SOURCES:
        times, strengths = choir_triggers(source, HARVESTED, 5.0, SR, 0)
        assert len(times) == len(strengths) and ((times >= 0) & (times < 5.0)).all() and (np.diff(times) >= 0).all()
        assert ((strengths >= 0) & (strengths <= 1)).all()


def test_an_unknown_source_is_refused_with_the_list():
    with pytest.raises(ValueError, match="gong, rolls, register"):
        choir_triggers("snare", FALLBACK, 1.0, SR, 0)


# --- what the choir reads: its own fields plus those of the source that fires it ---------------------------------

from tensixinet.music.devices import CHOIR_SOURCE_INPUTS, DEVICE_INPUTS, IDENTITY_FIELDS, PRESSURE_FIELDS  # noqa: E402
from tensixinet.music.ttchoir import ttchoir  # noqa: E402
from tests.telemetry_changes import CHANGES  # noqa: E402

OWN_CHOIR_FIELDS = frozenset({"asic_temperature", "gddr_temps", "aiclk"}) | PRESSURE_FIELDS | IDENTITY_FIELDS
CHOIR_PATCHES = {s: dataclasses.replace(DEFAULT_PATCH, choir_onset=s, rests=0.3) for s in CHOIR_SOURCES}
# Its gong follows a roll and its register fires. 0x3002 leaves some steps of the register firing and the changed mask
# (0x1235) fires others, so the mask moves the register's triggers.
CHOIR_BASE = {"tensix_mask": 0x3002, "ddr_status": (0, 0, 0, 0)}  # not (1, 1, 1, 1): that is the change under test, and a no-op as a base
LONG = 15.0


@pytest.mark.parametrize("source", CHOIR_SOURCES)
@pytest.mark.parametrize("field", sorted(CHANGES))
def test_the_choir_reads_its_own_fields_and_its_sources(source, field):
    base = dataclasses.replace(FALLBACK, **CHOIR_BASE)
    changed = dataclasses.replace(base, **{field: CHANGES[field]})
    p = CHOIR_PATCHES[source]
    differs = not np.allclose(ttchoir(base, LONG, SR, 0, p), ttchoir(changed, LONG, SR, 0, p))
    assert differs == (field in OWN_CHOIR_FIELDS | CHOIR_SOURCE_INPUTS[source]), (
        f"{source}: CHOIR_SOURCE_INPUTS says {sorted(CHOIR_SOURCE_INPUTS[source])}, but changing {field} {'did' if differs else 'did not'} change the audio")


def test_device_inputs_lists_the_choirs_own_fields_and_every_sources():
    assert DEVICE_INPUTS["ttchoir"] == OWN_CHOIR_FIELDS | frozenset().union(*CHOIR_SOURCE_INPUTS.values())
    assert {"heartbeat", "ddr_status", "axiclk"} <= DEVICE_INPUTS["ttchoir"]


@pytest.mark.parametrize("field", sorted(CHANGES))
def test_the_old_choir_reads_exactly_its_own_fields(field):
    # The default patch has no onset source, so it sings the old vowel choir on its own bar clock: no source fields, and every one
    # of its own fields moves the audio.
    assert DEFAULT_PATCH.choir_onset == ""
    base = dataclasses.replace(FALLBACK, **CHOIR_BASE)
    changed = dataclasses.replace(base, **{field: CHANGES[field]})
    differs = not np.allclose(ttchoir(base, LONG, SR, 0), ttchoir(changed, LONG, SR, 0))
    assert differs == (field in OWN_CHOIR_FIELDS), (
        f"the old choir reads {sorted(OWN_CHOIR_FIELDS)}, but changing {field} {'did' if differs else 'did not'} change the audio")
