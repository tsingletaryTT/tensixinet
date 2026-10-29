# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The lead voice: a short motif that is shifted or inverted when signals say so."""
import dataclasses

import numpy as np
import pytest

from tensixinet.music.lead import (
    MOTIF,
    Transform,
    apply_transform,
    degree_to_semitones,
    lead,
    lead_transforms,
    phrase_transform,
)
from tensixinet.music.mapping import aiclk_to_bpm
from tensixinet.telemetry import FALLBACK

SR = 8000
AICLK_120_BPM = 875.0  # 400-1350 MHz -> 60-180 BPM, so 875 MHz is 120 BPM: a bar is 2 s


# --- scale degrees -> semitones ------------------------------------------------

@pytest.mark.parametrize("degree, semitones", [
    (0, 0), (1, 2), (4, 9),      # inside the first octave
    (5, 12), (7, 16),            # the scale repeats an octave up
    (-1, -3), (-5, -12),         # and an octave down (degree -1 is the 9 of the octave below)
])
def test_degree_to_semitones_walks_the_pentatonic_scale_in_both_directions(degree, semitones):
    assert degree_to_semitones(degree) == semitones


# --- deciding what a reading means ---------------------------------------------

def test_a_reading_near_the_average_repeats_the_phrase():
    assert phrase_transform(0.3) == Transform("repeat", 0)
    assert phrase_transform(-0.9) == Transform("repeat", 0)


def test_a_hotter_reading_shifts_up_by_about_a_degree_per_degree_celsius():
    assert phrase_transform(2.4) == Transform("shift", 2)


def test_the_shift_is_capped_so_one_hot_channel_cannot_fly_off_the_scale():
    assert phrase_transform(40.0) == Transform("shift", 4)


def test_a_cooler_reading_inverts_the_phrase():
    assert phrase_transform(-3.0) == Transform("invert", 0)


# --- applying it to the motif --------------------------------------------------

def test_repeat_returns_the_motif_unchanged():
    assert apply_transform(MOTIF, Transform("repeat", 0)) == MOTIF


def test_shift_moves_every_note_by_the_amount():
    assert apply_transform((0, 2, 1), Transform("shift", 3)) == (3, 5, 4)


def test_invert_mirrors_the_motif_around_its_first_note():
    assert apply_transform((2, 4, 1), Transform("invert", 0)) == (2, 0, 3)


def test_inverting_twice_gives_the_motif_back():
    once = apply_transform(MOTIF, Transform("invert", 0))
    assert once != MOTIF
    assert apply_transform(once, Transform("invert", 0)) == MOTIF


# --- turning a reading list into one transform per phrase -----------------------

def test_flat_readings_loop_forever():
    assert lead_transforms((30.0,) * 8, 6) == [Transform("repeat", 0)] * 6


def test_each_phrase_reads_the_next_channel_and_wraps_around():
    temps = (30.0, 30.0, 40.0, 30.0)  # mean 32.5: channel 2 is hot, the rest are cool-ish
    got = lead_transforms(temps, 8)
    assert got[2] == got[6]  # phrase 6 reads channel 2 again
    assert got[2].kind == "shift"
    assert got[0].kind == "invert"  # 30 is 2.5 below the 32.5 mean


def test_a_single_channel_has_nothing_to_differ_from():
    assert lead_transforms((55.0,), 3) == [Transform("repeat", 0)] * 3


def test_no_phrases_means_no_transforms():
    assert lead_transforms((30.0, 31.0), 0) == []


# --- the device ----------------------------------------------------------------

def tel(gddr, aiclk=AICLK_120_BPM):
    return dataclasses.replace(FALLBACK, gddr_temps=gddr, aiclk=aiclk)


def bars(audio, n):
    size = 2 * SR  # a 2 s bar at 120 BPM
    return [audio[i * size:(i + 1) * size] for i in range(n)]


def test_bar_length_assumption_in_these_tests_is_right():
    assert aiclk_to_bpm(AICLK_120_BPM) == pytest.approx(120.0)


def test_lead_output_is_the_right_shape_bounded_and_deterministic():
    t = tel((30.0, 36.0, 28.0, 31.0))
    a = lead(t, 4.0, SR, 0)
    assert a.shape == (4 * SR,) and a.dtype == np.float32
    assert np.isfinite(a).all() and np.abs(a).max() <= 1.0 and np.abs(a).max() > 0.1
    assert np.array_equal(a, lead(t, 4.0, SR, 0))


def test_with_flat_gddr_temperatures_every_bar_is_the_same_loop():
    b = bars(lead(tel((30.0,) * 8), 6.0, SR, 0), 3)
    assert np.array_equal(b[0], b[1]) and np.array_equal(b[1], b[2])


def test_a_hot_channel_changes_the_bar_that_reads_it_and_only_that_bar():
    temps = (30.0, 40.0, 30.0, 30.0)  # channel 1 is hot; the mean is 32.5, 30 is 2.5 cooler
    flat = bars(lead(tel((32.5,) * 4), 8.0, SR, 0), 4)
    moved = bars(lead(tel(temps), 8.0, SR, 0), 4)
    # every bar differs from the flat loop: cooler channels invert, the hot one shifts
    assert not np.array_equal(moved[1], flat[1])
    # bars 0, 2 and 3 read the three equal cool channels, so they match each other
    assert np.array_equal(moved[0], moved[2]) and np.array_equal(moved[2], moved[3])
    assert not np.array_equal(moved[0], moved[1])


def test_a_shifted_phrase_is_higher_than_the_loop():
    flat = bars(lead(tel((30.0,) * 4), 2.0, SR, 0), 1)[0]
    hot_first = bars(lead(tel((70.0, 30.0, 30.0, 30.0)), 2.0, SR, 0), 1)[0]  # phrase 0 reads 70 C
    def centroid(x):
        spec = np.abs(np.fft.rfft(x))
        return float((spec * np.arange(len(spec))).sum() / spec.sum())
    assert centroid(hot_first) > centroid(flat)


def test_aiclk_sets_the_tempo_of_the_lead():
    slow = lead(tel((30.0,) * 4, 400.0), 4.0, SR, 0)
    fast = lead(tel((30.0,) * 4, 1350.0), 4.0, SR, 0)
    assert not np.allclose(slow, fast)


def test_temperature_sets_the_pitch_of_the_lead():
    cool = lead(dataclasses.replace(tel((30.0,) * 4), asic_temperature=30.0), 2.0, SR, 0)
    hot = lead(dataclasses.replace(tel((30.0,) * 4), asic_temperature=90.0), 2.0, SR, 0)
    assert not np.allclose(cool, hot)


# --- the patch moves the lead -----------------------------------------------------

from tensixinet.music.patch import DEFAULT_PATCH, SCALES  # noqa: E402


def test_the_default_patch_leaves_the_lead_as_it_was():
    t = tel((30.0, 36.0, 28.0, 31.0))
    assert np.array_equal(lead(t, 4.0, SR, 0), lead(t, 4.0, SR, 0, patch=DEFAULT_PATCH))


def test_a_patch_motif_replaces_the_default_motif():
    t = tel((30.0,) * 4)
    other = dataclasses.replace(DEFAULT_PATCH, motif=(0, 1, 0, 1, 0, 1, 0, 1))
    assert not np.allclose(lead(t, 2.0, SR, 0), lead(t, 2.0, SR, 0, patch=other))


def test_a_patch_scale_changes_the_pitches_of_the_same_motif():
    t = tel((30.0,) * 4)
    whole = dataclasses.replace(DEFAULT_PATCH, scale_name="whole_tone", scale=SCALES["whole_tone"])
    assert not np.allclose(lead(t, 2.0, SR, 0), lead(t, 2.0, SR, 0, patch=whole))


def test_degrees_walk_whatever_scale_they_are_given():
    assert degree_to_semitones(6, SCALES["whole_tone"]) == 12      # 6 steps of a 6-note scale is an octave
    assert degree_to_semitones(1, SCALES["dorian"]) == 2
    assert degree_to_semitones(-1, SCALES["dorian"]) == -2         # 10 semitones, one octave down


# --- rests: some of the motif's notes are left out ------------------------------------------------------------------

EIGHTH = SR // 4  # an eighth at 120 BPM


def _slots(audio, n_phrases, per_phrase):
    """The audio cut into one slot for each note of each phrase."""
    return np.array(audio[: n_phrases * per_phrase * EIGHTH]).reshape(n_phrases * per_phrase, EIGHTH)


def _played(audio, n_phrases, per_phrase):
    return np.abs(_slots(audio, n_phrases, per_phrase)).max(axis=1) > 1e-6


def test_lead_rests_leave_out_whole_notes_and_play_the_rest_exactly_as_before():
    t = tel((30.0,) * 4)
    plain = lead(t, 40.0, SR, 3)
    rested = lead(t, 40.0, SR, 3, patch=dataclasses.replace(DEFAULT_PATCH, lead_rests=0.4))
    per = len(DEFAULT_PATCH.motif)
    n_slots = 40 * SR // EIGHTH // per * per
    played = _played(rested, n_slots // per, per)
    assert 0.3 < 1 - played.mean() < 0.5
    keep = np.repeat(played, EIGHTH)
    assert np.array_equal(rested[: len(keep)][keep], plain[: len(keep)][keep])
    assert not rested[: len(keep)][~keep].any()


def test_the_first_note_of_every_phrase_plays_however_much_rests():
    t = tel((30.0,) * 4)
    per = len(DEFAULT_PATCH.motif)
    audio = lead(t, 40.0, SR, 3, patch=dataclasses.replace(DEFAULT_PATCH, lead_rests=0.97))
    played = _played(audio, 40 * SR // EIGHTH // per, per)
    assert played[::per].all() and 1 - played.mean() > 0.8


def test_the_lead_rests_in_a_pattern_of_its_own_seed_and_share():
    t = tel((30.0,) * 4)
    per = len(DEFAULT_PATCH.motif)
    patch = dataclasses.replace(DEFAULT_PATCH, lead_rests=0.4)
    a, b = lead(t, 20.0, SR, 3, patch=patch), lead(t, 20.0, SR, 4, patch=patch)
    assert np.array_equal(a, lead(t, 20.0, SR, 3, patch=patch)) and not np.array_equal(_played(a, 10, per), _played(b, 10, per))
    more = lead(t, 20.0, SR, 3, patch=dataclasses.replace(DEFAULT_PATCH, lead_rests=0.7))
    assert (_played(more, 10, per) <= _played(a, 10, per)).all() and _played(more, 10, per).sum() < _played(a, 10, per).sum()


def test_a_note_that_rests_is_not_replaced_by_its_neighbour_the_phrase_keeps_its_length():
    t = tel((30.0,) * 4)
    patch = dataclasses.replace(DEFAULT_PATCH, lead_rests=0.5)
    assert lead(t, 7.3, SR, 1, patch=patch).shape == (int(7.3 * SR),)


# --- characterization: the lead's audio is fixed byte for byte (MIDI-out task 2 split it into events and a renderer) ---
import hashlib as _hashlib  # noqa: E402

from tensixinet.music.patch import draw_patch as _draw_patch  # noqa: E402
from tensixinet.music.lead import lead_notes  # noqa: E402

_CHAR_TELEMETRY = (
    FALLBACK,
    dataclasses.replace(FALLBACK, asic_temperature=62.0, aiclk=1000.0, gddr_temps=(30., 34., 29., 38., 31., 27., 36., 30.), eth_mask=0x3F3, tensix_mask=0x2D5A),
    dataclasses.replace(FALLBACK, asic_temperature=85.0, aiclk=500.0, power=110.0, gddr_temps=(55., 50., 61., 48., 52., 59., 47., 53.), eth_mask=0x7FF, tensix_mask=0x3002),
)
_DRAWN = dataclasses.replace(_draw_patch(7, _CHAR_TELEMETRY), lead_rests=0.5, roll_rests=0.5, steam_routes=("accent",))
_CHAR_PATCHES = (DEFAULT_PATCH, _DRAWN, _DRAWN)
# SHA-256 of the float32 bytes of a 6 s render at 22050 Hz, taken from the code before the split.
_LEAD_DIGESTS = (
    "56300987641dc644ffeda8450c9eb36dc9ecd74d3a81488aaa32cb45b4799a1d",
    "ed161af13781f047b18a16cc8998dd2b908caa11aed04afd687880ce87e918ac",
    "360695e8287fd7d87ef491289fb568029bb0d1a152025cca3147f712ce27ff81",
)


@pytest.mark.parametrize("i", range(3))
def test_lead_audio_is_byte_identical_to_the_digest_taken_before_the_split(i):
    audio = lead(_CHAR_TELEMETRY[i], 6.0, 22050, 12345 + i * (1 << 20), _CHAR_PATCHES[i])
    assert audio.dtype == np.float32
    assert _hashlib.sha256(audio.tobytes()).hexdigest() == _LEAD_DIGESTS[i]


# --- lead_notes: the events the lead's audio is rendered from ---
def _notes(t, patch, seed=5, duration=6.0, sr=22050):
    return lead_notes(t, duration, sr, seed, patch)


def test_lead_notes_have_one_entry_per_motif_note_when_nothing_rests():
    t = tel((30.0,) * 8)  # 120 BPM: an eighth is 0.25 s, a bar is len(motif) eighths
    notes = _notes(t, DEFAULT_PATCH)
    eighth = round(0.25 * 22050) / 22050  # whole samples, as the audio places them
    bar = eighth * len(DEFAULT_PATCH.motif)
    per_bar = len(DEFAULT_PATCH.motif)
    expected = [p * bar + i * eighth for p in range(int(6.0 / bar) + 1) for i in range(per_bar) if p * bar + i * eighth < 6.0]
    assert [s for s, _, _ in notes] == pytest.approx(expected, abs=1 / 22050)
    assert all(d == pytest.approx(eighth, abs=1 / 22050) for _, d, _ in notes)


def test_lead_notes_never_start_at_or_after_the_duration():
    for dur in (0.3, 1.0, 2.77, 6.0):
        assert all(s < dur for s, _, _ in _notes(tel((30.0,) * 8), DEFAULT_PATCH, duration=dur))


def test_lead_notes_keep_the_first_note_of_each_phrase_when_rests_are_heavy():
    patch = dataclasses.replace(DEFAULT_PATCH, lead_rests=0.9)
    notes = _notes(tel((30.0,) * 8), patch)
    bar = 0.25 * len(patch.motif)
    starts = [s for s, _, _ in notes]
    for phrase in range(int(6.0 / bar)):
        assert any(abs(s - phrase * bar) < 1e-3 for s in starts)
    assert len(notes) < len(_notes(tel((30.0,) * 8), DEFAULT_PATCH))  # and the rests did remove some


def test_lead_notes_cap_the_pitch_at_the_ceiling_the_audio_uses():
    hot = dataclasses.replace(FALLBACK, asic_temperature=120.0, gddr_temps=(80.0,) * 4 + (20.0,) * 4)
    notes = lead_notes(hot, 4.0, 2000, 1, DEFAULT_PATCH)
    assert notes and max(p for _, _, p in notes) <= 0.45 * 2000 + 1e-9
    assert any(p == pytest.approx(0.45 * 2000) for _, _, p in notes)


def test_lead_notes_follow_the_phrase_transform_the_audio_hears():
    flat = _notes(tel((30.0,) * 8), DEFAULT_PATCH)
    hot = _notes(tel((30.0,) * 7 + (38.0,)), DEFAULT_PATCH)
    per_bar = len(DEFAULT_PATCH.motif)
    # phrase 0 reads channel 0, which is below the mean here and so inverts; the last channel (read by phrase 7) shifts up
    assert [p for _, _, p in hot[:per_bar]] != [p for _, _, p in flat[:per_bar]]
