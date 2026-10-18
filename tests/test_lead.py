"""The lead voice: a short motif that is shifted or inverted when signals say so."""
import dataclasses

import numpy as np
import pytest

from tensixinet.lead import (
    MOTIF,
    Transform,
    apply_transform,
    degree_to_semitones,
    lead,
    lead_transforms,
    phrase_transform,
)
from tensixinet.mapping import aiclk_to_bpm
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
    temps = (30.0, 30.0, 30.0, 70.0)  # mean 40; use only channel 3 hot by rotating
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
