# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The ttrolzer's rolls: barre tempos from E6 capacitors, stable even rolls, unstable odd ones."""
import numpy as np
import pytest

from tensixinet.music.ttrolzer import BARRE_BPM, REFERENCE_BARRE, barre_bpm, roll_pulse_times


def _cv(times):
    d = np.diff(times)
    return float(d.std() / d.mean())


def test_the_barre_tempos_are_the_guides_table_for_one_to_ten_microfarads():
    assert BARRE_BPM == (192, 122, 84, 56, 39, 27, 18)


def test_neighbouring_barres_run_at_about_two_thirds_and_skipping_one_gives_about_four_ninths():
    for a, b in zip(BARRE_BPM, BARRE_BPM[1:]):
        assert b / a == pytest.approx(2 / 3, abs=0.06)
    for a, b in zip(BARRE_BPM, BARRE_BPM[2:]):
        assert b / a == pytest.approx(4 / 9, abs=0.06)


def test_the_slowest_barre_is_about_ten_times_slower_than_the_fastest():
    assert BARRE_BPM[0] / BARRE_BPM[-1] == pytest.approx(10, rel=0.1)


def test_the_reference_barre_runs_at_the_chips_own_tempo_and_the_others_scale_from_it():
    assert barre_bpm(120.0, REFERENCE_BARRE) == pytest.approx(120.0)
    assert barre_bpm(120.0, REFERENCE_BARRE - 1) == pytest.approx(120.0 * 122 / 84)
    assert barre_bpm(120.0, REFERENCE_BARRE + 1) == pytest.approx(120.0 * 56 / 84)


@pytest.mark.parametrize("sides", [3, 4, 5, 6])
def test_a_roll_keeps_its_mean_tempo(sides):
    means = [np.diff(roll_pulse_times(sides, 90.0, 300.0, seed)).mean() for seed in range(5)]
    assert np.mean(means) == pytest.approx(60.0 / 90.0, rel=0.04)


def test_odd_rolls_are_less_regular_than_even_rolls():
    odd = np.mean([_cv(roll_pulse_times(s, 90.0, 120.0, seed)) for s in (3, 5) for seed in range(6)])
    even = np.mean([_cv(roll_pulse_times(s, 90.0, 120.0, seed)) for s in (4, 6) for seed in range(6)])
    assert odd > 1.3 * even


def test_even_rolls_swing_by_alternating_long_and_short_steps():
    d = np.diff(roll_pulse_times(4, 90.0, 120.0, 0, tolerance=0.0))
    assert np.allclose(d[::2], d[0]) and np.allclose(d[1::2], d[1]) and d[0] > 1.3 * d[1]


def test_a_tolerance_of_zero_makes_the_same_rolls_for_every_seed():
    a = roll_pulse_times(5, 90.0, 30.0, 1, tolerance=0.0)
    b = roll_pulse_times(5, 90.0, 30.0, 2, tolerance=0.0)
    assert np.array_equal(a, b)


def test_the_same_seed_repeats_and_another_seed_wanders_off():
    a = roll_pulse_times(6, 90.0, 30.0, 1)
    assert np.array_equal(a, roll_pulse_times(6, 90.0, 30.0, 1))
    assert not np.allclose(a[:20], roll_pulse_times(6, 90.0, 30.0, 2)[:20])


def test_pulses_start_at_zero_and_stay_inside_the_duration():
    t = roll_pulse_times(3, 120.0, 5.0, 0)
    assert t[0] == 0.0 and t.max() < 5.0 and np.all(np.diff(t) > 0)


@pytest.mark.parametrize("sides", [3, 5])
def test_an_odd_roll_spreads_its_steps_unevenly_over_exactly_one_cycle_of_its_own_length(sides):
    d = np.diff(roll_pulse_times(sides, 60.0, 60.0, 0, tolerance=0.0))
    assert np.allclose(d[:sides], d[sides: 2 * sides]) and len(set(np.round(d[:sides], 6))) > 1
    assert d[:sides].sum() == pytest.approx(sides * 1.0)  # uneven, but a cycle still lasts `sides` mean steps
