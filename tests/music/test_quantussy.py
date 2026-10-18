# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The Quantussy: five triangle oscillators, each with a castle that samples another petal's triangle."""
import numpy as np
import pytest

from tensixinet.music.quantussy import (
    N_PETALS,
    castle_leader,
    castle_output,
    crunch,
    petal_phases,
    petal_ratios,
    triangle,
)

SR = 1000


def test_a_triangle_runs_zero_to_one_and_back_at_its_frequency():
    t = np.arange(SR) / SR
    w = triangle(5.0, 0.0, t)
    assert w.min() == pytest.approx(0.0, abs=0.01) and w.max() == pytest.approx(1.0, abs=0.01) and w[0] == 0.0
    assert np.sum(np.diff(np.signbit(w - 0.5)) != 0) == 10  # five cycles cross the middle twice each


def test_the_start_phase_shifts_the_triangle():
    t = np.arange(10) / SR
    assert triangle(1.0, 0.25, t)[0] == pytest.approx(0.5)  # a quarter of a cycle in, halfway up


def test_every_petal_samples_the_one_two_positions_counter_clockwise_by_default():
    # guide: Gamma(alpha), Epsilon(gamma), Beta(epsilon), Delta(beta), Alpha(delta)
    assert [castle_leader(i) for i in range(N_PETALS)] == [3, 4, 0, 1, 2]
    assert N_PETALS == 5


def test_two_does_not_divide_five_so_the_chain_of_leaders_visits_every_petal():
    seen, p = [], 0
    for _ in range(N_PETALS):
        seen.append(p)
        p = castle_leader(p)
    assert sorted(seen) == list(range(N_PETALS))


def test_the_other_leader_offsets_are_three_four_and_one_places_counter_clockwise():
    assert [castle_leader(0, k) for k in (1, 3, 4)] == [4, 2, 1]


def test_the_castle_is_a_staircase_that_changes_only_at_its_own_petals_peaks_and_valleys():
    t = np.arange(10 * SR) / SR
    freqs = [2.0, 3.0, 5.0, 7.0, 11.0]
    c = castle_output(0, freqs, [0.0] * 5, t)
    changes = np.flatnonzero(np.diff(c) != 0) + 1
    assert len(changes) <= 2 * 2.0 * 10  # petal 0 turns round twice a cycle
    expected = {int(round(k / (2 * 2.0) * SR)) for k in range(1, 40)}
    assert set(changes.tolist()) <= expected


def test_the_castle_holds_the_leaders_triangle_as_it_was_at_the_last_turning_point():
    t = np.arange(4 * SR) / SR
    freqs = [2.0, 3.0, 5.0, 7.0, 11.0]
    phases = [0.0, 0.1, 0.2, 0.3, 0.4]
    c = castle_output(0, freqs, phases, t)
    leader = castle_leader(0)
    k = 5  # the sixth half cycle of petal 0
    sample_time = k / (2 * freqs[0])
    expected = triangle(freqs[leader], phases[leader], np.array([sample_time]))[0]
    assert c[int((sample_time + 0.01) * SR)] == pytest.approx(expected, abs=1e-9)


def test_castle_values_stay_between_zero_and_one():
    t = np.arange(5 * SR) / SR
    c = castle_output(2, [2.0, 3.0, 5.0, 7.0, 11.0], [0.0, 0.2, 0.4, 0.6, 0.8], t)
    assert c.min() >= 0.0 and c.max() <= 1.0


def test_petals_in_unison_and_in_phase_hold_only_their_own_peak_and_valley():
    t = np.arange(3 * SR) / SR
    c = castle_output(0, [4.0] * 5, [0.0] * 5, t)
    assert set(np.round(c, 9)) <= {0.0, 1.0}  # each castle samples an identical triangle at its turning points


def test_petals_with_low_prime_ratios_repeat_a_pattern_and_unrelated_ones_do_not():
    t = np.arange(40 * SR) / SR
    low = castle_output(0, [4.0, 6.0, 5.0, 16 / 3, 20 / 3], [0.0] * 5, t)
    steps_low = low[np.flatnonzero(np.diff(low) != 0) + 1]
    # four ratios: petal 3 runs at 4/3 of petal 0, so petal 0 sees the same three values over and over
    assert len(set(np.round(steps_low, 6))) <= 3
    odd = castle_output(0, [4.0, 6.1234, 5.0771, 5.4137, 6.6613], [0.0] * 5, t)
    assert len(set(np.round(odd[np.flatnonzero(np.diff(odd) != 0) + 1], 6))) > 10


def test_the_default_ratios_are_low_primes_nudged_by_the_seed():
    a, b = petal_ratios(1), petal_ratios(2)
    assert len(a) == N_PETALS and a == petal_ratios(1) and a != b
    assert a[0] == 1.0
    base = (1.0, 3 / 2, 5 / 4, 4 / 3, 5 / 3)
    assert all(abs(x / y - 1) <= 0.03 + 1e-9 for x, y in zip(a, base))


def test_crunch_rounds_to_eight_bits():
    x = np.linspace(-0.8, 0.8, 1001)
    y = crunch(x, 1.0)
    assert len(np.unique(y)) <= 256 and np.allclose(y * 127, np.round(y * 127))
    assert np.abs(y - x).max() <= 0.5 / 127 + 1e-9


def test_crunch_holds_each_sample_so_a_slower_clock_aliases():
    x = np.sin(2 * np.pi * 440 * np.arange(8000) / 8000)
    held = crunch(x, 0.25)  # a quarter of the sample rate: every sample is held four times
    assert np.all(held[0::4] == held[1::4]) and np.all(held[0::4] == held[3::4])
    assert not np.array_equal(crunch(x, 1.0), held)


def test_crunch_at_full_rate_does_not_hold_samples():
    x = np.sin(2 * np.pi * 50 * np.arange(8000) / 8000)
    assert np.count_nonzero(np.diff(crunch(x, 1.0))) > 7000


def test_the_start_phases_come_from_the_seed_and_lie_inside_one_cycle():
    p = petal_phases(4)
    assert len(p) == N_PETALS and p == petal_phases(4) and p != petal_phases(5)
    assert all(0.0 <= x < 1.0 for x in p)


def test_phases_and_ratios_do_not_share_one_random_stream():
    base = (1.0, 3 / 2, 5 / 4, 4 / 3, 5 / 3)
    unit = [((r / b) - 1) / (2 * 0.03) + 0.5 for r, b in zip(petal_ratios(7), base)]  # the draws behind the ratios, 0 to 1
    assert not np.allclose(unit[1:], petal_phases(7)[1:])


# --- the castle mod select: which petal each castle samples ----------------------------------------------------------------

from tensixinet.music.quantussy import castle_offset  # noqa: E402


def test_a_chips_identity_picks_one_of_the_four_leader_offsets_the_guide_cycles_through():
    assert [castle_offset(i) for i in range(4)] == [2, 3, 4, 1] and castle_offset(4) == castle_offset(0)


def test_the_castle_output_samples_whichever_leader_the_offset_names():
    t = np.arange(4 * SR) / SR
    freqs, phases = [2.0, 3.0, 5.0, 7.0, 11.0], [0.0, 0.1, 0.2, 0.3, 0.4]
    default = castle_output(0, freqs, phases, t)
    assert np.array_equal(default, castle_output(0, freqs, phases, t, offset=2))
    sample_time = 5 / (2 * freqs[0])
    for offset in (1, 3, 4):
        c = castle_output(0, freqs, phases, t, offset=offset)
        expected = triangle(freqs[castle_leader(0, offset)], phases[castle_leader(0, offset)], np.array([sample_time]))[0]
        assert c[int((sample_time + 0.01) * SR)] == pytest.approx(expected, abs=1e-9)
