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


def test_a_barre_index_past_the_table_wraps_so_a_bigger_rax_configuration_can_still_use_it():
    """Quatrax has fourteen barres; the guide's tempo table has seven. Beyond it the pattern repeats."""
    assert barre_bpm(120.0, 7) == pytest.approx(barre_bpm(120.0, 0))
    assert barre_bpm(120.0, 13) == pytest.approx(barre_bpm(120.0, 6))


# --- rolls as coupled circuits: persistent capacitors, drift, a marked cycle, and Browns that couple rolls -------------

from tensixinet.music.ttrolzer import (  # noqa: E402
    DOWNBEAT_GAIN, DRIFT_RANGE, LINK_PULL_RANGE, Link, Roll, draw_links, roll_drift, roll_ensemble,
)

BPM = 60.0  # one step a second keeps the arithmetic easy to read


def _one(sides, duration=400.0, seed=1, **kw):
    times, downbeats = roll_ensemble([Roll(sides, BPM)], [], duration, seed, **kw)[0]
    return times, downbeats


def _steps(sides, **kw):
    return np.diff(_one(sides, **kw)[0])


@pytest.mark.parametrize("sides", [3, 4, 5, 6])
def test_every_geometry_keeps_the_barres_mean_tempo_over_a_long_run(sides):
    assert _steps(sides).mean() == pytest.approx(1.0, abs=0.08)


def test_the_ensemble_repeats_for_the_same_inputs_and_wanders_off_for_another_seed():
    a = roll_ensemble([Roll(5, BPM), Roll(4, 90.0)], [Link(0, 1, 0.2)], 60.0, 3)
    b = roll_ensemble([Roll(5, BPM), Roll(4, 90.0)], [Link(0, 1, 0.2)], 60.0, 3)
    c = roll_ensemble([Roll(5, BPM), Roll(4, 90.0)], [Link(0, 1, 0.2)], 60.0, 4)
    assert all(np.array_equal(x[0], y[0]) and np.array_equal(x[1], y[1]) for x, y in zip(a, b))
    assert not np.array_equal(a[0][0], c[0][0])


def test_pulses_start_at_zero_stay_inside_the_duration_and_never_bunch_up():
    for sides in (3, 4, 5, 6):
        times, _ = _one(sides, duration=100.0)
        assert times[0] == 0.0 and times.max() < 100.0 and np.diff(times).min() > 0.1


@pytest.mark.parametrize("sides", [3, 4, 5, 6])
def test_the_first_step_of_each_cycle_is_marked_and_no_other(sides):
    times, downbeats = _one(sides, duration=100.0)
    assert downbeats.dtype == bool and len(downbeats) == len(times)
    assert np.flatnonzero(downbeats).tolist() == list(range(0, len(times), sides))


def test_an_even_roll_keeps_its_pattern_but_an_odd_roll_does_not_return_with_the_same_phase():
    def longest_share(sides):
        steps = _steps(sides, duration=600.0)
        cycles = len(steps) // sides
        pos = np.argmax(steps[: cycles * sides].reshape(cycles, sides), axis=1)
        return np.bincount(pos).max() / cycles
    assert longest_share(4) > 0.9           # the long step stays on its own capacitor
    assert longest_share(3) < 0.75          # the odd roll's long step slides round the cycle
    assert longest_share(5) < 0.6


def test_a_square_trots_in_pairs_and_a_hexagon_has_a_triplet_feel_in_its_swing():
    def corr(sides, lag, seed):
        d = _steps(sides, drift=0.0, jitter=0.0, seed=seed)
        d = d - d.mean()
        return float(np.dot(d[:-lag], d[lag:]) / np.dot(d, d))
    for seed in range(15):  # whatever the capacitors' mismatch, not just a lucky draw of it
        assert corr(4, 2, seed) > 0.75 and corr(4, 3, seed) < -0.75   # long-short-long-short
        assert corr(6, 3, seed) > 0.3                                   # two threes interleaved
        assert corr(6, 3, seed) > corr(4, 3, seed) + 1.0


def test_the_capacitors_drift_slowly_so_a_cycles_length_wanders_with_memory():
    def cycle_sums(drift):
        steps = _steps(4, duration=1200.0, drift=drift, jitter=0.0)
        n = len(steps) // 4
        return steps[: n * 4].reshape(n, 4).sum(axis=1)
    still, moving = cycle_sums(0.0), cycle_sums(0.04)
    assert still.std() < 1e-9                         # persistent capacitors and no drift: every cycle takes the same time
    d = moving - moving.mean()
    assert moving.std() > 0.02 and np.dot(d[:-1], d[1:]) / np.dot(d, d) > 0.3  # drift, and it has memory


def test_drift_grows_with_the_chips_temperature_inside_its_range():
    lo, hi = DRIFT_RANGE
    assert roll_drift(20.0) == pytest.approx(lo) and roll_drift(110.0) == pytest.approx(hi)
    assert roll_drift(40.0) < roll_drift(60.0) < roll_drift(80.0)


# --- Browns: a spike from one roll pulls another's next step early -------------------------------------------------------

def _pair(link, bpm_a=60.0, bpm_b=63.0, sides_a=4, sides_b=4, duration=300.0, **kw):
    return roll_ensemble([Roll(sides_a, bpm_a), Roll(sides_b, bpm_b)], link, duration, 2, drift=0.0, jitter=0.0, **kw)


def _offset(src, dst):
    """Mean distance of each of dst's pulses from the nearest of src's, over the second half of the run."""
    half = dst[len(dst) // 2:]
    return float(np.mean([np.min(np.abs(src - x)) for x in half]))


def test_a_link_leaves_its_source_alone_and_moves_its_target():
    free = _pair([])
    linked = _pair([Link(0, 1, 0.3)])
    assert np.array_equal(free[0][0], linked[0][0])
    assert not np.array_equal(free[1][0], linked[1][0])
    both = _pair([Link(0, 1, 0.3), Link(1, 0, 0.3)])
    assert not np.array_equal(free[0][0], both[0][0])           # a link the other way moves the source too


def test_a_faster_even_roll_patched_into_another_syncs_it_mostly():
    """The guide: 'If the brown is receiving a faster pulse from another roll, it will cause even rolls to sync, mostly.'"""
    ratios = []
    for seed in range(15):  # the capacitors' mismatch moves how tightly it locks, so look at the typical case
        free = roll_ensemble([Roll(4, 63.0), Roll(4, 60.0)], [], 300.0, seed, drift=0.0, jitter=0.0)
        linked = roll_ensemble([Roll(4, 63.0), Roll(4, 60.0)], [Link(0, 1, 0.5)], 300.0, seed, drift=0.0, jitter=0.0)
        ratios.append(_offset(linked[0][0], linked[1][0]) / _offset(free[0][0], free[1][0]))
    assert np.median(ratios) < 0.3 and max(ratios) < 0.9


def test_a_slower_source_does_not_sync_its_target_the_same_way():
    ratios = []
    for seed in range(15):
        free = roll_ensemble([Roll(4, 57.0), Roll(4, 60.0)], [], 300.0, seed, drift=0.0, jitter=0.0)
        linked = roll_ensemble([Roll(4, 57.0), Roll(4, 60.0)], [Link(0, 1, 0.3)], 300.0, seed, drift=0.0, jitter=0.0)
        ratios.append(_offset(linked[0][0], linked[1][0]) / _offset(free[0][0], free[1][0]))
    assert np.median(ratios) > 0.6


def test_a_stronger_link_pulls_harder():
    weak_offsets, strong_offsets = [], []
    for seed in range(15):
        weak = roll_ensemble([Roll(4, 63.0), Roll(4, 60.0)], [Link(0, 1, 0.05)], 300.0, seed, drift=0.0, jitter=0.0)
        strong = roll_ensemble([Roll(4, 63.0), Roll(4, 60.0)], [Link(0, 1, 0.3)], 300.0, seed, drift=0.0, jitter=0.0)
        weak_offsets.append(_offset(weak[0][0], weak[1][0]))
        strong_offsets.append(_offset(strong[0][0], strong[1][0]))
    assert np.median(strong_offsets) < 0.75 * np.median(weak_offsets)  # typically, not in every draw of the capacitors


def test_an_odd_rolls_spikes_push_an_even_roll_off_its_steady_pattern():
    """Measured on the length of a whole cycle: the square's swing is fixed, so its step lengths vary either way, but with
    no drift and no jitter its four steps always add up to the same time until something knocks it off."""
    def cycle_spread(links):
        steps = np.diff(_pair(links, bpm_a=60.0, bpm_b=60.0, sides_a=5, sides_b=4)[1][0])
        n = len(steps) // 4
        return float(steps[: n * 4].reshape(n, 4).sum(axis=1).std())
    assert cycle_spread([]) < 1e-9
    assert cycle_spread([Link(0, 1, 0.25)]) > 0.05


def test_the_link_is_bounded_so_a_target_is_never_pulled_onto_the_instant_it_fires():
    linked = _pair([Link(0, 1, 0.9), Link(1, 0, 0.9)], duration=100.0)
    for times, _ in linked:
        assert np.diff(times).min() > 0.05


def test_a_link_to_a_roll_that_is_not_there_is_refused():
    with pytest.raises(ValueError):
        roll_ensemble([Roll(3, BPM)], [Link(0, 1, 0.2)], 10.0, 0)
    with pytest.raises(ValueError):
        roll_ensemble([Roll(3, BPM), Roll(4, BPM)], [Link(1, 1, 0.2)], 10.0, 0)


# --- which Browns are patched to which --------------------------------------------------------------------------------

def test_the_links_are_distinct_pairs_that_the_chip_and_patch_decide_and_grow_with_the_network():
    links = draw_links(8, patch_seed=5, identity=77, pressure=0.0)
    assert 2 <= len(links) <= 5
    big = [len(draw_links(40, patch_seed=s, identity=s * 3 + 1, pressure=0.0)) for s in range(40)]
    assert min(big) >= 6 and max(big) <= 12 and len(set(big)) > 3   # 2 + 40//10 to 4 + 40//5
    assert all(l.source != l.target and 0 <= l.source < 8 and 0 <= l.target < 8 for l in links)
    assert len({(l.source, l.target) for l in links}) == len(links)
    assert links == draw_links(8, patch_seed=5, identity=77, pressure=0.0)
    others = [draw_links(8, patch_seed=s, identity=i, pressure=0.0) for s in range(6) for i in range(6)]
    assert len({tuple((l.source, l.target) for l in ls) for ls in others}) > 10   # chips and patches wire differently


def test_with_fewer_than_two_rolls_there_is_nothing_to_link_and_with_two_there_is_at_most_a_pair_each_way():
    assert draw_links(0, 1, 1, 0.0) == [] and draw_links(1, 1, 1, 0.0) == []
    assert len(draw_links(2, 1, 1, 0.0)) <= 2


def test_pressure_strengthens_the_links_inside_their_range():
    lo, hi = LINK_PULL_RANGE
    calm, strained = draw_links(8, 5, 77, 0.0), draw_links(8, 5, 77, 1.0)
    assert all(lo <= l.pull <= hi for l in calm + strained)
    assert [(l.source, l.target) for l in calm] == [(l.source, l.target) for l in strained]  # the same wiring
    assert all(s.pull > c.pull for c, s in zip(calm, strained))


def test_the_ranges_and_gain_are_what_the_docs_say():
    assert DRIFT_RANGE == (0.005, 0.05) and LINK_PULL_RANGE == (0.05, 0.3) and DOWNBEAT_GAIN == 0.6


# --- gaps a mutation run found ----------------------------------------------------------------------------------------

def test_a_hard_pull_still_leaves_every_step_at_least_the_minimum_share_of_its_own_barres_step():
    from tensixinet.music.ttrolzer import MIN_STEP
    rolls = [Roll(4, 60.0), Roll(4, 66.0), Roll(5, 60.0)]
    links = [Link(0, 1, 0.95), Link(1, 0, 0.95), Link(2, 0, 0.95), Link(1, 2, 0.95)]  # a source that fires just after its target did
    for (times, _), roll in zip(roll_ensemble(rolls, links, 200.0, 0, drift=0.0, jitter=0.0), rolls):
        assert np.diff(times).min() >= MIN_STEP * 60.0 / roll.bpm - 1e-9


def test_links_differ_in_strength_and_never_repeat_a_pair_whatever_the_seed():
    assert len({l.pull for l in draw_links(8, 5, 77, 0.5)}) > 1
    for seed in range(60):
        links = draw_links(3, seed, seed * 7 + 1, 0.3)  # three rolls have six ordered pairs and up to four links: repeats would show
        assert len({(l.source, l.target) for l in links}) == len(links) >= 2


# --- the Rolzer: a bank of rolls, and a gong patched to one of them ---------------------------------------------------------

def test_a_barre_carries_the_rolzers_set_of_rolls_a_triangle_a_square_a_pentagon_and_two_hexagons():
    from tensixinet.music.ttrolzer import BARRE_GEOMETRY
    assert BARRE_GEOMETRY == (3, 4, 5, 6, 6)
    assert len(BARRE_GEOMETRY) * 7 == 35  # seven barres: the 35 rolls of the real Rolzer


def test_a_roll_can_start_part_way_through_its_first_step_so_a_pair_of_hexagons_pulse_out_of_phase():
    a, b = roll_ensemble([Roll(6, BPM), Roll(6, BPM, phase=0.5)], [], 30.0, 1, drift=0.0, jitter=0.0)
    assert a[0][0] == 0.0 and b[0][0] == pytest.approx(0.5)
    assert len(a[0]) > 20 and len(b[0]) > 20
    assert bool(a[1][0]) and bool(b[1][0])  # the first pulse of each still begins its cycle


def test_the_seed_stride_is_a_prime_shared_with_the_chorus():
    from tensixinet.music.chorus import SEGMENT_SEED_STRIDE as from_chorus
    from tensixinet.music.ttrolzer import SEGMENT_SEED_STRIDE
    assert SEGMENT_SEED_STRIDE == from_chorus == 10_007


def test_a_gong_is_patched_to_one_roll_or_to_the_clock_and_the_chip_and_patch_decide_which():
    from tensixinet.music.ttrolzer import GONG_ROLL_ODDS, draw_gong_trigger
    assert draw_gong_trigger(0, 1, 1) is None                       # nothing to patch it to
    picks = [draw_gong_trigger(12, s, s * 31 + 7) for s in range(400)]
    assert draw_gong_trigger(12, 5, 77) == draw_gong_trigger(12, 5, 77)
    on_clock = sum(p is None for p in picks) / len(picks)
    assert abs(on_clock - (1 - GONG_ROLL_ODDS)) < 0.08 and GONG_ROLL_ODDS == 0.75
    rolls = [p for p in picks if p is not None]
    assert min(rolls) >= 0 and max(rolls) <= 11 and len(set(rolls)) >= 10  # any roll can be the one


def test_at_least_half_the_links_reach_a_roll_that_is_heard_so_the_hidden_rolls_shape_the_sound():
    heard = [3, 7, 11, 15]
    for seed in range(40):
        links = draw_links(40, seed, seed * 5 + 2, 0.4, heard)
        assert sum(l.target in heard for l in links) >= (len(links) + 1) // 2
        assert len({(l.source, l.target) for l in links}) == len(links) and all(l.source != l.target for l in links)
    # without a list of heard rolls the draw is the one it always was
    assert draw_links(40, 3, 9, 0.4) == draw_links(40, 3, 9, 0.4, ())
    assert [(l.source, l.target) for l in draw_links(40, 3, 9, 0.4, heard)] != [(l.source, l.target) for l in draw_links(40, 3, 9, 0.4)]


def test_a_hidden_roll_can_drive_a_heard_one_and_a_heard_one_can_drive_a_hidden_one():
    heard = [3, 7, 11, 15]
    links = [l for seed in range(60) for l in draw_links(40, seed, seed + 1, 0.4, heard)]
    assert any(l.source not in heard and l.target in heard for l in links)
    assert any(l.source in heard and l.target not in heard for l in links)


def test_heard_indexes_outside_the_network_are_ignored():
    assert draw_links(5, 3, 9, 0.2, [99, -1]) == draw_links(5, 3, 9, 0.2)
    links = draw_links(5, 3, 9, 0.2, [2, 99, -1])
    assert all(0 <= l.source < 5 and 0 <= l.target < 5 for l in links)
    assert sum(l.target == 2 for l in links) >= (len(links) + 1) // 2
