# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Live mode's two pure pieces: what changed between two readings, and joining segments without a click."""
import dataclasses
import math

import numpy as np
import pytest

from tensixinet.music.live import RATE_WINDOW_S, XFADE_S, Stitcher, Tracker, advance
from tensixinet.telemetry import FALLBACK

SR = 8000


# --- advance: two readings become rates and new events -----------------------------------------------------------

def test_the_first_reading_has_nothing_to_compare_with_and_is_returned_as_it_is():
    chips = [FALLBACK, dataclasses.replace(FALLBACK, power=90.0)]
    assert advance(None, chips, 2.0) == chips


def test_rates_are_the_change_in_temperature_and_power_per_second():
    before = [dataclasses.replace(FALLBACK, asic_temperature=50.0, power=40.0)]
    after = [dataclasses.replace(FALLBACK, asic_temperature=51.0, power=20.0)]
    (now,) = advance(before, after, 2.0)
    assert now.temp_rate == pytest.approx(0.5) and now.power_rate == pytest.approx(-10.0)
    assert (now.asic_temperature, now.power) == (51.0, 20.0)  # the readings themselves are untouched


def test_each_chip_is_compared_with_the_same_chip():
    before = [dataclasses.replace(FALLBACK, asic_temperature=50.0), dataclasses.replace(FALLBACK, asic_temperature=70.0)]
    after = [dataclasses.replace(FALLBACK, asic_temperature=52.0), dataclasses.replace(FALLBACK, asic_temperature=69.0)]
    a, b = advance(before, after, 1.0)
    assert (a.temp_rate, b.temp_rate) == (2.0, -1.0)


def test_no_time_between_readings_means_no_rate_and_no_division_by_zero():
    before, after = [FALLBACK], [dataclasses.replace(FALLBACK, asic_temperature=60.0)]
    for dt in (0.0, -1.0):
        (now,) = advance(before, after, dt)
        assert now.temp_rate == 0.0 and now.power_rate == 0.0


def test_a_different_number_of_chips_cannot_be_paired_so_the_new_readings_stand_alone():
    after = [FALLBACK, dataclasses.replace(FALLBACK, power=90.0)]
    assert advance([FALLBACK], after, 2.0) == after


def test_event_counters_become_what_is_new_since_the_last_reading():
    before = [dataclasses.replace(FALLBACK, corr_errs=5, uncorr_errs=1, thermal_trips=0)]
    after = [dataclasses.replace(FALLBACK, corr_errs=8, uncorr_errs=1, thermal_trips=2)]
    (now,) = advance(before, after, 2.0)
    assert (now.corr_errs, now.uncorr_errs, now.thermal_trips) == (3, 0, 2)


def test_a_counter_that_went_down_after_a_chip_reset_counts_as_nothing_new():
    (now,) = advance([dataclasses.replace(FALLBACK, corr_errs=9)], [dataclasses.replace(FALLBACK, corr_errs=2)], 1.0)
    assert now.corr_errs == 0


def test_every_counter_is_taken_since_the_last_reading_not_just_the_first_time_it_rises():
    before = [dataclasses.replace(FALLBACK, corr_errs=4, uncorr_errs=2, thermal_trips=1)]
    after = [dataclasses.replace(FALLBACK, corr_errs=4, uncorr_errs=5, thermal_trips=3)]
    (now,) = advance(before, after, 1.0)
    assert (now.corr_errs, now.uncorr_errs, now.thermal_trips) == (0, 3, 2)
    (reset,) = advance(before, [dataclasses.replace(FALLBACK, uncorr_errs=0, thermal_trips=0)], 1.0)
    assert (reset.uncorr_errs, reset.thermal_trips) == (0, 0)


def test_losing_a_chip_cannot_be_paired_any_more_than_gaining_one():
    before = [FALLBACK, dataclasses.replace(FALLBACK, power=90.0), dataclasses.replace(FALLBACK, power=60.0)]
    after = [dataclasses.replace(FALLBACK, asic_temperature=70.0), dataclasses.replace(FALLBACK, power=95.0)]
    assert advance(before, after, 2.0) == after


def test_nothing_else_is_changed():
    before = [FALLBACK]
    after = [dataclasses.replace(FALLBACK, asic_temperature=60.0, corr_errs=1)]
    (now,) = advance(before, after, 1.0)
    assert dataclasses.replace(now, temp_rate=0.0, corr_errs=1) == after[0]


# --- Tracker: rates over a window long enough to mean something ---------------------------------------------------

def _t(**kw):
    return [dataclasses.replace(FALLBACK, **kw)]


def test_the_rate_window_is_two_seconds():
    assert RATE_WINDOW_S == 2.0


def test_the_first_update_has_nothing_to_compare_with():
    tracker = Tracker()
    (now,) = tracker.update(_t(asic_temperature=50.0, corr_errs=3), 100.0)
    assert (now.temp_rate, now.corr_errs) == (0.0, 3)


def test_readings_closer_together_than_the_window_give_no_rate_but_the_baseline_waits():
    tracker = Tracker()
    tracker.update(_t(asic_temperature=50.0), 100.0)
    (soon,) = tracker.update(_t(asic_temperature=50.4), 100.3)  # 0.4 C in 0.3 s would be 1.3 C/s of mostly noise
    assert soon.temp_rate == 0.0
    (later,) = tracker.update(_t(asic_temperature=51.0), 102.0)  # measured from the first reading, 2 s back
    assert later.temp_rate == pytest.approx(0.5)


def test_after_a_rate_is_measured_the_next_one_starts_from_that_reading():
    tracker = Tracker()
    tracker.update(_t(asic_temperature=50.0), 0.0)
    tracker.update(_t(asic_temperature=52.0), 4.0)
    (third,) = tracker.update(_t(asic_temperature=52.0), 8.0)
    assert third.temp_rate == 0.0


def test_power_gets_a_rate_over_the_same_window():
    tracker = Tracker()
    tracker.update(_t(power=40.0), 0.0)
    (now,) = tracker.update(_t(power=100.0), 3.0)
    assert now.power_rate == pytest.approx(20.0)


def test_new_errors_are_counted_from_the_last_reading_even_inside_the_window():
    tracker = Tracker()
    tracker.update(_t(thermal_trips=0), 0.0)
    (a,) = tracker.update(_t(thermal_trips=2), 0.5)  # inside the window: the rate waits, the errors do not
    (b,) = tracker.update(_t(thermal_trips=2), 1.0)  # nothing new since the last reading
    (c,) = tracker.update(_t(thermal_trips=3), 5.0)
    assert (a.thermal_trips, b.thermal_trips, c.thermal_trips) == (2, 0, 1)


def test_a_different_number_of_chips_starts_the_tracker_over():
    tracker = Tracker()
    tracker.update(_t(asic_temperature=50.0) * 2, 0.0)
    now = tracker.update(_t(asic_temperature=60.0) * 3, 5.0)
    assert [c.temp_rate for c in now] == [0.0, 0.0, 0.0]
    (again,) = tracker.update(_t(asic_temperature=62.0), 6.0)[:1]
    assert again.temp_rate == 0.0  # the new baseline was only a second ago


def test_losing_chips_starts_the_tracker_over_too():
    tracker = Tracker()
    tracker.update(_t(asic_temperature=50.0) * 3, 0.0)
    now = tracker.update(_t(asic_temperature=60.0) * 2, 1.0)  # inside the window of the old baseline
    assert [c.temp_rate for c in now] == [0.0, 0.0]
    after = tracker.update(_t(asic_temperature=62.0) * 2, 3.0)  # the new baseline is the two-chip reading, 2 s ago
    assert [c.temp_rate for c in after] == [1.0, 1.0]


def test_each_chip_keeps_its_own_baseline():
    tracker = Tracker()
    tracker.update([FALLBACK, dataclasses.replace(FALLBACK, asic_temperature=70.0)], 0.0)
    a, b = tracker.update([dataclasses.replace(FALLBACK, asic_temperature=FALLBACK.asic_temperature + 2),
                           dataclasses.replace(FALLBACK, asic_temperature=69.0)], 2.0)
    assert (a.temp_rate, b.temp_rate) == (1.0, -0.5)


def test_a_window_of_your_own_changes_when_a_rate_is_first_measured():
    tracker = Tracker(window_s=10.0)
    tracker.update(_t(asic_temperature=50.0), 0.0)
    assert tracker.update(_t(asic_temperature=55.0), 9.0)[0].temp_rate == 0.0
    assert tracker.update(_t(asic_temperature=55.0), 10.0)[0].temp_rate == pytest.approx(0.5)


# --- Stitcher: segments overlap and cross-fade --------------------------------------------------------------------

def _chunk(value, seconds):
    return np.full(int(seconds * SR), value, dtype=np.float32)


def test_the_crossfade_is_a_fraction_of_a_second():
    assert 0.1 <= XFADE_S <= 1.0


def test_each_push_returns_its_length_minus_the_tail_it_keeps_for_the_next_segment():
    s = Stitcher(SR, 0.25)
    assert len(s.push(_chunk(1.0, 2.25))) == 2 * SR
    assert len(s.push(_chunk(1.0, 3.25))) == 3 * SR


def test_the_first_segment_starts_at_full_level_because_nothing_comes_before_it():
    out = Stitcher(SR, 0.25).push(_chunk(1.0, 2.25))
    assert out[0] == 1.0 and out[-1] == 1.0


def test_the_next_segment_fades_in_while_the_last_tail_fades_out_with_equal_power():
    s = Stitcher(SR, 0.25)
    s.push(_chunk(1.0, 2.25))
    out = s.push(_chunk(1.0, 2.25))
    n = int(0.25 * SR)
    k = (np.arange(n) + 0.5) / n  # the middle of each sample's slot in the fade
    expected = np.cos(0.5 * math.pi * k) + np.sin(0.5 * math.pi * k)
    assert np.allclose(out[:n], expected, atol=1e-4)
    assert out[n] == pytest.approx(1.0)  # after the join the new segment is on its own


def test_the_two_fade_curves_have_equal_power_everywhere():
    s = Stitcher(SR, 0.25)
    s.push(_chunk(1.0, 2.25))
    tail = s.flush()  # the faded-out tail of what was pushed
    n = int(0.25 * SR)
    k = (np.arange(n) + 0.5) / n
    assert np.allclose(tail, np.cos(0.5 * math.pi * k), atol=1e-4)
    assert np.allclose(tail ** 2 + np.sin(0.5 * math.pi * k) ** 2, 1.0, atol=2e-4)


def test_flush_gives_back_the_last_tail_once_and_then_nothing():
    s = Stitcher(SR, 0.25)
    assert len(s.flush()) == 0  # nothing was pushed
    s.push(_chunk(1.0, 2.25))
    assert len(s.flush()) == int(0.25 * SR) and len(s.flush()) == 0


def test_a_segment_too_short_to_hold_its_own_crossfade_is_refused():
    with pytest.raises(ValueError):
        Stitcher(SR, 0.25).push(_chunk(1.0, 0.4))


def test_pushing_does_not_change_the_array_it_was_given():
    chunk = _chunk(1.0, 2.25)
    s = Stitcher(SR, 0.25)
    s.push(chunk)
    s.push(chunk)
    assert np.all(chunk == 1.0)


def test_stitched_audio_keeps_its_dtype_and_stays_inside_one_for_real_segments():
    from tensixinet.music.chorus import render_chorus
    s = Stitcher(SR, 0.25)
    first = s.push(render_chorus([FALLBACK], 2.25, SR, segment=0))
    second = s.push(render_chorus([FALLBACK], 2.25, SR, segment=1))
    assert first.dtype == second.dtype == np.float32 and np.abs(second).max() <= 1.5  # equal-power can add up to 3 dB at the join


# --- Stitcher with two channels ----------------------------------------------------------------------------------------------------

def test_a_stereo_segment_is_crossfaded_in_both_ears_like_a_mono_one():
    n = int(0.25 * SR)
    left = np.random.default_rng(7).uniform(-1, 1, int(2.25 * SR))
    right = np.random.default_rng(8).uniform(-1, 1, int(2.25 * SR))
    mono_l, mono_r = Stitcher(SR, 0.25), Stitcher(SR, 0.25)
    st = Stitcher(SR, 0.25)
    stereo = np.stack([left, right], axis=1).astype(np.float32)
    first = st.push(stereo)
    assert first.shape == (int(2.25 * SR) - n, 2) and first.dtype == np.float32
    second = st.push(stereo)
    ref_l = np.concatenate([mono_l.push(left.astype(np.float32)), mono_l.push(left.astype(np.float32))])
    ref_r = np.concatenate([mono_r.push(right.astype(np.float32)), mono_r.push(right.astype(np.float32))])
    assert np.allclose(np.concatenate([first, second])[:, 0], ref_l, atol=1e-5) and np.allclose(np.concatenate([first, second])[:, 1], ref_r, atol=1e-5)
    tail = st.flush()
    assert tail.shape == (n, 2) and st.flush().shape == (0, 2)


def test_a_stitcher_that_has_taken_nothing_flushes_nothing_in_either_shape():
    assert Stitcher(SR, 0.25).flush().shape == (0,)
