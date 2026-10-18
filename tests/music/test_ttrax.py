# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""ttrax: seven barres whose oscillators bounce between each other, from the Sidrax guide."""
import dataclasses

import numpy as np
import pytest

from tensixinet.music.mapping import aiclk_to_bpm, temp_to_pitch_hz
from tensixinet.music.patch import DEFAULT_PATCH
from tensixinet.music.ttrolzer import barre_bpm
from tensixinet.music.ttrax import (
    BOUNDS_TABLE,
    N_BARRES,
    barre_bounds,
    barre_frequency,
    bounce_wave,
    gesture_envelopes,
    ttrax,
    windowed_bounce,
)
from tensixinet.telemetry import FALLBACK

SR = 8000


def calm(**kw):
    """A cool, idle chip, so the chaos knob (pressure) stays at zero."""
    return dataclasses.replace(FALLBACK, asic_temperature=30.0, power=10.0, **kw)


# --- the bounds table (guide, figure 1) ----------------------------------------------------------------------

def test_there_are_seven_barres_and_three_bound_assignments_each():
    assert N_BARRES == 7 and set(BOUNDS_TABLE) == {"3", "5", "35"}
    assert all(len(BOUNDS_TABLE[k]) == N_BARRES for k in BOUNDS_TABLE)


def test_the_bounds_match_the_guides_table_counting_barres_from_zero():
    assert BOUNDS_TABLE["3"][0] == (1, 2) and BOUNDS_TABLE["3"][6] == (0, 1)
    assert BOUNDS_TABLE["5"][0] == (3, 6) and BOUNDS_TABLE["5"][1] == (0, 4) and BOUNDS_TABLE["5"][4] == (3, 0)
    assert BOUNDS_TABLE["35"][0] == (4, 5) and BOUNDS_TABLE["35"][2] == (0, 6) and BOUNDS_TABLE["35"][6] == (3, 4)


def test_a_barre_is_never_its_own_bound_and_its_two_bounds_differ():
    for pairs in BOUNDS_TABLE.values():
        for barre, (lo, hi) in enumerate(pairs):
            assert barre not in (lo, hi) and lo != hi and 0 <= lo < N_BARRES and 0 <= hi < N_BARRES


# --- static bounds from the memory temperatures --------------------------------------------------------------

def test_a_barres_upper_bound_follows_its_gddr_channel_and_its_lower_bound_the_next_one():
    base = (40.0,) * 8
    hot = (40.0, 90.0) + (40.0,) * 6  # channel 1 hot
    lo0, hi0 = barre_bounds(base, 0)
    lo_a, hi_a = barre_bounds(hot, 0)  # barre 0's lower bound reads channel 1
    lo_b, hi_b = barre_bounds(hot, 1)  # barre 1's upper bound reads channel 1
    assert lo_a < lo0 and hi_a == hi0
    assert hi_b > barre_bounds(base, 1)[1] and lo_b == barre_bounds(base, 1)[0]
    assert lo0 < 0 < hi0


def test_the_last_barres_lower_bound_wraps_to_the_first_channel_and_fewer_channels_still_work():
    assert barre_bounds((30.0, 30.0, 30.0), 6)[0] == barre_bounds((30.0, 30.0, 30.0), 0)[0]
    assert barre_bounds((50.0,), 3) == barre_bounds((50.0,), 0)


# --- the bounce: frequency is bounce over bounds -----------------------------------------------------------------

def test_the_windowed_bounce_stays_between_its_bounds():
    n = SR
    t = np.arange(n) / SR
    lower = -0.5 + 0.1 * np.sin(2 * np.pi * 3 * t)
    upper = 0.5 + 0.1 * np.sin(2 * np.pi * 5 * t)
    w = windowed_bounce(100.0, lower, upper, 1.0, SR)
    assert np.all(w >= lower - 1e-9) and np.all(w <= upper + 1e-9)


def test_where_the_upper_bound_falls_below_the_lower_the_oscillator_rides_the_lower_bound():
    n = 400
    lower = np.full(n, 0.2)
    upper = np.where(np.arange(n) < 200, 0.8, 0.0)
    w = windowed_bounce(50.0, lower, upper, 0.6, SR)
    assert np.all(w[200:] == 0.2) and w[:200].max() > 0.5


def _crossings(x):
    return int(np.sum(np.diff(np.signbit(x - x.mean())) != 0))


def test_a_narrower_window_bounces_faster_in_proportion_so_frequency_is_bounce_over_bounds():
    n = 2 * SR
    wide = windowed_bounce(50.0, np.full(n, -0.8), np.full(n, 0.8), 1.6, SR)
    half = windowed_bounce(50.0, np.full(n, -0.4), np.full(n, 0.4), 1.6, SR)  # half as wide
    assert _crossings(half) == pytest.approx(2 * _crossings(wide), rel=0.05)


def test_a_converging_window_raises_the_pitch_and_lowers_the_level():
    n = 4 * SR
    lower = -np.linspace(0.8, 0.1, n)
    upper = np.linspace(0.8, 0.1, n)
    w = windowed_bounce(40.0, lower, upper, 1.6, SR)
    early, late = w[: SR // 2], w[-SR // 2:]
    assert _crossings(late) > 2 * _crossings(early) and np.ptp(late) < 0.5 * np.ptp(early)


def test_the_speed_up_is_capped_so_a_closing_window_cannot_alias():
    n = SR
    w = windowed_bounce(100.0, np.zeros(n), np.full(n, 1e-9), 1.0, SR)
    assert np.isfinite(w).all()


def test_chaos_modulates_the_frequency_of_the_bounce_and_zero_chaos_changes_nothing():
    n = SR
    lead = np.sin(2 * np.pi * 7 * np.arange(n) / SR)
    flat = windowed_bounce(60.0, np.full(n, -0.5), np.full(n, 0.5), 1.0, SR)
    assert np.array_equal(flat, windowed_bounce(60.0, np.full(n, -0.5), np.full(n, 0.5), 1.0, SR, chaos=0.0, leader=lead))
    assert not np.allclose(flat, windowed_bounce(60.0, np.full(n, -0.5), np.full(n, 0.5), 1.0, SR, chaos=0.5, leader=lead))


# --- gestures: a press opens one channel, the release the other -----------------------------------------------------

def test_a_barre_is_pressed_every_four_steps_of_its_own_ttrolzer_barre():
    bpm = 120.0
    press, release = gesture_envelopes(bpm, 2, 30.0, SR)
    period = 4 * 60.0 / barre_bpm(bpm, 2)
    starts = np.flatnonzero(np.diff(press) > 0.5) + 1
    assert np.diff(starts) / SR == pytest.approx(period, abs=2 / SR)


def test_the_press_envelope_peaks_at_the_press_and_the_release_envelope_at_the_release():
    bpm = 120.0
    press, release = gesture_envelopes(bpm, 3, 30.0, SR)
    period = 4 * 60.0 / barre_bpm(bpm, 3)
    assert press[0] == pytest.approx(1.0) and release[0] == 0.0
    mid = int(np.ceil(period / 2 * SR))
    assert release[mid] == pytest.approx(1.0, abs=0.01) and release[mid - 2] < 0.01
    assert press[mid - 1] < 0.5 and press.max() <= 1.0 and release.max() <= 1.0 and min(press.min(), release.min()) >= 0


def test_the_slower_barres_are_pressed_less_often():
    counts = [int(np.sum(np.diff(gesture_envelopes(120.0, b, 60.0, SR)[0]) > 0.5)) for b in range(N_BARRES)]
    assert all(a > b for a, b in zip(counts, counts[1:]))


# --- the device ---------------------------------------------------------------------------------------------------------

def test_the_seven_barres_are_tuned_to_the_first_seven_scale_degrees():
    base = temp_to_pitch_hz(30.0)
    freqs = [barre_frequency(base, b, DEFAULT_PATCH.scale) for b in range(N_BARRES)]
    semis = [12 * np.log2(f / base) for f in freqs]
    assert semis == pytest.approx([0, 2, 4, 7, 9, 12, 14], abs=1e-6)  # major pentatonic, then up an octave


def test_ttrax_is_bounded_deterministic_and_audible():
    out = ttrax(calm(), 4.0, SR, 0)
    assert out.dtype == np.float32 and np.abs(out).max() <= 1.0 and np.abs(out).max() > 0.1
    assert np.array_equal(out, ttrax(calm(), 4.0, SR, 0))


def _centroid(x, sr):
    spec = np.abs(np.fft.rfft(x))
    return float((spec * np.arange(len(spec))).sum() / spec.sum() * sr / len(x))


def test_a_faster_arc_clock_raises_every_barre_in_proportion(monkeypatch):
    from tensixinet.music import ttrax as ttrax_module
    def tuned(arcclk):
        seen = []
        real = ttrax_module.windowed_bounce
        monkeypatch.setattr(ttrax_module, "windowed_bounce", lambda f, *a, **k: seen.append(f) or real(f, *a, **k))
        ttrax(calm(arcclk=arcclk), 0.5, 44100, 0)
        monkeypatch.setattr(ttrax_module, "windowed_bounce", real)
        return seen[:N_BARRES]  # the seven walls are built first, one per barre
    slow, fast = tuned(800.0), tuned(1600.0)
    assert len(slow) == N_BARRES and fast == pytest.approx([2 * f for f in slow])


def test_pressure_is_the_chaos_knob_so_a_stressed_chip_is_noisier():
    def flatness(x):  # geometric over arithmetic mean of the spectrum: 1 for noise, near 0 for tones
        spec = np.abs(np.fft.rfft(x)) ** 2 + 1e-12
        return float(np.exp(np.mean(np.log(spec))) / np.mean(spec))
    cool = ttrax(calm(), 8.0, SR, 0)
    stressed = ttrax(dataclasses.replace(FALLBACK, asic_temperature=89.0, power=124.0), 8.0, SR, 0)
    assert flatness(stressed) > 2 * flatness(cool)


def test_the_gesture_rate_follows_the_chips_tempo():
    slow = ttrax(calm(aiclk=400.0), 8.0, SR, 0)
    fast = ttrax(calm(aiclk=1350.0), 8.0, SR, 0)
    assert not np.allclose(slow, fast)


def test_ttrax_follows_the_patch_scale():
    from tensixinet.music.patch import SCALES
    whole = dataclasses.replace(DEFAULT_PATCH, scale_name="whole_tone", scale=SCALES["whole_tone"])
    assert not np.allclose(ttrax(calm(), 4.0, SR, 0), ttrax(calm(), 4.0, SR, 0, patch=whole))


def test_wider_memory_spread_makes_a_wider_swing():
    narrow = ttrax(calm(gddr_temps=(50.0,) * 8), 2.0, SR, 0)
    wide = ttrax(calm(gddr_temps=(26.0, 99.0) * 4), 2.0, SR, 0)
    assert np.ptp(wide) > np.ptp(narrow)


def test_bounce_wave_is_still_a_triangle_between_two_constants():
    w = bounce_wave(freq_hz=50, lower=-0.2, upper=0.7, n=SR, sample_rate=SR)
    assert w.min() == pytest.approx(-0.2, abs=0.01) and w.max() == pytest.approx(0.7, abs=0.01)


# --- limits and wiring that the mutation checks found untested ---------------------------------------------------------

def test_a_collapsing_window_speeds_the_oscillator_up_by_eight_and_no_more():
    n = SR
    w = windowed_bounce(100.0, np.zeros(n), np.full(n, 1e-9), 1.0, SR)
    assert _crossings(w) == pytest.approx(2 * 100 * 8, rel=0.05)


def test_a_very_wide_window_slows_the_oscillator_to_a_quarter_and_no_further():
    n = SR
    w = windowed_bounce(100.0, np.zeros(n), np.full(n, 100.0), 1.0, SR)
    assert _crossings(w) == pytest.approx(2 * 100 * 0.25, rel=0.1)


def test_chaos_can_never_push_the_frequency_past_nyquist():
    n = SR
    w = windowed_bounce(2000.0, np.full(n, -0.5), np.full(n, 0.5), 1.0, SR, chaos=1.0, leader=np.ones(n))  # asks for 6 kHz at 8 kHz
    assert _crossings(w) == pytest.approx(2 * 0.45 * SR, rel=0.03)


def test_each_envelope_is_exactly_zero_outside_its_own_half_of_the_period():
    press, release = gesture_envelopes(120.0, 3, 30.0, SR)
    period = 4 * 60.0 / barre_bpm(120.0, 3)
    mid = int(np.ceil(period / 2 * SR))
    assert np.all(press[mid: mid + 50] == 0.0) and np.all(release[mid - 50: mid - 1] == 0.0)


@pytest.fixture
def traced(monkeypatch):
    """Every windowed_bounce call of one ttrax render: (args, returned array)."""
    from tensixinet.music import ttrax as ttrax_module
    calls = []
    real = ttrax_module.windowed_bounce
    def spy(*a, **k):
        out = real(*a, **k)
        calls.append((a, out))
        return out
    monkeypatch.setattr(ttrax_module, "windowed_bounce", spy)
    ttrax(calm(), 1.0, SR, 0)
    return calls


def test_a_release_glitch_bounces_between_the_two_barres_the_guides_table_gives_for_switch_three(traced):
    walls = [out for _, out in traced[:N_BARRES]]
    for b in range(N_BARRES):
        (f, lower, upper, width, sr, chaos, leader), _ = traced[N_BARRES + 2 * b + 1]  # barre b's glitch
        lo_idx, hi_idx = BOUNDS_TABLE["3"][b]
        assert np.array_equal(lower, walls[lo_idx]) and np.array_equal(upper, walls[hi_idx])


def test_each_barre_is_chaos_modulated_by_the_oscillator_two_barres_along(traced):
    walls = [out for _, out in traced[:N_BARRES]]
    for b in range(N_BARRES):
        for call in (traced[N_BARRES + 2 * b], traced[N_BARRES + 2 * b + 1]):  # its tone and its glitch
            assert np.allclose(call[0][6], walls[(b + 2) % N_BARRES] / 0.9)


def test_the_sum_of_the_barres_goes_through_a_saturating_tanh(monkeypatch):
    seen = []
    real = np.tanh
    monkeypatch.setattr(np, "tanh", lambda x: seen.append(len(x)) or real(x))
    ttrax(calm(), 0.5, SR, 0)
    assert seen == [SR // 2]


def test_the_chaos_knob_is_pressure_plus_how_fast_the_readings_move(monkeypatch):
    from tensixinet.music import ttrax as ttrax_module
    from tensixinet.music.mapping import RATE_FULL_TEMP_C_PER_S
    seen = []
    real = ttrax_module.windowed_bounce
    monkeypatch.setattr(ttrax_module, "windowed_bounce", lambda *a, **k: (seen.append(a[5]) if len(a) > 5 else None) or real(*a, **k))  # the fixed walls take no chaos
    ttrax(calm(), 0.5, SR, 0)
    assert set(seen) == {0.0}  # a calm chip that holds still
    seen.clear()
    ttrax(calm(temp_rate=0.5 * RATE_FULL_TEMP_C_PER_S), 0.5, SR, 0)
    assert set(seen) == {0.5}  # heating at half the full rate, no pressure: half chaos
