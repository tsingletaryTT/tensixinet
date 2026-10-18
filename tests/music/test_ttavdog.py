# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""ttavdog: a droning oscillator that a ringing filter opens, one voice on each half of the swing (Plumbutter guide, AVDog)."""
import dataclasses

import numpy as np
import pytest

from tensixinet.music.devices import SHOCK_DECAY_BEATS, SHOCK_RATE_BEATS, seconds_since, strike_times
from tensixinet.music.mapping import aiclk_to_bpm, chip_pressure, temp_to_pitch_hz
from tensixinet.music.ttavdog import (
    AVDOG_LEVEL,
    RED_GAIN,
    avdog_voices,
    skewed_triangle,
    ttavdog,
    undulation,
)
from tensixinet.telemetry import FALLBACK

SR = 8000


# --- the undulation: a damped swing that reverses ---------------------------------------------------------------------

def test_the_undulation_is_silent_before_the_first_strike_and_swings_positive_first():
    since = np.array([np.inf, np.inf, 0.0, 0.05])
    u = undulation(since, rate_hz=2.0, decay_s=1.0)
    assert u[:3].tolist() == [0.0, 0.0, 0.0] and u[3] > 0


def test_the_undulation_changes_direction_every_half_cycle_and_dies_away():
    tau = np.linspace(0.001, 3.0, 3000)
    u = undulation(tau, rate_hz=2.0, decay_s=1.0)
    crossings = tau[np.where(np.diff(np.sign(u)) != 0)[0]]
    assert crossings[:4] == pytest.approx([0.25, 0.5, 0.75, 1.0], abs=0.002)  # a half cycle at 2 Hz is a quarter second
    peaks = [np.max(np.abs(u[(tau > a) & (tau < a + 0.25)])) for a in (0.0, 0.5, 1.0, 1.5)]
    assert peaks == sorted(peaks, reverse=True) and peaks[-1] < 0.5 * peaks[0]


def test_the_undulation_peak_after_a_quarter_cycle_is_one_decayed_by_the_exponential():
    u = undulation(np.array([0.125]), rate_hz=2.0, decay_s=1.0)  # a quarter of a 0.5 s cycle
    assert u[0] == pytest.approx(np.exp(-0.125), rel=1e-9)


def test_the_undulation_swings_to_the_same_height_each_way_while_it_decays_only_slowly():
    tau = np.linspace(0, 0.9, 9001)
    u = undulation(tau, rate_hz=1.0, decay_s=1000.0)
    assert np.max(u) == pytest.approx(1.0, abs=1e-3) and np.min(u) == pytest.approx(-1.0, abs=1e-3)


# --- the dogtone: a triangle that skews toward a saw ---------------------------------------------------------------

def test_shape_zero_is_a_triangle_and_the_extremes_are_saws():
    x = np.linspace(0, 1, 1001)[:-1]
    tri = skewed_triangle(x, 0.0)
    assert tri[0] == pytest.approx(-1.0) and tri[500] == pytest.approx(1.0, abs=2e-3) and tri[250] == pytest.approx(0.0, abs=2e-3)
    up = skewed_triangle(x, 0.98)  # rises for most of the cycle, then falls fast: an up-sloped saw
    assert up[500] == pytest.approx(0.0, abs=0.05) and np.argmax(up) > 950


def test_the_inverse_shape_is_the_same_wave_run_backwards():
    x = np.linspace(0, 1, 2001)
    assert np.allclose(skewed_triangle(x, -0.6), skewed_triangle(1 - x, 0.6), atol=1e-12)


def test_the_wave_stays_between_minus_one_and_one_even_for_a_shape_past_the_ends():
    x = np.linspace(0, 1, 5000)
    for shape in (-5.0, -1.0, 0.3, 1.0, 5.0):
        y = skewed_triangle(x, shape)
        assert np.isfinite(y).all() and y.min() >= -1 - 1e-9 and y.max() <= 1 + 1e-9


# --- the two voices -------------------------------------------------------------------------------------------------

def chip(**kw):
    return dataclasses.replace(FALLBACK, **{"asic_temperature": 60.0, "power": 50.0, **kw})


def test_the_white_voice_sounds_on_one_half_of_the_swing_and_the_red_on_the_other():
    white, red = avdog_voices(chip(), 12.0, SR)
    assert np.any(white) and np.any(red)
    assert not np.any((white != 0) & (red != 0))  # call and response: never together


def test_the_voices_follow_the_undulations_direction():
    t = chip()
    white, red = avdog_voices(t, 12.0, SR)
    beat = 60.0 / aiclk_to_bpm(t.aiclk)
    since = seconds_since(strike_times(t, 12.0), len(white), SR)
    u = undulation(since, SHOCK_RATE_BEATS / beat, SHOCK_DECAY_BEATS * beat)
    assert not np.any(white[u <= 0]) and not np.any(red[u >= 0])


def test_nothing_sounds_before_the_first_strike():
    t = chip()
    first = strike_times(t, 12.0)[0]
    white, red = avdog_voices(t, 12.0, SR)
    assert not np.any(white[: int(first * SR)]) and not np.any(red[: int(first * SR)])


def test_the_red_voice_is_the_same_pitch_but_the_opposite_slope_and_a_little_quieter():
    t = chip(asic_temperature=40.0, power=120.0)  # pressure well above zero so the shape is skewed
    assert chip_pressure(t) > 0.3
    white, red = avdog_voices(t, 12.0, SR)
    wf = np.abs(np.fft.rfft(white))[1:]
    rf = np.abs(np.fft.rfft(red))[1:]
    assert np.argmax(wf) == np.argmax(rf)  # the same fundamental
    assert RED_GAIN < 1.0 and np.max(np.abs(red)) < np.max(np.abs(white)) * 1.1


def test_the_dogtone_is_at_the_chips_pitch():
    t = chip(asic_temperature=80.0)
    white, _ = avdog_voices(t, 12.0, SR)
    spec = np.abs(np.fft.rfft(white))
    peak = np.argmax(spec[20:]) + 20
    assert peak * SR / len(white) == pytest.approx(temp_to_pitch_hz(80.0), rel=0.03)


def test_a_chip_near_a_limit_gets_a_harder_wave_than_a_cool_one():
    def top_share(t):
        w, _ = avdog_voices(t, 12.0, SR)
        s = np.abs(np.fft.rfft(w)) ** 2
        return float(s[len(s) // 8:].sum() / s.sum())  # energy in the top of the spectrum: a saw has more of it than a triangle
    # the same temperature, so the same pitch: only the throttle limit, and so the pressure, differs
    relaxed, strained = chip(thm_limit_throttle=120.0), chip(thm_limit_throttle=61.0)
    assert chip_pressure(strained) > 0.9 > 0.3 > chip_pressure(relaxed)
    assert top_share(strained) > 1.3 * top_share(relaxed)


def test_each_voice_is_the_dogtone_under_its_own_half_of_the_undulation_with_opposite_slopes():
    t = chip(thm_limit_throttle=65.0)
    shape = chip_pressure(t)
    assert shape > 0.5
    white, red = avdog_voices(t, 12.0, SR)
    n = len(white)
    beat = 60.0 / aiclk_to_bpm(t.aiclk)
    u = undulation(seconds_since(strike_times(t, 12.0), n, SR), SHOCK_RATE_BEATS / beat, SHOCK_DECAY_BEATS * beat)
    phase = temp_to_pitch_hz(t.asic_temperature) * np.arange(n) / SR
    up, down = u > 0.3, u < -0.3
    assert up.any() and down.any()
    assert np.allclose(white[up] / u[up], skewed_triangle(phase[up], shape), atol=1e-9)
    assert np.allclose(red[down] / -u[down], skewed_triangle(phase[down], -shape), atol=1e-9)


# --- the device ---------------------------------------------------------------------------------------------------------

def test_the_levels_are_a_half_and_the_red_is_four_fifths_of_the_white():
    assert AVDOG_LEVEL == 0.5 and RED_GAIN == 0.8
    audio = ttavdog(chip(), 12.0, SR, 0)
    white, _ = avdog_voices(chip(), 12.0, SR)
    assert float(np.max(np.abs(audio))) == pytest.approx(0.5 * float(np.max(np.abs(white))), rel=0.3)


def test_the_device_mixes_the_voices_inside_full_scale():
    t = chip()
    audio = ttavdog(t, 12.0, SR, 0)
    white, red = avdog_voices(t, 12.0, SR)
    assert audio.dtype == np.float32 and audio.shape == (12 * SR,)
    assert np.allclose(audio, AVDOG_LEVEL * (white + RED_GAIN * red), atol=1e-6) and np.max(np.abs(audio)) <= 1.0


def test_the_device_is_deterministic_and_never_nan():
    t = chip()
    assert np.array_equal(ttavdog(t, 4.0, SR, 3), ttavdog(t, 4.0, SR, 3)) and np.isfinite(ttavdog(t, 4.0, SR, 0)).all()


def test_the_parameters_are_the_pitch_the_shape_and_the_undulations_rate_and_decay():
    from tensixinet.music.ttavdog import avdog_parameters
    t = chip(thm_limit_throttle=65.0)
    pitch, shape, rate_hz, decay_s = avdog_parameters(t)
    beat = 60.0 / aiclk_to_bpm(t.aiclk)
    assert pitch == temp_to_pitch_hz(t.asic_temperature) and shape == chip_pressure(t)
    assert rate_hz == pytest.approx(SHOCK_RATE_BEATS / beat) and decay_s == pytest.approx(SHOCK_DECAY_BEATS * beat)

