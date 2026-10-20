# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Space: where a voice sits (left to right, near to far, narrow to wide) and how it moves, from the chips' own signals."""
import dataclasses

import numpy as np
import pytest

from tensixinet.music.chorus import device_names
from tensixinet.music.space import (
    DEVICE_SPACE, chip_azimuth, device_space, motion_azimuth, pan_gains, smooth, spatialize,
)
from tensixinet.telemetry import FALLBACK

SR = 8000
RNG = np.random.default_rng(1)
NOISE = RNG.uniform(-1, 1, SR * 2)


def tel(**kw):
    return dataclasses.replace(FALLBACK, **kw)


# --- panning -----------------------------------------------------------------------------------------------------------------

def test_the_pan_law_keeps_the_power_constant_from_hard_left_to_hard_right():
    az = np.linspace(-1, 1, 41)
    left, right = pan_gains(az)
    assert np.allclose(left ** 2 + right ** 2, 1.0)
    assert (left[0], right[0]) == pytest.approx((1.0, 0.0), abs=1e-12) and (left[-1], right[-1]) == pytest.approx((0.0, 1.0), abs=1e-12)
    assert left[20] == pytest.approx(np.sqrt(0.5)) == pytest.approx(right[20])
    assert (np.diff(left) <= 1e-12).all() and (np.diff(right) >= -1e-12).all()   # moving right takes the left down and the right up


def test_a_position_past_the_edge_is_the_edge():
    assert pan_gains(np.array([-3.0, 7.0]))[0].tolist() == pytest.approx([1.0, 0.0], abs=1e-12)
    assert pan_gains(np.array([-3.0, 7.0]))[1].tolist() == pytest.approx([0.0, 1.0], abs=1e-12)


def test_smoothing_removes_a_jump_and_keeps_the_ends_and_the_length():
    step = np.concatenate([np.full(1000, -1.0), np.full(1000, 1.0)])
    s = smooth(step, SR, 20.0)
    assert len(s) == len(step) and abs(s[0] + 1) < 1e-9 and abs(s[-1] - 1) < 1e-9
    assert np.abs(np.diff(s)).max() < 0.1 * np.abs(np.diff(step)).max()


# --- spatialising one voice ---------------------------------------------------------------------------------------------------

def test_a_centred_narrow_near_voice_is_the_same_in_both_ears_and_keeps_its_power():
    out = spatialize(NOISE, np.zeros(len(NOISE)), 0.0, 0.0, SR)
    assert out.shape == (len(NOISE), 2) and out.dtype == np.float32
    assert np.allclose(out[:, 0], out[:, 1])
    assert float(np.sum(out.astype(np.float64) ** 2)) == pytest.approx(float(np.sum(NOISE ** 2)), rel=1e-4)


def test_a_hard_left_voice_is_silent_on_the_right():
    out = spatialize(NOISE, np.full(len(NOISE), -1.0), 0.0, 0.0, SR)
    assert np.abs(out[:, 1]).max() < 1e-6 and np.allclose(out[:, 0], NOISE, atol=1e-6)


def test_a_voice_that_travels_from_left_to_right_is_heard_on_the_left_first():
    az = np.linspace(-1, 1, len(NOISE))
    out = spatialize(NOISE, az, 0.0, 0.0, SR)
    half = len(NOISE) // 2
    assert np.sum(out[:half, 0] ** 2) > 3 * np.sum(out[:half, 1] ** 2) and np.sum(out[half:, 1] ** 2) > 3 * np.sum(out[half:, 0] ** 2)


def test_width_makes_the_two_ears_differ_for_a_centred_voice_without_changing_the_loudness_much():
    narrow, wide = spatialize(NOISE, np.zeros(len(NOISE)), 0.0, 0.0, SR), spatialize(NOISE, np.zeros(len(NOISE)), 1.0, 0.0, SR)
    assert np.allclose(narrow[:, 0], narrow[:, 1]) and not np.allclose(wide[:, 0], wide[:, 1], atol=0.05)
    corr = float(np.corrcoef(wide[:, 0], wide[:, 1])[0, 1])
    assert 0.3 < corr < 0.95                                                       # wide, not two unrelated noises
    assert float(np.sum(wide ** 2)) == pytest.approx(float(np.sum(narrow ** 2)), rel=0.2)


def test_depth_pushes_a_voice_back_quieter_and_darker():
    def centroid(x):
        spec = np.abs(np.fft.rfft(x)); return float((spec * np.arange(len(spec))).sum() / spec.sum())
    near, far = spatialize(NOISE, np.zeros(len(NOISE)), 0.0, 0.0, SR), spatialize(NOISE, np.zeros(len(NOISE)), 0.0, 1.0, SR)
    assert float(np.sum(far ** 2)) < 0.6 * float(np.sum(near ** 2))
    assert centroid(far[:, 0]) < 0.8 * centroid(near[:, 0])
    assert np.isfinite(far).all()


def test_depth_changing_over_time_moves_a_voice_back_smoothly():
    depth = np.linspace(0.0, 1.0, len(NOISE))
    out = spatialize(NOISE, np.zeros(len(NOISE)), 0.0, depth, SR)
    half = len(NOISE) // 2
    assert np.sum(out[:half] ** 2) > 1.4 * np.sum(out[half:] ** 2)


def test_spatialize_accepts_scalars_or_curves_and_a_short_voice():
    assert spatialize(NOISE[:50], 0.3, 0.2, 0.4, SR).shape == (50, 2)
    assert spatialize(np.zeros(0), 0.0, 0.0, 0.0, SR).shape == (0, 2)
    with pytest.raises(ValueError):
        spatialize(NOISE, np.zeros(10), 0.0, 0.0, SR)                              # a curve must be as long as the voice


# --- where things sit: fixed positions -----------------------------------------------------------------------------------------------

def test_chips_are_spread_across_the_field_in_order_and_one_chip_sits_in_the_middle():
    t = tel()
    homes = [chip_azimuth(i, 4, t) for i in range(4)]
    assert homes == sorted(homes) and len(set(round(h, 3) for h in homes)) == 4
    assert homes[0] < -0.5 and homes[-1] > 0.5 and all(abs(h) <= 0.9 for h in homes)
    assert abs(chip_azimuth(0, 1, t)) < 0.15
    assert chip_azimuth(2, 4, t) == chip_azimuth(2, 4, t)


def test_a_chips_identity_nudges_its_place_a_little_and_never_past_its_neighbour():
    a, b = chip_azimuth(1, 4, tel(tensix_mask=0x3FFF)), chip_azimuth(1, 4, tel(tensix_mask=0x1235))
    assert a != b and abs(a - b) < 0.1


def test_every_device_has_a_place_and_the_air_is_wider_and_farther_than_the_lead():
    assert set(device_names()) <= set(DEVICE_SPACE)
    for name in device_names():
        az, width, depth = device_space(name)
        assert -1 <= az <= 1 and 0 <= width <= 1 and 0 <= depth <= 1
    assert device_space("ultrasonic_shimmer")[1] > device_space("lead")[1] and device_space("ultrasonic_shimmer")[2] > device_space("lead")[2]
    assert device_space("lead")[2] <= 0.1 and device_space("ttdrone")[1] >= 0.8
    assert device_space("not a device") == (0.0, 0.5, 0.5)


# --- motion: some things sit, some travel -----------------------------------------------------------------------------------------

def test_a_calm_chip_barely_moves_and_a_chaotic_one_travels_further():
    n = SR * 20
    calm = motion_azimuth(tel(), "lead", chip=0, n=n, sample_rate=SR, chaos=0.0)
    wild = motion_azimuth(tel(), "lead", chip=0, n=n, sample_rate=SR, chaos=1.0)
    assert np.ptp(calm) < 0.15 and np.ptp(wild) > 0.4 and len(calm) == n
    assert np.array_equal(calm, motion_azimuth(tel(), "lead", chip=0, n=n, sample_rate=SR, chaos=0.0))


def test_a_faster_clock_makes_the_orbit_faster_and_chips_and_devices_orbit_out_of_step():
    n = SR * 30
    def crossings(x):
        return int(np.sum(np.diff(np.sign(x - x.mean())) != 0))
    slow = motion_azimuth(tel(aiclk=400.0), "lead", 0, n, SR, chaos=1.0)
    fast = motion_azimuth(tel(aiclk=1350.0), "lead", 0, n, SR, chaos=1.0)
    assert crossings(fast) > 1.5 * crossings(slow)
    assert not np.allclose(motion_azimuth(tel(), "lead", 0, n, SR, 1.0), motion_azimuth(tel(), "gong", 0, n, SR, 1.0))
    assert not np.allclose(motion_azimuth(tel(), "lead", 0, n, SR, 1.0), motion_azimuth(tel(), "lead", 1, n, SR, 1.0))


# --- gaps a mutation run found ---------------------------------------------------------------------------------------------------------

def test_a_chips_identity_never_moves_it_more_than_the_nudge_and_does_use_most_of_it():
    homes = [chip_azimuth(1, 4, tel(tensix_mask=m)) for m in range(0x3000, 0x3400)]
    base = -0.75 + 1.5 / 3
    assert max(abs(h - base) for h in homes) <= 0.04 + 1e-12 and max(abs(h - base) for h in homes) > 0.032   # written out, not read from the module


def test_a_far_voice_is_quieter_by_the_depth_level_even_when_it_has_no_highs_to_lose():
    from tensixinet.music.space import REFLECTION_S
    t = np.arange(SR * 2) / SR
    delay = int(REFLECTION_S * SR)
    low = np.sin(2 * np.pi * (SR / delay) * t)                          # far below the darkening, and one cycle long in the reflection's delay,
    near, far = spatialize(low, 0.0, 0.0, 0.0, SR), spatialize(low, 0.0, 0.0, 1.0, SR)   # so the reflection adds to the body in step
    ratio = float(np.sqrt(np.mean(far.astype(float) ** 2)) / np.sqrt(np.mean(near.astype(float) ** 2)))
    assert 0.70 < ratio < 0.90                                          # 1 - 0.55 for the body, 0.35 for the reflection


def test_a_far_voice_carries_an_early_reflection_a_near_one_does_not():
    from tensixinet.music.space import REFLECTION_S
    click = np.zeros(SR)
    click[10] = 1.0
    at = 10 + int(REFLECTION_S * SR)
    far, near = spatialize(click, 0.0, 0.0, 1.0, SR), spatialize(click, 0.0, 0.0, 0.0, SR)
    assert np.abs(far[at - 2:at + 6]).max() > 0.02 and np.abs(near[at - 2:at + 6]).max() < 1e-6
