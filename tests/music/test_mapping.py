# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The README's hardware -> music table, as executable spec."""
import pytest

from tensixinet.music.mapping import (
    aiclk_to_bpm,
    ddr_lane_divisions,
    norm_temp,
    power_to_volume,
    temp_to_pitch_hz,
)

C3, C5 = 130.81, 523.25


def test_temperature_maps_25_to_100C_onto_C3_to_C5():
    assert temp_to_pitch_hz(25) == pytest.approx(C3, rel=1e-3)
    assert temp_to_pitch_hz(100) == pytest.approx(C5, rel=1e-3)


def test_pitch_is_linear_in_semitones_so_midpoint_is_one_octave_up():
    assert temp_to_pitch_hz(62.5) == pytest.approx(C3 * 2, rel=1e-3)


@pytest.mark.parametrize("t", [-40, 0, 24.9])
def test_pitch_clamps_below_range(t):
    assert temp_to_pitch_hz(t) == pytest.approx(C3, rel=1e-3)


def test_pitch_clamps_above_range():
    assert temp_to_pitch_hz(400) == pytest.approx(C5, rel=1e-3)


def test_aiclk_maps_400_to_1350MHz_onto_60_to_180_bpm():
    assert aiclk_to_bpm(400) == pytest.approx(60)
    assert aiclk_to_bpm(1350) == pytest.approx(180)
    assert aiclk_to_bpm(800) == pytest.approx(60 + 120 * 400 / 950)


def test_bpm_clamps_outside_the_aiclk_range():
    assert aiclk_to_bpm(0) == 60
    assert aiclk_to_bpm(5000) == 180


def test_power_maps_0_to_200W_onto_volume_0p3_to_1():
    assert power_to_volume(0) == pytest.approx(0.3)
    assert power_to_volume(200) == pytest.approx(1.0)


def test_volume_curve_is_exponential_not_linear():
    # Convex: halfway in power is below halfway in volume (0.65).
    assert power_to_volume(100) < 0.65
    assert power_to_volume(50) < power_to_volume(100) < power_to_volume(150)


def test_volume_clamps():
    assert power_to_volume(-5) == pytest.approx(0.3)
    assert power_to_volume(900) == pytest.approx(1.0)


def test_norm_temp_is_0_to_1_over_25_to_100():
    assert norm_temp(25) == 0 and norm_temp(100) == 1
    assert norm_temp(62.5) == pytest.approx(0.5)


def test_ddr_lanes_use_the_readme_divisions_and_mute_status_zero():
    # Lane i pulses (3, 4, 5, 7) times per bar; status 0 mutes the lane.
    assert ddr_lane_divisions((1, 1, 1, 1)) == (3, 4, 5, 7)
    assert ddr_lane_divisions((1, 0, 2, 0)) == (3, 0, 5, 0)
