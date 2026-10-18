# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The README's hardware -> music table, as executable spec."""
import pytest

from tensixinet.music.mapping import (
    aiclk_to_bpm,
    ddr_lane_sides,
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


def test_ddr_lanes_are_rolls_of_three_to_six_sides_and_status_zero_mutes_a_lane():
    assert ddr_lane_sides((1, 1, 1, 1)) == (3, 4, 5, 6)
    assert ddr_lane_sides((1, 0, 2, 0)) == (3, 0, 5, 0)


# --- the wider readings -------------------------------------------------------------------

from tensixinet.music.mapping import (  # noqa: E402
    chip_identity,
    drone_voices,
    eth_lane_levels,
    heartbeat_phase,
    pressure,
    turn_weight,
    voltage_to_cents,
)


def test_pressure_is_zero_for_a_cool_idle_chip_and_one_at_either_limit():
    assert pressure(40, 0, 90, 125) == 0.0
    assert pressure(90, 0, 90, 125) == 1.0  # at the throttle temperature
    assert pressure(40, 125, 90, 125) == 1.0  # at the power limit
    assert pressure(150, 400, 90, 125) == 1.0  # clamped


def test_pressure_takes_whichever_limit_is_closer():
    hot_only = pressure(65, 0, 90, 125)  # halfway up the temperature ramp
    loaded_only = pressure(40, 0.625 * 125, 90, 125)  # halfway up the power ramp (31 to 125 W)
    assert hot_only == pytest.approx(0.5)
    assert loaded_only == pytest.approx(0.5)
    assert pressure(65, 0.625 * 125, 90, 125) == pytest.approx(0.5)


def test_a_lower_throttle_limit_means_more_pressure_at_the_same_temperature():
    assert pressure(60, 0, 70, 125) > pressure(60, 0, 90, 125)


def test_eth_lanes_are_four_nibbles_each_scaled_by_links_up():
    assert eth_lane_levels(0xFFFF) == (1.0, 1.0, 1.0, 1.0)
    assert eth_lane_levels(0x0000) == (0.0, 0.0, 0.0, 0.0)
    assert eth_lane_levels(0x3EDF) == (1.0, 0.75, 0.75, 0.5)  # nibbles f, d, e, 3 from the low end
    assert eth_lane_levels(0b0000_0000_0000_0111) == (0.75, 0.0, 0.0, 0.0)


def test_chip_identity_differs_between_chips_and_is_stable():
    assert chip_identity(0x37F7, 0x3EDF) == chip_identity(0x37F7, 0x3EDF)
    ids = {chip_identity(a, b) for a, b in [(0x37F7, 0x3EDF), (0x37FE, 0x3EDF), (0x37F7, 0x3EEF)]}
    assert len(ids) == 3  # a change in either mask changes it


def test_voltage_detunes_around_0p75_and_stays_within_forty_cents():
    assert voltage_to_cents(0.75) == 0.0
    assert voltage_to_cents(0.71) < 0 < voltage_to_cents(0.82)
    assert voltage_to_cents(0.82) == pytest.approx(28.0)
    assert voltage_to_cents(5.0) == 40.0 and voltage_to_cents(0.0) == -40.0


def test_drone_voices_follow_pcie_usage_between_one_and_six():
    assert [drone_voices(u) for u in (0, 1, 4, 6, 40)] == [1, 1, 4, 6, 6]


def test_heartbeat_phase_cycles_through_sixteenths():
    assert heartbeat_phase(0) == 0.0 and heartbeat_phase(8) == 0.5 and heartbeat_phase(16) == 0.0


def test_turn_weight_is_plain_power_for_an_ordinary_chip():
    assert turn_weight(power=40, board_power=120, mean_board_power=120, pressure=0.0, busy=None) == 40


def test_turn_weight_rises_with_each_of_its_inputs():
    base = dict(power=40, board_power=120, mean_board_power=120, pressure=0.0, busy=None)
    assert turn_weight(**dict(base, board_power=240)) == 80  # twice the average board load
    assert turn_weight(**dict(base, pressure=1.0)) == 120  # pressure 1 triples it
    assert turn_weight(**dict(base, busy=2)) == 80  # each process adds half, up to three
    assert turn_weight(**dict(base, busy=9)) == 100
    assert turn_weight(**dict(base, busy=0)) == 40  # an idle chip is not penalised below plain power


# --- live mode: how fast the readings are moving -------------------------------------------

import dataclasses  # noqa: E402

from tensixinet.music.mapping import (  # noqa: E402
    RATE_FULL_POWER_W_PER_S,
    RATE_FULL_TEMP_C_PER_S,
    chip_chaos,
    chip_pressure,
    rate_activity,
)
from tensixinet.telemetry import FALLBACK  # noqa: E402


def test_a_single_snapshot_has_no_rates():
    assert FALLBACK.temp_rate == 0.0
    assert FALLBACK.power_rate == 0.0


def test_rate_activity_is_zero_when_nothing_moves_and_one_at_the_full_rate_of_either_reading():
    assert rate_activity(0.0, 0.0) == 0.0
    assert rate_activity(RATE_FULL_TEMP_C_PER_S, 0.0) == 1.0
    assert rate_activity(0.0, RATE_FULL_POWER_W_PER_S) == 1.0
    assert rate_activity(10 * RATE_FULL_TEMP_C_PER_S, 0.0) == 1.0  # clamped


def test_rate_activity_does_not_care_whether_the_reading_rises_or_falls():
    assert rate_activity(-0.5 * RATE_FULL_TEMP_C_PER_S, 0.0) == pytest.approx(0.5)
    assert rate_activity(0.0, -0.25 * RATE_FULL_POWER_W_PER_S) == pytest.approx(0.25)


def test_rate_activity_takes_whichever_reading_is_moving_faster():
    assert rate_activity(0.2 * RATE_FULL_TEMP_C_PER_S, 0.6 * RATE_FULL_POWER_W_PER_S) == pytest.approx(0.6)


def test_chaos_is_pressure_for_a_chip_that_holds_still():
    t = dataclasses.replace(FALLBACK, asic_temperature=65.0, thm_limit_throttle=90.0)
    assert chip_chaos(t) == chip_pressure(t) == pytest.approx(0.5)


def test_a_cool_idle_chip_that_heats_quickly_gets_chaotic():
    cool = dataclasses.replace(FALLBACK, asic_temperature=40.0, power=0.0)
    assert chip_pressure(cool) == 0.0
    heating = dataclasses.replace(cool, temp_rate=0.5 * RATE_FULL_TEMP_C_PER_S)
    assert chip_chaos(heating) == pytest.approx(0.5)


def test_rates_add_to_pressure_and_chaos_stops_at_one():
    hot = dataclasses.replace(FALLBACK, asic_temperature=65.0, thm_limit_throttle=90.0, power_rate=0.25 * RATE_FULL_POWER_W_PER_S)
    assert chip_chaos(hot) == pytest.approx(0.75)
    assert chip_chaos(dataclasses.replace(hot, temp_rate=RATE_FULL_TEMP_C_PER_S)) == 1.0


def test_the_full_rates_are_one_degree_a_second_and_twenty_five_watts_a_second():
    assert rate_activity(1.0, 0.0) == 1.0 and rate_activity(0.5, 0.0) == 0.5
    assert rate_activity(0.0, 25.0) == 1.0 and rate_activity(0.0, 5.0) == pytest.approx(0.2)


def test_a_limit_at_or_below_the_floor_is_full_pressure_for_a_chip_past_it_and_none_before_it():
    assert pressure(60, 0, 40, 125) == 1.0 and pressure(60, 0, 30, 125) == 1.0  # past a limit that sits on or under the 40 C floor
    assert pressure(35, 0, 40, 125) == 0.0 and pressure(35, 0, 30, 125) == 1.0  # not yet at the floor; 35 is past a 30 C limit
    assert pressure(40, 0, 40, 125) == 1.0  # exactly at the limit counts
