# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Synthetic chips, for anyone without N Tenstorrent chips."""
import pytest

from tensixinet.telemetry import Telemetry, mock_telemetry


@pytest.mark.parametrize("n", [1, 4, 16])
def test_mock_telemetry_has_n_distinct_plausible_chips(n):
    chips = mock_telemetry(n, seed=0)
    assert len(chips) == n and all(isinstance(c, Telemetry) for c in chips)
    assert all(25 <= c.asic_temperature <= 100 and 400 <= c.aiclk <= 1350 for c in chips)
    assert len(chips[0].gddr_temps) == 8
    if n > 1:
        assert len({c.asic_temperature for c in chips}) > 1


def test_mock_telemetry_is_deterministic_per_seed():
    assert mock_telemetry(4, 1) == mock_telemetry(4, 1)
    assert mock_telemetry(4, 1) != mock_telemetry(4, 2)


def test_mock_rejects_non_positive_counts():
    with pytest.raises(ValueError):
        mock_telemetry(0)


# --- the wider readings in synthetic chips -------------------------------------------------

def test_mock_chips_each_have_their_own_harvesting_and_links():
    chips = mock_telemetry(6, seed=2)
    assert len({(c.tensix_mask, c.eth_mask) for c in chips}) == 6
    assert all(bin(c.tensix_mask).count("1") >= 10 for c in chips)  # a few columns harvested, not most


def test_mock_chips_are_healthy_unless_faults_are_asked_for():
    assert all(c.corr_errs == c.uncorr_errs == c.thermal_trips == 0 for c in mock_telemetry(8, seed=1))


def test_mock_faults_put_errors_on_some_chips_and_leave_others_healthy():
    chips = mock_telemetry(8, seed=1, faults=True)
    faulty = [c for c in chips if c.corr_errs or c.uncorr_errs or c.thermal_trips]
    assert 1 <= len(faulty) < 8
    assert chips == mock_telemetry(8, seed=1, faults=True)


def test_mock_reports_consistent_limits_and_board_power():
    for c in mock_telemetry(5, seed=3):
        assert c.board_power >= c.power and c.thm_limit_throttle > c.asic_temperature - 40 and c.tdp_limit > 0
        assert c.busy in (0, 1, 2)


def test_hundreds_of_mock_chips_still_each_have_their_own_masks():
    chips = mock_telemetry(400, seed=0)
    assert len({(c.tensix_mask, c.eth_mask) for c in chips}) == 400


def test_faults_always_hit_at_least_one_chip_and_never_all_of_several():
    for seed in range(30):
        chips = mock_telemetry(2, seed=seed, faults=True)
        faulty = [c.corr_errs or c.uncorr_errs or c.thermal_trips for c in chips]
        assert sum(map(bool, faulty)) == 1, seed
