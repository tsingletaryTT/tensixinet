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


# --- a load cycle, so eviction can be heard without a model -------------------------------------------------------------------------------------

def test_a_load_cycle_takes_each_chip_from_rest_up_to_a_load_and_back_staggered_from_the_others():
    import numpy as np
    from tensixinet.telemetry.mock import mock_telemetry
    period = 90.0
    series = np.array([[c.power for c in mock_telemetry(3, 1, load_period=period, elapsed=e)] for e in np.arange(0.0, 90.0, 1.0)])
    assert series.min() == pytest.approx(14.0) and series.max() == pytest.approx(62.0)
    assert all(series[:, i].min() == pytest.approx(14.0) and series[:, i].max() == pytest.approx(62.0) for i in range(3))
    assert len({int(np.argmax(series[:, i])) for i in range(3)}) == 3                         # the chips are not loaded together
    on = (series > 38.0).mean(axis=0)
    assert np.all((on > 0.3) & (on < 0.7))                                                      # loaded for about half the cycle
    assert np.allclose(series[0], [c.power for c in mock_telemetry(3, 1, load_period=period, elapsed=period)])   # and it repeats


def test_the_load_cycle_moves_current_and_board_power_with_the_power_and_nothing_else_changes():
    from tensixinet.telemetry.mock import mock_telemetry
    base = mock_telemetry(2, 1)
    rest = mock_telemetry(2, 1, load_period=90.0, elapsed=0.0)
    for b, r in zip(base, rest):
        assert r.current == pytest.approx(r.power * 1.3) and r.gddr_temps == b.gddr_temps and r.tensix_mask == b.tensix_mask
    assert mock_telemetry(2, 1) == base and mock_telemetry(2, 1, load_period=None, elapsed=5.0) == base


def test_a_load_cycle_needs_a_positive_period():
    from tensixinet.telemetry.mock import mock_telemetry
    with pytest.raises(ValueError):
        mock_telemetry(2, 1, load_period=0.0, elapsed=1.0)
