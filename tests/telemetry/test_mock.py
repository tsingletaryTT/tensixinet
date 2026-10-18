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
