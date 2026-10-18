# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The governor for load kernels: a hard wall-clock cap and a live pressure ceiling that aborts a burst early.

Pure host-side logic, no device and no `ttnn`, so it is tested here like any other module."""
import dataclasses

import pytest

from tensixinet.kernels.governor import (
    ABORT_PRESSURE,
    MAX_BURST_S,
    may_start,
    should_stop,
)
from tensixinet.telemetry import FALLBACK


def chip(**kw):
    return dataclasses.replace(FALLBACK, **kw)


COOL = chip(asic_temperature=40.0, power=20.0, thm_limit_throttle=90.0, tdp_limit=100.0)
HOT = chip(asic_temperature=89.0, power=95.0, thm_limit_throttle=90.0, tdp_limit=100.0)  # already near both limits


def test_a_cool_chip_may_start_a_burst():
    assert may_start(COOL) is True


def test_a_chip_already_near_its_limit_may_not_start_a_burst():
    assert may_start(HOT) is False


def test_the_start_check_uses_the_abort_pressure_as_its_line_not_some_other_number():
    from tensixinet.music.mapping import pressure
    # a chip exactly at the abort pressure may not start; a hair below it may
    at_limit = chip(asic_temperature=40.0 + ABORT_PRESSURE * (90.0 - 40.0), power=20.0, thm_limit_throttle=90.0, tdp_limit=100.0)
    just_under = chip(asic_temperature=40.0 + (ABORT_PRESSURE - 0.01) * (90.0 - 40.0), power=20.0, thm_limit_throttle=90.0, tdp_limit=100.0)
    assert pressure(at_limit.asic_temperature, at_limit.power, at_limit.thm_limit_throttle, at_limit.tdp_limit) == pytest.approx(ABORT_PRESSURE)
    assert may_start(at_limit) is False
    assert may_start(just_under) is True


def test_a_burst_stops_at_the_hard_wall_clock_cap_even_on_a_cool_chip():
    stop, reason = should_stop(MAX_BURST_S, COOL)
    assert stop is True and "cap" in reason
    stop, reason = should_stop(MAX_BURST_S - 0.01, COOL)
    assert stop is False


def test_a_burst_stops_early_if_the_chip_crosses_the_abort_pressure():
    stop, reason = should_stop(0.1, HOT)
    assert stop is True and "pressure" in reason


def test_a_burst_on_a_cool_chip_within_the_cap_does_not_stop():
    stop, reason = should_stop(0.1, COOL)
    assert stop is False and reason == ""


def test_the_cap_is_at_most_three_seconds_and_the_abort_pressure_is_below_full():
    assert 0 < MAX_BURST_S <= 3.0
    assert 0.0 < ABORT_PRESSURE < 1.0


def test_the_cap_is_exactly_two_seconds_and_the_ceiling_exactly_085():
    # pinned so a change to either constant is a deliberate, tested decision
    assert MAX_BURST_S == 2.0
    assert ABORT_PRESSURE == 0.85


def test_the_cap_and_ceiling_are_reached_at_the_boundary_itself_not_just_past_it():
    stop, reason = should_stop(MAX_BURST_S, COOL)
    assert stop is True and "cap" in reason
    at_ceiling = chip(asic_temperature=40.0 + ABORT_PRESSURE * (90.0 - 40.0), power=20.0, thm_limit_throttle=90.0, tdp_limit=100.0)
    stop, reason = should_stop(0.1, at_ceiling)
    assert stop is True and "pressure" in reason
