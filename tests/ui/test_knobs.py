# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The Play tab's knobs: the slider values and events that stand in for chip readings."""
import dataclasses
import math

import pytest

from tensixinet.music.live import Tracker
from tensixinet.ui.knobs import EVENTS, KNOB_RANGES, LOAD_BURST_S, LOAD_BURST_W, MAX_CHIPS, Knobs, to_telemetry

# Written out here, not read from the module, so changing a range in the module turns this red.
RANGES = {
    "temperature": (20.0, 105.0, 50.0),
    "power": (5.0, 150.0, 30.0),
    "aiclk": (200.0, 1350.0, 800.0),
    "gddr_spread": (0.0, 30.0, 8.0),
    "voltage": (0.65, 0.95, 0.75),
}
# What each knob may move. Current and board power follow power, as they do in mock_telemetry.
MOVES = {
    "temperature": {"asic_temperature"},
    "power": {"power", "current", "board_power"},
    "aiclk": {"aiclk"},
    "gddr_spread": {"gddr_temps"},
    "voltage": {"voltage"},
}


def _read(knobs, now=0.0):
    return to_telemetry(knobs, Tracker(), now)


def _differs(a, b):
    return {f.name for f in dataclasses.fields(a) if getattr(a, f.name) != getattr(b, f.name)}


def test_the_ranges_are_the_ones_the_plan_chose():
    assert KNOB_RANGES == RANGES
    assert (LOAD_BURST_W, LOAD_BURST_S, MAX_CHIPS) == (70.0, 4.0, 4)
    assert EVENTS == ("memory_error", "thermal_trip", "load_burst")


@pytest.mark.parametrize("name", sorted(RANGES))
def test_each_knob_moves_only_the_fields_it_owns_and_only_on_its_chip(name):
    low, high, default = RANGES[name]
    base = _read(Knobs(2))
    knobs = Knobs(2)
    knobs.set(0, name, high if default < high else low)
    moved = _read(knobs)
    assert _differs(moved[0], base[0]) == MOVES[name]
    assert moved[1] == base[1]


@pytest.mark.parametrize("name,field", [("temperature", "asic_temperature"), ("power", "power"), ("aiclk", "aiclk"), ("voltage", "voltage")])
def test_a_scalar_knob_lands_on_its_field_exactly(name, field):
    knobs = Knobs(1)
    knobs.set(0, name, RANGES[name][0])
    assert getattr(_read(knobs)[0], field) == RANGES[name][0]


@pytest.mark.parametrize("spread", [0.0, 5.0, 12.0, 30.0])
def test_gddr_spread_is_max_minus_min_of_the_eight_channels(spread):
    knobs = Knobs(1)
    knobs.set(0, "gddr_spread", spread)
    temps = _read(knobs)[0].gddr_temps
    assert len(temps) == 8 and max(temps) - min(temps) == pytest.approx(spread)


def test_chips_keep_distinct_identities():
    t = _read(Knobs(4))
    assert len({(c.tensix_mask, c.eth_mask) for c in t}) == 4


def test_values_are_clamped_and_nan_is_ignored():
    knobs = Knobs(1)
    knobs.set(0, "power", 1e9)
    assert knobs.get(0, "power") == 150.0
    knobs.set(0, "power", -5)
    assert knobs.get(0, "power") == 5.0
    knobs.set(0, "power", 77.0)
    knobs.set(0, "power", float("nan"))
    assert knobs.get(0, "power") == 77.0


def test_an_unknown_knob_or_chip_is_ignored_and_does_not_crash():
    knobs = Knobs(2)
    knobs.set(0, "no_such_knob", 1.0)
    knobs.set(9, "power", 1.0)
    knobs.trigger(9, "memory_error")
    knobs.trigger(0, "no_such_event")
    assert knobs.take_events() == []
    assert knobs.get(9, "power") == 30.0  # a chip that is not there reads as the default


def test_the_chip_count_is_one_to_four():
    for bad in (0, 5):
        with pytest.raises(ValueError):
            Knobs(bad)
    with pytest.raises(ValueError):
        Knobs(2).resize(5)


def test_an_event_is_taken_once_and_in_order():
    knobs = Knobs(2)
    knobs.trigger(1, "thermal_trip")
    knobs.trigger(0, "memory_error")
    assert knobs.take_events() == [(1, "thermal_trip"), (0, "memory_error")]
    assert knobs.take_events() == []


def test_a_memory_error_shows_in_one_block_only():
    knobs, tracker = Knobs(2), Tracker()
    to_telemetry(knobs, tracker, 0.0)
    knobs.trigger(1, "memory_error")
    first = to_telemetry(knobs, tracker, 1.0)
    second = to_telemetry(knobs, tracker, 2.0)
    assert (first[1].corr_errs, first[0].corr_errs) == (1, 0)
    assert second[1].corr_errs == 0


def test_a_thermal_trip_shows_in_one_block_only_and_two_trips_count_twice():
    knobs, tracker = Knobs(1), Tracker()
    to_telemetry(knobs, tracker, 0.0)
    knobs.trigger(0, "thermal_trip")
    knobs.trigger(0, "thermal_trip")
    assert to_telemetry(knobs, tracker, 1.0)[0].thermal_trips == 2
    assert to_telemetry(knobs, tracker, 2.0)[0].thermal_trips == 0


def test_a_load_burst_adds_power_for_four_seconds_then_stops():
    knobs, tracker = Knobs(1), Tracker()
    base = RANGES["power"][2]
    knobs.trigger(0, "load_burst")
    assert to_telemetry(knobs, tracker, 0.0)[0].power == pytest.approx(base + 70.0)
    assert to_telemetry(knobs, tracker, 3.9)[0].power == pytest.approx(base + 70.0)
    assert to_telemetry(knobs, tracker, 4.1)[0].power == pytest.approx(base)


def test_a_second_load_burst_restarts_the_four_seconds():
    knobs, tracker = Knobs(1), Tracker()
    knobs.trigger(0, "load_burst")
    to_telemetry(knobs, tracker, 0.0)
    knobs.trigger(0, "load_burst")
    to_telemetry(knobs, tracker, 3.0)
    assert to_telemetry(knobs, tracker, 6.9)[0].power == pytest.approx(30.0 + 70.0)


def test_a_burst_follows_the_knob_it_sits_on():
    knobs = Knobs(1)
    knobs.set(0, "power", 100.0)
    knobs.trigger(0, "load_burst")
    t = to_telemetry(knobs, Tracker(), 0.0)[0]
    assert t.power == pytest.approx(170.0) and t.current == pytest.approx(170.0 * 1.3)


def test_resize_keeps_the_chips_that_remain_and_their_identity():
    knobs = Knobs(3)
    knobs.set(1, "temperature", 90.0)
    masks = [(c.tensix_mask, c.eth_mask) for c in _read(knobs)]
    knobs.resize(2)
    assert knobs.get(1, "temperature") == 90.0 and len(knobs.values()) == 2
    knobs.resize(3)
    assert knobs.get(2, "temperature") == 50.0
    assert [(c.tensix_mask, c.eth_mask) for c in _read(knobs)] == masks


def test_a_chip_hidden_by_resize_gets_its_values_back_when_it_returns():
    knobs = Knobs(4)
    knobs.set(3, "power", 120.0)
    knobs.resize(2)
    assert len(knobs.values()) == 2 and len(_read(knobs)) == 2
    assert knobs.get(3, "power") == 120.0  # the page's slider for it still shows 120
    knobs.set(3, "power", 130.0)  # and moving it while hidden is remembered too
    knobs.resize(4)
    assert knobs.get(3, "power") == 130.0 and _read(knobs)[3].power == 130.0


def test_resize_drops_the_events_of_chips_that_go():
    knobs = Knobs(3)
    knobs.trigger(2, "memory_error")
    knobs.resize(2)
    assert knobs.take_events() == []


def test_moving_a_slider_gives_a_rate():
    knobs, tracker = Knobs(1), Tracker()
    knobs.set(0, "temperature", 50.0)
    to_telemetry(knobs, tracker, 0.0)
    knobs.set(0, "temperature", 60.0)
    assert to_telemetry(knobs, tracker, 2.5)[0].temp_rate == pytest.approx(4.0)


def test_every_field_is_finite_with_every_knob_at_each_bound_and_a_burst_on():
    for which in (0, 1):
        knobs = Knobs(4)
        for chip in range(4):
            for name, bounds in RANGES.items():
                knobs.set(chip, name, bounds[which])
            for event in EVENTS:
                knobs.trigger(chip, event)
        for t in _read(knobs):
            for f in dataclasses.fields(t):
                v = getattr(t, f.name)
                for x in (v if isinstance(v, tuple) else (v,)):
                    assert x is None or math.isfinite(x), (f.name, x)


def test_a_burst_has_ended_at_exactly_four_seconds():
    knobs, tracker = Knobs(1), Tracker()
    knobs.trigger(0, "load_burst")
    to_telemetry(knobs, tracker, 0.0)
    assert to_telemetry(knobs, tracker, 4.0)[0].power == pytest.approx(30.0)


def test_the_knobs_never_report_an_uncorrectable_error():
    knobs = Knobs(4)
    for chip in range(4):
        knobs.trigger(chip, "memory_error")
    assert [t.uncorr_errs for t in _read(knobs)] == [0, 0, 0, 0]


def test_an_event_for_a_chip_that_is_not_being_played_is_dropped():
    knobs = Knobs(2)
    knobs.trigger(3, "memory_error")
    assert knobs.take_events() == []


def test_knobs_hold_a_syllable_pick_and_ignore_unknown_values():
    k = Knobs(2)
    assert k.syllable() == ("", "")
    k.set_syllable("t", "i")
    assert k.syllable() == ("t", "i")
    k.set_syllable("x", "a")  # an unknown onset is ignored, and so is its vowel's change with it
    assert k.syllable() == ("t", "i")
    k.set_syllable("s", "y")
    assert k.syllable() == ("t", "i")
    k.set_syllable("", "")  # empty is a value: back to the readings
    assert k.syllable() == ("", "")


def test_the_pick_lists_are_any_plus_every_consonant_and_vowel():
    from tensixinet.music.consonants import CONSONANTS
    from tensixinet.music.ttchoir import VOWEL_ORDER
    from tensixinet.ui.knobs import CHOIR_ONSETS, CHOIR_VOWELS
    assert CHOIR_ONSETS == ("",) + tuple(CONSONANTS) and CHOIR_VOWELS == ("", "u", "o", "a", "e", "i") == ("",) + tuple(VOWEL_ORDER)
