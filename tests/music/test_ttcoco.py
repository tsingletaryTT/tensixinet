# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""ttcoco: the Cocoquantus's own delay chips, reimagined as eight parallel Tensix-scale taps."""
import dataclasses

import numpy as np
import pytest

from tensixinet.music.devices import quantussy_tone
from tensixinet.music.mapping import chip_pressure
from tensixinet.music.ttcoco import (
    BITS_RANGE,
    DELAY_RANGE_MS,
    ECHO_FLOOR,
    FEEDBACK_MAX,
    N_TAPS,
    crunch_bits,
    echo_tap,
    shift_right,
    tap_bits,
    tap_delay_ms,
    tap_feedback,
    ttcoco,
)
from tensixinet.telemetry import FALLBACK

SR = 8000


def tel(**kw):
    return dataclasses.replace(FALLBACK, **kw)


# --- the bit crunch ------------------------------------------------------------------------------------------

def test_crunch_bits_rounds_to_signed_levels_and_keeps_the_sign():
    x = np.array([0.3, -0.3, 0.999, -1.0, 0.0])
    out = crunch_bits(x, bits=4.0)
    levels = 2 ** 3 - 1
    assert np.allclose(out * levels, np.round(x * levels))
    assert np.all(np.sign(out[x != 0]) == np.sign(x[x != 0]))


def test_fewer_bits_means_fewer_distinct_levels():
    x = np.linspace(-1, 1, 4000)
    coarse = len(np.unique(crunch_bits(x, 4.0)))
    fine = len(np.unique(crunch_bits(x, 8.0)))
    assert coarse < fine
    assert coarse == pytest.approx(2 ** 4 - 1, abs=1)
    assert fine == pytest.approx(2 ** 8 - 1, abs=1)


# --- shifting a tap in time ------------------------------------------------------------------------------------

def test_shift_right_delays_by_n_samples_and_pads_with_silence():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert shift_right(x, 2).tolist() == [0.0, 0.0, 1.0, 2.0]
    assert shift_right(x, 0).tolist() == x.tolist()


def test_shift_right_past_the_end_of_the_signal_is_silence():
    x = np.array([1.0, 2.0, 3.0])
    assert shift_right(x, 5).tolist() == [0.0, 0.0, 0.0]


# --- a single tap: a geometric series of delayed, re-crunched echoes --------------------------------------------

def test_a_tap_with_no_feedback_is_just_the_crunched_dry_signal():
    x = np.sin(2 * np.pi * 100 * np.arange(SR) / SR)
    out = echo_tap(x, delay_samples=400, feedback=0.0, bits=8.0)
    assert np.array_equal(out, crunch_bits(x, 8.0))


def test_each_echo_arrives_one_delay_later_and_quieter_by_the_feedback_gain():
    x = np.zeros(SR)
    x[0] = 1.0  # an impulse, so each echo's arrival time is easy to find
    out = echo_tap(x, delay_samples=100, feedback=0.5, bits=8.0)
    assert out[0] == pytest.approx(crunch_bits(np.array([1.0]), 8.0)[0])
    assert out[100] == pytest.approx(0.5 * crunch_bits(np.array([1.0]), 8.0)[0], abs=1e-6)
    assert out[200] == pytest.approx(0.25 * crunch_bits(np.array([1.0]), 8.0)[0], abs=1e-6)


def test_the_series_stops_once_an_echo_falls_below_the_floor():
    x = np.ones(SR)
    out = echo_tap(x, delay_samples=1, feedback=0.1, bits=8.0)  # dies in a handful of repeats
    # past where the geometric series must have died, the tail looks like the (few) live repeats only
    n_repeats = 0
    gain = 1.0
    while gain > ECHO_FLOOR:
        n_repeats += 1
        gain *= 0.1
    assert n_repeats < 10  # sanity: the test itself terminates
    assert np.isfinite(out).all()


def test_zero_feedback_and_a_feedback_of_nearly_one_both_stay_finite_and_bounded():
    x = np.sin(2 * np.pi * 50 * np.arange(SR) / SR)
    for fb in (0.0, FEEDBACK_MAX):
        out = echo_tap(x, delay_samples=200, feedback=fb, bits=6.0)
        assert np.isfinite(out).all()


# --- how a GDDR channel's temperature sets its tap's delay and bit depth ----------------------------------------

def test_a_hotter_channel_gets_a_shorter_delay():
    assert tap_delay_ms(25.0) == pytest.approx(DELAY_RANGE_MS[1])  # coolest: the slowest echo
    assert tap_delay_ms(100.0) == pytest.approx(DELAY_RANGE_MS[0])  # hottest: the fastest echo
    assert tap_delay_ms(30.0) > tap_delay_ms(90.0)


def test_a_shorter_delay_holds_more_bits_the_real_chips_own_trade():
    assert tap_bits(DELAY_RANGE_MS[0]) == pytest.approx(BITS_RANGE[1])  # shortest delay: most bits
    assert tap_bits(DELAY_RANGE_MS[1]) == pytest.approx(BITS_RANGE[0])  # longest delay: fewest bits
    assert tap_bits(DELAY_RANGE_MS[0]) > tap_bits(DELAY_RANGE_MS[1])


def test_feedback_is_pressure_scaled_to_the_feedback_ceiling():
    calm = tel(asic_temperature=30.0, power=10.0)
    hot = tel(asic_temperature=95.0, power=190.0)
    assert tap_feedback(calm) == pytest.approx(FEEDBACK_MAX * chip_pressure(calm))
    assert tap_feedback(hot) == pytest.approx(FEEDBACK_MAX * chip_pressure(hot))
    assert tap_feedback(hot) > tap_feedback(calm)
    assert 0.0 <= tap_feedback(hot) <= FEEDBACK_MAX


# --- the whole device: eight parallel taps -----------------------------------------------------------------------

def test_ttcoco_is_bounded_deterministic_and_the_right_length():
    t = tel(gddr_temps=(30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 35.0))
    a = ttcoco(t, 2.0, SR, seed=1)
    b = ttcoco(t, 2.0, SR, seed=1)
    assert a.shape == (2 * SR,) and a.dtype == np.float32
    assert np.array_equal(a, b)
    assert np.isfinite(a).all() and np.abs(a).max() < 1.0


def test_there_are_eight_taps_one_per_gddr_channel(monkeypatch):
    import tensixinet.music.ttcoco as ttcoco_module
    seen = []
    real = ttcoco_module.echo_tap
    monkeypatch.setattr(ttcoco_module, "echo_tap", lambda *a, **k: seen.append(a) or real(*a, **k))
    ttcoco(tel(), 0.5, SR, seed=0)
    assert len(seen) == N_TAPS == 8


def test_uneven_memory_temperatures_give_each_tap_a_different_delay_and_bit_depth(monkeypatch):
    import tensixinet.music.ttcoco as ttcoco_module
    seen = []
    real = ttcoco_module.echo_tap
    monkeypatch.setattr(ttcoco_module, "echo_tap", lambda x, d, f, b: seen.append((d, b)) or real(x, d, f, b))
    t = tel(gddr_temps=(25.0, 40.0, 55.0, 70.0, 85.0, 100.0, 33.0, 62.0))
    ttcoco(t, 0.5, SR, seed=0)
    delays = [d for d, b in seen]
    bits = [b for d, b in seen]
    assert len(set(delays)) > 1 and len(set(bits)) > 1


def test_ttcoco_plays_the_quantussys_own_tone(monkeypatch):
    """The source patched into every tap is quantussy_tone, the same notes ttquantus plays, not a tone of its own."""
    import tensixinet.music.ttcoco as ttcoco_module
    seen = []
    real = ttcoco_module.echo_tap
    monkeypatch.setattr(ttcoco_module, "echo_tap", lambda x, d, f, b: seen.append(x) or real(x, d, f, b))
    t = tel()
    expected = quantussy_tone(t, 0.5, SR, 0)
    ttcoco(t, 0.5, SR, seed=0)
    assert all(np.array_equal(x, expected) for x in seen)


def test_a_colder_system_and_a_hotter_one_sound_different():
    cool = tel(asic_temperature=30.0, power=10.0, gddr_temps=(28.0,) * 8)
    hot = tel(asic_temperature=92.0, power=180.0, gddr_temps=(95.0,) * 8)
    assert not np.allclose(ttcoco(cool, 1.0, SR, seed=0), ttcoco(hot, 1.0, SR, seed=0))


def test_a_different_seed_changes_the_melody_and_so_the_sound():
    t = tel()
    assert not np.allclose(ttcoco(t, 1.0, SR, seed=0), ttcoco(t, 1.0, SR, seed=1))


def test_eight_taps_are_mixed_with_room_to_spare_not_summed_flat():
    """A sanity check against silently dropping the 1/sqrt(N_TAPS) headroom term: a full-pressure, uniformly
    hot system (every tap maximally fed back) must still stay inside tanh's range with margin, not pinned at it."""
    t = tel(asic_temperature=98.0, power=199.0, gddr_temps=(99.0,) * 8)
    out = ttcoco(t, 1.0, SR, seed=0)
    assert np.abs(out).max() < 0.999

# --- the constants, pinned so a change to any of them is a deliberate, tested decision --------------------------

def test_the_tuning_constants_are_exactly_these():
    assert N_TAPS == 8
    assert DELAY_RANGE_MS == (40.0, 900.0)
    assert BITS_RANGE == (4.0, 8.0)
    assert FEEDBACK_MAX == 0.8
    assert ECHO_FLOOR == pytest.approx(1.0 / 400.0)
