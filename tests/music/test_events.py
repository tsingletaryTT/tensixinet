# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Event sounds: memory errors and thermal trips, silent on a healthy chip."""
import dataclasses

import numpy as np
import pytest

from tensixinet.music.events import EVENT_INPUTS, error_events
from tensixinet.telemetry import FALLBACK

SR = 16000


def play(**kw):
    return error_events(dataclasses.replace(FALLBACK, **kw), 6.0, SR, seed=1)


def _rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


def test_a_healthy_chip_is_silent():
    audio = play()
    assert audio.shape == (6 * SR,) and audio.dtype == np.float32 and not audio.any()


@pytest.mark.parametrize("field,value", [("corr_errs", 3), ("uncorr_errs", 1), ("thermal_trips", 1)])
def test_each_kind_of_event_makes_sound_and_stays_bounded(field, value):
    audio = play(**{field: value})
    assert _rms(audio) > 0 and np.abs(audio).max() <= 1.0 and np.isfinite(audio).all()


def test_the_three_kinds_sound_different():
    a, b, c = play(corr_errs=3), play(uncorr_errs=3), play(thermal_trips=3)
    assert not np.allclose(a, b) and not np.allclose(b, c) and not np.allclose(a, c)


def test_correctable_errors_are_ticks_one_per_error_up_to_a_limit():
    def ticks(n):
        win = SR // 100  # smooth over 10 ms so one tick is one burst, not one per wave
        env = np.convolve(np.abs(play(corr_errs=n)), np.ones(win) / win, mode="same") > 0.01
        return int(np.sum(np.diff(env.astype(int)) == 1))  # count bursts that start
    assert ticks(2) < ticks(6)
    assert ticks(500) == ticks(40)  # capped, so a flood of errors cannot fill the run


def test_more_errors_are_louder_or_denser_up_to_the_cap():
    assert _rms(play(uncorr_errs=3)) > _rms(play(uncorr_errs=1))
    assert _rms(play(thermal_trips=3)) > _rms(play(thermal_trips=1))


def test_a_thermal_trip_sweeps_downward():
    audio = play(thermal_trips=1)
    start = np.flatnonzero(np.abs(audio) > 0.01)[0]
    seg = audio[start:start + SR // 2]
    first, last = seg[: len(seg) // 4], seg[-len(seg) // 4:]
    def hz(x):
        return np.sum(np.diff(np.signbit(x)) != 0) * SR / (2 * len(x))
    assert hz(first) > 1.5 * hz(last)


def test_events_are_deterministic_per_seed_and_vary_across_seeds():
    t = dataclasses.replace(FALLBACK, corr_errs=4, uncorr_errs=1)
    assert np.array_equal(error_events(t, 5.0, SR, 3), error_events(t, 5.0, SR, 3))
    assert not np.array_equal(error_events(t, 5.0, SR, 3), error_events(t, 5.0, SR, 4))


def test_events_read_only_the_error_counters():
    assert EVENT_INPUTS == frozenset({"corr_errs", "uncorr_errs", "thermal_trips"})
    quiet = error_events(FALLBACK, 2.0, SR, 0)
    assert np.array_equal(quiet, error_events(dataclasses.replace(FALLBACK, asic_temperature=99.0, power=190.0), 2.0, SR, 0))


def test_events_that_last_longer_than_the_run_are_cut_to_fit():
    assert error_events(dataclasses.replace(FALLBACK, thermal_trips=3, uncorr_errs=3), 0.4, SR, 0).shape == (int(0.4 * SR),)


def test_every_kind_at_its_cap_at_once_still_stays_below_full_scale():
    audio = error_events(dataclasses.replace(FALLBACK, corr_errs=500, uncorr_errs=9, thermal_trips=9), 3.0, SR, 5)
    assert np.abs(audio).max() < 1.0 and _rms(audio) > 0
