# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Pan curves: how a device's own internals move it through the field. Each is checked against the pieces it is built from."""
import dataclasses

import numpy as np
import pytest

from tensixinet.music.devices import SHOCK_DECAY_BEATS, SHOCK_RATE_BEATS, seconds_since, shimmer_parameters, strike_times, ttrollz_pulse_times
from tensixinet.music.mapping import aiclk_to_bpm
from tensixinet.music.patch import DEFAULT_PATCH
from tensixinet.music.spatial import depth_curve, pan_curve
from tensixinet.music.ttavdog import avdog_parameters, avdog_pings, undulation
from tensixinet.music.ttrolzer import SEGMENT_SEED_STRIDE
from tensixinet.telemetry import FALLBACK

SR = 8000


def tel(**kw):
    return dataclasses.replace(FALLBACK, **kw)


def curve(name, t=None, duration=20.0, seed=0, patch=DEFAULT_PATCH):
    return pan_curve(name, t or tel(), duration, SR, seed, patch)


def test_a_device_with_no_curve_of_its_own_gets_none_and_every_curve_is_as_long_as_the_voice_and_inside_the_field():
    assert curve("lead") is None and curve("ttrax") is None and curve("ttchoir") is None and curve("not a device") is None
    for name in ("ttavdog", "ultrasonic_shimmer", "gong", "ttquantus", "ttrollz", "ttsteam"):
        c = curve(name, tel(gddr_temps=(30.0, 50.0) * 4, ddr_status=(1, 1, 1, 1), eth_mask=0xFFFF, tensix_mask=0x37F7))
        assert c is not None and len(c) == int(20.0 * SR) and np.isfinite(c).all() and np.abs(c).max() <= 1.0, name


# --- ttavdog: white on the left half of the swing, red on the right ----------------------------------------------------------------

def test_the_avdogs_white_voice_is_left_and_its_red_answer_is_right_following_the_undulation():
    t = tel(aiclk=800.0, heartbeat=0)
    c = curve("ttavdog", t)
    pitch, shape, rate_hz, decay_s = avdog_parameters(t)
    u = undulation(seconds_since(avdog_pings(t, 20.0, DEFAULT_PATCH, 0), len(c), SR), rate_hz, decay_s)
    assert (c[u > 0.3] < 0).all() and (c[u < -0.3] > 0).all()           # white (the up half) left, red (the down half) right
    assert np.allclose(c[u == 0], 0.0) and np.abs(c).max() > 0.2
    assert np.allclose(c, -0.8 * u)


def test_the_avdogs_swing_follows_its_pings_including_the_steam_routes():
    t = tel(tensix_mask=0x3EDF, ddr_status=(0, 0, 0, 0), eth_mask=0, aiclk=800.0)
    plain, routed = curve("ttavdog", t), curve("ttavdog", t, patch=dataclasses.replace(DEFAULT_PATCH, steam_routes=("avdog",)))
    assert not np.allclose(plain, routed)


# --- shimmer: the two tones beat, and the sound pans at the beat -----------------------------------------------------------------------

def test_equal_memory_temperatures_give_a_steady_shimmer_and_a_spread_makes_it_pan_at_the_beat():
    assert np.allclose(curve("ultrasonic_shimmer", tel(gddr_temps=(40.0,) * 8)), 0.0)
    t = tel(gddr_temps=(30.0, 50.0) * 4)
    c = curve("ultrasonic_shimmer", t, duration=30.0)
    f1, f2, _ = shimmer_parameters(t, SR, DEFAULT_PATCH)
    spectrum = np.abs(np.fft.rfft(c - c.mean()))
    peak = np.fft.rfftfreq(len(c), 1 / SR)[np.argmax(spectrum)]
    assert peak == pytest.approx(f2 - f1, abs=0.1) and np.abs(c).max() == pytest.approx(0.8, abs=0.02)


# --- gong: each strike lands somewhere and settles ------------------------------------------------------------------------------------

def test_each_gong_strike_lands_at_its_own_place_and_drifts_back_to_the_middle():
    t = tel(aiclk=800.0, heartbeat=0, ddr_status=(0, 0, 0, 0), eth_mask=0)
    c = curve("gong", t, duration=40.0)
    strikes = strike_times(t, 40.0)
    assert len(strikes) >= 4
    landings = [c[int(s * SR) + 1] for s in strikes]
    assert len({round(x, 2) for x in landings}) == len(landings) and max(abs(x) for x in landings) <= 0.8 + 1e-9
    assert np.allclose(c[: int(strikes[0] * SR)], 0.0)                                       # nothing before the first strike
    first = int(strikes[0] * SR)
    assert abs(c[first + int(2.5 * 4 * 60.0 / aiclk_to_bpm(800.0) * SR) - 1]) < 0.2 * abs(landings[0]) + 1e-3   # settled by the end of the ring
    assert np.abs(c[first + 1]) > np.abs(c[first + int(0.5 * SR)]) or landings[0] == 0


def test_a_new_segment_lands_its_strikes_elsewhere_so_a_live_run_does_not_repeat_itself():
    t = tel(aiclk=800.0, heartbeat=0, ddr_status=(0, 0, 0, 0), eth_mask=0)
    a, b = curve("gong", t, 20.0, seed=0), curve("gong", t, 20.0, seed=SEGMENT_SEED_STRIDE)
    assert not np.allclose(a, b)


# --- ttquantus: pitch is position --------------------------------------------------------------------------------------------------------

def test_ttquantus_pans_with_its_pitch_low_notes_left_and_high_notes_right():
    from tensixinet.music.devices import ttquantus_steps
    t = tel(aiclk=800.0, axiclk=960.0)
    c = curve("ttquantus", t)
    steps = ttquantus_steps(t, 20.0, SR, 0, DEFAULT_PATCH)
    assert np.corrcoef(c, steps)[0, 1] > 0.99 and c.max() <= 0.7 + 1e-9 and c.min() >= -0.7 - 1e-9
    assert c[np.argmax(steps)] > 0 > c[np.argmin(steps)]


# --- ttrollz: a lane's blip comes from the lane's own place --------------------------------------------------------------------------------

def test_ttrollz_blips_hop_from_lane_to_lane_across_the_field_and_the_clock_sits_in_the_middle():
    t = tel(ddr_status=(1, 1, 1, 1), eth_mask=0xFFFF, aiclk=800.0)
    c = curve("ttrollz", t, duration=30.0)
    assert len({round(x, 2) for x in c}) >= 5 and c.min() < -0.5 and c.max() > 0.5
    clock = ttrollz_pulse_times(aiclk_to_bpm(800.0), 30.0)
    assert any(abs(c[int(p * SR) + 1]) < 1e-9 for p in clock)                                 # the clock's own blips are central
    assert curve("ttrollz", tel(ddr_status=(0, 0, 0, 0), eth_mask=0)) is None or np.allclose(curve("ttrollz", tel(ddr_status=(0, 0, 0, 0), eth_mask=0)), 0)


# --- ttsteam: the snare alternates sides ----------------------------------------------------------------------------------------------------

def test_the_snares_hits_alternate_left_and_right():
    from tensixinet.music.register import step_times
    t = tel(tensix_mask=0x37F7, aiclk=800.0)
    c = curve("ttsteam", t)
    hits = step_times(t, 20.0)
    sides = [np.sign(c[int(h * SR) + 1]) for h in hits[:8]]
    assert sides[0] != 0 and all(a == -b for a, b in zip(sides, sides[1:]))
    assert curve("ttsteam", tel(tensix_mask=0x3FFF)) is None or np.allclose(curve("ttsteam", tel(tensix_mask=0x3FFF)), 0)


# --- the drone comes forward as it ripples --------------------------------------------------------------------------------------------------

def test_the_drone_moves_toward_the_listener_while_it_ripples_and_nothing_else_has_a_depth_curve():
    t = tel(aiclk=800.0, heartbeat=0)
    d = depth_curve("ttdrone", t, 20.0, SR, 0, DEFAULT_PATCH)
    assert d is not None and len(d) == int(20.0 * SR) and d.max() <= 1e-12 and d.min() >= -0.3 - 1e-9 and d.min() < -0.05
    assert np.allclose(d[: int(strike_times(t, 20.0)[0] * SR)], 0.0) if len(strike_times(t, 20.0)) else True
    for name in ("lead", "gong", "ttavdog", "ttrollz", "ultrasonic_shimmer"):
        assert depth_curve(name, t, 20.0, SR, 0, DEFAULT_PATCH) is None


def test_ttquantus_puts_the_middle_of_its_two_octaves_in_the_middle_of_the_field():
    from tensixinet.music.devices import ttquantus_steps
    t = tel(aiclk=800.0, axiclk=960.0)
    c = curve("ttquantus", t, duration=40.0)
    steps = ttquantus_steps(t, 40.0, SR, 0, DEFAULT_PATCH)
    assert (steps == 12).any() and np.allclose(c[steps == 12], 0.0)      # the octave is the centre
