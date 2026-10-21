# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The shift register and its noise patched into other devices: a snare-driven clock, AVDog pings, accents and held notes.

Each route is off in the default patch, so the sound from before routes existed does not move. Every test here
checks the wiring (what changes when a route is on, and what does not), not just the arithmetic under it."""
import dataclasses

import numpy as np
import pytest

from tensixinet.music.devices import DEVICE_INPUTS, ROUTE_INPUTS, gong, ttdrone, ttrollz, strike_times, ultrasonic_shimmer
from tensixinet.music.mapping import aiclk_to_bpm
from tensixinet.music.patch import DEFAULT_PATCH, STEAM_ROUTES
from tensixinet.music.register import STEPS, held_semitones, step_times
from tensixinet.music.ttavdog import avdog_voices, ttavdog
from tensixinet.telemetry import FALLBACK

SR = 8000


def chip(**kw):
    """Fires on steps 5 and 8. Step 8 is on the ttrollz grid (every fourth step), so clock pulses land on it, and with
    heartbeat 9 the first gong window opens where a register hit comes before the next clock pulse. A mask that fires
    only on odd steps never lines up with the clock, and most masks give a hit-only strike rarely or never. No DDR lane
    or ethernet link is up, so the chip has no rolls and its gong follows the clock."""
    return dataclasses.replace(FALLBACK, tensix_mask=0x3EDF, heartbeat=9, aiclk=800.0, ddr_status=(0, 0, 0, 0), eth_mask=0x0000, **kw)


def routed(*routes, **kw):
    return dataclasses.replace(DEFAULT_PATCH, steam_routes=tuple(routes), **kw)


# --- strikes: the register's hits join the clock the gong waits on --------------------------------------------------

def test_a_register_hit_can_set_off_the_gong_where_the_rollz_clock_has_no_pulse():
    t = chip()
    hits, clock = step_times(t, 60.0), ttrollz_clock(t, 60.0)
    plain, joined = strike_times(t, 60.0), strike_times(t, 60.0, routed("strikes"))
    assert any(np.isclose(hits, s).any() and not np.isclose(clock, s).any() for s in joined)
    assert not any(np.isclose(hits, s).any() and not np.isclose(clock, s).any() for s in plain)


def test_with_the_strikes_route_every_strike_is_a_clock_pulse_or_a_register_hit_and_the_window_still_holds():
    t = chip()
    allowed = np.concatenate([ttrollz_clock(t, 60.0), step_times(t, 60.0)])
    joined = strike_times(t, 60.0, routed("strikes"))
    assert len(joined) >= 2 and all(np.isclose(allowed, s).any() for s in joined)
    assert np.diff(joined).min() >= 2 * 4 * 60.0 / aiclk_to_bpm(800.0) - 1e-9


def test_other_routes_leave_the_strike_times_alone():
    t = chip()
    assert np.array_equal(strike_times(t, 60.0, routed("avdog", "accent", "pitch")), strike_times(t, 60.0))


def test_the_shared_clock_carries_the_strikes_route_to_the_gong_the_shimmer_and_the_drone():
    t, patch = chip(), routed("strikes")
    for device in (gong, ultrasonic_shimmer, ttdrone):
        assert not np.array_equal(device(t, 15.0, SR, 0, patch), device(t, 15.0, SR, 0)), device.__name__


# --- avdog: the register's hits ping the filter themselves -----------------------------------------------------------

def _energy_after(x, start, length):
    return float(np.sum(x[int(start * SR): int((start + length) * SR)] ** 2))


def test_the_avdog_route_makes_the_dogtone_sound_after_a_register_hit_with_no_strike_near_it():
    t = chip()
    strikes = strike_times(t, 30.0)
    hits = step_times(t, 30.0)
    beat = 60.0 / aiclk_to_bpm(t.aiclk)  # the ripple dies to a third in 2 beats, so 5 beats after a strike it is gone
    far = [h for h in hits if not ((strikes > h - 5 * beat) & (strikes < h + 0.5)).any() and h < 25.0]
    assert far, "the test needs a register hit well away from every strike"
    h = far[0]
    plain = np.concatenate(avdog_voices(t, 30.0, SR)).reshape(2, -1).sum(axis=0)
    routed_out = np.concatenate(avdog_voices(t, 30.0, SR, routed("avdog"))).reshape(2, -1).sum(axis=0)
    assert _energy_after(routed_out, h, 0.5) > 20 * max(_energy_after(plain, h, 0.5), 1e-9)


def test_the_avdog_route_changes_nothing_before_the_first_hit_or_strike():
    t = chip()
    first = min(step_times(t, 30.0)[0], strike_times(t, 30.0)[0])
    a, b = avdog_voices(t, 30.0, SR)[0], avdog_voices(t, 30.0, SR, routed("avdog"))[0]
    assert np.array_equal(a[: int(first * SR)], b[: int(first * SR)])


# --- pitch: held notes steer the dogtone ------------------------------------------------------------------------------

def test_the_pitch_route_with_a_one_note_scale_is_the_plain_dogtone_and_with_a_real_scale_it_is_not():
    t = chip()
    plain = ttavdog(t, 20.0, SR, 0)
    one_note = routed("pitch", scale=(0,), scale_name="x", mode=0)
    assert np.allclose(ttavdog(t, 20.0, SR, 0, one_note), plain, atol=1e-6)
    assert not np.allclose(ttavdog(t, 20.0, SR, 0, routed("pitch")), plain)


def test_the_pitch_route_runs_the_dogtone_at_the_chips_pitch_moved_by_the_held_note_of_each_step():
    from tensixinet.music.devices import seconds_since
    from tensixinet.music.ttavdog import avdog_parameters, skewed_triangle, undulation
    t, patch = chip(), routed("pitch")
    n = int(20.0 * SR)
    semis = held_semitones(t, n, SR, patch)
    assert len(np.unique(semis)) >= 3
    pitch, shape, rate_hz, decay_s = avdog_parameters(t)
    f = pitch * 2.0 ** (semis / 12.0)
    phase = (np.cumsum(f) - f) / SR  # each sample's phase is the sum of the frequencies before it
    u = undulation(seconds_since(strike_times(t, 20.0), n, SR), rate_hz, decay_s)
    white, red = avdog_voices(t, 20.0, SR, patch)
    assert np.allclose(white, skewed_triangle(phase, shape) * np.maximum(u, 0.0), atol=1e-6)
    assert np.allclose(red, skewed_triangle(phase, -shape) * np.maximum(-u, 0.0), atol=1e-6)


# --- accent: the register accents the clock pulses it lands on ---------------------------------------------------

def ttrollz_clock(t, duration):
    from tensixinet.music.devices import ttrollz_pulse_times
    return ttrollz_pulse_times(aiclk_to_bpm(t.aiclk), duration)


def test_the_accent_route_adds_to_a_clock_pulse_that_lands_on_a_register_hit_and_to_nothing_else():
    t = chip()
    clock, hits = ttrollz_clock(t, 30.0), step_times(t, 30.0)
    landed = [c for c in clock if np.abs(hits - c).min() < 0.005]
    assert landed, "the test needs a clock pulse that lands on a register hit"
    plain, accented = ttrollz(t, 30.0, SR, 0), ttrollz(t, 30.0, SR, 0, routed("accent"))
    first = landed[0]
    assert np.array_equal(plain[: int(first * SR)], accented[: int(first * SR)])           # nothing before the first accent
    assert np.abs(accented[int(first * SR): int((first + 0.05) * SR)]).max() > np.abs(plain[int(first * SR): int((first + 0.05) * SR)]).max()


def test_the_accent_route_does_nothing_for_a_chip_whose_register_never_fires():
    t = dataclasses.replace(chip(), tensix_mask=0x3FFF)
    assert np.array_equal(ttrollz(t, 20.0, SR, 0, routed("accent")), ttrollz(t, 20.0, SR, 0))


# --- the claim about what each route reads ----------------------------------------------------------------------------

def test_the_routes_name_the_fields_they_make_a_device_read():
    assert set(ROUTE_INPUTS) == set(STEAM_ROUTES)
    assert "tensix_mask" in ROUTE_INPUTS["accent"]["ttrollz"] and "aiclk" in ROUTE_INPUTS["accent"]["ttrollz"]


@pytest.mark.parametrize("route", STEAM_ROUTES)
def test_a_route_makes_exactly_the_devices_it_names_read_the_mask_and_clock(route):
    from tensixinet.music.chorus import DEVICES
    base = chip()
    changed = dataclasses.replace(base, tensix_mask=0x1235)
    patch = routed(route)
    for device in DEVICES:
        name = device.__name__
        moved_plain = not np.allclose(device(base, 15.0, SR, 0), device(changed, 15.0, SR, 0))
        moved = not np.allclose(device(base, 15.0, SR, 0, patch), device(changed, 15.0, SR, 0, patch))
        named = "tensix_mask" in ROUTE_INPUTS[route].get(name, ())
        if named:
            assert moved, f"{route} says {name} reads tensix_mask, but changing it changed nothing"
        else:
            assert moved == moved_plain, f"{route} does not name {name}, but changing tensix_mask moved it differently"


# --- the gaps a mutation run found ---------------------------------------------------------------------------------

def test_a_clock_pulse_counts_as_landing_on_a_hit_whether_it_comes_just_after_it_or_just_before_it():
    from tensixinet.music.devices import accented_pulses
    hits = np.array([1.0, 2.0])
    assert accented_pulses(np.array([1.003]), hits).tolist() == [1.003]   # just after the hit
    assert accented_pulses(np.array([1.997]), hits).tolist() == [1.997]   # just before the next
    assert accented_pulses(np.array([1.5]), hits).tolist() == []          # between hits
    assert accented_pulses(np.array([3.0]), hits).tolist() == []          # past the last hit
    assert accented_pulses(np.array([0.5]), hits).tolist() == []          # before the first
    assert accented_pulses(np.array([1.0]), np.array([])).tolist() == []  # a register that never fires


def test_the_avdog_route_adds_the_register_to_the_strikes_and_does_not_replace_them():
    t = dataclasses.replace(chip(), tensix_mask=0x3FFF)  # a chip whose register never fires
    plain, routed_voices = avdog_voices(t, 20.0, SR), avdog_voices(t, 20.0, SR, routed("avdog"))
    assert np.abs(plain[0]).max() > 0.1  # the gong strikes still ping it
    assert all(np.array_equal(a, b) for a, b in zip(plain, routed_voices))


def test_the_strikes_route_reaches_the_avdog_through_the_shared_clock():
    t = chip()
    assert not np.array_equal(avdog_voices(t, 20.0, SR, routed("strikes"))[0], avdog_voices(t, 20.0, SR)[0])


def test_the_register_never_fires_at_or_after_the_end_of_the_render():
    t = chip()
    for duration in np.linspace(3.0, 20.0, 60):
        hits = step_times(t, duration)
        assert len(hits) == 0 or hits.max() < duration, duration


def test_a_hit_exactly_at_the_end_of_the_render_is_left_out():
    t = dataclasses.replace(chip(), tensix_mask=0x3FFE)  # step 0 fires, so the lap's first step lands on the end of a 2-lap render
    duration = 28 * 30.0 / aiclk_to_bpm(t.aiclk)
    hits = step_times(t, duration)
    assert len(hits) == 2 and hits.max() < duration


def test_the_strikes_route_joins_the_register_to_the_rolls_the_gong_follows_too():
    from tensixinet.music.devices import gong_trigger_roll, roll_pulses
    t = dataclasses.replace(chip(), tensix_mask=0x3002, ddr_status=(1, 1, 1, 1), eth_mask=0xFFFF)
    which = gong_trigger_roll(t, DEFAULT_PATCH)
    assert which is not None, "the test needs a chip whose gong follows a roll"
    allowed = np.concatenate([roll_pulses(t, 60.0, DEFAULT_PATCH)[which][0], step_times(t, 60.0)])
    joined = strike_times(t, 60.0, routed("strikes"))
    assert len(joined) >= 2 and all(np.isclose(allowed, s).any() for s in joined)
    assert not np.array_equal(joined, strike_times(t, 60.0))


def test_a_clock_pulse_that_rests_is_not_accented_even_where_it_lands_on_a_register_hit(monkeypatch):
    from tensixinet.music import devices
    t, real = chip(), devices._blips
    landed = [c for c in ttrollz_clock(t, 30.0) if np.abs(step_times(t, 30.0) - c).min() < 0.005]
    rested_one = False
    for seed in range(6):
        calls = []
        monkeypatch.setattr(devices, "_blips", lambda times, *a, **kw: calls.append(np.array(times)) or real(times, *a, **kw))
        ttrollz(t, 30.0, SR, seed, routed("accent", roll_rests=0.6))
        heard, accented = calls[0], calls[1]
        assert set(accented.tolist()) <= set(heard.tolist())
        rested_one = rested_one or any(c not in heard for c in landed)
    assert rested_one, "the test needs a landed pulse that rests"
