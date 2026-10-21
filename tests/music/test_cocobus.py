# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The coco bus: the Cocoquantus as the real one is used, with other sounds patched into its delay and the petals moving the echoes, plus a room.

The real instrument takes audio in, puts it through its two 8-bit delay chips, and its petals' castles set the delay clocks, so the echoes bend as the
castles step. Here the chip's other voices are sent to eight crunchy taps whose delay times the castles sway, and the echoes and the dry send go through a reverb."""
import dataclasses

import numpy as np
import pytest

from tensixinet.music.chorus import DEVICES, device_names, render_chorus
from tensixinet.music.cocobus import (
    BUS_TAPS, MAX_REPEATS, SEND_LEVELS, SWAY_DEPTH, coco_bus, fft_convolve, petal_sway, reverb_ir, send_level, swayed_tap,
)
from tensixinet.music.evict import ChipDefrag
from tensixinet.music.mapping import chip_pressure
from tensixinet.music.patch import DEFAULT_PATCH
from tensixinet.music.ttcoco import ECHO_FLOOR, crunch_bits, echo_tap
from tensixinet.telemetry import FALLBACK

SR = 8000
BUS_ON = dataclasses.replace(DEFAULT_PATCH, coco_send=0.6, coco_reverb=0.3, reverb_s=1.0)


def tel(**kw):
    return dataclasses.replace(FALLBACK, **kw)


def impulse(n, at=0, height=0.5):
    x = np.zeros(n)
    x[at] = height
    return x


# --- what gets sent ----------------------------------------------------------------

def test_every_device_has_a_send_level_and_the_cocos_own_devices_send_nothing():
    assert set(SEND_LEVELS) == set(device_names())
    assert SEND_LEVELS["ttquantus"] == 0.0 and SEND_LEVELS["ttcoco"] == 0.0  # they are the instrument, not its input
    assert all(0.0 <= v <= 1.0 for v in SEND_LEVELS.values())
    assert send_level("gong") > 0 and send_level("lead") > 0 and send_level("ttchoir") > 0
    assert send_level("not_a_device") == 0.0


# --- the petals sway the delay -----------------------------------------------------

def test_the_sway_is_a_castle_held_between_zero_and_one_that_steps_and_differs_by_tap():
    n = SR * 6
    a = petal_sway(tel(), n, SR, seed=3, tap=0)
    b = petal_sway(tel(), n, SR, seed=3, tap=1)
    assert a.shape == (n,) and 0.0 <= a.min() and a.max() <= 1.0
    assert len(np.unique(np.round(a, 6))) >= 4          # it moves, and in steps (a sample and hold)
    assert not np.allclose(a, b)                         # each tap follows another petal
    assert np.array_equal(a, petal_sway(tel(), n, SR, seed=3, tap=0))
    assert not np.array_equal(a, petal_sway(tel(), n, SR, seed=4, tap=0))


def test_the_chips_identity_changes_which_petal_each_castle_samples():
    n = SR * 6
    assert not np.array_equal(petal_sway(tel(tensix_mask=0x3FFF), n, SR, 3, 0), petal_sway(tel(tensix_mask=0x37F7), n, SR, 3, 0))


def test_a_faster_clock_makes_the_castles_step_faster():
    n = SR * 8
    steps = lambda s: int(np.count_nonzero(np.diff(s) != 0))
    assert steps(petal_sway(tel(aiclk=1300), n, SR, 3, 0)) > 1.5 * steps(petal_sway(tel(aiclk=500), n, SR, 3, 0))


# --- the swayed tap ----------------------------------------------------------------

def test_with_the_sway_held_at_the_middle_a_tap_is_the_plain_echo_without_its_dry_signal():
    x = np.random.default_rng(0).uniform(-0.5, 0.5, SR)
    got = swayed_tap(x, 800, np.full(SR, 0.5), 0.5, 0.5, 6.0)
    want = echo_tap(x, 800, 0.5, 6.0) - crunch_bits(x, 6.0)
    assert np.allclose(got, want, atol=1e-9)


def test_the_first_echo_lands_one_delay_later_and_the_sway_moves_it():
    x = impulse(SR)
    first = lambda sway: int(np.argmax(np.abs(swayed_tap(x, 1000, np.full(SR, sway), 0.5, 0.5, 8.0))))
    assert first(0.5) == 1000
    assert first(1.0) == 1500 and first(0.0) == 500      # a depth of 0.5 moves it by half the delay either way
    assert 0 < SWAY_DEPTH <= 0.5


def test_the_dry_signal_is_not_in_the_tap_and_repeats_fall_by_the_feedback():
    x = impulse(SR * 2)
    y = swayed_tap(x, 500, np.full(SR * 2, 0.5), 0.5, 0.5, 8.0)
    assert y[0] == 0.0
    peaks = [abs(y[500 * k]) for k in (1, 2, 3)]
    assert peaks[1] == pytest.approx(peaks[0] * 0.5, rel=0.1) and peaks[2] == pytest.approx(peaks[1] * 0.5, rel=0.1)


def test_a_tap_stops_repeating_once_a_repeat_is_inaudible_or_past_the_limit():
    assert MAX_REPEATS >= 8 and ECHO_FLOOR == 1 / 400
    x = impulse(SR * 4)
    y = swayed_tap(x, 100, np.full(SR * 4, 0.5), 0.5, 0.79, 8.0)
    assert np.count_nonzero(y) <= MAX_REPEATS


def test_zero_feedback_is_no_echo_and_a_delay_past_the_end_is_silence():
    x = impulse(SR)
    assert not np.any(swayed_tap(x, 700, np.full(SR, 0.5), 0.5, 0.0, 8.0))
    assert not np.any(swayed_tap(x, SR * 2, np.full(SR, 0.5), 0.5, 0.5, 8.0))


# --- the room ----------------------------------------------------------------------

def test_the_reverb_impulse_response_decays_by_sixty_decibels_over_its_time_and_has_unit_energy():
    ir = reverb_ir(1.0, SR, seed=1, ear=0)
    assert np.sum(ir ** 2) == pytest.approx(1.0)
    pre = int(0.02 * SR)
    assert not np.any(ir[:pre])                                           # a pre-delay before the room answers
    early, late = np.sqrt(np.mean(ir[pre:pre + SR // 10] ** 2)), np.sqrt(np.mean(ir[pre + int(0.9 * SR):pre + SR] ** 2))
    assert 0 < late / early < 0.02                                        # about -60 dB at one second, to within noise


def test_a_longer_time_is_a_longer_tail_and_the_two_ears_have_their_own_room():
    short, long_ = reverb_ir(0.5, SR, 1, 0), reverb_ir(2.0, SR, 1, 0)
    assert len(long_) > len(short) * 3
    assert not np.array_equal(reverb_ir(1.0, SR, 1, 0), reverb_ir(1.0, SR, 1, 1))
    assert np.array_equal(reverb_ir(1.0, SR, 1, 0), reverb_ir(1.0, SR, 1, 0))


def test_the_reverb_is_rolled_off_so_the_tail_is_darker_than_noise():
    ir = reverb_ir(1.0, SR, 1, 0)
    spec = np.abs(np.fft.rfft(ir))
    assert spec[len(spec) * 3 // 4:].mean() < 0.5 * spec[: len(spec) // 4].mean()


def test_fft_convolution_is_the_plain_convolution_cut_to_the_input_length():
    rng = np.random.default_rng(2)
    x, h = rng.normal(size=300), rng.normal(size=70)
    assert np.allclose(fft_convolve(x, h), np.convolve(x, h)[:300], atol=1e-9)


# --- the bus -----------------------------------------------------------------------

def test_silence_in_is_silence_out():
    assert not np.any(coco_bus(np.zeros(SR), tel(), SR, 1, BUS_ON, stereo=False))


def test_a_sound_sent_in_comes_back_later_as_echoes_and_then_a_tail():
    x = impulse(SR * 4, at=0, height=0.8)
    wet = coco_bus(x, tel(), SR, 1, BUS_ON, stereo=False)
    assert wet.shape == x.shape and np.isfinite(wet).all()
    assert np.abs(wet[: int(0.02 * SR)]).max() < 1e-9               # nothing before the first echo or the room answers
    assert np.abs(wet[SR // 2: SR * 2]).max() > 0.001            # the echoes


def test_the_room_rings_on_after_the_last_echo_and_without_it_nothing_does():
    x = impulse(SR * 4, at=0, height=0.8)
    hot, cool = (95.0,) * 8, tel(power=20.0, asic_temperature=30.0)
    short = dataclasses.replace(cool, gddr_temps=hot)           # fast taps and few repeats: every echo is over by half a second
    window = slice(int(0.5 * SR), int(0.9 * SR))
    with_room = coco_bus(x, short, SR, 1, BUS_ON, stereo=False)[window]
    without = coco_bus(x, short, SR, 1, dataclasses.replace(BUS_ON, coco_reverb=0.0), stereo=False)[window]
    assert np.sum(without ** 2) < 1e-12 and np.sum(with_room ** 2) > 1e-9


def test_with_no_reverb_the_wet_is_only_the_echoes_and_they_end_by_the_longest_repeat():
    x = impulse(SR * 6)
    wet = coco_bus(x, tel(gddr_temps=(30.0,) * 8), SR, 1, dataclasses.replace(BUS_ON, coco_reverb=0.0), stereo=False)
    longest = int(MAX_REPEATS * 0.9 * (1 + SWAY_DEPTH) * SR) + 1
    assert not np.any(wet[longest:])


def test_stereo_puts_alternate_taps_in_alternate_ears_with_each_ears_own_room():
    x = np.random.default_rng(0).uniform(-0.5, 0.5, SR * 2)
    wet = coco_bus(x, tel(), SR, 1, BUS_ON, stereo=True)
    assert wet.shape == (len(x), 2)
    assert not np.allclose(wet[:, 0], wet[:, 1]) and abs(np.corrcoef(wet[:, 0], wet[:, 1])[0, 1]) < 0.9
    mono = coco_bus(x, tel(), SR, 1, BUS_ON, stereo=False)
    assert mono.shape == (len(x),) and np.allclose(mono, wet[:, 0] + wet[:, 1])   # the mono bus is the two ears added
    left_only = coco_bus(x, tel(gddr_temps=(30.0,) * 8), SR, 1, dataclasses.replace(BUS_ON, coco_reverb=0.0), stereo=True)
    assert np.any(left_only[:, 0]) and np.any(left_only[:, 1])


def test_the_bus_is_deterministic_and_the_seed_changes_it():
    x = np.random.default_rng(0).uniform(-0.5, 0.5, SR)
    a = coco_bus(x, tel(), SR, 1, BUS_ON, stereo=False)
    assert np.array_equal(a, coco_bus(x, tel(), SR, 1, BUS_ON, stereo=False))
    assert not np.array_equal(a, coco_bus(x, tel(), SR, 2, BUS_ON, stereo=False))


def test_pressure_feeds_the_echoes_back_harder():
    x = impulse(SR * 6, height=0.8)
    cool, hot = tel(asic_temperature=30.0, power=20.0, tdp_limit=300.0), tel(asic_temperature=99.0, thm_limit_throttle=100.0)
    assert chip_pressure(hot) > 0.9 > 0.2 > chip_pressure(cool)
    rooms = dataclasses.replace(BUS_ON, coco_reverb=0.0)
    late = lambda t: np.sum(coco_bus(x, t, SR, 1, rooms, stereo=False)[SR * 2:] ** 2)
    assert late(hot) > 3 * late(cool)


def test_a_cooler_tap_echoes_later_than_a_hotter_one():
    x = impulse(SR * 3)
    rooms = dataclasses.replace(BUS_ON, coco_reverb=0.0)
    first_echo = lambda g: int(np.argmax(np.abs(coco_bus(x, tel(gddr_temps=(g,) * 8), SR, 1, rooms, stereo=False)) > 1e-6))
    assert first_echo(35.0) > first_echo(75.0)


# --- in the chorus -----------------------------------------------------------------

def _chips(n=4):
    return [tel(asic_temperature=40.0 + 8 * i, power=40.0 + 10 * i) for i in range(n)]


def test_a_patch_with_no_send_renders_exactly_as_it_did_before_the_bus_existed():
    base = render_chorus(_chips(), 1.0, SR)
    assert np.array_equal(base, render_chorus(_chips(), 1.0, SR, patch=dataclasses.replace(DEFAULT_PATCH, coco_send=0.0, coco_reverb=0.9)))


def test_a_send_changes_the_mix_and_a_bigger_send_changes_it_more():
    base = render_chorus(_chips(), 2.0, SR)
    small = render_chorus(_chips(), 2.0, SR, patch=dataclasses.replace(BUS_ON, coco_send=0.2))
    big = render_chorus(_chips(), 2.0, SR, patch=dataclasses.replace(BUS_ON, coco_send=0.8))
    assert not np.array_equal(base, small)
    assert np.sqrt(np.mean((big - base) ** 2)) > 1.5 * np.sqrt(np.mean((small - base) ** 2))
    assert np.isfinite(big).all() and np.abs(big).max() < 1.0


def test_the_bus_works_in_stereo_and_keeps_the_dry_mix_in_it():
    base = render_chorus(_chips(), 2.0, SR, stereo=True)
    wet = render_chorus(_chips(), 2.0, SR, patch=BUS_ON, stereo=True)
    assert wet.shape == base.shape == (2 * SR, 2) and not np.array_equal(wet, base)
    assert np.corrcoef(wet[:, 0], base[:, 0])[0, 1] > 0.5


def test_auditioning_one_device_with_only_plays_it_without_the_bus():
    solo = lambda p: render_chorus(_chips(), 1.0, SR, patch=p, only=["gong"])
    assert np.array_equal(solo(DEFAULT_PATCH), solo(BUS_ON))


def test_a_chip_that_is_idle_sends_nothing():
    idle = [ChipDefrag("idle", 5.0)] * 4
    out = render_chorus(_chips(), 1.0, SR, patch=BUS_ON, defrag=idle)
    assert not np.any(out)


def test_the_send_follows_what_is_heard_not_what_is_made():
    """A voice with no floor is sent at the floor's level: with the floor tiny, the bus has little to echo."""
    quiet = dataclasses.replace(BUS_ON, floor=0.01)
    loud = dataclasses.replace(BUS_ON, floor=0.5)
    plain = lambda p: render_chorus(_chips(), 2.0, SR, patch=dataclasses.replace(p, coco_send=0.0))
    gap = lambda p: np.sqrt(np.mean((render_chorus(_chips(), 2.0, SR, patch=p) - plain(p)) ** 2))
    assert gap(loud) > 1.5 * gap(quiet)


# --- what the bus hands to its taps, and what is mixed -----------------------------------

@pytest.fixture
def taps(monkeypatch):
    """Every call coco_bus makes to swayed_tap, recorded, with the real tap still run."""
    import tensixinet.music.cocobus as cb
    calls, real = [], cb.swayed_tap
    monkeypatch.setattr(cb, "swayed_tap", lambda *a: calls.append(a) or real(*a))
    return calls


def test_there_are_four_taps_one_for_every_second_gddr_channel_each_with_its_own_delay_and_bits(taps):
    from tensixinet.music.ttcoco import tap_bits, tap_delay_ms
    temps = (30.0, 90.0, 40.0, 80.0, 50.0, 70.0, 60.0, 35.0)
    coco_bus(impulse(SR), tel(gddr_temps=temps), SR, 1, BUS_ON, stereo=False)
    assert BUS_TAPS == 4 and len(taps) == 4
    for call, ch in zip(taps, (0, 2, 4, 6)):
        _, base, _, depth, _, bits = call
        ms = tap_delay_ms(temps[ch])
        assert base == pytest.approx(ms / 1000 * SR) and bits == tap_bits(ms) and depth == SWAY_DEPTH == 0.35


def test_the_channels_the_taps_do_not_use_change_nothing():
    a = tel(gddr_temps=(30.0, 90.0, 40.0, 80.0, 50.0, 70.0, 60.0, 35.0))
    b = dataclasses.replace(a, gddr_temps=(30.0, 55.0, 40.0, 45.0, 50.0, 85.0, 60.0, 99.0))
    x = np.random.default_rng(0).uniform(-0.5, 0.5, SR)
    assert np.array_equal(coco_bus(x, a, SR, 1, BUS_ON, stereo=False), coco_bus(x, b, SR, 1, BUS_ON, stereo=False))


def test_pressure_raises_the_feedback_from_a_quarter_to_the_coco_ceiling(taps):
    from tensixinet.music.ttcoco import FEEDBACK_MAX
    cool, hot = tel(asic_temperature=30.0, power=20.0, tdp_limit=300.0), tel(asic_temperature=100.0, thm_limit_throttle=100.0, power=400.0)
    coco_bus(impulse(SR), cool, SR, 1, BUS_ON, stereo=False)
    coco_bus(impulse(SR), hot, SR, 1, BUS_ON, stereo=False)
    assert chip_pressure(cool) == 0.0 and chip_pressure(hot) == 1.0
    assert taps[0][4] == pytest.approx(0.25) and taps[4][4] == pytest.approx(FEEDBACK_MAX)


def test_the_sway_is_glided_so_a_castle_step_is_not_a_jump(taps):
    coco_bus(impulse(SR * 6), tel(), SR, 1, BUS_ON, stereo=False)
    raw = petal_sway(tel(), SR * 6, SR, 1, 0)
    sway = taps[0][2]
    assert np.abs(np.diff(raw)).max() > 0.2 and np.abs(np.diff(sway)).max() < 0.1 * np.abs(np.diff(raw)).max() + 0.02
    assert np.abs(sway - raw).max() > 0.05 and abs(sway.mean() - raw.mean()) < 0.02   # smoothed, not moved


def test_the_reverb_takes_the_dry_send_as_well_as_the_echoes_and_its_share_scales_it():
    x = impulse(SR * 3, height=0.8)
    hot = tel(gddr_temps=(99.0,) * 8)
    room = lambda share: coco_bus(x, hot, SR, 1, dataclasses.replace(BUS_ON, coco_reverb=share), stereo=False)
    pre, first_echo = int(0.02 * SR), int(0.02 * SR) + 40
    assert np.sum(room(0.4)[pre + 1: pre + 40] ** 2) > 1e-9                           # the room answers the dry hit before any echo is back
    r0, r1, r2 = room(0.0), room(0.25), room(0.5)
    assert np.allclose(r2 - r0, 2 * (r1 - r0), atol=1e-12) and np.abs(r1 - r0).max() > 1e-4


def test_each_ear_has_its_own_room():
    x = impulse(SR * 4, height=0.8)
    hot = tel(asic_temperature=30.0, power=20.0, gddr_temps=(99.0,) * 8)
    wet = coco_bus(x, hot, SR, 1, BUS_ON, stereo=True)[int(0.5 * SR): int(0.9 * SR)]
    assert abs(np.corrcoef(wet[:, 0], wet[:, 1])[0, 1]) < 0.5


def test_the_reverb_falls_by_sixty_decibels_over_its_time():
    ir = reverb_ir(2.0, SR, 1, 0)
    pre = int(0.02 * SR)
    rms = lambda a, b: np.sqrt(np.mean(ir[pre + int(a * SR): pre + int(b * SR)] ** 2))
    ratio = rms(1.5, 1.6) / rms(0.1, 0.2)                                              # 1.4 s later: 10 ** (-60 * 1.4 / 2 / 20) = 0.0079
    assert 0.004 < ratio < 0.016


def test_the_reverb_cutoff_is_four_kilohertz_at_a_high_rate_and_a_share_of_the_rate_at_a_low_one():
    import tensixinet.music.cocobus as cb
    assert cb.REVERB_CUTOFF_HZ == 4000.0 and cb.MIN_FEEDBACK == 0.25 and cb.TAP_LEVEL == 0.5
    ir = reverb_ir(1.0, 44100, 1, 0)
    spec = np.abs(np.fft.rfft(ir))
    freqs = np.fft.rfftfreq(len(ir), 1 / 44100)
    assert spec[(freqs > 10000) & (freqs < 15000)].mean() < 0.2 * spec[(freqs > 500) & (freqs < 1500)].mean()


def test_the_bus_is_seeded_by_the_segment_and_the_chip(monkeypatch):
    import tensixinet.music.chorus as ch
    seen = []
    monkeypatch.setattr(ch, "coco_bus", lambda send, t, sr, seed, patch, stereo: seen.append(seed) or np.zeros_like(send))
    render_chorus(_chips(), 0.5, SR, patch=BUS_ON, segment=0)
    render_chorus(_chips(), 0.5, SR, patch=BUS_ON, segment=1)
    assert len(seen) == 8 and len(set(seen[:4])) == 4 and not set(seen[:4]) & set(seen[4:])
