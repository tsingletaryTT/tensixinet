# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Ciat-Lonbarde-style devices. Each is a pure function of one chip's telemetry."""
import numpy as np
import pytest

from tensixinet.music.devices import (
    cocoquantus,
    cocoquantus_steps,
    current_mix,
    PENTATONIC,
    plumbutter_drone,
    quantize_to_scale,
    rollz,
    rollz_pulse_times,
    sidrax,
)
from tensixinet.telemetry import FALLBACK, Telemetry

SR = 8000


def tel(**kw):
    return Telemetry(**{**FALLBACK.__dict__, **kw})


# --- Rollz: asymmetric pulses --------------------------------------------------

def test_rollz_mean_interval_matches_the_bpm():
    times = rollz_pulse_times(bpm=120, duration=60, asymmetry=0.3)
    assert np.mean(np.diff(times)) == pytest.approx(0.5, rel=0.02)


def test_rollz_intervals_are_not_all_equal():
    d = np.diff(rollz_pulse_times(bpm=120, duration=10, asymmetry=0.3))
    assert d.max() > d.min() * 1.5


def test_rollz_zero_asymmetry_is_a_plain_clock():
    d = np.diff(rollz_pulse_times(bpm=60, duration=10, asymmetry=0.0))
    assert np.allclose(d, 1.0)


def test_rollz_pulses_stay_inside_the_duration():
    assert rollz_pulse_times(90, 5, 0.2).max() < 5


def test_rollz_faster_aiclk_means_more_onsets():
    slow = rollz(tel(aiclk=400), 4.0, SR, seed=0)
    fast = rollz(tel(aiclk=1350), 4.0, SR, seed=0)
    assert np.sum(np.abs(fast)) > np.sum(np.abs(slow))


def test_rollz_ddr_lanes_add_voices():
    none = rollz(tel(ddr_status=(0, 0, 0, 0)), 4.0, SR, seed=0)
    all4 = rollz(tel(ddr_status=(1, 1, 1, 1)), 4.0, SR, seed=0)
    assert not np.allclose(none, all4)


# --- Cocoquantus: sample and hold ----------------------------------------------

def test_quantize_puts_every_value_on_the_scale():
    v = np.linspace(0, 0.999, 500)
    semis = quantize_to_scale(v, PENTATONIC, octaves=2)
    allowed = {12 * o + d for o in range(2) for d in PENTATONIC}
    assert set(semis.tolist()) <= allowed
    assert set(semis.tolist()) == allowed  # and the whole range is reachable


def test_cocoquantus_steps_faster_when_aiclk_is_higher():
    def changes(aiclk):
        semis = cocoquantus_steps(tel(aiclk=aiclk), 8.0, SR, seed=5)
        return int(np.count_nonzero(np.diff(semis)))
    assert changes(1350) > 2 * changes(400)


def test_cocoquantus_plays_audible_stepped_pitches():
    audio = cocoquantus(tel(asic_temperature=25.0), 2.0, SR, seed=3)
    assert audio.dtype == np.float32 and np.abs(audio).max() > 0.1


# --- Plumbutter: current-based mixing ------------------------------------------

def test_current_mix_never_exceeds_full_scale_however_many_sources():
    # tanh rounds to exactly 1.0 at this drive, so the bound is <=, not <.
    loud = [np.ones(100) for _ in range(500)]
    assert np.abs(current_mix(loud, [10.0] * 500)).max() <= 1.0


def test_current_mix_of_a_realistic_chorus_stays_strictly_below_full_scale():
    loud = [np.ones(100) for _ in range(32)]
    assert np.abs(current_mix(loud, [40.0] * 32)).max() < 1.0


def test_current_mix_compresses_rather_than_adds():
    one = current_mix([np.full(10, 0.5)], [10.0])
    many = current_mix([np.full(10, 0.5)] * 8, [10.0] * 8)
    assert many.max() < 8 * one.max() * 0.6


def test_current_mix_is_order_independent():
    a, b = np.sin(np.arange(100) / 5), np.cos(np.arange(100) / 7)
    assert np.allclose(current_mix([a, b], [5, 20]), current_mix([b, a], [20, 5]))


def test_higher_current_source_dominates_the_mix():
    a, b = np.ones(10) * 0.5, -np.ones(10) * 0.5
    assert current_mix([a, b], [40.0, 10.0]).mean() > 0
    assert current_mix([a, b], [10.0, 40.0]).mean() < 0


def test_zero_current_everywhere_falls_back_to_equal_weights():
    out = current_mix([np.ones(4) * 0.2, np.ones(4) * 0.2], [0.0, 0.0])
    assert np.all(out > 0)


def test_plumbutter_drone_has_the_chips_pitch_voices():
    audio = plumbutter_drone(tel(), 1.0, SR, seed=0)
    assert audio.dtype == np.float32 and np.abs(audio).max() > 0


# --- gong: inharmonic, long-ringing ---------------------------------------------

from tensixinet.music.devices import GONG_PARTIALS, gong, ultrasonic_shimmer  # noqa: E402
from tensixinet.music.mapping import temp_to_pitch_hz as _pitch  # noqa: E402
from tensixinet.telemetry import FALLBACK as _FB  # noqa: E402
import dataclasses as _dc  # noqa: E402

_SR = 8000


def _peak_hz(x, sr):
    spec = np.abs(np.fft.rfft(x))
    return float(np.argmax(spec) * sr / len(x))


def test_gong_partials_after_the_first_are_not_whole_multiples():
    assert GONG_PARTIALS[0] == 1.0
    assert all(abs(r - round(r)) > 0.1 for r in GONG_PARTIALS[1:])


def test_gong_rings_at_the_temperature_pitch_an_octave_down():
    t = _dc.replace(_FB, asic_temperature=60.0, aiclk=400.0)  # 60 BPM: first strike rings for 2 s
    out = gong(t, 2.0, _SR, 0)
    assert _peak_hz(out, _SR) == pytest.approx(_pitch(60.0) / 2, abs=1.5)


def test_gong_strike_decays():
    out = gong(_dc.replace(_FB, aiclk=400.0), 6.0, _SR, 0)
    rms = lambda x: float(np.sqrt(np.mean(x ** 2)))
    assert rms(out[: _SR // 2]) > 2 * rms(out[3 * _SR: int(3.5 * _SR)])


def test_gong_is_bounded_and_deterministic():
    out = gong(_FB, 3.0, _SR, 0)
    assert out.dtype == np.float32 and np.abs(out).max() <= 1.0 and np.abs(out).max() > 0.1
    assert np.array_equal(out, gong(_FB, 3.0, _SR, 0))


def test_gong_strikes_come_at_the_aiclk_tempo():
    assert not np.allclose(gong(_dc.replace(_FB, aiclk=400.0), 6.0, _SR, 0),
                           gong(_dc.replace(_FB, aiclk=1350.0), 6.0, _SR, 0))


# --- ultrasonic shimmer ---------------------------------------------------------

_HI = 44100


def _hi_fraction(x, sr, above):
    spec = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(len(x), 1 / sr)
    return float(spec[freqs >= above].sum() / spec.sum())


def test_shimmer_lives_in_the_top_of_the_spectrum():
    out = ultrasonic_shimmer(_dc.replace(_FB, asic_temperature=50.0), 1.0, _HI, 0)
    assert _hi_fraction(out, _HI, 12000) > 0.95


def test_a_hotter_chip_shimmers_higher():
    cool = ultrasonic_shimmer(_dc.replace(_FB, asic_temperature=30.0), 1.0, _HI, 0)
    hot = ultrasonic_shimmer(_dc.replace(_FB, asic_temperature=95.0), 1.0, _HI, 0)
    assert _peak_hz(hot, _HI) > _peak_hz(cool, _HI) + 1000


def test_shimmer_stays_below_nyquist_at_any_sample_rate():
    for sr in (8000, 22050, 48000, 96000):
        out = ultrasonic_shimmer(_dc.replace(_FB, asic_temperature=100.0), 0.5, sr, 0)
        assert _peak_hz(out, sr) < 0.46 * sr
        assert np.abs(out).max() <= 1.0


def test_gddr_spread_makes_the_shimmer_beat():
    flat = ultrasonic_shimmer(_dc.replace(_FB, gddr_temps=(40.0,) * 8), 1.0, _HI, 0)
    wide = ultrasonic_shimmer(_dc.replace(_FB, gddr_temps=(30.0, 60.0) * 4), 1.0, _HI, 0)
    env = lambda x: np.abs(x[: _HI // 2]).reshape(-1, 441).max(axis=1)  # 100 slices of 10 ms
    assert env(wide).std() > 3 * env(flat).std()  # a beating pair swells and fades; a lone tone does not


def test_gong_fundamental_is_a_beating_pair_not_a_single_line():
    t = _dc.replace(_FB, asic_temperature=60.0, aiclk=400.0)
    x = gong(t, 8.0, _SR, 0)
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    freqs = np.fft.rfftfreq(len(x), 1 / _SR)
    f0 = _pitch(60.0) / 2
    near = (freqs > f0 - 1.5) & (freqs < f0 + 1.5)
    s = spec[near]
    peaks = [i for i in range(1, len(s) - 1) if s[i] > s[i - 1] and s[i] > s[i + 1] and s[i] > 0.3 * s.max()]
    assert len(peaks) >= 2


# --- the patch reaches the devices ------------------------------------------------

from tensixinet.music.patch import DEFAULT_PATCH, SCALES  # noqa: E402


def test_every_device_accepts_a_patch_and_the_default_changes_nothing():
    for dev in (rollz, sidrax, cocoquantus, plumbutter_drone, gong, ultrasonic_shimmer):
        assert np.array_equal(dev(_FB, 1.0, _SR, 3), dev(_FB, 1.0, _SR, 3, patch=DEFAULT_PATCH)), dev.__name__


def test_cocoquantus_quantises_onto_the_patch_scale():
    patch = _dc.replace(DEFAULT_PATCH, scale_name="whole_tone", scale=SCALES["whole_tone"])
    steps = cocoquantus_steps(_FB, 4.0, _SR, 0, patch=patch)
    allowed = {12 * o + d for o in range(2) for d in SCALES["whole_tone"]}
    assert set(np.unique(steps)) <= allowed and len(set(np.unique(steps))) > 3


def test_a_patch_scale_changes_cocoquantus_audio():
    patch = _dc.replace(DEFAULT_PATCH, scale_name="whole_tone", scale=SCALES["whole_tone"])
    assert not np.allclose(cocoquantus(_FB, 2.0, _SR, 0), cocoquantus(_FB, 2.0, _SR, 0, patch=patch))


def test_gong_uses_the_patch_partials():
    t = _dc.replace(_FB, asic_temperature=60.0, aiclk=400.0)
    f0 = _pitch(60.0) / 2
    patched = _dc.replace(DEFAULT_PATCH, gong_partials=(1.0, 3.3), gong_amps=(1.0, 1.0))
    x = gong(t, 4.0, _SR, 0, patch=patched)
    spec = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(len(x), 1 / _SR)
    near = lambda f: spec[(freqs > f - 3) & (freqs < f + 3)].max()
    assert near(3.3 * f0) > 5 * near(2.76 * f0)  # the default's second partial is gone


def test_gong_uses_the_patch_beat():
    t = _dc.replace(_FB, asic_temperature=60.0, aiclk=400.0)
    slow = gong(t, 4.0, _SR, 0, patch=_dc.replace(DEFAULT_PATCH, gong_beat_hz=0.4))
    fast = gong(t, 4.0, _SR, 0, patch=_dc.replace(DEFAULT_PATCH, gong_beat_hz=1.2))
    assert not np.allclose(slow, fast)


def test_the_shimmer_sits_in_the_patch_band():
    patch = _dc.replace(DEFAULT_PATCH, shimmer_band=(6000.0, 8000.0))
    out = ultrasonic_shimmer(_dc.replace(_FB, asic_temperature=60.0), 1.0, _HI, 0, patch=patch)
    assert 6000 <= _peak_hz(out, _HI) <= 8100


# --- the wider readings reach the devices --------------------------------------------------

import dataclasses as dc  # noqa: E402

from tensixinet.music import devices as devices_module  # noqa: E402
from tensixinet.music.mapping import aiclk_to_bpm  # noqa: E402
from tensixinet.music.devices import ultrasonic_shimmer as shimmer, gong as gong_dev  # noqa: E402
from tensixinet.music.lead import lead as lead_dev  # noqa: E402


def _rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


def _centroid(x, sr):
    spec = np.abs(np.fft.rfft(x))
    return float((spec * np.arange(len(spec))).sum() / spec.sum() * sr / len(x))


def _rollz_base(**kw):
    return tel(ddr_status=(0, 0, 0, 0), **kw)


def test_rollz_ethernet_links_add_lanes_and_each_link_counts():
    none = rollz(_rollz_base(eth_mask=0x0000), 4.0, SR, 0)
    full = rollz(_rollz_base(eth_mask=0xFFFF), 4.0, SR, 0)
    assert _rms(full) > _rms(none)
    assert not np.allclose(rollz(_rollz_base(eth_mask=0xFFFF), 4.0, SR, 0), rollz(_rollz_base(eth_mask=0xFFFE), 4.0, SR, 0))
    # a lane belongs to a group of four links: losing the whole top group changes it again
    assert not np.allclose(rollz(_rollz_base(eth_mask=0xFFFF), 4.0, SR, 0), rollz(_rollz_base(eth_mask=0x0FFF), 4.0, SR, 0))


def test_cocoquantus_holds_each_note_for_less_time_when_axiclk_is_higher():
    def changes(axi):
        return int(np.sum(np.diff(cocoquantus_steps(tel(axiclk=axi), 8.0, SR, 3)) != 0))
    assert changes(1920.0) > 1.5 * changes(960.0)


def _dominant_hz(x, sr, lo=40.0, hi=400.0):
    spec = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(len(x), 1 / sr)
    band = (freqs >= lo) & (freqs <= hi)
    return float(freqs[band][np.argmax(spec[band])])


def test_a_higher_core_voltage_detunes_the_drone_upward():
    low = plumbutter_drone(tel(voltage=0.71), 8.0, SR, 0)
    high = plumbutter_drone(tel(voltage=0.90), 8.0, SR, 0)
    assert _dominant_hz(high, SR) > _dominant_hz(low, SR)


def test_the_drone_asks_for_as_many_voices_as_the_pcie_usage_gives(monkeypatch):
    seen = []
    real = devices_module.build_voices
    monkeypatch.setattr(devices_module, "build_voices", lambda *a, **k: seen.append(k.get("voices_per_chip")) or real(*a, **k))
    plumbutter_drone(tel(pcie_usage=4), 0.2, SR, 0)
    plumbutter_drone(tel(pcie_usage=1), 0.2, SR, 0)
    assert seen == [4, 1]


def test_lead_is_brighter_under_pressure():
    cool = lead_dev(tel(asic_temperature=40.0, power=10.0), 6.0, 16000, 0)
    hot = lead_dev(tel(asic_temperature=88.0, power=10.0, aiclk=800.0), 6.0, 16000, 0)
    same_pitch_cool = lead_dev(tel(asic_temperature=88.0, power=10.0, aiclk=800.0, thm_limit_throttle=400.0), 6.0, 16000, 0)
    assert _centroid(hot, 16000) > _centroid(same_pitch_cool, 16000)  # only the throttle limit differs
    assert np.abs(hot).max() <= 1.0 and np.abs(cool).max() <= 1.0


def test_chips_with_different_harvesting_play_the_lead_differently():
    from tensixinet.music.lead import identity_offset
    from tensixinet.music.mapping import chip_identity
    # the offset is one of three values, so pick two harvesting masks that land on different ones
    other = next(m for m in range(0x3000, 0x3100) if identity_offset(chip_identity(m, 0x3FFF)) != identity_offset(chip_identity(0x37F7, 0x3FFF)))
    a = lead_dev(tel(tensix_mask=0x37F7), 3.0, SR, 0)
    b = lead_dev(tel(tensix_mask=other), 3.0, SR, 0)
    assert not np.allclose(a, b)
    assert np.allclose(a, lead_dev(tel(tensix_mask=0x37F7), 3.0, SR, 0))


def test_shimmer_swells_under_pressure_and_stays_bounded():
    calm = shimmer(tel(power=10.0), 1.0, 48000, 0)
    loaded = shimmer(tel(power=125.0), 1.0, 48000, 0)
    assert _rms(loaded) > 1.5 * _rms(calm) and np.abs(loaded).max() <= 1.0


def test_chips_with_different_links_tune_their_gongs_differently():
    a = gong_dev(tel(eth_mask=0x3EDF), 4.0, SR, 0)
    b = gong_dev(tel(eth_mask=0x3EEF), 4.0, SR, 0)
    assert not np.allclose(a, b) and np.allclose(a, gong_dev(tel(eth_mask=0x3EDF), 4.0, SR, 0))


def _first_sound(x, sr):
    return int(np.argmax(np.abs(x) > 0.02)) / sr


def test_the_heartbeat_sets_when_the_first_gong_strike_lands():
    on_the_beat = _first_sound(gong_dev(tel(heartbeat=0), 20.0, SR, 0), SR)
    late = _first_sound(gong_dev(tel(heartbeat=8), 20.0, SR, 0), SR)
    interval = 2 * 4 * 60.0 / aiclk_to_bpm(tel().aiclk)  # two bars at the chip's tempo
    assert on_the_beat < 0.05 and late == pytest.approx(interval / 4, rel=0.1)


# --- the Gongue's trigger response: a blocking window over a clock (Plumbutter guide, Gongues [2]) ----------

from tensixinet.music.devices import gate_triggers, strike_times  # noqa: E402
from tensixinet.music.mapping import personality  # noqa: E402


def test_a_trigger_inside_the_window_is_ignored_so_the_clock_is_divided():
    # The guide's example: a 250 bpm clock with a window tuned for 1.1 s gongs on every fifth pulse.
    pulses = np.arange(0, 6.0, 0.24)
    assert np.allclose(gate_triggers(pulses, window=1.1, start=0.0), [0.0, 1.2, 2.4, 3.6, 4.8])


def test_the_window_is_a_time_and_not_a_count_so_a_faster_clock_divides_by_more():
    slow = gate_triggers(np.arange(0, 6.0, 0.6), window=1.1, start=0.0)  # 100 bpm
    fast = gate_triggers(np.arange(0, 6.0, 0.24), window=1.1, start=0.0)  # 250 bpm
    assert np.allclose(np.diff(slow), 1.2) and np.allclose(np.diff(fast), 1.2)  # the same gap, divided by 2 and by 5


def test_pulses_before_the_start_never_trigger():
    assert gate_triggers(np.arange(0, 5.0, 1.0), window=0.5, start=2.5)[0] == 3.0


def test_no_pulse_after_the_start_means_no_strike():
    assert len(gate_triggers(np.arange(0, 3.0, 1.0), window=0.5, start=10.0)) == 0


def test_strikes_sit_on_the_chips_own_clock():
    t = tel(aiclk=800.0, heartbeat=0)
    clock = rollz_pulse_times(aiclk_to_bpm(800.0), 30.0)
    strikes = strike_times(t, 30.0)
    assert len(strikes) >= 2 and all(np.isclose(clock, s).any() for s in strikes)


def test_a_chips_identity_sets_its_trigger_response():
    gaps = {tuple(np.round(np.diff(strike_times(tel(tensix_mask=m, heartbeat=0), 60.0)), 3)) for m in range(0x3000, 0x3040)}
    assert len(gaps) >= 3  # different harvesting divides the clock differently


def test_a_strike_is_never_closer_than_two_bars():
    t = tel(aiclk=800.0)
    interval = 2 * 4 * 60.0 / aiclk_to_bpm(800.0)
    assert np.diff(strike_times(t, 60.0)).min() >= interval - 1e-9


# --- one clock, three modules: the Deconstructed Drum and Shock Wave patches ------------------------------------

from tensixinet.music.devices import seconds_since, shock_wave  # noqa: E402


def test_seconds_since_counts_from_the_latest_strike_and_is_infinite_before_the_first():
    since = seconds_since(np.array([1.0, 3.0]), n=8, sample_rate=2)  # samples at 0, .5, 1, ... 3.5 s
    assert np.isinf(since[:2]).all() and np.allclose(since[2:6], [0.0, 0.5, 1.0, 1.5]) and np.allclose(since[6:], [0.0, 0.5])


def test_seconds_since_with_no_strikes_is_infinite_everywhere():
    assert np.isinf(seconds_since(np.array([]), n=5, sample_rate=10)).all()


def test_the_shock_wave_is_silent_until_a_strike_then_rings_out_and_stays_inside_zero_to_one():
    since = seconds_since(np.array([1.0]), n=1000, sample_rate=100)
    w = shock_wave(since, rate_hz=3.0, decay_s=1.0)
    assert np.all(w[:100] == 0) and w.min() >= 0 and w.max() <= 1.0
    assert w[100:200].max() > 0.5 and w[800:].max() < 0.1  # it rings after the strike and has died away by the end


def test_the_shock_wave_swings_both_ways_so_the_rectified_ripple_runs_at_twice_the_rate():
    since = seconds_since(np.array([0.0]), n=1000, sample_rate=1000)
    w = shock_wave(since, rate_hz=5.0, decay_s=10.0)
    peaks = int(np.sum((w[1:-1] > w[:-2]) & (w[1:-1] > w[2:]) & (w[1:-1] > 0.2)))
    assert peaks == pytest.approx(10, abs=1)  # five undulations a second, a peak for each half: white then red


def _block_rms(x, sr, start, length):
    seg = x[int(start * sr): int((start + length) * sr)]
    return float(np.sqrt(np.mean(seg ** 2)))


def test_the_shimmer_snaps_at_each_gong_strike():
    t = _dc.replace(_FB, asic_temperature=60.0, aiclk=800.0, gddr_temps=(40.0,) * 8, heartbeat=0)
    out = ultrasonic_shimmer(t, 6.0, _HI, 0)
    s = strike_times(t, 6.0)[1]
    assert _block_rms(out, _HI, s, 0.005) > 1.2 * _block_rms(out, _HI, s + 0.3, 0.005)
    assert _block_rms(out, _HI, s - 0.1, 0.005) == pytest.approx(_block_rms(out, _HI, s + 0.3, 0.005), rel=0.05)


def test_the_shimmer_snap_moves_with_the_strike_times():
    a = ultrasonic_shimmer(_dc.replace(_FB, gddr_temps=(40.0,) * 8, heartbeat=0), 4.0, _HI, 0)
    b = ultrasonic_shimmer(_dc.replace(_FB, gddr_temps=(40.0,) * 8, heartbeat=8), 4.0, _HI, 0)
    assert not np.allclose(a, b)


def test_the_drone_throbs_after_a_strike_and_settles_back():
    t = tel(aiclk=800.0, heartbeat=0, pcie_usage=1)
    out = plumbutter_drone(t, 8.0, 4000, 0)
    s = strike_times(t, 8.0)[0]
    ripple = lambda start: np.abs(out[int(start * 4000): int((start + 0.9) * 4000)]).reshape(-1, 100).max(axis=1)
    assert ripple(s + 0.05).std() > 3 * ripple(s + interval_s(t) - 0.95).std()  # steady again before the next strike


def interval_s(t):
    return 2 * 4 * 60.0 / aiclk_to_bpm(t.aiclk)


def test_a_chip_with_other_strike_times_throbs_at_other_moments():
    a = plumbutter_drone(tel(heartbeat=0), 6.0, SR, 0)
    b = plumbutter_drone(tel(heartbeat=8), 6.0, SR, 0)
    assert not np.allclose(a, b)


def test_the_shock_wave_dies_to_a_third_after_one_decay_time():
    since = seconds_since(np.array([0.0]), n=401, sample_rate=100)
    w = shock_wave(since, rate_hz=0.25, decay_s=2.0)  # the swing peaks at one second, half a decay time in
    assert w[100] == pytest.approx(np.exp(-0.5), abs=1e-6)


def test_the_drone_ripples_at_one_and_a_half_undulations_a_beat_and_rings_for_two_beats(monkeypatch):
    seen = []
    real = devices_module.shock_wave
    monkeypatch.setattr(devices_module, "shock_wave", lambda since, rate_hz, decay_s: seen.append((rate_hz, decay_s)) or real(since, rate_hz, decay_s))
    t = tel(aiclk=800.0)
    plumbutter_drone(t, 0.2, SR, 0)
    beat = 60.0 / aiclk_to_bpm(800.0)
    assert seen == [pytest.approx((1.5 / beat, 2.0 * beat))]


# --- Rollz lanes are rolls on the Rolzer's barres --------------------------------------------------------------

from tensixinet.music.rolzer import barre_bpm, DDR_LANE_BARRES, ETH_LANE_BARRES, ETH_LANE_SIDES  # noqa: E402


def _lane_calls(monkeypatch, t, seed=3):
    seen = []
    real = devices_module.roll_pulse_times
    monkeypatch.setattr(devices_module, "roll_pulse_times", lambda sides, bpm, duration, s: seen.append((sides, bpm, s)) or real(sides, bpm, duration, s))
    rollz(t, 2.0, SR, seed)
    return seen


def test_each_ddr_lane_is_a_roll_on_its_own_barre_at_that_barres_tempo(monkeypatch):
    seen = _lane_calls(monkeypatch, tel(ddr_status=(1, 0, 1, 1), eth_mask=0x0000, aiclk=800.0))
    bpm = aiclk_to_bpm(800.0)
    assert [(s, b) for s, b, _ in seen] == [
        (3, pytest.approx(barre_bpm(bpm, DDR_LANE_BARRES[0]))),
        (5, pytest.approx(barre_bpm(bpm, DDR_LANE_BARRES[2]))),
        (6, pytest.approx(barre_bpm(bpm, DDR_LANE_BARRES[3]))),
    ]


def test_each_ethernet_lane_with_a_link_up_is_a_slow_roll_on_a_grave_barre(monkeypatch):
    seen = _lane_calls(monkeypatch, tel(ddr_status=(0, 0, 0, 0), eth_mask=0x00F1, aiclk=800.0))  # groups 0 and 1 have links up
    bpm = aiclk_to_bpm(800.0)
    assert [(s, b) for s, b, _ in seen] == [
        (ETH_LANE_SIDES[0], pytest.approx(barre_bpm(bpm, ETH_LANE_BARRES[0]))),
        (ETH_LANE_SIDES[1], pytest.approx(barre_bpm(bpm, ETH_LANE_BARRES[1]))),
    ]
    assert min(ETH_LANE_BARRES) > max(DDR_LANE_BARRES) - 2  # the grave end of the Rolzer


def test_no_two_lanes_share_a_seed_so_their_tolerances_differ(monkeypatch):
    seen = _lane_calls(monkeypatch, tel(ddr_status=(1, 1, 1, 1), eth_mask=0xFFFF))
    assert len(seen) == 8 and len({s for _, _, s in seen}) == 8


def test_rollz_lanes_are_not_locked_to_the_bar():
    # the old lanes fired a whole number of times per bar; a roll on a barre drifts against the bar
    t = tel(ddr_status=(0, 1, 0, 0), eth_mask=0x0000, aiclk=800.0)
    bar = 4 * 60.0 / aiclk_to_bpm(800.0)
    onsets = devices_module.roll_pulse_times(4, barre_bpm(aiclk_to_bpm(800.0), DDR_LANE_BARRES[1]), 60.0, 0)
    assert len(set(np.round((onsets % bar) / bar, 2))) > len(onsets) / 2
    assert rollz(t, 4.0, SR, 0).dtype == np.float32


# --- Cocoquantus: the Quantussy chain and the 8-bit buffer ---------------------------------------------------------------

from tensixinet.music.quantussy import petal_phases, petal_ratios  # noqa: E402


def test_the_notes_come_from_the_first_petals_castle_with_every_petal_at_its_ratio_to_the_note_rate(monkeypatch):
    seen = []
    real = devices_module.castle_output
    monkeypatch.setattr(devices_module, "castle_output", lambda petal, freqs, phases, t, **k: seen.append((petal, freqs, phases)) or real(petal, freqs, phases, t, **k))
    cocoquantus_steps(tel(aiclk=800.0, axiclk=1920.0), 1.0, SR, 7)
    rate = aiclk_to_bpm(800.0) / 60.0 * 2 * 2.0  # eighth notes a second, doubled by an AXI clock at twice the reference
    petal, freqs, phases = seen[0]
    assert petal == 0
    assert list(freqs) == pytest.approx([rate / 2 * r for r in petal_ratios(7)])
    assert list(phases) == list(petal_phases(7))


def test_a_note_is_held_for_one_eighth_when_the_castle_turns_round_twice_a_cycle():
    steps = cocoquantus_steps(tel(aiclk=800.0), 8.0, 4000, 3)
    changes = np.flatnonzero(np.diff(steps) != 0) + 1
    eighth = 30.0 / aiclk_to_bpm(800.0)
    assert np.all(np.diff(changes) / 4000 >= eighth - 2 / 4000)  # no note is shorter than an eighth
    assert len(changes) > 8


def test_the_notes_repeat_a_pattern_because_the_petal_ratios_are_low_primes():
    steps = cocoquantus_steps(tel(aiclk=800.0), 30.0, 4000, 0)
    eighth = int(30.0 / aiclk_to_bpm(800.0) * 4000)
    notes = steps[eighth // 2:: eighth]  # one reading per held note
    assert len(set(notes.tolist())) < len(notes) / 3  # a small set of notes comes round again and again


@pytest.mark.parametrize("axi, rate", [(960.0, 0.5), (1920.0, 1.0), (3000.0, 1.0), (100.0, 0.25)])
def test_a_slower_axi_clock_holds_the_buffers_samples_longer(monkeypatch, axi, rate):
    seen = []
    real = devices_module.crunch
    monkeypatch.setattr(devices_module, "crunch", lambda x, r: seen.append(r) or real(x, r))
    cocoquantus(tel(axiclk=axi), 0.1, SR, 0)
    assert seen == [pytest.approx(rate)]


def test_the_cocoquantus_voice_is_eight_bit_and_stays_under_its_level():
    out = cocoquantus(tel(axiclk=1920.0), 1.0, SR, 0)
    assert np.allclose(out * 127, np.round(out * 127), atol=1e-4) and np.abs(out).max() <= 0.8 + 1 / 127


def test_a_castle_at_the_very_top_of_its_range_plays_the_highest_note_and_does_not_wrap():
    assert quantize_to_scale(np.array([1.0]), PENTATONIC, octaves=2)[0] == 12 + PENTATONIC[-1]


def test_the_cocoquantus_castle_samples_the_petal_its_chips_identity_names(monkeypatch):
    from tensixinet.music.quantussy import castle_offset
    seen = []
    real = devices_module.castle_output
    monkeypatch.setattr(devices_module, "castle_output", lambda *a, **k: seen.append(k.get("offset")) or real(*a, **k))
    for mask in (0x3FFF, 0x1235, 0x37F7):
        cocoquantus_steps(tel(tensix_mask=mask), 0.2, SR, 0)
    from tensixinet.music.mapping import personality as person
    assert seen == [castle_offset(person(tel(tensix_mask=m))) for m in (0x3FFF, 0x1235, 0x37F7)]
    assert len(set(seen)) > 1
