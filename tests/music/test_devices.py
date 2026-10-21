# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Ciat-Lonbarde-style devices. Each is a pure function of one chip's telemetry."""
import dataclasses

import numpy as np
import pytest

from tensixinet.music.devices import (
    ttquantus,
    ttquantus_steps,
    current_mix,
    PENTATONIC,
    ttdrone,
    quantize_to_scale,
    ttrollz,
    ttrollz_pulse_times,
    ttrax,
)
from tensixinet.telemetry import FALLBACK, Telemetry

SR = 8000


def tel(**kw):
    return Telemetry(**{**FALLBACK.__dict__, **kw})


# --- ttrollz: asymmetric pulses --------------------------------------------------

def test_ttrollz_mean_interval_matches_the_bpm():
    times = ttrollz_pulse_times(bpm=120, duration=60, asymmetry=0.3)
    assert np.mean(np.diff(times)) == pytest.approx(0.5, rel=0.02)


def test_ttrollz_intervals_are_not_all_equal():
    d = np.diff(ttrollz_pulse_times(bpm=120, duration=10, asymmetry=0.3))
    assert d.max() > d.min() * 1.5


def test_ttrollz_zero_asymmetry_is_a_plain_clock():
    d = np.diff(ttrollz_pulse_times(bpm=60, duration=10, asymmetry=0.0))
    assert np.allclose(d, 1.0)


def test_ttrollz_pulses_stay_inside_the_duration():
    assert ttrollz_pulse_times(90, 5, 0.2).max() < 5


def test_ttrollz_faster_aiclk_means_more_onsets():
    slow = ttrollz(tel(aiclk=400), 4.0, SR, seed=0)
    fast = ttrollz(tel(aiclk=1350), 4.0, SR, seed=0)
    assert np.sum(np.abs(fast)) > np.sum(np.abs(slow))


def test_ttrollz_ddr_lanes_add_voices():
    none = ttrollz(tel(ddr_status=(0, 0, 0, 0)), 4.0, SR, seed=0)
    all4 = ttrollz(tel(ddr_status=(1, 1, 1, 1)), 4.0, SR, seed=0)
    assert not np.allclose(none, all4)


# --- ttquantus: sample and hold ----------------------------------------------

def test_quantize_puts_every_value_on_the_scale():
    v = np.linspace(0, 0.999, 500)
    semis = quantize_to_scale(v, PENTATONIC, octaves=2)
    allowed = {12 * o + d for o in range(2) for d in PENTATONIC}
    assert set(semis.tolist()) <= allowed
    assert set(semis.tolist()) == allowed  # and the whole range is reachable


def test_ttquantus_steps_faster_when_aiclk_is_higher():
    def changes(aiclk):
        semis = ttquantus_steps(tel(aiclk=aiclk), 8.0, SR, seed=5)
        return int(np.count_nonzero(np.diff(semis)))
    assert changes(1350) > 2 * changes(400)


def test_ttquantus_plays_audible_stepped_pitches():
    audio = ttquantus(tel(asic_temperature=25.0), 2.0, SR, seed=3)
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


def test_ttdrone_has_the_chips_pitch_voices():
    audio = ttdrone(tel(), 1.0, SR, seed=0)
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
    for dev in (ttrollz, ttrax, ttquantus, ttdrone, gong, ultrasonic_shimmer):
        assert np.array_equal(dev(_FB, 1.0, _SR, 3), dev(_FB, 1.0, _SR, 3, patch=DEFAULT_PATCH)), dev.__name__


def test_ttquantus_quantises_onto_the_patch_scale():
    patch = _dc.replace(DEFAULT_PATCH, scale_name="whole_tone", scale=SCALES["whole_tone"])
    steps = ttquantus_steps(_FB, 4.0, _SR, 0, patch=patch)
    allowed = {12 * o + d for o in range(2) for d in SCALES["whole_tone"]}
    assert set(np.unique(steps)) <= allowed and len(set(np.unique(steps))) > 3


def test_a_patch_scale_changes_ttquantus_audio():
    patch = _dc.replace(DEFAULT_PATCH, scale_name="whole_tone", scale=SCALES["whole_tone"])
    assert not np.allclose(ttquantus(_FB, 2.0, _SR, 0), ttquantus(_FB, 2.0, _SR, 0, patch=patch))


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


def _ttrollz_base(**kw):
    return tel(ddr_status=(0, 0, 0, 0), **kw)


def test_ttrollz_ethernet_links_add_lanes_and_each_link_counts():
    none = ttrollz(_ttrollz_base(eth_mask=0x0000), 4.0, SR, 0)
    full = ttrollz(_ttrollz_base(eth_mask=0xFFFF), 4.0, SR, 0)
    assert _rms(full) > _rms(none)
    assert not np.allclose(ttrollz(_ttrollz_base(eth_mask=0xFFFF), 4.0, SR, 0), ttrollz(_ttrollz_base(eth_mask=0xFFFE), 4.0, SR, 0))
    # a lane belongs to a group of four links: losing the whole top group changes it again
    assert not np.allclose(ttrollz(_ttrollz_base(eth_mask=0xFFFF), 4.0, SR, 0), ttrollz(_ttrollz_base(eth_mask=0x0FFF), 4.0, SR, 0))


def test_ttquantus_holds_each_note_for_less_time_when_axiclk_is_higher():
    def changes(axi):
        return int(np.sum(np.diff(ttquantus_steps(tel(axiclk=axi), 8.0, SR, 3)) != 0))
    assert changes(1920.0) > 1.5 * changes(960.0)


def _dominant_hz(x, sr, lo=40.0, hi=400.0):
    spec = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(len(x), 1 / sr)
    band = (freqs >= lo) & (freqs <= hi)
    return float(freqs[band][np.argmax(spec[band])])


def _band_centroid_hz(x, sr, lo=40.0, hi=400.0):
    spec = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(len(x), 1 / sr)
    band = (freqs >= lo) & (freqs <= hi)
    return float((freqs[band] * spec[band]).sum() / spec[band].sum())


def test_a_higher_core_voltage_detunes_the_drone_upward():
    """Measured as the centre of the drone's spectrum, not its single tallest peak: the ripple after each strike makes the
    chord's voices swap places as the tallest, so the tallest peak can jump a whole voice either way."""
    low = ttdrone(tel(voltage=0.71), 8.0, SR, 0)
    high = ttdrone(tel(voltage=0.90), 8.0, SR, 0)
    assert _band_centroid_hz(high, SR) > _band_centroid_hz(low, SR)


def test_the_drone_asks_for_as_many_voices_as_the_pcie_usage_gives(monkeypatch):
    seen = []
    real = devices_module.build_voices
    monkeypatch.setattr(devices_module, "build_voices", lambda *a, **k: seen.append(k.get("voices_per_chip")) or real(*a, **k))
    ttdrone(tel(pcie_usage=4), 0.2, SR, 0)
    ttdrone(tel(pcie_usage=1), 0.2, SR, 0)
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
    on_the_beat = _first_sound(gong_dev(_chip_whose_gong_is_on("clock", heartbeat=0), 20.0, SR, 0), SR)
    late = _first_sound(gong_dev(_chip_whose_gong_is_on("clock", heartbeat=8), 20.0, SR, 0), SR)
    interval = 2 * 4 * 60.0 / aiclk_to_bpm(800.0)  # two bars at the chip's tempo
    assert on_the_beat < 0.05 and late == pytest.approx(interval / 4, rel=0.1)


def test_a_gong_on_a_roll_first_strikes_on_the_first_pulse_of_that_roll_after_the_heartbeats_start():
    from tensixinet.music.mapping import heartbeat_phase
    t = _chip_whose_gong_is_on("roll", heartbeat=8)
    mine = roll_pulses(t, 30.0, DEFAULT_PATCH)[gong_trigger_roll(t, DEFAULT_PATCH)][0]
    start = heartbeat_phase(8) * (2 * 4 * 60.0 / aiclk_to_bpm(800.0)) / 2
    assert strike_times(t, 30.0)[0] == pytest.approx(mine[mine >= start][0])


# --- the Gongue's trigger response: a blocking window over a clock (Plumbutter guide, Gongues [2]) ----------

from tensixinet.music.devices import gate_triggers, gong_trigger_roll, roll_pulses, strike_times  # noqa: E402
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


def _chip_whose_gong_is_on(kind, **kw):
    """A fully wired chip whose gong is patched to the clock ('clock') or to one of its rolls ('roll'). `kw` overrides the
    chip's fields; the choice depends only on the identity and the patch, so a heartbeat or tempo does not change it."""
    for mask in range(0x3000, 0x3400):
        fields = dict(tensix_mask=mask, ddr_status=(1, 1, 1, 1), eth_mask=0xFFFF, heartbeat=0, aiclk=800.0)
        fields.update(kw)
        t = tel(**fields)
        if (gong_trigger_roll(t, DEFAULT_PATCH) is None) == (kind == "clock"):
            return t
    raise AssertionError(f"no chip found whose gong is on a {kind}")


def test_a_gong_patched_to_the_clock_strikes_on_the_chips_own_clock():
    t = _chip_whose_gong_is_on("clock")
    clock = ttrollz_pulse_times(aiclk_to_bpm(800.0), 30.0)
    strikes = strike_times(t, 30.0)
    assert len(strikes) >= 2 and all(np.isclose(clock, s).any() for s in strikes)


def test_a_gong_patched_to_a_roll_strikes_on_that_rolls_pulses_and_not_on_the_clock():
    t = _chip_whose_gong_is_on("roll")
    which = gong_trigger_roll(t, DEFAULT_PATCH)
    mine = roll_pulses(t, 30.0, DEFAULT_PATCH)[which][0]
    clock = ttrollz_pulse_times(aiclk_to_bpm(800.0), 30.0)
    strikes = strike_times(t, 30.0)
    assert len(strikes) >= 2 and all(np.isclose(mine, s).any() for s in strikes)
    assert not all(np.isclose(clock, s).any() for s in strikes)     # the roll wanders off the clock's grid
    assert np.diff(strikes).min() >= 2 * 4 * 60.0 / aiclk_to_bpm(800.0) - 1e-9  # the blocking window still holds


def test_a_chip_with_no_rolls_up_has_its_gong_on_the_clock():
    t = tel(ddr_status=(0, 0, 0, 0), eth_mask=0x0000, aiclk=800.0)
    assert gong_trigger_roll(t, DEFAULT_PATCH) is None
    clock = ttrollz_pulse_times(aiclk_to_bpm(800.0), 30.0)
    assert all(np.isclose(clock, s).any() for s in strike_times(t, 30.0))


def test_the_roll_a_gong_follows_is_the_roll_the_next_segment_follows_with_new_drift():
    """Every device reads its segment out of its seed, so a chip's gong, shimmer, drone and AVDog and its ttrollz agree."""
    from tensixinet.music.ttrolzer import SEGMENT_SEED_STRIDE
    t = _chip_whose_gong_is_on("roll")
    seg0, seg1 = strike_times(t, 30.0, DEFAULT_PATCH, segment=0), strike_times(t, 30.0, DEFAULT_PATCH, segment=1)
    assert not np.array_equal(seg0, seg1)                           # new segment, new wander
    assert np.array_equal(seg0, strike_times(t, 30.0, DEFAULT_PATCH))
    assert not np.array_equal(gong(t, 12.0, SR, 3, DEFAULT_PATCH), gong(t, 12.0, SR, 3 + SEGMENT_SEED_STRIDE, DEFAULT_PATCH))
    # a seed within one segment (the slot added to the segment's base) does not change the strikes
    assert np.array_equal(gong(t, 12.0, SR, 3, DEFAULT_PATCH), gong(t, 12.0, SR, 5, DEFAULT_PATCH))


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
    out = ttdrone(t, 8.0, 4000, 0)
    s = strike_times(t, 8.0)[0]
    ripple = lambda start: np.abs(out[int(start * 4000): int((start + 0.9) * 4000)]).reshape(-1, 100).max(axis=1)
    assert ripple(s + 0.05).std() > 3 * ripple(s + interval_s(t) - 0.95).std()  # steady again before the next strike


def interval_s(t):
    return 2 * 4 * 60.0 / aiclk_to_bpm(t.aiclk)


def test_a_chip_with_other_strike_times_throbs_at_other_moments():
    a = ttdrone(tel(heartbeat=0), 6.0, SR, 0)
    b = ttdrone(tel(heartbeat=8), 6.0, SR, 0)
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
    ttdrone(t, 0.2, SR, 0)
    beat = 60.0 / aiclk_to_bpm(800.0)
    assert seen == [pytest.approx((1.5 / beat, 2.0 * beat))]


# --- ttrollz lanes are rolls on the ttrolzer's barres --------------------------------------------------------------

from tensixinet.music.ttrolzer import (  # noqa: E402
    DDR_LANE_BARRES, DOWNBEAT_GAIN, ETH_LANE_BARRES, ETH_LANE_SIDES, barre_bpm, draw_links, roll_drift, roll_ensemble,
)
from tensixinet.music.devices import ttrollz_lanes, ttrollz_network, ttrollz_pulse_times  # noqa: E402
from tensixinet.music.mapping import chip_pressure, personality, temp_to_pitch_hz  # noqa: E402
from tensixinet.music.patch import DEFAULT_PATCH  # noqa: E402


def _lanes(**kw):
    return [(l.roll.sides, l.roll.bpm) for l in ttrollz_lanes(tel(**kw))]


def test_each_ddr_lane_is_a_roll_on_its_own_barre_at_that_barres_tempo():
    bpm = aiclk_to_bpm(800.0)
    assert _lanes(ddr_status=(1, 0, 1, 1), eth_mask=0x0000, aiclk=800.0) == [
        (3, pytest.approx(barre_bpm(bpm, DDR_LANE_BARRES[0]))),
        (5, pytest.approx(barre_bpm(bpm, DDR_LANE_BARRES[2]))),
        (6, pytest.approx(barre_bpm(bpm, DDR_LANE_BARRES[3]))),
    ]


def test_each_ethernet_lane_with_a_link_up_is_a_slow_roll_on_a_grave_barre():
    bpm = aiclk_to_bpm(800.0)
    assert _lanes(ddr_status=(0, 0, 0, 0), eth_mask=0x00F1, aiclk=800.0) == [  # groups 0 and 1 have links up
        (ETH_LANE_SIDES[0], pytest.approx(barre_bpm(bpm, ETH_LANE_BARRES[0]))),
        (ETH_LANE_SIDES[1], pytest.approx(barre_bpm(bpm, ETH_LANE_BARRES[1]))),
    ]
    assert min(ETH_LANE_BARRES) > max(DDR_LANE_BARRES) - 2  # the grave end of the ttrolzer


def test_a_chip_with_everything_up_plays_eight_lanes_ddr_first_and_their_levels_follow_the_links():
    lanes = ttrollz_lanes(tel(ddr_status=(1, 1, 1, 1), eth_mask=0xFFFF))
    assert len(lanes) == 8 and [l.roll.sides for l in lanes[:4]] == [3, 4, 5, 6]
    assert {l.level for l in lanes[:4]} == {0.5} and {l.level for l in lanes[4:]} == {0.35}
    assert {l.decay_s for l in lanes[:4]} == {0.05} and {l.decay_s for l in lanes[4:]} == {0.04}  # DDR blips ring a little longer
    half = ttrollz_lanes(tel(ddr_status=(0, 0, 0, 0), eth_mask=0x0003))  # two of four links in the first group
    assert half[0].level == pytest.approx(0.35 * 0.5)


def test_each_lane_is_a_barre_of_the_rolzers_rolls_and_a_chip_with_everything_up_has_forty():
    from tensixinet.music.ttrolzer import BARRE_GEOMETRY
    net = ttrollz_network(tel(ddr_status=(1, 1, 1, 1), eth_mask=0xFFFF, aiclk=800.0))
    assert len(net.rolls) == 40 and len(net.lanes) == 8
    assert [r.sides for r in net.rolls[:5]] == list(BARRE_GEOMETRY)
    for lane in net.lanes:   # a lane's own roll is the member of its barre with its sides, and it is the one that is heard
        assert net.rolls[lane.index].sides == lane.roll.sides and net.rolls[lane.index].bpm == pytest.approx(lane.roll.bpm)
    # a barre's rolls all run at the barre's tempo, and its second hexagon pulses out of phase with the first
    first_barre = net.rolls[:5]
    assert {round(r.bpm, 6) for r in first_barre} == {round(first_barre[0].bpm, 6)}
    assert [r.phase for r in first_barre] == [0.0, 0.0, 0.0, 0.0, 0.5]


def test_an_ethernet_lane_brings_as_many_of_its_barres_rolls_as_its_links_but_always_its_own():
    net = ttrollz_network(tel(ddr_status=(0, 0, 0, 0), eth_mask=0x0001, aiclk=800.0))   # one link of group 0: a quarter
    assert len(net.lanes) == 1 and net.lanes[0].roll.sides == ETH_LANE_SIDES[0]
    assert len(net.rolls) == 4 and net.lanes[0].index == 3          # the 6-roll is the fourth of 3, 4, 5, 6, 6
    full = ttrollz_network(tel(ddr_status=(0, 0, 0, 0), eth_mask=0x000F, aiclk=800.0))
    assert len(full.rolls) == 5
    last = ttrollz_network(tel(ddr_status=(0, 0, 0, 0), eth_mask=0x1000, aiclk=800.0))  # group 3's own roll is the 3-roll
    assert len(last.rolls) == 2 and last.lanes[0].index == 0 and last.lanes[0].roll.sides == 3


def test_grounded_ddr_lanes_leave_no_rolls_and_no_network():
    net = ttrollz_network(tel(ddr_status=(0, 0, 0, 0), eth_mask=0x0000))
    assert net.rolls == [] and net.lanes == []


def test_ttrollz_is_its_lanes_blips_with_the_first_step_of_each_cycle_accented_an_octave_up():
    """Rebuilds the output from the public pieces, so a wrong link, drift, pitch, level or accent changes it."""
    t = tel(ddr_status=(1, 1, 0, 1), eth_mask=0x00F3, aiclk=800.0, asic_temperature=55.0, heartbeat=0)
    patch = dataclasses.replace(DEFAULT_PATCH, seed=9)
    n = int(6.0 * SR)
    net = ttrollz_network(t)
    links = draw_links(len(net.rolls), 9, personality(t), chip_pressure(t), [lane.index for lane in net.lanes])
    pulses = roll_ensemble(net.rolls, links, 6.0, 9 * 7919 + 4, drift=roll_drift(55.0), cap_seed=9 * 1_000_003 + personality(t))
    pitch = temp_to_pitch_hz(55.0)
    expected = devices_module._blips(ttrollz_pulse_times(aiclk_to_bpm(800.0), 6.0), pitch, n, SR)
    for lane in net.lanes:
        times, downbeats = pulses[lane.index]
        expected += lane.level * devices_module._blips(times, pitch * lane.pitch, n, SR, lane.decay_s)
        expected += DOWNBEAT_GAIN * lane.level * devices_module._blips(times[downbeats], pitch * lane.pitch * 2, n, SR, lane.decay_s)
    assert np.allclose(ttrollz(t, 6.0, SR, 4 * 10_007, patch), np.tanh(expected), atol=1e-6)
    # the roll_pulses the gong reads are those very pulses
    assert all(np.array_equal(a[0], b[0]) for a, b in zip(roll_pulses(t, 6.0, patch, 4), pulses))


def test_a_chips_capacitors_are_its_own_and_do_not_change_from_one_segment_to_the_next():
    """Live mode re-renders with a new seed each segment; the capacitor mismatch comes from the patch and the chip, so
    the roll keeps its trot, while the drift is free to wander differently."""
    t = tel(ddr_status=(0, 1, 0, 0), eth_mask=0x0000, aiclk=800.0)
    steps = []
    for seed in (1, 2, 3):
        lane = ttrollz_lanes(t)[0].roll
        times, _ = roll_ensemble([lane], [], 40.0, seed, drift=0.0, jitter=0.0, cap_seed=5)[0]
        steps.append(np.diff(times)[:8])
    assert np.allclose(steps[0], steps[1]) and np.allclose(steps[1], steps[2])


def test_ttrollz_lanes_are_not_locked_to_the_bar():
    # the old lanes fired a whole number of times per bar; a roll on a barre drifts against the bar
    t = tel(ddr_status=(0, 1, 0, 0), eth_mask=0x0000, aiclk=800.0)
    bar = 4 * 60.0 / aiclk_to_bpm(800.0)
    onsets = roll_ensemble([ttrollz_lanes(t)[0].roll], [], 60.0, 0)[0][0]
    assert len(set(np.round((onsets % bar) / bar, 2))) > 20  # a lane locked to the bar would land on a handful of positions
    assert ttrollz(t, 4.0, SR, 0).dtype == np.float32


def test_the_chips_pressure_and_wiring_reach_the_rolls():
    base = tel(ddr_status=(1, 1, 1, 1), eth_mask=0xFFFF, aiclk=800.0)
    assert not np.allclose(ttrollz(base, 8.0, SR, 0), ttrollz(dataclasses.replace(base, power=125.0), 8.0, SR, 0))      # pressure: link strength
    assert not np.allclose(ttrollz(base, 8.0, SR, 0), ttrollz(dataclasses.replace(base, asic_temperature=90.0), 8.0, SR, 0))  # temperature: drift
    assert not np.allclose(ttrollz(base, 8.0, SR, 0), ttrollz(dataclasses.replace(base, tensix_mask=0x1235), 8.0, SR, 0))     # identity: wiring
    other = dataclasses.replace(DEFAULT_PATCH, seed=77)
    assert not np.allclose(ttrollz(base, 8.0, SR, 0), ttrollz(base, 8.0, SR, 0, other))                                      # patch: wiring


# --- ttquantus: the ttquantussy chain and the 8-bit buffer ---------------------------------------------------------------

from tensixinet.music.ttquantussy import petal_phases, petal_ratios  # noqa: E402


def test_the_notes_come_from_the_first_petals_castle_with_every_petal_at_its_ratio_to_the_note_rate(monkeypatch):
    seen = []
    real = devices_module.castle_output
    monkeypatch.setattr(devices_module, "castle_output", lambda petal, freqs, phases, t, **k: seen.append((petal, freqs, phases)) or real(petal, freqs, phases, t, **k))
    ttquantus_steps(tel(aiclk=800.0, axiclk=1920.0), 1.0, SR, 7)
    rate = aiclk_to_bpm(800.0) / 60.0 * 2 * 2.0  # eighth notes a second, doubled by an AXI clock at twice the reference
    petal, freqs, phases = seen[0]
    assert petal == 0
    assert list(freqs) == pytest.approx([rate / 2 * r for r in petal_ratios(7)])
    assert list(phases) == list(petal_phases(7))


def test_a_note_is_held_for_one_eighth_when_the_castle_turns_round_twice_a_cycle():
    steps = ttquantus_steps(tel(aiclk=800.0), 8.0, 4000, 3)
    changes = np.flatnonzero(np.diff(steps) != 0) + 1
    eighth = 30.0 / aiclk_to_bpm(800.0)
    assert np.all(np.diff(changes) / 4000 >= eighth - 2 / 4000)  # no note is shorter than an eighth
    assert len(changes) > 8


def test_the_notes_repeat_a_pattern_because_the_petal_ratios_are_low_primes():
    steps = ttquantus_steps(tel(aiclk=800.0), 30.0, 4000, 0)
    eighth = int(30.0 / aiclk_to_bpm(800.0) * 4000)
    notes = steps[eighth // 2:: eighth]  # one reading per held note
    assert len(set(notes.tolist())) < len(notes) / 3  # a small set of notes comes round again and again


@pytest.mark.parametrize("axi, rate", [(960.0, 0.5), (1920.0, 1.0), (3000.0, 1.0), (100.0, 0.25)])
def test_a_slower_axi_clock_holds_the_buffers_samples_longer(monkeypatch, axi, rate):
    seen = []
    real = devices_module.crunch
    monkeypatch.setattr(devices_module, "crunch", lambda x, r: seen.append(r) or real(x, r))
    ttquantus(tel(axiclk=axi), 0.1, SR, 0)
    assert seen == [pytest.approx(rate)]


def test_the_ttquantus_voice_is_eight_bit_and_stays_under_its_level():
    out = ttquantus(tel(axiclk=1920.0), 1.0, SR, 0)
    assert np.allclose(out * 127, np.round(out * 127), atol=1e-4) and np.abs(out).max() <= 0.8 + 1 / 127


def test_a_castle_at_the_very_top_of_its_range_plays_the_highest_note_and_does_not_wrap():
    assert quantize_to_scale(np.array([1.0]), PENTATONIC, octaves=2)[0] == 12 + PENTATONIC[-1]


def test_the_ttquantus_castle_samples_the_petal_its_chips_identity_names(monkeypatch):
    from tensixinet.music.ttquantussy import castle_offset
    seen = []
    real = devices_module.castle_output
    monkeypatch.setattr(devices_module, "castle_output", lambda *a, **k: seen.append(k.get("offset")) or real(*a, **k))
    for mask in (0x3FFF, 0x1235, 0x37F7):
        ttquantus_steps(tel(tensix_mask=mask), 0.2, SR, 0)
    from tensixinet.music.mapping import personality as person
    assert seen == [castle_offset(person(tel(tensix_mask=m))) for m in (0x3FFF, 0x1235, 0x37F7)]
    assert len(set(seen)) > 1


# --- the shimmer's parameters, shared with the kernel's host side ---------------------------------------------------

def test_shimmer_parameters_are_the_two_tones_and_the_level_the_device_plays():
    import dataclasses
    from tensixinet.music.devices import shimmer_parameters
    from tensixinet.music.mapping import chip_pressure
    from tensixinet.telemetry import FALLBACK
    t = dataclasses.replace(FALLBACK, asic_temperature=70.0, gddr_temps=(30.0, 40.0) * 4, power=100.0)
    f1, f2, level = shimmer_parameters(t, 44100, DEFAULT_PATCH)
    lo, hi = DEFAULT_PATCH.shimmer_band
    assert lo <= f1 <= hi and f2 == pytest.approx(f1 + 0.2 * 10.0)  # a 10 C GDDR spread beats at 2 Hz
    assert level == pytest.approx(0.5 * (1 + chip_pressure(t)))
    f1_low, _, _ = shimmer_parameters(t, 8000, DEFAULT_PATCH)
    assert f1_low <= 0.45 * 8000  # the band shrinks at a low sample rate


def test_gong_parameters_are_the_numbers_the_strike_is_built_from():
    import dataclasses
    from tensixinet.music.devices import gong_parameters
    from tensixinet.music.mapping import aiclk_to_bpm
    t = dataclasses.replace(FALLBACK, aiclk=800.0)
    g = gong_parameters(t, 44100, DEFAULT_PATCH)
    interval = 2 * 4 * 60.0 / aiclk_to_bpm(800.0)
    assert g.length == int(2.5 * interval * 44100) and g.beat_hz == DEFAULT_PATCH.gong_beat_hz
    assert len(g.partials) == len(DEFAULT_PATCH.gong_partials)
    freq, decay, amp = g.partials[1]
    assert freq == pytest.approx(g.f0 * DEFAULT_PATCH.gong_partials[1]) and amp == DEFAULT_PATCH.gong_amps[1]
    assert decay == pytest.approx(0.3 * interval / (1 + 0.25 * (DEFAULT_PATCH.gong_partials[1] - 1)))
    assert gong_parameters(t, 4000, DEFAULT_PATCH).partials[-1][0] <= 0.45 * 4000  # partials above Nyquist are held at 0.45 of it


def test_quantussy_tone_is_the_plain_sine_before_the_cocos_crunch(monkeypatch):
    """A regression lock for the ttcoco refactor: ttquantus is exactly crunch(quantussy_tone(...), rate)."""
    from tensixinet.music.devices import quantussy_tone
    t = tel(axiclk=1920.0)
    tone = quantussy_tone(t, 1.0, SR, 0)
    assert np.abs(tone).max() <= 0.8 + 1e-9
    expected = devices_module.crunch(tone, 1.0)
    assert np.array_equal(ttquantus(t, 1.0, SR, 0), expected.astype(np.float32))


def test_every_device_that_follows_the_gongs_strikes_moves_with_the_segment_and_not_with_its_slot():
    """The gong, the shimmer's snap, the drone's ripple and the AVDog all read the segment out of their seed, so they follow
    the same rolls the gong does; each slot of a segment adds a small number to that seed, which must change nothing."""
    from tensixinet.music.ttrolzer import SEGMENT_SEED_STRIDE
    from tensixinet.music.devices import ttdrone, ultrasonic_shimmer
    from tensixinet.music.ttavdog import ttavdog
    t = _chip_whose_gong_is_on("roll")
    for device in (gong_dev, ultrasonic_shimmer, ttdrone, ttavdog):
        a = device(t, 20.0, SR, 0, DEFAULT_PATCH)
        assert np.array_equal(a, device(t, 20.0, SR, 7, DEFAULT_PATCH)), device.__name__
        assert not np.array_equal(a, device(t, 20.0, SR, SEGMENT_SEED_STRIDE, DEFAULT_PATCH)), device.__name__


# --- rests: the castle that decides which held notes are not played ---------------------------------------------

def _rested(rests):
    return dataclasses.replace(DEFAULT_PATCH, rests=rests)


def _open_share(rests, seeds=range(12), duration=20.0, **kw):
    from tensixinet.music.devices import ttquantus_gate
    return np.mean([np.mean(ttquantus_gate(tel(**kw), duration, SR, s, _rested(rests)) > 0.5) for s in seeds])


def test_a_patch_with_no_rests_has_every_note_played_so_the_sound_is_as_it_was():
    from tensixinet.music.devices import quantussy_tone, ttquantus_gate
    gate = ttquantus_gate(tel(), 4.0, SR, 0, DEFAULT_PATCH)
    assert DEFAULT_PATCH.rests == 0.0 and gate.shape == (4 * SR,) and np.all(gate == 1.0)
    assert np.array_equal(quantussy_tone(tel(), 4.0, SR, 0, DEFAULT_PATCH), quantussy_tone(tel(), 4.0, SR, 0, _rested(0.0)))


def test_the_share_of_held_notes_that_rest_is_about_the_patchs_rests():
    for rests in (0.15, 0.3, 0.45):
        assert _open_share(rests) == pytest.approx(1.0 - rests, abs=0.1), rests


def test_a_rest_is_true_silence_and_a_played_note_is_untouched():
    from tensixinet.music.devices import quantussy_tone, ttquantus_gate
    p = _rested(0.4)
    tone, gate, plain = quantussy_tone(tel(), 10.0, SR, 3, p), ttquantus_gate(tel(), 10.0, SR, 3, p), quantussy_tone(tel(), 10.0, SR, 3, DEFAULT_PATCH)
    shut, open_ = gate == 0.0, gate == 1.0
    assert shut.sum() > 0.1 * len(gate) and open_.sum() > 0.3 * len(gate)
    assert not np.any(tone[shut]) and np.array_equal(tone[open_], plain[open_])


def test_the_gate_changes_only_between_notes_and_glides_so_a_rest_does_not_click():
    from tensixinet.music.devices import REST_FADE_S, ttquantus_gate
    gate = ttquantus_gate(tel(), 20.0, SR, 3, _rested(0.4))
    assert REST_FADE_S == 0.005
    assert np.abs(np.diff(gate)).max() < 2.0 / (REST_FADE_S * SR)             # a fade over about 5 ms, not a jump
    assert np.any((gate > 0.05) & (gate < 0.95))
    steps = np.count_nonzero(np.diff(ttquantus_steps(tel(), 20.0, SR, 3)) != 0)
    edges = np.count_nonzero(np.diff((gate > 0.5).astype(int)) != 0)
    assert 0 < edges <= steps + 1                                              # a note's own steps, and no more often


def test_the_rest_pattern_is_a_castle_so_it_repeats_for_a_seed_and_drifts_between_seeds():
    from tensixinet.music.devices import ttquantus_gate
    p = _rested(0.4)
    g = ttquantus_gate(tel(), 12.0, SR, 3, p)
    assert np.array_equal(g, ttquantus_gate(tel(), 12.0, SR, 3, p))
    assert not np.array_equal(g, ttquantus_gate(tel(), 12.0, SR, 4, p))


def test_the_rest_castle_uses_the_chips_rest_leader_and_the_note_castle_keeps_its_own(monkeypatch):
    from tensixinet.music.devices import ttquantus_gate
    from tensixinet.music.mapping import personality as person
    from tensixinet.music.ttquantussy import castle_offset, rest_offset
    seen = []
    real = devices_module.castle_output
    args = []
    monkeypatch.setattr(devices_module, "castle_output", lambda *a, **k: (seen.append(k.get("offset")), args.append(a[:3])) and real(*a, **k))
    t = tel(tensix_mask=0x37F7)
    ttquantus_gate(t, 0.5, SR, 0, _rested(0.3))
    ttquantus_steps(t, 0.5, SR, 0)
    assert seen == [rest_offset(person(t)), castle_offset(person(t))]
    assert args[0][0] == args[1][0] == 0 and list(args[0][1]) == list(args[1][1]) and list(args[0][2]) == list(args[1][2])   # the same petals, the same phases
    seen.clear()
    ttquantus_gate(t, 0.5, SR, 0, DEFAULT_PATCH)
    assert seen == []                                                          # no rests: no extra castle


def test_more_rests_make_ttquantus_quieter_and_the_voice_stays_under_its_level():
    rms = lambda r: float(np.sqrt(np.mean(ttquantus(tel(), 12.0, SR, 2, patch=_rested(r)) ** 2)))
    assert rms(0.0) > rms(0.2) > rms(0.45) > 0.0
    assert np.abs(ttquantus(tel(), 4.0, SR, 2, patch=_rested(0.45))).max() <= 0.8 + 1 / 127   # the 8-bit buffer rounds to the nearest level


def test_ttcoco_echoes_the_rests_too_so_its_echoes_carry_the_gaps_and_it_is_quieter():
    """The echoes fill some of a rest, as a delay does, so the gaps are softer than in ttquantus; the delay still has less to repeat."""
    from tensixinet.music.ttcoco import ttcoco
    rms = lambda p: float(np.sqrt(np.mean(ttcoco(tel(), 12.0, SR, 2, p) ** 2)))
    assert rms(_rested(0.45)) < 0.92 * rms(DEFAULT_PATCH) and rms(_rested(0.3)) < rms(DEFAULT_PATCH)
    assert not np.array_equal(ttcoco(tel(), 4.0, SR, 2, _rested(0.45)), ttcoco(tel(), 4.0, SR, 2, DEFAULT_PATCH))


def test_a_rest_begins_and_ends_on_the_notes_own_boundaries_with_the_glide_centred_on_them():
    from tensixinet.music.devices import REFERENCE_AXICLK, ttquantus_gate
    from tensixinet.music.mapping import aiclk_to_bpm
    t = tel()
    steps_per_s = aiclk_to_bpm(t.aiclk) / 60.0 * 2 * t.axiclk / REFERENCE_AXICLK      # petal 0 turns half a cycle for every note
    gate = ttquantus_gate(t, 20.0, SR, 3, _rested(0.4))
    crossings = np.nonzero(np.diff((gate > 0.5).astype(int)))[0]
    edges = np.round(np.arange(0, 20.0 * steps_per_s + 1) * SR / steps_per_s)
    assert len(crossings) > 5
    assert all(np.abs(edges - c).min() <= 2 for c in crossings)
