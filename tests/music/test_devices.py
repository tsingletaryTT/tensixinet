"""Ciat-Lonbarde-style devices. Each is a pure function of one chip's telemetry."""
import numpy as np
import pytest

from tensixinet.music.devices import (
    bounce_wave,
    cocoquantus,
    cocoquantus_steps,
    current_mix,
    PENTATONIC,
    plumbutter_drone,
    quantize_to_scale,
    rollz,
    rollz_pulse_times,
    sample_hold,
    sidrax,
    sidrax_bounds,
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


# --- Sidrax: bounds-modulated oscillator ---------------------------------------

def test_bounce_wave_reaches_but_never_leaves_its_bounds():
    w = bounce_wave(freq_hz=50, lower=-0.2, upper=0.7, n=SR, sample_rate=SR)
    assert w.min() == pytest.approx(-0.2, abs=0.01)
    assert w.max() == pytest.approx(0.7, abs=0.01)
    assert w.min() >= -0.2 - 1e-9 and w.max() <= 0.7 + 1e-9


def test_bounce_wave_period_follows_frequency():
    w = bounce_wave(freq_hz=100, lower=-1, upper=1, n=SR, sample_rate=SR)
    spec = np.abs(np.fft.rfft(w - w.mean()))
    assert np.argmax(spec) * SR / len(w) == pytest.approx(100.0, abs=1.0)


def test_sidrax_wider_gddr_spread_gives_wider_swing():
    narrow = sidrax(tel(gddr_temps=(50.0,) * 8), 2.0, SR, seed=0)
    wide = sidrax(tel(gddr_temps=(26.0, 99.0) * 4), 2.0, SR, seed=0)
    assert np.ptp(wide) > np.ptp(narrow)


def test_sidrax_lower_bound_follows_coolest_channel_and_upper_the_hottest():
    lo_a, hi_a = sidrax_bounds((26.0, 60.0, 99.0))
    lo_b, hi_b = sidrax_bounds((60.0, 60.0, 99.0))  # cooler channel warmed up
    lo_c, hi_c = sidrax_bounds((26.0, 60.0, 60.0))  # hottest channel cooled down
    assert lo_b < lo_a and hi_b == hi_a  # warmer minimum -> deeper negative bound
    assert hi_c < hi_a and lo_c == lo_a  # cooler maximum -> lower positive bound
    assert lo_a < 0 < hi_a


# --- Cocoquantus: sample and hold ----------------------------------------------

def test_sample_hold_holds_each_value_for_one_step():
    v = sample_hold(n=SR, sample_rate=SR, rate_hz=4, rng=np.random.default_rng(0))
    assert len(np.unique(v)) == 4


def test_sample_hold_is_deterministic_per_seed_and_differs_across_seeds():
    a = sample_hold(SR, SR, 8, np.random.default_rng(1))
    b = sample_hold(SR, SR, 8, np.random.default_rng(1))
    c = sample_hold(SR, SR, 8, np.random.default_rng(2))
    assert np.array_equal(a, b) and not np.array_equal(a, c)


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
