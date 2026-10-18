# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The EQ on the final mix: quiet lows lifted, subsonic rumble and very high shimmer taken down."""
import numpy as np
import pytest

from tensixinet.music.eq import EQ_POINTS, EQ_TAPS, Equaliser, apply_eq, design_fir

SR = 44100


def response_db(sample_rate, freqs):
    """The filter's gain in dB at each frequency, measured from its impulse response."""
    h = design_fir(sample_rate)
    n = 1 << 17
    spectrum = np.abs(np.fft.rfft(h, n))
    grid = np.fft.rfftfreq(n, 1 / sample_rate)
    return 20 * np.log10(np.interp(freqs, grid, spectrum))


def tone_gain_db(freq, sample_rate=SR, seconds=2.0):
    """The gain a steady sine gets, measured on its second half so the filter has settled."""
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    x = (0.1 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    y = apply_eq(x, sample_rate)
    half = len(x) // 2
    rms = lambda v: float(np.sqrt(np.mean(np.square(v[half:], dtype=np.float64))))
    return 20 * np.log10(rms(y) / rms(x))


# --- the curve ---------------------------------------------------------------------------------------------------

def test_the_curve_is_flat_through_the_middle_where_the_ear_is_most_at_home():
    assert response_db(SR, [500, 1000, 2000]) == pytest.approx([0, 0, 0], abs=1.0)


def test_the_lows_are_lifted_where_small_speakers_and_the_ear_lose_them():
    low = response_db(SR, [70, 100, 160])
    assert all(2.0 <= g <= 6 for g in low)


def test_subsonic_rumble_is_taken_out():
    sub = response_db(SR, [20, 30])
    assert sub[0] < -15 and sub[1] < -8


def test_the_shimmer_band_is_brought_down_and_the_top_of_it_most_of_all():
    top = response_db(SR, [9000, 12000, 16000, 20000])
    assert top[0] > -6 and top[1] < -5 and top[2] < -15 and top[3] < -25
    assert list(top) == sorted(top, reverse=True)  # it only falls from 9 kHz up


def test_the_points_are_in_hertz_and_decibels_in_rising_order_of_frequency():
    freqs = [f for f, _ in EQ_POINTS]
    assert freqs == sorted(freqs) and len(set(freqs)) == len(freqs) and freqs[0] <= 20 and freqs[-1] >= 20000
    assert all(-60 <= g <= 6 for _, g in EQ_POINTS)  # nothing is boosted hard


@pytest.mark.parametrize("rate", [8000, 22050, 44100, 48000, 96000])
def test_every_sample_rate_gets_a_finite_filter_with_the_same_midrange(rate):
    h = design_fir(rate)
    assert len(h) == EQ_TAPS and np.isfinite(h).all()
    assert response_db(rate, [1000]) == pytest.approx([0.0], abs=1.0)


def test_the_filter_starts_at_once_with_no_pre_ringing():
    h = design_fir(SR)
    assert np.sum(h[: EQ_TAPS // 8] ** 2) > 0.95 * np.sum(h ** 2)  # minimum phase: the energy comes first
    assert abs(h[-1]) < 1e-6 * np.max(np.abs(h))  # and the tail has faded out


# --- applying it -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("freq,low,high", [(1000, -0.7, 0.7), (100, 2.5, 6.0), (15000, -40, -10), (25, -60, -8)])
def test_tones_come_out_at_the_gain_the_curve_gives_them(freq, low, high):
    assert low <= tone_gain_db(freq) <= high


def test_the_output_is_as_long_as_the_input_and_keeps_its_type():
    x = np.random.default_rng(0).standard_normal(12345).astype(np.float32) * 0.1
    y = apply_eq(x, SR)
    assert y.shape == x.shape and y.dtype == np.float32 and np.isfinite(y).all()


def test_silence_stays_silent_and_a_dc_offset_is_removed():
    assert not apply_eq(np.zeros(5000, dtype=np.float32), SR).any()
    y = apply_eq(np.full(SR * 2, 0.2, dtype=np.float32), SR)
    assert abs(float(np.mean(y[-SR:]))) < 0.01


def test_an_empty_array_is_fine():
    assert len(apply_eq(np.zeros(0, dtype=np.float32), SR)) == 0


# --- streaming: the same filter in pieces -----------------------------------------------------------------------

def test_processing_in_chunks_gives_the_same_audio_as_all_at_once():
    x = (np.random.default_rng(1).standard_normal(30000) * 0.1).astype(np.float32)
    whole = Equaliser(SR)
    expected = np.concatenate([whole.process(x), whole.flush()])
    pieces = Equaliser(SR)
    out, i = [], 0
    for size in (1, 777, 4096, 1000, 9999, 14127):
        out.append(pieces.process(x[i:i + size]))
        i += size
    assert i == len(x)
    got = np.concatenate(out + [pieces.flush()])
    assert got.shape == expected.shape and np.allclose(got, expected, atol=1e-5)


def test_flush_gives_the_ring_out_once_then_nothing():
    eq = Equaliser(SR)
    eq.process(np.ones(1000, dtype=np.float32))
    assert len(eq.flush()) == EQ_TAPS - 1 and len(eq.flush()) == 0


def test_apply_eq_is_what_one_pass_of_the_equaliser_gives():
    x = (np.random.default_rng(2).standard_normal(8000) * 0.1).astype(np.float32)
    assert np.allclose(apply_eq(x, SR), Equaliser(SR).process(x), atol=1e-6)


# The curve as it is meant to be, written out here and not read back from the module under test.
INTENDED = [(35, -12.0), (50, -3.0), (70, 2.5), (100, 4.0), (160, 3.5), (250, 1.5), (500, 0.0), (1000, 0.0),
            (2500, -1.0), (4000, -2.0), (6000, -2.0), (8000, -3.0), (11000, -7.0), (14000, -14.0), (17000, -24.0)]


@pytest.mark.parametrize("rate", [44100, 96000])
def test_the_filter_follows_the_intended_curve_at_every_point(rate):
    for freq, gain in INTENDED:
        assert response_db(rate, [freq])[0] == pytest.approx(gain, abs=0.6), f"{freq} Hz"


def test_the_module_lists_exactly_the_intended_points_and_the_two_below_35_hz():
    assert [p for p in EQ_POINTS if p[0] >= 35 and p[0] < 20000] == INTENDED and EQ_POINTS[0] == (10, -60.0) and EQ_POINTS[1] == (20, -30.0)
