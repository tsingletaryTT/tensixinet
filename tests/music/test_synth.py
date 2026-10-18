"""Voice layout and rendering for any number of chips. Pure numpy, no hardware."""
import numpy as np
import pytest

from tensixinet.music.synth import Voice, build_voices, render
from tensixinet.telemetry import FALLBACK, Telemetry

SR = 8000  # low rate keeps tests fast; the synth takes sample_rate as an argument


def tel(**kw):
    return Telemetry(**{**FALLBACK.__dict__, **kw})


# --- layout -------------------------------------------------------------------

@pytest.mark.parametrize("n_chips", [1, 2, 4, 32])
def test_voice_count_scales_with_chip_count(n_chips):
    assert len(build_voices(n_chips, voices_per_chip=4)) == 4 * n_chips


def test_every_chip_gets_the_same_number_of_voices():
    voices = build_voices(5, voices_per_chip=3)
    for chip in range(5):
        assert sum(v.chip_id == chip for v in voices) == 3


def test_voices_per_chip_is_configurable():
    assert len(build_voices(3, voices_per_chip=1)) == 3


def test_frequencies_stay_inside_the_requested_band_even_for_huge_systems():
    voices = build_voices(64, voices_per_chip=4, f_low=65.41, f_high=4186.0)
    assert all(65.41 <= v.base_freq <= 4186.0 for v in voices)


def test_single_chip_layout_is_a_playable_chord():
    freqs = sorted(v.base_freq for v in build_voices(1, voices_per_chip=3))
    # C major triad from the default low note: root, major third, fifth.
    assert freqs[1] / freqs[0] == pytest.approx(2 ** (4 / 12))
    assert freqs[2] / freqs[0] == pytest.approx(2 ** (7 / 12))


def test_layout_is_deterministic():
    assert build_voices(3, 4) == build_voices(3, 4)


@pytest.mark.parametrize("bad", [0, -1])
def test_rejects_non_positive_chip_count(bad):
    with pytest.raises(ValueError):
        build_voices(bad)


# --- rendering ----------------------------------------------------------------

def _peak_hz(audio):
    spec = np.abs(np.fft.rfft(audio * np.hanning(len(audio))))
    return np.argmax(spec) * SR / len(audio)


def test_render_returns_requested_length_float32_and_normalised_peak():
    audio = render(build_voices(2, 2), [FALLBACK, FALLBACK], duration=1.0, sample_rate=SR)
    assert audio.shape == (SR,)
    assert audio.dtype == np.float32
    assert np.max(np.abs(audio)) == pytest.approx(0.7, abs=1e-3)


def test_render_is_deterministic():
    v = build_voices(2, 2)
    a = render(v, [FALLBACK] * 2, 0.5, SR)
    b = render(v, [FALLBACK] * 2, 0.5, SR)
    assert np.array_equal(a, b)


def test_hotter_chip_raises_pitch_by_the_documented_drift():
    voice = [Voice(chip_id=0, base_freq=440.0, lfo_rate=0.0)]
    cold = render(voice, [tel(asic_temperature=25.0)], 2.0, SR, drift=0.1)  # temp_norm 0
    hot = render(voice, [tel(asic_temperature=100.0)], 2.0, SR, drift=0.1)  # temp_norm 1
    assert _peak_hz(cold) == pytest.approx(440.0, abs=1.0)
    assert _peak_hz(hot) == pytest.approx(484.0, abs=1.0)  # 440 * 1.1


def test_each_voice_follows_its_own_chips_telemetry():
    voices = [Voice(0, 440.0, 0.0), Voice(1, 440.0, 0.0)]
    both_cold = render(voices, [tel(asic_temperature=25.0)] * 2, 1.0, SR)
    one_hot = render(voices, [tel(asic_temperature=25.0), tel(asic_temperature=100.0)], 1.0, SR)
    assert not np.allclose(both_cold, one_hot)


def test_voice_for_a_missing_chip_is_an_error_not_a_silent_wrap():
    with pytest.raises(ValueError, match="chip 2"):
        render([Voice(2, 440.0, 0.0)], [FALLBACK, FALLBACK], 0.1, SR)


def test_partials_above_nyquist_do_not_alias_into_the_audible_band():
    # Fundamental 3600 Hz at SR 8000: its 2nd partial (7200 Hz) is above Nyquist
    # (4000) and would fold to 800 Hz if it were rendered.
    audio = render([Voice(0, SR * 0.45, 0.0)], [tel(asic_temperature=25.0)], 1.0, SR)
    spec = np.abs(np.fft.rfft(audio * np.hanning(len(audio))))
    bin_hz = SR / len(audio)
    alias = spec[int(780 / bin_hz): int(820 / bin_hz)].max()
    assert alias < 0.01 * spec.max()
