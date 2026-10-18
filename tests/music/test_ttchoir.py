# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""ttchoir: a chip as a singer. A glottal source through a vocal tract of formant resonances, sung on vowels."""
import dataclasses

import numpy as np
import pytest

from tensixinet.music.mapping import aiclk_to_bpm, temp_to_pitch_hz
from tensixinet.music.patch import DEFAULT_PATCH, SCALES
from tensixinet.music.ttchoir import (
    BANDWIDTHS,
    VOWEL_ORDER,
    VOWELS,
    choir_pitch,
    formant_response,
    harmonic_amplitudes,
    ttchoir,
    vowel_sequence,
    voice_scale,
)
from tensixinet.telemetry import FALLBACK

SR = 16000


def singer(**kw):
    return dataclasses.replace(FALLBACK, **{"asic_temperature": 50.0, "power": 40.0, **kw})


def band_share(audio, lo, hi, sr=SR):
    spec = np.abs(np.fft.rfft(audio.astype(np.float64))) ** 2
    f = np.fft.rfftfreq(len(audio), 1 / sr)
    return float(spec[(f >= lo) & (f < hi)].sum() / spec.sum())


# --- the vowels ---------------------------------------------------------------------------------------------------

def test_the_five_vowels_have_the_textbook_formants():
    assert set(VOWELS) == set("aeiou") and VOWEL_ORDER == "uoaei"
    for f1, f2, f3 in VOWELS.values():
        assert f1 < f2 < f3
    assert VOWELS["i"][1] > VOWELS["e"][1] > VOWELS["a"][1] > VOWELS["u"][1]  # the second formant is highest in "ee", lowest in "oo"
    assert VOWELS["a"][0] > VOWELS["o"][0] > VOWELS["u"][0] and VOWELS["a"][0] > VOWELS["i"][0]  # the jaw is most open in "ah"
    assert VOWELS["a"] == (730, 1090, 2440) and VOWELS["i"] == (270, 2290, 3010)


def test_bandwidths_widen_up_the_tract():
    assert len(BANDWIDTHS) == 5 and list(BANDWIDTHS) == sorted(BANDWIDTHS)


def test_the_tract_resonates_at_its_formants_and_is_flat_far_below_them():
    formants = (500.0, 1500.0, 2500.0, 3400.0, 4400.0)
    freqs = np.array([50.0, 500.0, 1000.0, 1500.0, 2000.0, 2500.0])
    gain = formant_response(freqs, formants)
    assert gain[1] > 2 * gain[2] and gain[3] > 2 * gain[2] and gain[5] > 1.5 * gain[4]  # peaks at the formants, dips between
    assert gain[0] == pytest.approx(1.0, abs=0.05)  # unity well below the first formant


def test_the_tract_shapes_a_flat_source_into_the_vowel_it_is_set_to():
    freqs = np.arange(100, 4000, 100.0)
    for vowel, (f1, f2, f3) in VOWELS.items():
        gain = formant_response(freqs, (f1, f2, f3, 3400.0, 4400.0))
        assert abs(freqs[np.argmax(gain)] - f1) <= 150 or abs(freqs[np.argmax(gain)] - f2) <= 150, vowel


# --- what is sung -----------------------------------------------------------------------------------------------

def test_equal_memory_temperatures_give_a_steady_ah():
    assert vowel_sequence((40.0,) * 8, 6, 0) == ["a"] * 6


def test_a_hot_channel_sings_toward_ee_and_a_cool_one_toward_oo():
    temps = (40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 50.0)
    hot = vowel_sequence(temps, 8, 0)[7]
    cool = vowel_sequence((50.0,) * 7 + (40.0,), 8, 0)[7]
    assert VOWEL_ORDER.index(hot) > VOWEL_ORDER.index("a") and VOWEL_ORDER.index(cool) < VOWEL_ORDER.index("a")


def test_every_vowel_in_the_sequence_is_one_of_the_five_and_the_offset_moves_the_start():
    temps = (30.0, 35.0, 40.0, 45.0, 50.0, 38.0, 32.0, 48.0)
    seq = vowel_sequence(temps, 16, 0)
    assert set(seq) <= set("aeiou")
    assert vowel_sequence(temps, 8, 3) == vowel_sequence(temps, 16, 0)[3:11]


def test_the_pitch_is_the_chips_pitch_snapped_to_a_note_of_the_patchs_scale():
    pent = SCALES["major_pentatonic"]
    for temp in (30.0, 45.0, 60.0, 75.0, 90.0):
        f = choir_pitch(temp, pent)
        semis = 12 * np.log2(f / 130.8128)
        assert min(abs(semis - (12 * o + d)) for o in range(3) for d in pent) < 1e-6
        assert abs(12 * np.log2(f / temp_to_pitch_hz(temp))) <= 1.51  # no more than a minor third away in a pentatonic scale
    assert choir_pitch(30.0, pent) < choir_pitch(60.0, pent) < choir_pitch(90.0, pent)


def test_voice_types_stretch_the_vocal_tract_four_ways():
    scales = {voice_scale(i) for i in range(64)}
    assert scales == {1.0, 1.08, 1.17, 1.25}


# --- the sound ---------------------------------------------------------------------------------------------------

def test_ah_has_its_energy_near_the_first_two_formants_and_ee_has_it_high_and_low():
    def sing(vowel_temp):
        t = singer(gddr_temps=(40.0,) * 7 + (vowel_temp,))
        return ttchoir(t, 12.0, SR, 0)
    # one chip: steady vowels. Equal temperatures give ah.
    ah = ttchoir(singer(gddr_temps=(40.0,) * 8), 8.0, SR, 0)
    assert band_share(ah, 600, 1300) > band_share(ah, 2800, 7000) * 3  # the first two formants carry the sound
    assert band_share(ah, 300, 4000) > 0.8


def test_the_harmonics_of_each_vowel_pile_up_at_its_formants():
    f0, sr = 130.0, 16000
    def amps(vowel, pressure=0.0):
        return harmonic_amplitudes(f0, VOWELS[vowel] + (3400.0, 4400.0), pressure, sr)
    freq = lambda k: k * f0  # harmonic number k is at k * f0 Hz (k counts from 1)
    a, i, u = amps("a"), amps("i"), amps("u")
    at = lambda arr, hz: arr[int(round(hz / f0)) - 1]
    assert at(i, 2290) > 3 * at(i, 1200) and at(i, 270) > 3 * at(i, 1200)  # ee: a low and a very high formant, a hole between
    assert at(a, 1090) > 3 * at(a, 3000) and at(a, 730) > 3 * at(a, 3000)
    above = lambda arr: float(arr[int(1500 / f0):].sum())  # the share of the voice that sits above 1500 Hz
    assert above(u) < 0.02 and above(a) < 0.06 and above(i) > 0.08 and above(amps("e")) > 0.12  # dark to bright: oo, ah, ee, eh


def test_the_harmonic_amplitudes_sum_to_one_and_stop_below_the_nyquist_limit():
    amps = harmonic_amplitudes(200.0, VOWELS["a"] + (3400.0, 4400.0), 0.0, 8000)
    assert amps.sum() == pytest.approx(1.0) and (amps >= 0).all()
    assert len(amps) * 200.0 <= 0.44 * 8000 / 1.0 + 1e-9 and len(amps) >= 15


def test_pressure_flattens_the_glottal_slope_so_the_top_harmonics_gain():
    calm = harmonic_amplitudes(130.0, VOWELS["a"] + (3400.0, 4400.0), 0.0, 16000)
    strained = harmonic_amplitudes(130.0, VOWELS["a"] + (3400.0, 4400.0), 1.0, 16000)
    assert strained[20:].sum() > 2 * calm[20:].sum()


def test_the_fundamental_is_at_the_chosen_pitch():
    t = singer(asic_temperature=60.0, gddr_temps=(40.0,) * 8)
    audio = ttchoir(t, 6.0, SR, 0)
    spec = np.abs(np.fft.rfft(audio * np.hanning(len(audio))))
    f = np.fft.rfftfreq(len(audio), 1 / SR)
    lo, hi = 80, 700
    band = (f >= lo) & (f <= hi)
    peak = f[band][np.argmax(spec[band])]
    expected = choir_pitch(60.0, DEFAULT_PATCH.scale)
    assert peak == pytest.approx(expected, rel=0.04) or peak == pytest.approx(2 * expected, rel=0.04)  # ah can put the loudest partial an octave up


def test_nothing_is_sung_above_the_nyquist_limit_at_any_sample_rate():
    for sr in (8000, 16000, 44100):
        audio = ttchoir(singer(asic_temperature=95.0), 3.0, sr, 0)
        assert np.isfinite(audio).all() and np.abs(audio).max() <= 1.0
        assert band_share(audio, 0.46 * sr, sr / 2, sr) < 1e-3


def test_the_voice_starts_softly_and_breathes_between_syllables():
    t = singer(gddr_temps=(40.0,) * 8)
    audio = ttchoir(t, 12.0, SR, 0)
    assert np.abs(audio[: int(0.02 * SR)]).max() < 0.2 * np.abs(audio).max()  # an attack, not a click
    syllable = 4 * 60.0 / aiclk_to_bpm(t.aiclk)
    around = lambda c: np.sqrt(np.mean(audio[int((c - 0.05) * SR): int((c + 0.05) * SR)] ** 2))
    middle = np.sqrt(np.mean(audio[int((syllable * 0.5 - 0.1) * SR): int((syllable * 0.5 + 0.1) * SR)] ** 2))
    assert around(syllable) < 0.7 * middle  # a dip where one syllable ends and the next begins


def test_a_strained_chip_sings_brighter_and_a_bigger_tract_sings_higher_formants():
    calm = ttchoir(singer(thm_limit_throttle=120.0, gddr_temps=(40.0,) * 8), 8.0, SR, 0)
    strained = ttchoir(singer(thm_limit_throttle=61.0, gddr_temps=(40.0,) * 8), 8.0, SR, 0)
    assert band_share(strained, 2000, 6000) > band_share(calm, 2000, 6000)


def test_the_choir_is_deterministic_finite_and_inside_full_scale():
    t = singer()
    a, b = ttchoir(t, 5.0, SR, 7), ttchoir(t, 5.0, SR, 7)
    assert a.dtype == np.float32 and a.shape == (5 * SR,) and np.array_equal(a, b)
    assert np.isfinite(a).all() and np.abs(a).max() <= 1.0


# --- pinned values and wiring -------------------------------------------------------------------------------------

def test_the_bandwidths_and_the_level_are_the_designed_numbers():
    from tensixinet.music.ttchoir import CHOIR_LEVEL, UPPER_FORMANTS, VOICE_SCALES
    assert BANDWIDTHS == (80.0, 100.0, 120.0, 160.0, 200.0) and UPPER_FORMANTS == (3400.0, 4400.0)
    assert CHOIR_LEVEL == 0.5 and VOICE_SCALES == (1.0, 1.08, 1.17, 1.25)
    audio = ttchoir(singer(), 6.0, SR, 0)
    assert 0.3 < float(np.abs(audio).max()) <= 0.5  # a level of a half, and the harmonics add to at most 1


def test_the_vibrato_is_at_five_and_a_half_hertz_and_deeper_under_pressure():
    from tensixinet.music.ttchoir import pitch_modulation
    t = np.arange(0, 6.0, 1 / 1000)
    calm, strained = pitch_modulation(t, 0.0, 1), pitch_modulation(t, 1.0, 1)
    assert float(np.abs(calm).max()) == pytest.approx(0.25, abs=0.1) and float(np.abs(strained).max()) == pytest.approx(0.6, abs=0.1)
    spectrum = np.abs(np.fft.rfft(calm))
    assert np.fft.rfftfreq(len(t), 1 / 1000)[np.argmax(spectrum[1:]) + 1] == pytest.approx(5.5, abs=0.3)


def test_the_drift_depends_on_the_seed_and_stays_small():
    from tensixinet.music.ttchoir import pitch_modulation
    t = np.arange(0, 6.0, 1 / 1000)
    a, b = pitch_modulation(t, 0.0, 1), pitch_modulation(t, 0.0, 2)
    assert not np.allclose(a, b) and float(np.abs(a - b).max()) < 0.2


def test_a_syllable_is_four_beats_and_nothing_dips_in_the_middle_of_one():
    t = singer(gddr_temps=(40.0,) * 8)
    audio = ttchoir(t, 14.0, SR, 0)
    beat = 60.0 / aiclk_to_bpm(t.aiclk)
    rms = lambda c: np.sqrt(np.mean(audio[int((c - 0.1) * SR): int((c + 0.1) * SR)] ** 2))
    boundary, half_way = rms(4 * beat), rms(2 * beat)  # two beats in: the middle of a syllable
    assert half_way > 1.3 * boundary and rms(6 * beat) > 1.3 * boundary


def test_the_breath_between_syllables_is_brief_and_the_voice_is_back_well_before_the_middle():
    t = singer(gddr_temps=(40.0,) * 8)
    audio = ttchoir(t, 14.0, SR, 0)
    beat = 60.0 / aiclk_to_bpm(t.aiclk)
    rms = lambda c: np.sqrt(np.mean(audio[int((c - 0.05) * SR): int((c + 0.05) * SR)] ** 2))
    assert rms(4 * beat + 0.3) > 0.85 * rms(6 * beat) and rms(4 * beat) < 0.7 * rms(6 * beat)


def test_a_longer_vocal_tract_sings_lower_formants_than_a_shorter_one():
    bass = ttchoir(singer(tensix_mask=12288, gddr_temps=(40.0,) * 8), 8.0, SR, 0)  # voice type 0: the longest tract
    soprano = ttchoir(singer(tensix_mask=12473, gddr_temps=(40.0,) * 8), 8.0, SR, 0)  # voice type 3: 25% shorter
    assert band_share(soprano, 1200, 1800) != pytest.approx(band_share(bass, 1200, 1800), rel=0.1)
    assert band_share(soprano, 900, 1800) > band_share(bass, 900, 1800)  # the second formant of "ah" moved up into this band


def test_the_chips_identity_picks_which_channel_its_first_syllable_reads(monkeypatch):
    from tensixinet.music import ttchoir as module
    from tensixinet.music.mapping import personality
    seen = []
    real = module.vowel_sequence
    monkeypatch.setattr(module, "vowel_sequence", lambda temps, count, offset: seen.append(offset) or real(temps, count, offset))
    t = singer(tensix_mask=12362)
    ttchoir(t, 2.0, SR, 0)
    assert seen == [personality(t) % 8]



@pytest.mark.parametrize("duration", [0.0, 0.001, 0.05, 0.2, 0.31])
def test_very_short_renders_come_out_the_right_length_and_finite(duration):
    audio = ttchoir(singer(), duration, SR, 0)
    assert audio.shape == (int(duration * SR),) and np.isfinite(audio).all()
