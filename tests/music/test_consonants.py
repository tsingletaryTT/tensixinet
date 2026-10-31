# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Consonants for the choir: band noise, fricatives, plosive bursts, nasal notches and the choice of consonant."""
import hashlib
import warnings
from pathlib import Path

import numpy as np
import pytest

from tensixinet.music.consonants import (
    BURST_ATTACK_S,
    BURST_S,
    CONSONANTS,
    EDGE_FADE_S,
    FAMILIES,
    FRICATIVES,
    NASALS,
    PLOSIVES,
    band_noise,
    burst,
    channel_sharpness,
    fricative_noise,
    nasal_gain,
    pick_consonant,
)


def spectral_centre(x, sr):
    """The power-weighted mean frequency of the rfft."""
    power = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(len(x), 1 / sr)
    return float((freqs * power).sum() / power.sum())


def band_share(x, lo, hi, sr):
    """The share of the power between `lo` and `hi` Hz."""
    power = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(len(x), 1 / sr)
    return float(power[(freqs >= lo) & (freqs <= hi)].sum() / power.sum())


def quarter_centres(x, sr):
    q = len(x) // 4
    return spectral_centre(x[:q], sr), spectral_centre(x[-q:], sr)


def test_the_inventory_is_three_families_ordered_soft_to_sharp():
    assert set(CONSONANTS) == set(PLOSIVES + FRICATIVES + NASALS) and FAMILIES == ("plosive", "fricative", "nasal")
    assert PLOSIVES == ("b", "d", "g", "p", "t", "k") and FRICATIVES == ("h", "f", "sh", "s") and NASALS == ("m", "n")
    assert {c for c, v in CONSONANTS.items() if v.voiced} == {"b", "d", "g", "m", "n"}
    assert CONSONANTS["s"][2:] == (6500.0, 2500.0, 0.35) and CONSONANTS["m"].centre_hz == 1000.0
    for name in PLOSIVES:
        assert CONSONANTS[name].family == "plosive"
    for name in FRICATIVES:
        assert CONSONANTS[name].family == "fricative"
    for name in NASALS:
        assert CONSONANTS[name].family == "nasal"


def test_the_pinned_inventory_numbers():
    expected = {
        "s": (6500, 2500, 0.35), "sh": (3200, 1600, 0.4), "f": (5000, 5000, 0.15), "h": (1500, 2500, 0.2),
        "p": (900, 1200, 0.5), "b": (900, 1200, 0.5), "t": (4500, 3000, 0.5), "d": (4500, 3000, 0.5),
        "k": (2200, 1200, 0.5), "g": (2200, 1200, 0.5),
    }
    for name, (centre, width, level) in expected.items():
        assert CONSONANTS[name][2:] == (centre, width, level)
    assert CONSONANTS["m"][2:4] == (1000.0, 300.0) and CONSONANTS["n"][2:4] == (1800.0, 400.0)


def test_s_is_brighter_than_sh_and_each_fricative_sits_at_its_centre():
    sr, rng = 44100, np.random.default_rng(0)
    s, sh = fricative_noise("s", 4000, sr, rng), fricative_noise("sh", 4000, sr, rng)
    assert spectral_centre(s, sr) == pytest.approx(6500 * 1.075, rel=0.15)  # the centre sweeps up 15% over the sound
    assert spectral_centre(sh, sr) == pytest.approx(3200 * 1.075, rel=0.15)
    assert spectral_centre(s, sr) > 1.6 * spectral_centre(sh, sr)


def test_a_fricative_is_at_its_level():
    rng = np.random.default_rng(3)
    x = fricative_noise("sh", 20000, 44100, rng)
    middle = x[1000:-1000]  # away from the edge fades
    assert np.sqrt(np.mean(middle ** 2)) == pytest.approx(0.4, rel=0.05)


def test_a_band_moves_with_its_sweep():
    sr = 44100
    up = band_noise(44100, 2000.0, 400.0, 0.15, sr, np.random.default_rng(4))
    first, last = quarter_centres(up, sr)
    assert last > 1.10 * first
    down = band_noise(44100, 2000.0, 400.0, -0.15, sr, np.random.default_rng(4))
    first, last = quarter_centres(down, sr)
    assert last < 0.92 * first


def test_band_noise_is_unit_rms_and_inside_its_band():
    sr = 44100
    x = band_noise(20000, 3000.0, 1000.0, 0.0, sr, np.random.default_rng(5))
    assert len(x) == 20000 and np.sqrt(np.mean(x ** 2)) == pytest.approx(1.0, rel=1e-6)
    assert band_share(x, 2500, 3500, sr) > 0.97
    assert len(band_noise(0, 3000.0, 1000.0, 0.0, sr, np.random.default_rng(5))) == 0


def test_a_fricative_rises_or_falls():
    sr = 44100
    rising = fricative_noise("sh", 44100, sr, np.random.default_rng(6), rising=True)
    falling = fricative_noise("sh", 44100, sr, np.random.default_rng(6), rising=False)
    first, last = quarter_centres(rising, sr)
    assert last > 1.08 * first
    first, last = quarter_centres(falling, sr)
    assert last < 0.93 * first


def test_bands_are_clamped_under_the_nyquist_limit():
    for name in CONSONANTS:
        if CONSONANTS[name].family != "nasal":
            x = fricative_noise(name, 800, 8000, np.random.default_rng(1)) if name in FRICATIVES else burst(name, 8000, np.random.default_rng(1))
            assert band_share(x, 0.46 * 8000, 4000, 8000) < 0.02 and np.isfinite(x).all()


def test_the_clamp_holds_for_every_draw_at_the_fast_mode_rates():
    # Every draw of every fricative stays clear at the fast-mode rates, and at 8000 Hz too. Bursts below 16 kHz have
    # their own test (`test_bursts_at_8000_hz_stay_under_the_limit_on_every_draw`).
    for sr in (16000, 22050):
        for seed in range(30):
            for name in ("t", "d", "k"):
                assert band_share(burst(name, sr, np.random.default_rng(seed)), 0.46 * sr, sr / 2, sr) < 1e-3
            for name in FRICATIVES:
                assert band_share(fricative_noise(name, 800, sr, np.random.default_rng(seed)), 0.46 * sr, sr / 2, sr) < 1e-3
    for seed in range(30):
        for name in FRICATIVES:
            assert band_share(fricative_noise(name, 800, 8000, np.random.default_rng(seed)), 0.46 * 8000, 4000, 8000) < 1e-3


def test_bursts_at_8000_hz_stay_under_the_limit_on_every_draw():
    # Below 16 kHz a burst's band tops out at 0.40 of the rate, so its short envelope has room to spread.
    for name in PLOSIVES:
        for seed in range(200):
            b = burst(name, 8000, np.random.default_rng(seed))
            assert band_share(b, 0.46 * 8000, 4000, 8000) < 0.02, (name, seed)
            assert abs(b[-1]) < 1e-3 * np.abs(b).max() and abs(b[0]) < 0.05 * np.abs(b).max()


def test_a_burst_asks_for_the_lower_top_edge_only_below_16_khz(monkeypatch):
    # From 15 kHz up every plosive's band (at most 4500 + 1500 Hz) is under 0.40 of the rate anyway, so the
    # output cannot show which limit was asked for. This checks the limit burst hands to band_noise.
    from tensixinet.music import consonants
    asked = []
    real = consonants.band_noise

    def spy(*args, **kwargs):
        asked.append(kwargs.get("top_limit"))
        return real(*args, **kwargs)

    monkeypatch.setattr(consonants, "band_noise", spy)
    for sr in (8000, 15999, 16000, 22050, 44100):
        consonants.burst("t", sr, np.random.default_rng(0))
    assert asked == [0.40, 0.40, 0.45, 0.45, 0.45]


# Every fourth sample of each hashed burst, saved while both hashes held, in tests/fixtures/burst_samples.npz. The hashes are
# bit-exact and numpy's rfft can differ in the last bits between numpy versions and CPUs; the samples tell that apart from a change.
BURST_SAMPLES = Path(__file__).resolve().parents[1] / "fixtures" / "burst_samples.npz"
_HASH_MESSAGE = "burst bytes differ: a numpy or CPU difference, or burst() changed. "


def _bursts_close(cases):
    """True when every case's burst agrees with its saved samples to 1e-6."""
    with np.load(BURST_SAMPLES) as data:
        return all(np.allclose(burst(name, sr, np.random.default_rng(seed))[::4], data[f"{name}|{sr}|{seed}"], rtol=0.0, atol=1e-6)
                   for name, sr, seed in cases)


HIGH_RATE_CASES = [(name, sr, seed) for sr in (22050, 44100) for name in PLOSIVES for seed in range(3)]
AT_16K_CASES = [(name, 16000, 0) for name in PLOSIVES]


def _verdict(cases):
    return _HASH_MESSAGE + ("The samples agree to 1e-6: a platform difference." if _bursts_close(cases) else
                            "The samples disagree too: burst() changed.")


def _same_or_platform(expected, got, cases):
    """Equal hashes pass. Different hashes with samples that agree to 1e-6 are the platform's (CI's numpy and CPU), so they warn; samples
    that disagree too mean `burst()` changed, and fail."""
    if got != expected:
        assert _bursts_close(cases), _verdict(cases)
        warnings.warn(_HASH_MESSAGE + "The samples agree to 1e-6: a platform difference.")


def test_bursts_from_16_khz_up_are_as_they_were():
    # Hashes of the bursts before the low-rate rule was added (sha256 of the float64 bytes).
    digest = hashlib.sha256()
    for name, sr, seed in HIGH_RATE_CASES:
        digest.update(np.ascontiguousarray(burst(name, sr, np.random.default_rng(seed)), dtype=np.float64).tobytes())
    _same_or_platform("e4d8842091c2319910f0e029e262cfff2bc17c75954706c6211916b93ae634c3", digest.hexdigest(), HIGH_RATE_CASES)
    at_16k = hashlib.sha256()
    for name, sr, seed in AT_16K_CASES:
        at_16k.update(burst(name, sr, np.random.default_rng(seed)).astype(np.float64).tobytes())
    # 16000 is not below 16000
    _same_or_platform("22f810f43a1968a5bd81d4db397434b96d514795ad19d56fe6ae7f893ce65a8e", at_16k.hexdigest(), AT_16K_CASES)


def test_bursts_from_16_khz_up_are_within_a_millionth_of_their_saved_samples():
    assert _bursts_close(HIGH_RATE_CASES + AT_16K_CASES), "burst() changed (a platform difference in numpy would stay under 1e-6)"


def test_a_band_past_the_limit_sits_just_under_it():
    sr = 8000
    x = band_noise(8000, 6500.0, 2500.0, 0.0, sr, np.random.default_rng(7))
    assert spectral_centre(x, sr) == pytest.approx(0.42 * sr, rel=0.03)
    assert band_share(x, 0.45 * sr, sr / 2, sr) < 0.01


def test_a_burst_has_a_short_attack_so_it_does_not_click():
    b = burst("t", 44100, np.random.default_rng(2))
    assert len(b) == int(0.015 * 44100) and abs(b[0]) < 0.05 * np.abs(b).max()
    assert np.argmax(np.abs(b)) >= int(0.001 * 44100) - 1 and np.abs(b[-10:]).max() < 0.05 * np.abs(b).max()


def test_a_burst_rises_over_its_attack_and_ends_at_zero_for_every_seed():
    sr = 44100
    attack = int(BURST_ATTACK_S * sr)
    bursts = [burst("t", sr, np.random.default_rng(seed)) for seed in range(60)]
    for b in bursts:
        assert abs(b[-1]) < 1e-3 * np.abs(b).max()
    # The envelope, averaged over many draws so one draw's noise does not decide it.
    envelope = np.mean([b ** 2 for b in bursts], axis=0)
    assert envelope[: attack // 2].mean() < 0.4 * envelope[attack: 2 * attack].mean()


def test_a_burst_has_its_energy_at_its_place():
    sr = 44100
    for name, centre in (("p", 900), ("t", 4500), ("k", 2200)):
        centres = [spectral_centre(burst(name, sr, np.random.default_rng(seed)), sr) for seed in range(10)]
        assert np.mean(centres) == pytest.approx(centre, rel=0.02)  # a burst holds its place: it does not sweep
    assert len(burst("b", 16000, np.random.default_rng(8))) == int(BURST_S * 16000)


def test_a_fricative_fades_in_and_out_so_its_ends_do_not_click():
    sr = 44100
    x = fricative_noise("s", 4000, sr, np.random.default_rng(9))
    peak = np.abs(x).max()
    assert abs(x[0]) < 1e-3 * peak and abs(x[-1]) < 1e-3 * peak
    fade = int(EDGE_FADE_S * sr)
    assert np.abs(x[: fade // 4]).mean() < 0.3 * np.abs(x[fade: 2 * fade]).mean()


def test_the_wrong_family_is_refused():
    with pytest.raises(ValueError):
        fricative_noise("t", 100, 8000, np.random.default_rng(0))
    with pytest.raises(ValueError):
        burst("s", 8000, np.random.default_rng(0))
    with pytest.raises(ValueError):
        nasal_gain(np.array([100.0]), "s")


def test_a_nasal_cuts_a_notch_and_leaves_the_rest():
    f = np.array([250.0, 1000.0, 2000.0, 4000.0])
    g = nasal_gain(f, "m")
    assert g[1] <= 0.1 and g[0] >= 0.9 and g[2] >= 0.9 and nasal_gain(np.array([1800.0]), "n")[0] <= 0.1


def test_a_nasal_notch_is_as_deep_as_pinned():
    assert nasal_gain(np.array([1000.0]), "m")[0] == pytest.approx(0.05, abs=1e-6)
    # The width is the notch's full width at half depth.
    edges = nasal_gain(np.array([1000.0 - 150.0, 1000.0 + 150.0]), "m")
    assert edges == pytest.approx([1 - 0.95 / 2] * 2, abs=1e-6)


def test_a_hot_channel_and_a_strong_trigger_give_a_sharp_plosive():
    # u_member 0.5 is the middle of the nudge, so it moves nothing.
    assert pick_consonant(1.0, 1.0, (1.0, 1.0, 1.0), 0.5, 0.5) == "k"
    assert pick_consonant(0.0, 1.0, (1.0, 1.0, 1.0), 0.5, 0.5) == "b"


def test_the_member_is_the_nearest_step_of_sharpness():
    assert pick_consonant(0.75, 1.0, (1.0, 0.0, 0.0), 0.5, 0.5) == "t"  # 0.75 * 5 = 3.75 rounds to 4
    assert pick_consonant(0.75, 0.0, (0.0, 1.0, 0.0), 0.5, 0.5) == "sh"  # 0.75 * 3 = 2.25 rounds to 2


def test_a_weak_trigger_leans_to_fricatives_and_nasals_and_a_cool_channel_to_soft_ones():
    # At strength 0.2 the weights are 0.3 + 0.14 for plosives and 1 - 0.14 for each of the others: 79.6% fricatives and nasals.
    picks = [pick_consonant(0.0, 0.2, (1.0, 1.0, 1.0), u, 0.5) for u in np.linspace(0, 0.999, 200)]
    assert sum(c in FRICATIVES + NASALS for c in picks) > 0.75 * len(picks) and set(picks) <= {"b", "h", "m"}


def test_the_mix_weights_the_families():
    assert all(pick_consonant(0.5, s, (0.0, 0.0, 1.0), u, 0.5) in NASALS for s in (0.0, 0.5, 1.0) for u in (0.0, 0.5, 0.999))
    assert pick_consonant(0.0, 0.0, (1.0, 0.0, 0.0), 0.5, 0.5) == "b"  # a family missing from the mix is never sung
    assert pick_consonant(1.0, 1.0, (0.0, 1.0, 0.0), 0.5, 0.5) == "s"  # even at the strength that leans hardest away from it


def _family_at(strength, u, mix=(1.0, 1.0, 1.0)):
    return CONSONANTS[pick_consonant(0.5, strength, mix, u, 0.5)].family


def test_every_family_keeps_a_floor_at_full_strength():
    # Weights at strength s: plosive mix0 * (0.3 + 0.7 s), fricative mix1 * (1 - 0.7 s), nasal mix2 * (1 - 0.7 s).
    # At s = 1 on an equal mix that is 1, 0.3, 0.3: plosives up to 1 / 1.6 = 0.625, fricatives up to 1.3 / 1.6 = 0.8125.
    assert _family_at(1.0, 0.62) == "plosive" and _family_at(1.0, 0.63) == "fricative"
    assert _family_at(1.0, 0.81) == "fricative" and _family_at(1.0, 0.82) == "nasal"
    # At s = 0 it is 0.3, 1, 1: plosives up to 0.3 / 2.3 = 0.1304, fricatives up to 1.3 / 2.3 = 0.5652.
    assert _family_at(0.0, 0.125) == "plosive" and _family_at(0.0, 0.135) == "fricative"
    assert _family_at(0.0, 0.56) == "fricative" and _family_at(0.0, 0.57) == "nasal"


def test_a_full_strength_trigger_on_an_enya_mix_still_sings_every_family():
    # enya's mix is 0.1, 0.3, 0.6. At strength 1 the weights are 0.1, 0.09, 0.18: plosives 27%, three times their 10% in
    # the mix, and fricatives and nasals still sung.
    rng = np.random.default_rng(0)
    picks = [CONSONANTS[pick_consonant(0.5, 1.0, (0.1, 0.3, 0.6), u, v)].family for u, v in rng.random((400, 2))]
    shares = {f: picks.count(f) / len(picks) for f in FAMILIES}
    assert shares["fricative"] > 0.15 and shares["nasal"] > 0.35
    assert 0.2 < shares["plosive"] < 0.35


def test_a_second_draw_nudges_the_member_around_the_sharpness():
    # member = round(clip(sharpness + 0.5 * (u_member - 0.5), 0, 1) * (n - 1))
    assert pick_consonant(0.5, 1.0, (1.0, 0.0, 0.0), 0.5, 0.99) == "t"  # 0.745 * 5 = 3.725 rounds to 4
    assert pick_consonant(0.5, 1.0, (1.0, 0.0, 0.0), 0.5, 0.01) == "d"  # 0.255 * 5 = 1.275 rounds to 1
    assert pick_consonant(0.5, 1.0, (1.0, 0.0, 0.0), 0.5, 0.5) == "g"  # 2.5 rounds to 2 (to even)
    # The size of the nudge: half of 0.5 either way at most. 0.65 + 0.245 = 0.895 (4.475 rounds to 4) and 0.7 + 0.245 = 0.945.
    assert pick_consonant(0.65, 1.0, (1.0, 0.0, 0.0), 0.5, 0.99) == "t"
    assert pick_consonant(0.7, 1.0, (1.0, 0.0, 0.0), 0.5, 0.99) == "k"
    # The nudged sharpness is held inside 0 to 1, so the softest and sharpest stay the ends of the family.
    assert pick_consonant(0.0, 1.0, (1.0, 0.0, 0.0), 0.5, 0.0) == "b"
    assert pick_consonant(1.0, 1.0, (1.0, 0.0, 0.0), 0.5, 0.999) == "k"
    assert pick_consonant(0.0, 0.0, (0.0, 1.0, 0.0), 0.5, 0.0) == "h"


def test_equal_channels_give_many_syllables_over_many_draws():
    # Equal channels give every syllable sharpness 0.5. At strength 1 the old rule sang "g" every time.
    rng = np.random.default_rng(1)
    picks = {pick_consonant(0.5, 1.0, (1.0, 1.0, 1.0), u, v) for u, v in rng.random((200, 2))}
    assert len(picks) >= 6, picks


def test_channel_sharpness_follows_vowel_sequences_rule():
    assert channel_sharpness((40.0,) * 8, 3) == 0.5
    assert channel_sharpness((40.0, 40.4) + (40.0,) * 6, 1) == 0.5  # half spread 0.2 C is under 0.25 C
    assert channel_sharpness((40.0, 40.6) + (40.0,) * 6, 1) == 1.0  # half spread 0.3 C is over it
    temps = (30.0, 35.0, 40.0, 45.0, 50.0, 38.0, 32.0, 48.0)
    assert channel_sharpness(temps, 0) == 0.0 and channel_sharpness(temps, 4) == 1.0
    assert channel_sharpness(temps, 2) == pytest.approx(0.5)
    assert channel_sharpness(temps, 12) == 1.0  # channels wrap round
