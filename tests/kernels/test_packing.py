# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The host side of the kernels: packing audio into tiles, and splitting a device's maths between host and chip."""
import math

import numpy as np
import pytest

from tensixinet.kernels.packing import (
    TILE,
    TILE_SAMPLES,
    pack,
    padded_length,
    shock_inputs,
    shock_on_host,
    unpack,
    within_tolerance,
)
from tensixinet.music.devices import shock_wave


def test_a_tile_is_32_by_32_so_1024_samples():
    assert TILE == 32 and TILE_SAMPLES == 1024


@pytest.mark.parametrize("n,expected", [(0, 1024), (1, 1024), (1024, 1024), (1025, 2048), (5000, 5120)])
def test_the_length_is_rounded_up_to_whole_tiles(n, expected):
    assert padded_length(n) == expected


def test_packing_makes_a_float32_matrix_of_whole_tiles_and_keeps_the_samples_in_order():
    x = np.arange(3000, dtype=np.float64)
    tiles = pack(x)
    assert tiles.dtype == np.float32 and tiles.shape == (3072 // TILE, TILE) and tiles.shape[0] % TILE == 0
    assert np.array_equal(tiles.ravel()[:3000], x.astype(np.float32)) and not tiles.ravel()[3000:].any()


def test_unpacking_returns_the_samples_that_went_in():
    x = np.random.default_rng(0).standard_normal(2500)
    assert np.allclose(unpack(pack(x), 2500), x, atol=1e-6) and len(unpack(pack(x), 2500)) == 2500


def test_a_matrix_that_is_not_whole_tiles_is_refused():
    with pytest.raises(ValueError):
        unpack(np.zeros((33, 32), dtype=np.float32), 10)


# --- the shock wave: what the host prepares, what the chip computes ---------------------------------------------

def test_the_host_hands_the_chip_the_phase_the_decay_and_a_mask_all_packed_the_same_way():
    since = np.array([np.inf, 0.0, 0.25, 1.0])
    x, neg_e, live = shock_inputs(since, rate_hz=2.0, decay_s=0.5)
    assert x.shape == neg_e.shape == live.shape and x.dtype == neg_e.dtype == live.dtype == np.float32
    assert x.ravel()[:4] == pytest.approx([0.0, 0.0, 2 * np.pi * 2.0 * 0.25, 2 * np.pi * 2.0 * 1.0], rel=1e-6)
    assert neg_e.ravel()[:4] == pytest.approx([0.0, 0.0, -0.5, -2.0], rel=1e-6)  # before the first strike nothing is live
    assert live.ravel()[:4].tolist() == [0.0, 1.0, 1.0, 1.0]


def test_the_chips_formula_on_the_hosts_inputs_is_the_devices_shock_wave():
    rng = np.random.default_rng(1)
    since = rng.uniform(0, 4, 3000)
    since[:200] = np.inf
    got = shock_on_host(*shock_inputs(since, 1.5 * 2.0, 0.8), n=3000)
    assert np.allclose(got, shock_wave(since, 3.0, 0.8), atol=2e-5)


def test_tolerance_is_checked_on_the_worst_sample_and_reports_it():
    expected = np.linspace(0, 1, 100)
    ok, worst = within_tolerance(expected + 0.001, expected, atol=0.002)
    assert ok and worst == pytest.approx(0.001)
    ok, worst = within_tolerance(expected + np.r_[np.zeros(99), 0.1], expected, atol=0.002)
    assert not ok and worst == pytest.approx(0.1)
    assert within_tolerance(np.zeros(3), np.zeros(4), atol=1)[0] is False  # different lengths never match


# --- the shimmer: two tones with a snap on every strike ---------------------------------------------------------

from tensixinet.kernels.packing import (  # noqa: E402
    COCO_TAPS,
    SNAP_FLOOR,
    coco_inputs,
    coco_on_host,
    mix_inputs,
    mix_on_host,
    shimmer_inputs,
    shimmer_on_host,
)
from tensixinet.music.devices import SNAP_DECAY_S, SNAP_DEPTH, seconds_since, ultrasonic_shimmer  # noqa: E402
from tensixinet.music.patch import DEFAULT_PATCH  # noqa: E402
from tensixinet.telemetry import FALLBACK  # noqa: E402


def test_the_snap_never_sends_the_chip_minus_infinity():
    since = np.array([np.inf, 0.0, 0.012])
    _, _, e, _, _ = shimmer_inputs(since, 15000.0, 15002.0, 0.8, 3, 44100)
    assert np.isfinite(e).all() and e.ravel()[0] == SNAP_FLOOR and e.ravel()[1] == 0.0
    assert e.ravel()[2] == pytest.approx(-0.012 / SNAP_DECAY_S, rel=1e-6)


def test_the_phases_are_reduced_to_one_turn_so_float32_can_hold_them():
    p1, p2, *_ = shimmer_inputs(np.full(44100, np.inf), 15000.0, 15002.0, 1.0, 44100, 44100)
    assert 0 <= p1.min() and p1.max() <= 2 * np.pi + 1e-6 and 0 <= p2.min() and p2.max() <= 2 * np.pi + 1e-6


def test_the_two_constants_are_the_level_with_and_without_the_snap_already_halved_for_two_tones():
    *_, a0, a1 = shimmer_inputs(np.array([1.0]), 15000.0, 15002.0, 0.8, 1, 44100)
    assert a0.ravel()[0] == pytest.approx(0.5 * 0.8 * (1 - SNAP_DEPTH)) and a1.ravel()[0] == pytest.approx(0.5 * 0.8 * SNAP_DEPTH)


@pytest.mark.parametrize("rate", [44100, 8000])
def test_the_chips_formula_on_the_hosts_inputs_is_the_shimmer_device(rate):
    import dataclasses
    from tensixinet.music.devices import shimmer_parameters, strike_times
    t = dataclasses.replace(FALLBACK, asic_temperature=72.0, gddr_temps=(30.0, 41.0) * 4, power=90.0, heartbeat=5)
    n = rate * 6
    f1, f2, level = shimmer_parameters(t, rate, DEFAULT_PATCH)
    since = seconds_since(strike_times(t, 6.0), n, rate)
    got = shimmer_on_host(*shimmer_inputs(since, f1, f2, level, n, rate), n=n)
    assert np.allclose(got, ultrasonic_shimmer(t, 6.0, rate, 0), atol=2e-3)  # float32 phases at 15 kHz are good to a few thousandths


# --- the mixer: the chips' signals summed and squashed -----------------------------------------------------------

def test_the_mixer_inputs_are_weighted_and_scaled_on_the_host_and_padded_to_four():
    from tensixinet.music.devices import current_mix
    a, b = np.full(10, 0.5), np.full(10, -0.25)
    tiles = mix_inputs([a, b], [30.0, 10.0], n=10)
    assert len(tiles) == 4 and all(t.shape == tiles[0].shape for t in tiles)
    assert not tiles[2].any() and not tiles[3].any()  # unused inputs are silence
    assert np.allclose(mix_on_host(tiles, 10), current_mix([a, b], [30.0, 10.0])[:10], atol=1e-6)


def test_more_than_four_signals_are_refused_for_now():
    with pytest.raises(ValueError):
        mix_inputs([np.zeros(4)] * 5, [1.0] * 5, n=4)


def test_a_mix_with_no_current_weighs_every_signal_the_same_as_the_device_does():
    from tensixinet.music.devices import current_mix
    sigs = [np.full(6, 0.2), np.full(6, 0.4), np.full(6, -0.1)]
    assert np.allclose(mix_on_host(mix_inputs(sigs, [0.0, 0.0, 0.0], n=6), 6), current_mix(sigs, [0.0, 0.0, 0.0]), atol=1e-6)


def test_a_negative_current_counts_as_none_in_the_kernels_mix_as_in_the_device():
    from tensixinet.music.devices import current_mix
    sigs = [np.full(6, 0.3), np.full(6, -0.2)]
    got = mix_on_host(mix_inputs(sigs, [-5.0, 10.0], n=6), 6)
    assert np.allclose(got, current_mix(sigs, [-5.0, 10.0]), atol=1e-6)


# --- the gong: partials ringing after each strike -----------------------------------------------------------------

from tensixinet.kernels.packing import (  # noqa: E402
    GONG_PARTIAL_SLOTS,
    GONG_OVERLAP,
    gong_combine,
    gong_inputs,
    gong_on_host,
    strike_ages,
)
from tensixinet.music.devices import gong, gong_parameters, strike_times  # noqa: E402


def test_strike_ages_count_samples_since_the_nth_latest_strike_the_way_the_device_places_it():
    # onsets at 1 ms and 3.5 ms at 1 kHz begin at samples 1 and 3, because the device places a strike at int(onset * rate)
    ages = strike_ages(np.array([0.001, 0.0035]), n=60, sample_rate=1000, length=100, back=0)
    assert ages[0] == -1 and ages[1] == 0 and ages[2] == 1 and ages[3] == 0 and ages[4] == 1 and ages[59] == 56
    previous = strike_ages(np.array([0.001, 0.0035]), n=60, sample_rate=1000, length=100, back=1)
    assert previous[2] == -1 and previous[3] == 2 and previous[4] == 3 and previous[59] == 58


def test_a_strike_that_has_rung_out_is_no_longer_live():
    ages = strike_ages(np.array([0.0]), n=50, sample_rate=1000, length=20, back=0)
    assert ages[19] == 19 and ages[20] == -1 and ages[49] == -1


def test_there_are_five_partial_slots_and_three_overlapping_strikes():
    assert GONG_PARTIAL_SLOTS == 5 and GONG_OVERLAP == 3


def test_gong_inputs_are_sixteen_tiles_with_unused_partials_silent_and_dead_samples_silent():
    import dataclasses
    t = dataclasses.replace(FALLBACK, aiclk=800.0)
    g = gong_parameters(t, 8000, DEFAULT_PATCH)
    ages = np.array([-1, 0, 100, 400])
    tiles = gong_inputs(ages, g, 8000)
    assert len(tiles) == 16 and all(x.shape == tiles[0].shape for x in tiles)
    a, c, e, att = tiles[0:5], tiles[5:10], tiles[10:15], tiles[15]
    assert att.ravel()[0] == 0.0 and att.ravel()[1] == 0.0 and att.ravel()[2] == 1.0  # dead, then the first sample of the attack, then past it
    for slot in range(len(g.partials), GONG_PARTIAL_SLOTS):
        assert (e[slot].ravel()[:4] == SNAP_FLOOR).all()  # a slot with no partial adds nothing
    assert (e[0].ravel()[0] == SNAP_FLOOR) and e[0].ravel()[1] > SNAP_FLOOR  # live samples have a real exponent
    assert all(0 <= x.min() and x.max() <= 2 * np.pi + 1e-5 for x in (*a, *c))


def test_each_detuned_copy_has_its_own_phase_the_partials_plus_the_beat():
    """The chip may not add two input tiles that also feed `sin` (one buffer cannot serve both kinds of unit), so the host
    gives each partial's detuned copy as a tile of its own. On the first live sample the sag sum is exactly 1, so the
    copy's phase is the partial's plus the beat's, both in radians per sample."""
    import dataclasses
    t = dataclasses.replace(FALLBACK, aiclk=800.0)
    g = gong_parameters(t, 8000, DEFAULT_PATCH)
    tiles = gong_inputs(np.array([0]), g, 8000)
    a, c = tiles[0:5], tiles[5:10]
    for slot, (freq, _, _) in enumerate(g.partials):
        assert a[slot].ravel()[0] == pytest.approx(freq * 2 * np.pi / 8000, abs=1e-5)
        assert c[slot].ravel()[0] - a[slot].ravel()[0] == pytest.approx(g.beat_hz * 2 * np.pi / 8000, abs=1e-5)
    assert (c[len(g.partials)].ravel()[0] == 0.0)  # an empty slot has no copy either


@pytest.mark.parametrize("tuning", ["bell", "tamtam", "singing_bowl"])
@pytest.mark.parametrize("rate", [8000, 44100])
def test_the_chips_formula_on_the_hosts_inputs_is_the_gong_device(tuning, rate):
    import dataclasses
    from tensixinet.music.patch import GONG_TUNINGS
    partials, amps = GONG_TUNINGS[tuning]
    patch = dataclasses.replace(DEFAULT_PATCH, gong_partials=partials, gong_amps=amps)
    t = dataclasses.replace(FALLBACK, aiclk=1350.0, asic_temperature=60.0, heartbeat=3)
    duration = 9.0
    n = int(duration * rate)
    g = gong_parameters(t, rate, patch)
    strikes = strike_times(t, duration)
    assert len(strikes) >= 2
    runs = [gong_on_host(gong_inputs(strike_ages(strikes, n, rate, g.length, back), g, rate), n) for back in range(GONG_OVERLAP)]
    assert np.allclose(gong_combine(runs), gong(t, duration, rate, 0, patch), atol=2e-4)


def test_three_overlapping_strikes_are_enough_for_the_closest_strikes_the_device_makes():
    import dataclasses
    t = dataclasses.replace(FALLBACK, aiclk=400.0)
    g = gong_parameters(t, 8000, DEFAULT_PATCH)
    strikes = strike_times(t, 60.0)
    spacing = np.diff(strikes).min()
    assert g.length / 8000 / spacing <= GONG_OVERLAP  # a strike never overlaps more than the three runs can hold


# --- the AVDog: the dogtone under the two halves of the undulation ------------------------------------------------------

from tensixinet.kernels.packing import avdog_inputs, avdog_on_host  # noqa: E402
from tensixinet.music.ttavdog import AVDOG_LEVEL, RED_GAIN, avdog_parameters, ttavdog  # noqa: E402


def test_the_avdog_inputs_are_ten_tiles_with_the_constants_folded_so_no_scalar_reaches_the_chip():
    import dataclasses
    t = dataclasses.replace(FALLBACK, asic_temperature=60.0, thm_limit_throttle=65.0)
    pitch, shape, rate_hz, decay_s = avdog_parameters(t)
    since = np.array([np.inf, 0.0, 0.1])
    tiles = avdog_inputs(since, rate_hz, decay_s, pitch, shape, 3, 8000)
    assert len(tiles) == 10 and all(x.shape == tiles[0].shape and x.dtype == np.float32 for x in tiles)
    x, ox, ip, iq, pu, e, aw2, aw, ar2, ar = (tile.ravel()[:3] for tile in tiles)
    assert (x + ox == pytest.approx([1, 1, 1])) and e[0] == SNAP_FLOOR and e[1] == 0.0 and pu[0] == 0.0
    peak = (1 + shape) / 2
    assert ip[0] == pytest.approx(1 / peak) and iq[0] == pytest.approx(1 / (1 - peak))
    assert aw2[0] == pytest.approx(2 * AVDOG_LEVEL) and aw[0] == pytest.approx(AVDOG_LEVEL)
    assert ar2[0] == pytest.approx(2 * AVDOG_LEVEL * RED_GAIN) and ar[0] == pytest.approx(AVDOG_LEVEL * RED_GAIN)


@pytest.mark.parametrize("pressure_limit", [120.0, 65.0, 61.0, 40.0])  # the last is full pressure, past the point where a saw would need an infinite slope
def test_the_chips_formula_on_the_hosts_inputs_is_the_ttavdog_device(pressure_limit):
    import dataclasses
    t = dataclasses.replace(FALLBACK, asic_temperature=60.0, power=50.0, thm_limit_throttle=pressure_limit)
    n = 8000 * 10
    pitch, shape, rate_hz, decay_s = avdog_parameters(t)
    since = seconds_since(strike_times(t, 10.0), n, 8000)
    got = avdog_on_host(avdog_inputs(since, rate_hz, decay_s, pitch, shape, n, 8000), n)
    assert np.allclose(got, ttavdog(t, 10.0, 8000, 0), atol=2e-4)



# --- ttcoco: eight taps summed and squashed, the same shape as the mixer but with eight inputs --------------------

def test_there_are_exactly_eight_tap_slots():
    assert COCO_TAPS == 8


def test_coco_inputs_scale_each_tap_by_the_devices_own_headroom_term_and_pack_all_eight():
    from tensixinet.music.ttcoco import LEVEL, N_TAPS
    taps = [np.full(10, 0.1 * (i + 1)) for i in range(N_TAPS)]
    tiles = coco_inputs(taps, n=10)
    assert len(tiles) == COCO_TAPS == N_TAPS and all(t.shape == tiles[0].shape for t in tiles)
    for i, tap in enumerate(taps):
        assert np.allclose(coco_on_host([tiles[i]] + [np.zeros_like(tiles[0])] * (COCO_TAPS - 1), 10),
                            np.tanh(LEVEL / math.sqrt(N_TAPS) * tap), atol=1e-6)


def test_coco_on_host_is_tanh_of_the_sum_and_matches_ttcoco_on_a_real_chip():
    import dataclasses
    from tensixinet.music.ttcoco import echo_tap, tap_bits, tap_delay_ms, tap_feedback, ttcoco
    t = dataclasses.replace(FALLBACK, gddr_temps=(28.0, 55.0, 70.0, 40.0, 90.0, 33.0, 60.0, 48.0), power=95.0)
    n = 24000
    from tensixinet.music.devices import quantussy_tone
    tone = quantussy_tone(t, n / 8000, 8000, 0)
    feedback = tap_feedback(t)
    taps = []
    for ch in range(COCO_TAPS):
        delay_ms = tap_delay_ms(t.gddr_temps[ch])
        bits = tap_bits(delay_ms)
        delay_samples = int(round(delay_ms / 1000.0 * 8000))
        taps.append(echo_tap(tone, delay_samples, feedback, bits))
    got = coco_on_host(coco_inputs(taps, n), n)
    assert np.allclose(got, ttcoco(t, n / 8000, 8000, 0), atol=1e-6)


def test_more_than_eight_taps_are_refused():
    with pytest.raises(ValueError):
        coco_inputs([np.zeros(4)] * 9, n=4)


def test_fewer_than_eight_taps_are_padded_with_silence():
    tiles = coco_inputs([np.full(5, 0.3)], n=5)
    assert len(tiles) == COCO_TAPS and not tiles[1].any() and not tiles[-1].any()


# --- ttchoir: the voiced part, a sum of sines, in groups of eight harmonics -------------------------------------------

from tensixinet.kernels.packing import CHOIR_GROUP, choir_compact, choir_expand, choir_groups, choir_inputs, choir_on_host  # noqa: E402


def test_a_choir_group_is_eight_harmonics_so_sixteen_tiles():
    assert CHOIR_GROUP == 8
    tiles = choir_inputs(np.zeros(10), [np.ones(10)], [1], 10)
    assert len(tiles) == 2 * CHOIR_GROUP and all(t.shape == tiles[0].shape and t.dtype == np.float32 for t in tiles)


def test_choir_inputs_reduce_each_harmonics_phase_to_one_turn_and_carry_its_level():
    """A phase of thousands of radians times a harmonic number up to 40: the chip must get it reduced to [0, 2 pi) in float64,
    because float32 alone would be off by about a radian at that size."""
    n = 30 * 8000
    phase = 2 * math.pi * 400.0 * np.arange(n) / 8000  # 75000 rad by the end
    levels = [np.linspace(0.0, 1.0, n), np.full(n, 0.25), np.linspace(1.0, 0.0, n)]
    numbers = [1, 17, 40]
    tiles = choir_inputs(phase, levels, numbers, n)
    p, a = tiles[:CHOIR_GROUP], tiles[CHOIR_GROUP:]
    for j, (lv, k) in enumerate(zip(levels, numbers)):
        got_p = unpack(p[j], n)
        assert got_p.min() >= 0.0 and got_p.max() < 2 * math.pi + 1e-6
        assert np.allclose(np.sin(got_p), np.sin(k * phase), atol=2e-6)
        assert np.allclose(unpack(a[j], n), lv, atol=1e-7)
    want = sum(lv * np.sin(k * phase) for lv, k in zip(levels, numbers))
    assert np.allclose(choir_on_host(tiles, n), want, atol=5e-6)


def test_unused_slots_of_a_group_are_silent():
    n = 2000
    tiles = choir_inputs(np.linspace(0, 50, n), [np.full(n, 0.5)], [3], n)
    for j in range(1, CHOIR_GROUP):
        assert not tiles[CHOIR_GROUP + j].any()  # level 0
    assert np.allclose(choir_on_host(tiles, n), 0.5 * np.sin(3 * np.linspace(0, 50, n)), atol=2e-6)


def test_more_than_eight_harmonics_in_one_group_are_refused():
    with pytest.raises(ValueError):
        choir_inputs(np.zeros(4), [np.zeros(4)] * 9, list(range(1, 10)), 4)


def _choir_tables(count=40, n=24000, sr=8000, seed=0):
    rng = np.random.default_rng(seed)
    sample_t = np.arange(n) / sr
    frame_t = np.arange(int(n / sr * 200) + 2) / 200
    harmonics = rng.uniform(0, 1, (len(frame_t), count)) / count
    phase = 2 * math.pi * np.cumsum(np.full(n, 220.0) * (1 + 0.01 * np.sin(sample_t))) / sr
    return sample_t, frame_t, harmonics, phase


@pytest.mark.parametrize("ks", [list(range(40)), list(range(13)), [0, 2, 5, 9, 11, 30, 31, 32, 33, 39], [7]])
def test_the_groups_cover_the_harmonics_in_order_eight_at_a_time_and_add_up_to_the_numpy_sum(ks):
    from tensixinet.music.ttchoir import harmonic_sum
    sample_t, frame_t, harmonics, phase = _choir_tables()
    n = len(sample_t)
    groups = list(choir_groups(sample_t, frame_t, harmonics, phase, ks))
    assert len(groups) == math.ceil(len(ks) / CHOIR_GROUP)
    got = sum(choir_on_host(g, n) for g in groups)
    want = harmonic_sum(sample_t, frame_t, harmonics, phase, ks)
    assert np.allclose(got, want, atol=1e-5) and np.abs(want).max() > 0.01  # one harmonic of level <= 1/40 still sounds


def test_no_harmonics_means_no_groups():
    assert list(choir_groups(*_choir_tables(), [])) == []


@pytest.mark.parametrize("onset", ["", "gong"], ids=["vowel", "syllable"])
def test_the_choir_on_the_host_formula_matches_ttchoir_on_a_real_chip(onset):
    """The host/chip split proven on its own: `choir_voice` with the kernel's formula in place of the sum is `ttchoir`."""
    import dataclasses
    from tensixinet.music.patch import DEFAULT_PATCH
    from tensixinet.music.ttchoir import choir_voice, ttchoir
    t = dataclasses.replace(FALLBACK, tensix_mask=0x3002, ddr_status=(1, 1, 1, 1), asic_temperature=55.0)
    patch = dataclasses.replace(DEFAULT_PATCH, choir_onset=onset, choir_coda=onset, choir_drift=0.4)

    def on_host(sample_t, frame_t, harmonics, phase, ks):
        n = len(sample_t)
        return sum((choir_on_host(g, n) for g in choir_groups(sample_t, frame_t, harmonics, phase, ks)), np.zeros(n))
    for sr in (8000, 44100):
        got, want = choir_voice(t, 4.0, sr, 2, patch, sines=on_host), ttchoir(t, 4.0, sr, 2, patch)
        assert np.abs(want).max() > 0.05 and np.allclose(got, want, atol=1e-5), float(np.abs(got - want).max())


# --- the choir's compact form: the phase once, the level tables at frame rate, expanded next to the chip -----------------------

def test_the_compact_choir_expands_to_the_same_tiles_as_the_groups_the_host_packs():
    sr, n = 8000, 24000
    sample_t, frame_t, harmonics, phase = _choir_tables(count=19, n=n, sr=sr)
    ks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18, 3, 9]
    compact = choir_compact(sr, frame_t, harmonics, phase, ks)
    want = list(choir_groups(sample_t, frame_t, harmonics, phase, ks))
    got = list(choir_expand(compact, n))
    assert len(got) == len(want) == 3
    for g, w in zip(got, want):
        assert len(g) == len(w) == 2 * CHOIR_GROUP
        for a, b in zip(g, w):
            assert a.shape == b.shape and np.array_equal(a, b)


def test_the_compact_choir_is_a_small_fraction_of_the_tiles_it_stands_for():
    sr, n = 8000, 24000
    sample_t, frame_t, harmonics, phase = _choir_tables(count=40, n=n, sr=sr)
    compact = choir_compact(sr, frame_t, harmonics, phase, range(40))
    sent = sum(a.nbytes for a in compact)
    tiles = sum(a.nbytes for g in choir_groups(sample_t, frame_t, harmonics, phase, range(40)) for a in g)
    assert sent < tiles / 20, (sent, tiles)
    assert compact[0].dtype == np.float64 and len(compact) == 5  # the phase keeps float64: a harmonic's turn is cut from it


def test_a_compact_choir_with_no_harmonics_expands_to_nothing():
    sample_t, frame_t, harmonics, phase = _choir_tables(count=4, n=800, sr=8000)
    assert list(choir_expand(choir_compact(8000, frame_t, harmonics, phase, []), 800)) == []
