# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Man With The Red Steam: a shift register stepping a noise snare, with the Tensix holes as its pattern."""
import dataclasses

import numpy as np
import pytest

from tensixinet.music.mapping import aiclk_to_bpm
from tensixinet.music.mwtrs import STEPS, dust_per_second, mwtrs, snare_decay, step_pattern
from tensixinet.telemetry import FALLBACK

SR = 8000


def tel(**kw):
    return dataclasses.replace(FALLBACK, **kw)


def test_the_register_is_as_long_as_the_chips_tensix_columns():
    assert STEPS == 14 and len(step_pattern(0x3FFF)) == STEPS


def test_a_step_fires_where_a_tensix_column_is_harvested():
    p = step_pattern(0x37F7)  # the live chip 0: columns 3 and 11 are off
    assert [i for i, hit in enumerate(p) if hit] == [3, 11]
    assert [i for i, hit in enumerate(step_pattern(0x3EDF)) if hit] == [5, 8]


def test_a_chip_with_every_column_enabled_never_fires_and_bits_above_the_columns_are_ignored():
    assert not any(step_pattern(0x3FFF)) and step_pattern(0xFFFF) == step_pattern(0x3FFF)
    assert all(step_pattern(0x0000))


def test_pressure_tightens_the_snare_from_three_quarters_of_a_second_to_75_ms():
    assert snare_decay(0.0) == pytest.approx(0.75) and snare_decay(1.0) == pytest.approx(0.075)
    assert snare_decay(0.5) == pytest.approx((0.75 + 0.075) / 2)


def test_pcie_traffic_sets_how_dense_the_dust_is():
    assert dust_per_second(1) < dust_per_second(3) < dust_per_second(6)
    assert dust_per_second(0) == 0 and dust_per_second(60) == dust_per_second(6)


def _rms(x, sr, start, length):
    seg = x[int(start * sr): int((start + length) * sr)]
    return float(np.sqrt(np.mean(seg ** 2)))


def _one_hit(**kw):
    """A chip with one column harvested (bit 0), so the only snare hit is on the first step; no dust."""
    return tel(tensix_mask=0x3FFE, pcie_usage=0, asic_temperature=30.0, power=10.0, aiclk=800.0, **kw)


def test_a_hit_is_a_noise_burst_on_its_step_that_decays():
    out = mwtrs(_one_hit(), 3.0, SR, 0)
    assert _rms(out, SR, 0.0, 0.05) > 0.2
    assert _rms(out, SR, 0.0, 0.05) > 1.5 * _rms(out, SR, 0.5, 0.05)  # the snare is fading half a second on
    assert _rms(out, SR, 2.5, 0.5) < 0.02  # and has all but died after three decay times


def test_a_hit_falls_to_a_third_after_one_decay_time():
    t = _one_hit()
    tau = snare_decay(0.0)
    out = mwtrs(t, 3.0, 44100, 0)
    first, later = _rms(out, 44100, 0.02, 0.05), _rms(out, 44100, 0.02 + tau, 0.05)
    assert later / first == pytest.approx(np.exp(-1), rel=0.25)


def test_hits_come_round_every_fourteen_steps_and_land_on_the_steps_of_the_chips_clock():
    t = tel(tensix_mask=0x3FF7, pcie_usage=0, asic_temperature=30.0, power=10.0, aiclk=1350.0)  # column 3 off
    step = 30.0 / aiclk_to_bpm(1350.0)
    out = mwtrs(t, 14 * step * 2 + 1.0, SR, 0)
    first = 3 * step
    assert _rms(out, SR, first, 0.02) > 0.1 and _rms(out, SR, first - 0.05, 0.03) < 0.02
    assert _rms(out, SR, first + 14 * step, 0.02) > 0.1


def test_a_hotter_chip_makes_a_tighter_snare():
    cool = mwtrs(_one_hit(), 2.0, SR, 0)
    hot = mwtrs(dataclasses.replace(_one_hit(), asic_temperature=88.0, power=124.0), 2.0, SR, 0)
    assert _rms(hot, SR, 0.3, 0.1) < 0.3 * _rms(cool, SR, 0.3, 0.1)


def test_the_dust_is_a_sparse_crackle_that_a_healthy_chip_plays_between_no_hits():
    healthy = tel(tensix_mask=0x3FFF, pcie_usage=6)
    out = mwtrs(healthy, 4.0, SR, 0)
    assert 0 < np.abs(out).max() <= 0.5 and np.mean(np.abs(out) > 0) < 0.05
    assert not np.any(mwtrs(tel(tensix_mask=0x3FFF, pcie_usage=0), 1.0, SR, 0))


def test_more_pcie_traffic_means_more_dust():
    few = np.count_nonzero(mwtrs(tel(tensix_mask=0x3FFF, pcie_usage=1), 8.0, SR, 0))
    many = np.count_nonzero(mwtrs(tel(tensix_mask=0x3FFF, pcie_usage=6), 8.0, SR, 0))
    assert many > 3 * few


def test_the_device_is_deterministic_bounded_and_the_seed_changes_the_noise():
    t = tel(tensix_mask=0x37F7)
    a = mwtrs(t, 4.0, SR, 3)
    assert np.array_equal(a, mwtrs(t, 4.0, SR, 3)) and not np.array_equal(a, mwtrs(t, 4.0, SR, 4))
    assert a.dtype == np.float32 and np.abs(a).max() <= 1.0


def test_pressure_and_pcie_readings_outside_their_ranges_hold_the_nearest_end():
    assert snare_decay(2.0) == pytest.approx(0.075) and snare_decay(-1.0) == pytest.approx(0.75)
    assert dust_per_second(-5) == 0


def test_dust_pops_fifteen_times_a_second_for_each_pcie_step():
    assert dust_per_second(3) == pytest.approx(45.0)
    pops = np.count_nonzero(mwtrs(tel(tensix_mask=0x3FFF, pcie_usage=6), 20.0, SR, 0))
    assert pops == pytest.approx(90 * 20, rel=0.15)


def test_a_hit_opens_over_two_milliseconds_and_does_not_click_in():
    out = mwtrs(_one_hit(), 1.0, SR, 0)
    assert np.sqrt(np.mean(out[:6] ** 2)) < 0.4 * np.sqrt(np.mean(out[20:40] ** 2))


def test_a_snare_hit_peaks_at_its_level_and_never_above_it():
    out = mwtrs(_one_hit(), 1.0, SR, 0)
    assert 0.5 < np.abs(out).max() <= 0.6 + 1e-6


def test_the_snare_noise_is_rolled_off_above_four_kilohertz_so_it_does_not_hiss():
    out = mwtrs(_one_hit(), 1.0, 44100, 0)
    spec = np.abs(np.fft.rfft(out)) ** 2
    freqs = np.fft.rfftfreq(len(out), 1 / 44100)
    assert spec[freqs > 12000].sum() / spec.sum() < 0.08  # white noise would put 46% there
    assert spec[freqs < 6000].sum() / spec.sum() > 0.6


def test_the_roll_off_does_nothing_at_a_sample_rate_too_low_to_have_anything_above_it():
    from tensixinet.music.mwtrs import lowpass
    x = np.random.default_rng(0).uniform(-1, 1, 1000)
    assert np.allclose(lowpass(x, 6000.0, 8000), x, atol=0.05)


def test_the_dust_stays_sharp_and_is_not_rolled_off():
    out = mwtrs(tel(tensix_mask=0x3FFF, pcie_usage=6), 4.0, 44100, 0)
    pops = out[out != 0]
    assert pops.size > 100 and np.abs(pops).max() > 0.2  # each crackle is a single full-height sample


def test_the_roll_off_passes_a_steady_level_unchanged():
    from tensixinet.music.mwtrs import lowpass
    assert np.allclose(lowpass(np.ones(500), 4000.0, 44100)[100:], 1.0, atol=0.01)
