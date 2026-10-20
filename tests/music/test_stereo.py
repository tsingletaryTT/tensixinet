# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The stereo chorus: every device placed in the field by its chip, its own place and the signals, then mixed by current in each ear."""
import dataclasses

import numpy as np
import pytest

from tensixinet.music.chorus import DEVICES, assign_device, render_chorus, slot_chip
from tensixinet.music.patch import DEFAULT_PATCH
from tensixinet.music.space import chip_azimuth, device_space
from tensixinet.music.stereo import TURN_DEPTH, place_voice, turn_depth
from tensixinet.telemetry import FALLBACK
from tensixinet.telemetry.mock import mock_telemetry

SR = 8000
CHIPS = mock_telemetry(4, 3)


def render(**kw):
    return render_chorus(CHIPS, kw.pop("duration", 4.0), SR, kw.pop("patch", DEFAULT_PATCH), **kw)


def test_a_stereo_render_is_two_channels_and_the_mono_render_is_what_it_was():
    mono, stereo = render(), render(stereo=True)
    assert mono.ndim == 1 and stereo.shape == (len(mono), 2) and stereo.dtype == np.float32 and np.isfinite(stereo).all()
    assert np.array_equal(mono, render(stereo=False))
    assert np.array_equal(stereo, render(stereo=True))


def test_the_two_ears_are_not_the_same_and_the_whole_has_a_comparable_loudness():
    stereo, mono = render(stereo=True), render()
    assert not np.allclose(stereo[:, 0], stereo[:, 1], atol=1e-3)
    ratio = float(np.sqrt(np.mean(stereo ** 2)) / np.sqrt(np.mean(mono ** 2)))
    assert 0.4 < ratio < 2.5


@pytest.mark.parametrize("device, side", [("lead", "left"), ("ttsteam", "right")])
def test_a_device_is_heard_from_the_side_its_chip_sits_on(device, side):
    """With four chips the dealt order puts the lead on chip 0 (left of the field) and ttsteam on chip 3 (right)."""
    out = render(stereo=True, duration=8.0, only=[device], cables=[])
    left, right = float(np.sum(out[:, 0] ** 2)), float(np.sum(out[:, 1] ** 2))
    assert (left > 1.5 * right) if side == "left" else (right > 1.5 * left)


def test_one_chip_plays_near_the_middle_and_a_chips_home_is_where_the_spread_says():
    one = mock_telemetry(1, 3)
    out = render_chorus(one, 6.0, SR, DEFAULT_PATCH, stereo=True)
    assert 0.5 < float(np.sum(out[:, 0] ** 2) / np.sum(out[:, 1] ** 2)) < 2.0
    assert [slot_chip(s, 4) for s in range(4)] == [0, 1, 2, 3]
    assert chip_azimuth(0, 4, CHIPS[0]) < chip_azimuth(3, 4, CHIPS[3])


def test_the_voice_that_has_the_floor_is_nearer_than_the_ones_that_have_not():
    assert turn_depth(np.ones(100), 0.35).max() == 0.0
    d = turn_depth(np.full(100, 0.35), 0.35)
    assert np.allclose(d, TURN_DEPTH)
    mid = turn_depth(np.full(100, 0.675), 0.35)
    assert 0 < mid[0] < TURN_DEPTH and turn_depth(np.array([1.0, 0.35]), 0.35)[1] > turn_depth(np.array([1.0, 0.35]), 0.35)[0]
    assert np.allclose(turn_depth(np.full(5, 0.2), 0.35), TURN_DEPTH)  # below the floor is the floor


def test_place_voice_puts_a_device_at_its_chips_home_plus_its_own_offset_and_stays_inside_the_field():
    t, n = CHIPS[3], int(4.0 * SR)
    voice = np.random.default_rng(2).uniform(-1, 1, n)
    out = place_voice("lead", 3, 4, t, voice, 4.0, SR, 0, DEFAULT_PATCH, chaos=0.0, depth_offset=np.zeros(n))
    assert out.shape == (n, 2)
    expected = chip_azimuth(3, 4, t) + device_space("lead")[0]
    left, right = float(np.sum(out[:, 0] ** 2)), float(np.sum(out[:, 1] ** 2))
    assert (right > left) == (expected > 0)
    far = place_voice("lead", 3, 4, dataclasses.replace(t, aiclk=1350.0), voice, 4.0, SR, 0, DEFAULT_PATCH, chaos=1.0, depth_offset=np.zeros(n))
    assert np.isfinite(far).all() and np.abs(far).max() <= np.abs(voice).max() * 1.5


def test_a_chaotic_chip_moves_its_voices_more_than_a_calm_one():
    t, n = CHIPS[1], int(20.0 * SR)
    voice = np.random.default_rng(2).uniform(-1, 1, n)
    def balance(chaos):
        out = place_voice("ttrax", 1, 4, t, voice, 20.0, SR, 0, DEFAULT_PATCH, chaos=chaos, depth_offset=np.zeros(n))
        blocks = out.reshape(-1, SR * 2, 2)
        return float(np.std([np.log((np.sum(b[:, 1] ** 2) + 1e-9) / (np.sum(b[:, 0] ** 2) + 1e-9)) for b in blocks]))
    assert balance(1.0) > 2 * balance(0.0)


def test_the_stereo_render_follows_the_mono_devices_choices_the_same_cables_and_the_same_soloist():
    soloist = [5]
    a, b = render(stereo=True, soloists=soloist, duration=6.0), render(stereo=True, soloists=soloist, duration=6.0, cables=[])
    assert not np.allclose(a, b)                                   # cables still bend the voices before they are placed
    mono_a, mono_b = render(soloists=soloist, duration=6.0), render(soloists=soloist, duration=6.0, cables=[])
    assert (np.abs(a - b).mean() > 0) == (np.abs(mono_a - mono_b).mean() > 0)


def test_stereo_works_with_one_device_auditioned_and_with_no_floor_issue_for_a_live_segment():
    out = render(stereo=True, only=["ttavdog"], duration=6.0)
    assert out.shape[1] == 2 and float(np.abs(out).max()) > 0
    seg = render(stereo=True, segment=3, soloists=[2], duration=2.0)
    assert seg.shape == (int(2.0 * SR), 2)


# --- gaps a mutation run found ---------------------------------------------------------------------------------------------------------

def _voice(n):
    return np.random.default_rng(9).uniform(-1, 1, n)


def test_a_dissolved_voice_has_left_the_field_by_the_left_edge_and_fallen_back():
    t, n = CHIPS[3], int(4.0 * SR)
    v = _voice(n)
    def place(d):
        return place_voice("lead", 3, 4, t, v, 4.0, SR, 0, DEFAULT_PATCH, chaos=0.0, depth_offset=np.zeros(n), dissolve=None if d is None else np.full(n, d))
    whole, gone, half = place(0.0), place(1.0), place(0.5)
    assert np.array_equal(whole, place(None))
    assert float(np.sum(gone[:, 1] ** 2)) < 0.02 * float(np.sum(gone ** 2))                            # hard left (a little width keeps some in the right ear)
    home_share = float(np.sum(whole[:, 1] ** 2) / np.sum(whole ** 2))
    assert home_share > 0.6 > float(np.sum(half[:, 1] ** 2) / np.sum(half ** 2)) > float(np.sum(gone[:, 1] ** 2) / np.sum(gone ** 2))
    assert float(np.sum(gone ** 2)) < 0.8 * float(np.sum(whole ** 2))                                   # and farther back, so quieter and darker


def test_a_hop_between_places_is_smoothed_so_it_does_not_click():
    t = dataclasses.replace(CHIPS[1], ddr_status=(1, 1, 1, 1), eth_mask=0xFFFF, aiclk=800.0)
    n = int(6.0 * SR)
    out = place_voice("ttrollz", 1, 4, t, np.ones(n), 6.0, SR, 0, DEFAULT_PATCH, chaos=0.0, depth_offset=np.zeros(n))
    settled = out[1000:, 0]                                       # past the first moments, where the voice itself starts from silence
    assert float(np.abs(np.diff(settled)).max()) < 0.01           # a lane hop of 0.8 of the field takes ~120 samples, not one
    assert float(np.ptp(settled)) > 0.2                           # and it does move
