# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Patch cables: one device's sound bends, crunches, swells or folds another's, in the manner of the purple jacks."""
import math

import numpy as np
import pytest

from tensixinet.music.cables import (
    ANCHOR,
    KINDS,
    Cable,
    apply_cables,
    draw_cables,
    envelope,
    strength,
)

SR = 8000
NAMES = ["a", "b", "c", "d", ANCHOR, "e"]


def sine(freq, seconds=1.0, amp=0.5):
    t = np.arange(int(seconds * SR)) / SR
    return amp * np.sin(2 * math.pi * freq * t)


def rms(x):
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))


def patched(kind, depth=1.0, source=None, target=None, chaos=1.0):
    """The target voice after one cable from a source voice, and the untouched target."""
    source = sine(50.0, amp=0.8) if source is None else source
    target = sine(440.0) if target is None else target
    out = apply_cables([source, target], ["src", "dst"], [Cable("src", "dst", kind, depth)], [chaos, chaos], SR)
    return out[1], target


# --- drawing the cables -----------------------------------------------------------------------------------------

def test_the_same_seed_draws_the_same_cables_and_another_seed_does_not():
    assert draw_cables(7, NAMES) == draw_cables(7, NAMES)
    assert any(draw_cables(s, NAMES) != draw_cables(7, NAMES) for s in range(8, 20))


@pytest.mark.parametrize("seed", range(30))
def test_a_drawing_has_five_to_eight_cables_between_two_different_devices_with_every_kind_in_use(seed):
    cables = draw_cables(seed, NAMES)
    assert 5 <= len(cables) <= 8
    assert all(c.source != c.target and c.source in NAMES and c.target in NAMES for c in cables)
    assert len({(c.source, c.target) for c in cables}) == len(cables)
    assert {c.kind for c in cables} == set(KINDS)
    assert all(0.3 <= c.depth <= 1.0 for c in cables)


@pytest.mark.parametrize("seed", range(30))
def test_the_choir_always_modulates_another_voice_and_is_modulated_by_one(seed):
    cables = draw_cables(seed, NAMES)
    assert any(c.source == ANCHOR for c in cables)
    assert any(c.target == ANCHOR for c in cables)


def test_a_cable_drawing_needs_two_devices():
    with pytest.raises(ValueError):
        draw_cables(1, ["only"])


def test_without_a_choir_the_drawing_still_works():
    cables = draw_cables(3, ["a", "b", "c"])
    assert len(cables) >= 5 and all(c.source != ANCHOR and c.target != ANCHOR for c in cables)


# --- how hard a cable pushes ------------------------------------------------------------------------------------

def test_a_quiet_source_chip_still_pushes_a_quarter_and_a_chaotic_one_pushes_in_full():
    c = Cable("a", "b", "fm", 0.8)
    assert strength(c, 0.0) == pytest.approx(0.2)
    assert strength(c, 1.0) == pytest.approx(0.8)
    assert strength(c, 0.5) == pytest.approx(0.8 * 0.625)


# --- the envelope that drives the cable --------------------------------------------------------------------------

def test_the_envelope_follows_loudness_and_peaks_at_one():
    x = np.concatenate([np.zeros(SR // 2), sine(200.0, 0.5)])
    e = envelope(x, SR)
    assert e.shape == x.shape
    assert e.max() == pytest.approx(1.0)
    assert e[: SR // 4].max() < 0.01 and e[-SR // 8:].min() > 0.5


def test_a_silent_source_has_no_envelope():
    assert not np.any(envelope(np.zeros(SR), SR))


def test_a_quiet_source_drives_as_hard_as_a_loud_one():
    assert np.allclose(envelope(sine(100.0, amp=0.01), SR), envelope(sine(100.0, amp=0.9), SR), atol=1e-6)


# --- what each kind does ----------------------------------------------------------------------------------------

@pytest.mark.parametrize("kind", KINDS)
def test_a_silent_source_leaves_the_target_alone(kind):
    out, target = patched(kind, source=np.zeros(SR))
    assert np.allclose(out, target, atol=1e-9)


@pytest.mark.parametrize("kind", KINDS)
def test_a_cable_of_no_depth_leaves_the_target_alone(kind):
    out, target = patched(kind, depth=0.0)
    assert np.allclose(out, target, atol=1e-9)


@pytest.mark.parametrize("kind", KINDS)
def test_every_kind_changes_the_target_and_keeps_it_finite_and_below_full_scale(kind):
    out, target = patched(kind)
    assert not np.allclose(out, target, atol=1e-3)
    assert np.isfinite(out).all() and np.abs(out).max() <= 1.0


def spectrum_share(x, freq, width=15.0):
    s = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(len(x), 1 / SR)
    return float(s[np.abs(f - freq) < width].sum() / s.sum())


def test_fm_spreads_the_targets_energy_into_sidebands():
    out, target = patched("fm")
    assert spectrum_share(target, 440.0) > 0.95
    assert spectrum_share(out, 440.0) < 0.6


def spread_around(x, freq):
    s = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(len(x), 1 / SR)
    return float(np.sqrt((s * (f - freq) ** 2).sum() / s.sum()))


def test_fm_follows_the_source_waveform_itself_so_a_faster_source_makes_wider_sidebands():
    slow, _ = patched("fm", source=sine(20.0, amp=0.8))
    fast, _ = patched("fm", source=sine(200.0, amp=0.8))
    assert spread_around(fast, 440.0) > 3 * spread_around(slow, 440.0)


def test_inflate_makes_the_target_louder_and_keeps_its_sign():
    out, target = patched("inflate")
    assert rms(out) > 1.3 * rms(target)
    big = np.abs(target) > 0.05
    assert np.all(np.sign(out[big]) == np.sign(target[big]))


def test_inflate_swells_only_while_the_source_sounds():
    source = np.concatenate([np.zeros(SR // 2), sine(100.0, 0.5, 0.8)])
    out, target = patched("inflate", source=source)
    assert rms(out[: SR // 4]) == pytest.approx(rms(target[: SR // 4]), rel=1e-3)
    assert rms(out[-SR // 4:]) > 1.3 * rms(target[-SR // 4:])


def test_crunch_leaves_fewer_distinct_levels_and_adds_error():
    out, target = patched("crunch")
    # the step follows the source's loudness, which is steady after its first 20 ms
    late = slice(SR // 2, None)
    assert len(np.unique(np.round(out[late], 9))) < len(np.unique(np.round(target[late], 9))) / 4
    assert rms(out - target) > 0.02


def test_crunch_coarsens_more_when_the_source_is_louder_in_the_moment():
    source = np.concatenate([np.zeros(SR // 2), sine(100.0, 0.5, 0.8)])
    out, target = patched("crunch", source=source)
    assert np.allclose(out[: SR // 4], target[: SR // 4], atol=1e-9)
    assert rms(out[-SR // 4:] - target[-SR // 4:]) > 0.02


def test_fold_adds_harmonics_the_target_did_not_have():
    out, target = patched("fold")
    assert spectrum_share(target, 1320.0) < 1e-6
    assert spectrum_share(out, 1320.0) > 0.01


def test_a_deeper_cable_changes_the_target_more():
    for kind in KINDS:
        light, target = patched(kind, depth=0.2)
        deep, _ = patched(kind, depth=1.0)
        assert rms(deep - target) > rms(light - target), kind


def test_a_chaotic_source_chip_changes_the_target_more_than_a_calm_one():
    for kind in KINDS:
        calm, target = patched(kind, chaos=0.0)
        wild, _ = patched(kind, chaos=1.0)
        assert rms(wild - target) > rms(calm - target), kind


# --- the plumbing -----------------------------------------------------------------------------------------------

def test_the_sources_are_read_as_they_were_rendered_and_not_changed_by_a_later_cable():
    a, b = sine(50.0, amp=0.8), sine(440.0)
    a0, b0 = a.copy(), b.copy()
    out = apply_cables([a, b], ["a", "b"], [Cable("a", "b", "fm", 1.0), Cable("b", "a", "inflate", 1.0)], [1.0, 1.0], SR)
    assert np.array_equal(a, a0) and np.array_equal(b, b0)
    # a is driven by the raw b, not by the b that the first cable bent
    alone = apply_cables([a, b], ["a", "b"], [Cable("b", "a", "inflate", 1.0)], [1.0, 1.0], SR)
    assert np.array_equal(out[0], alone[0])


def test_cables_to_one_target_apply_in_order():
    src, dst = sine(50.0, amp=0.8), sine(440.0)
    both = apply_cables([src, dst], ["s", "d"], [Cable("s", "d", "fm", 1.0), Cable("s", "d", "crunch", 1.0)], [1.0, 1.0], SR)[1]
    fm_only = apply_cables([src, dst], ["s", "d"], [Cable("s", "d", "fm", 1.0)], [1.0, 1.0], SR)[1]
    crunch_only = apply_cables([src, dst], ["s", "d"], [Cable("s", "d", "crunch", 1.0)], [1.0, 1.0], SR)[1]
    assert not np.allclose(both, fm_only) and not np.allclose(both, crunch_only)
    # crunch ran on the bent voice: it is the crunch of the fm result, not of the plain target
    again = apply_cables([src, fm_only], ["s", "d"], [Cable("s", "d", "crunch", 1.0)], [1.0, 1.0], SR)[1]
    assert np.allclose(both, again)


def test_a_cable_to_or_from_a_device_that_is_not_there_is_skipped():
    a = sine(440.0)
    cables = [Cable("a", "b", "fm", 1.0), Cable("b", "a", "fm", 1.0), Cable("zz", "a", "fm", 1.0)]
    out = apply_cables([a, None], ["a", "b"], cables, [1.0, 1.0], SR)
    assert np.array_equal(out[0], a) and out[1] is None


def test_every_slot_of_a_target_device_is_modulated_and_the_first_slot_of_the_source_drives_them():
    src = sine(50.0, amp=0.8)
    other = sine(30.0, amp=0.8)
    dst = sine(440.0)
    out = apply_cables([src, dst, other, dst.copy()], ["s", "d", "s", "d"], [Cable("s", "d", "inflate", 1.0)], [1.0, 1.0, 0.0, 1.0], SR)
    assert np.array_equal(out[1], out[3]) and not np.allclose(out[1], dst)
    assert np.array_equal(out[2], other)


def test_the_source_slots_own_chip_chaos_sets_the_strength():
    src, dst = sine(50.0, amp=0.8), sine(440.0)
    cable = [Cable("s", "d", "fm", 1.0)]
    first = apply_cables([src, dst, src], ["s", "d", "s"], cable, [0.0, 0.0, 1.0], SR)[1]
    calm = apply_cables([src, dst], ["s", "d"], cable, [0.0, 0.0], SR)[1]
    assert np.array_equal(first, calm)


def test_the_length_of_the_voices_must_match_the_names():
    with pytest.raises(ValueError):
        apply_cables([sine(1.0)], ["a", "b"], [], [1.0], SR)


# --- the sizes of the effects, pinned to numbers -------------------------------------------------------------------
# A steady full-scale source gives an envelope of 1 once its first 20 ms have passed, and a chaos of 1 gives the whole depth.

STEADY = np.full(SR, 0.8)
LATE = slice(SR // 2, SR - SR // 4)


def steady_patch(kind, target):
    return apply_cables([STEADY, target], ["s", "d"], [Cable("s", "d", kind, 1.0)], [1.0, 1.0], SR)[1]


def test_fm_at_full_strength_moves_the_read_point_by_three_milliseconds():
    ramp = np.arange(SR, dtype=np.float64)  # one unit per sample, so the shift in samples is the change in value
    out = steady_patch("fm", ramp)
    assert np.allclose((out - ramp)[LATE], 0.003 * SR)


def test_crunch_at_full_strength_rounds_to_steps_of_a_half():
    out = steady_patch("crunch", np.linspace(-0.7, 0.7, SR))
    assert np.allclose(out[LATE] * 2, np.round(out[LATE] * 2))


def test_inflate_at_full_strength_triples_a_quiet_level():
    assert np.allclose(steady_patch("inflate", np.full(SR, 0.1))[LATE], 0.3)


def test_inflate_passes_a_level_up_to_nine_tenths_and_bends_what_is_above_it_toward_one():
    out = steady_patch("inflate", np.full(SR, 0.5))[LATE]  # 1.5 after the gain
    assert np.allclose(out, 0.9 + 0.1 * math.tanh(0.6 / 0.1))


def test_fold_reflects_what_passes_nine_tenths_back_inside():
    out = steady_patch("fold", np.full(SR, 0.5))[LATE]  # 2.5 after the drive
    assert np.allclose(out, -0.7)


def test_the_envelope_averages_over_twenty_milliseconds():
    x = np.concatenate([np.zeros(800), np.ones(800)])
    e = envelope(x, SR)
    assert e[800 + 79] == pytest.approx(0.5, abs=0.01) and e[800 + 159] == pytest.approx(1.0)
