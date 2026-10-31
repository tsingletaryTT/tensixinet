# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Styles: named patches for listeners who want a genre before they want a circuit (chiptune, enya, blues, aphex).

A style narrows what a patch may draw (the scales, the gong tunings, the shape of the motif, how long a turn is, which steam routes
are on) and sets a balance for the band. The seed still decides the rest, and the system still picks which of the style's scales
and tunings from the chips' readings, so no two runs of a style match."""
import dataclasses
import json

import numpy as np
import pytest

from tensixinet.music.chorus import device_names, render_chorus, slot_weights
from tensixinet.music.patch import (
    DEFAULT_PATCH, GONG_TUNINGS, SCALES, STEAM_ROUTES, draw_patch, load_patch, rotate_mode, save_patch,
)
from tensixinet.music.styles import STYLES, draw_styled_patch
from tensixinet.telemetry import FALLBACK
from tensixinet.telemetry.mock import mock_telemetry

CHIPS = mock_telemetry(4, 2)


def chips(spread=24.0):
    return [dataclasses.replace(FALLBACK, gddr_temps=(30.0, 30.0 + spread) * 4) for _ in range(2)]


def test_there_are_four_styles_each_described_and_made_of_real_pieces():
    assert tuple(STYLES) == ("chiptune", "enya", "blues", "aphex")
    for name, s in STYLES.items():
        assert s.name == name and s.blurb and len(s.blurb) > 20
        assert set(s.scales) <= set(SCALES) and set(s.gong_tunings) <= set(GONG_TUNINGS) and set(s.routes) <= set(STEAM_ROUTES)
        assert set(dict(s.emphasis)) <= set(device_names()) and all(v > 0 for _, v in s.emphasis)
        assert s.modes in ("root", "any") and s.bars and s.motif_lengths and s.rax_bars


@pytest.mark.parametrize("name", list(STYLES))
@pytest.mark.parametrize("seed", range(40))
def test_a_styled_patch_stays_inside_its_style_and_is_playable(name, seed):
    s, p = STYLES[name], draw_styled_patch(seed, CHIPS, name)
    assert p.style == name and p.seed == seed
    assert p.scale_name in s.scales and p.gong_tuning in s.gong_tunings
    assert (s.modes == "any") or p.mode == 0
    assert p.scale == rotate_mode(SCALES[p.scale_name], p.mode)
    assert len(p.motif) in s.motif_lengths and p.motif[0] == 0 and len(set(p.motif)) >= 3
    assert p.bars_per_turn in s.bars and p.rax_bars in s.rax_bars
    assert s.floor[0] <= p.floor <= s.floor[1] and s.beat[0] <= p.gong_beat_hz <= s.beat[1]
    lo, hi = p.shimmer_band
    assert s.band_low[0] <= lo <= s.band_low[1] and lo < hi <= 20500
    assert set(p.steam_routes) == set(s.routes) and dict(p.emphasis) == dict(s.emphasis) and p.snare == s.snare
    assert p.gong_partials[0] == 1.0 and len(p.gong_partials) <= 5


@pytest.mark.parametrize("name", list(STYLES))
def test_a_style_is_deterministic_and_still_varies_with_the_seed(name):
    assert draw_styled_patch(7, CHIPS, name) == draw_styled_patch(7, CHIPS, name)
    drawn = [draw_styled_patch(s, CHIPS, name) for s in range(60)]
    assert len({p.motif for p in drawn}) >= 40 and len({p.gong_beat_hz for p in drawn}) >= 40 and len({p.floor for p in drawn}) >= 30


def test_the_system_still_picks_which_of_a_styles_scales_from_the_memory_temperatures():
    names = {draw_styled_patch(1, chips(spread), "blues").scale_name for spread in (0.5, 4, 8, 12, 16, 20)}
    assert len(names) >= 3 and names <= set(STYLES["blues"].scales)
    assert len({draw_styled_patch(s, chips(8), "blues").scale_name for s in range(30)}) == 1   # the seed does not move it


def test_the_styles_really_sound_different_in_what_they_draw():
    def mean(name, field):
        return float(np.mean([getattr(draw_styled_patch(s, CHIPS, name), field) for s in range(60)]))
    assert mean("enya", "bars_per_turn") > 4 > mean("aphex", "bars_per_turn") and mean("chiptune", "bars_per_turn") < 2.1
    assert mean("enya", "floor") > mean("aphex", "floor") + 0.15
    assert mean("aphex", "gong_beat_hz") > mean("enya", "gong_beat_hz") + 1.0
    assert len(STYLES["aphex"].routes) == 2 and len(STYLES["enya"].routes) <= 1
    assert "blues" in STYLES["blues"].scales and "octatonic" in STYLES["aphex"].scales and "dorian" in STYLES["enya"].scales
    emphasis = {n: dict(STYLES[n].emphasis) for n in STYLES}
    assert emphasis["enya"]["ttchoir"] > emphasis["enya"]["ttsteam"] and emphasis["aphex"]["ttsteam"] > emphasis["aphex"]["ttchoir"]
    assert emphasis["chiptune"]["ttquantus"] > 1 > emphasis["chiptune"]["ttchoir"] and emphasis["blues"]["ttrax"] > 1


def test_an_unknown_style_is_refused_with_the_list():
    with pytest.raises(ValueError, match="chiptune, enya, blues, aphex"):
        draw_styled_patch(1, CHIPS, "polka")


# --- emphasis: the band's balance, in the mix and in who gets the floor ---------------------------------------------------------

def _weights(patch, chips=CHIPS):
    return slot_weights(chips, patch)


def test_emphasis_scales_how_often_a_device_gets_the_floor_and_nothing_else():
    base = _weights(DEFAULT_PATCH)
    from tensixinet.music.chorus import assign_device, n_slots
    boosted = _weights(dataclasses.replace(DEFAULT_PATCH, emphasis=(("ttchoir", 3.0),)))
    for slot in range(n_slots(len(CHIPS))):
        factor = 3.0 if assign_device(slot).__name__ == "ttchoir" else 1.0
        assert boosted[slot] == pytest.approx(base[slot] * factor)
    assert _weights(DEFAULT_PATCH) == slot_weights(CHIPS)


def test_emphasis_scales_a_devices_level_in_the_mix():
    silent = dataclasses.replace(DEFAULT_PATCH, emphasis=tuple((n, 1e-9) for n in device_names()))
    quiet = render_chorus(CHIPS, 2.0, 8000, silent, cables=[])
    assert float(np.abs(quiet).max()) < 1e-4 < float(np.abs(render_chorus(CHIPS, 2.0, 8000, DEFAULT_PATCH, cables=[])).max())
    same = dataclasses.replace(DEFAULT_PATCH, emphasis=(("gong", 1.0),))
    assert np.array_equal(render_chorus(CHIPS, 2.0, 8000, same), render_chorus(CHIPS, 2.0, 8000, DEFAULT_PATCH))
    changed = dataclasses.replace(DEFAULT_PATCH, emphasis=(("gong", 0.0),))
    assert not np.allclose(render_chorus(CHIPS, 6.0, 8000, changed), render_chorus(CHIPS, 6.0, 8000, DEFAULT_PATCH))


def test_auditioning_with_only_ignores_the_balance_so_a_device_is_heard_as_it_is():
    loud = dataclasses.replace(DEFAULT_PATCH, emphasis=(("gong", 5.0),))
    assert np.array_equal(render_chorus(CHIPS, 4.0, 8000, loud, only=["gong"]), render_chorus(CHIPS, 4.0, 8000, DEFAULT_PATCH, only=["gong"]))


# --- saving and loading ------------------------------------------------------------------------------------------------------

def test_a_styled_patch_round_trips_and_an_old_patch_has_no_style(tmp_path):
    p = draw_styled_patch(3, CHIPS, "enya")
    save_patch(p, tmp_path / "e.json")
    assert load_patch(tmp_path / "e.json") == p
    d = json.loads(json.dumps(dataclasses.asdict(DEFAULT_PATCH)))
    del d["style"], d["emphasis"]
    (tmp_path / "old.json").write_text(json.dumps(d))
    old = load_patch(tmp_path / "old.json")
    assert old.style == "" and old.emphasis == ()


@pytest.mark.parametrize("bad", [[["gong", 0.0]], [["gong", -1.0]], [["gong"]], "gong"])
def test_a_broken_emphasis_is_refused(tmp_path, bad):
    d = json.loads(json.dumps(dataclasses.asdict(DEFAULT_PATCH)))
    d["emphasis"] = bad
    (tmp_path / "b.json").write_text(json.dumps(d))
    with pytest.raises(ValueError):
        load_patch(tmp_path / "b.json")


def test_a_plain_draw_is_untouched_by_styles_existing():
    assert draw_patch(11, CHIPS).style == "" and draw_patch(11, CHIPS).emphasis == ()


def test_no_style_turns_on_more_than_two_steam_routes_and_none_gives_the_snare_a_boost_above_one():
    for name, style in STYLES.items():
        assert len(style.routes) <= 2, name
        assert dict(style.emphasis)["ttsteam"] <= 1.0, name


def test_each_style_sets_how_much_of_the_snare_plays_and_only_aphex_and_blues_keep_much_of_it():
    assert {n: s.snare for n, s in STYLES.items()} == {"chiptune": 0.5, "enya": 0.1, "blues": 0.5, "aphex": 0.6}


def test_the_gong_carries_the_beat_in_every_style_as_much_as_the_snare_does():
    for name, style in STYLES.items():
        e = dict(style.emphasis)
        assert e["gong"] >= e["ttsteam"], name
    for name in ("chiptune", "blues", "aphex"):
        assert dict(STYLES[name].emphasis)["gong"] >= 1.0, name


def test_each_style_sets_how_much_goes_through_the_coco_and_how_big_the_room_is():
    assert {n: s.bus for n, s in STYLES.items()} == {
        "chiptune": (0.4, 0.1, 0.6), "enya": (0.35, 0.15, 1.6), "blues": (0.3, 0.3, 1.4), "aphex": (0.7, 0.25, 1.0)}
    for name in STYLES:
        p = draw_styled_patch(3, CHIPS, name)
        assert (p.coco_send, p.coco_reverb, p.reverb_s) == STYLES[name].bus


def test_each_style_sets_how_many_of_the_cocos_notes_are_rests():
    assert {n: s.rests for n, s in STYLES.items()} == {"chiptune": 0.25, "enya": 0.3, "blues": 0.35, "aphex": 0.4}
    for name in STYLES:
        assert draw_styled_patch(3, CHIPS, name).rests == STYLES[name].rests


def test_each_style_sets_how_many_of_the_leads_notes_and_the_rolls_pulses_rest():
    assert {n: s.pulse_rests for n, s in STYLES.items()} == {
        "chiptune": (0.1, 0.15), "enya": (0.2, 0.2), "blues": (0.25, 0.2), "aphex": (0.15, 0.3)}
    for name in STYLES:
        p = draw_styled_patch(3, CHIPS, name)
        assert (p.lead_rests, p.roll_rests) == STYLES[name].pulse_rests


def test_each_style_sets_how_much_the_looping_lines_pitch_is_moved():
    assert {n: s.mod for n, s in STYLES.items()} == {"chiptune": 0.15, "enya": 0.8, "blues": 0.45, "aphex": 0.9}
    for name in STYLES:
        assert draw_styled_patch(3, CHIPS, name).quantus_mod == STYLES[name].mod


def test_in_enya_the_looping_coco_devices_are_in_the_background_by_balance_and_by_what_is_heard():
    emphasis = dict(STYLES["enya"].emphasis)
    others = [d for d in emphasis if d not in ("ttcoco", "ttquantus", "ttsteam", "ttrollz", "ttavdog")]
    assert all(emphasis["ttcoco"] < emphasis[d] and emphasis["ttquantus"] < emphasis[d] for d in others)
    assert emphasis["ttcoco"] <= 0.4 and emphasis["ttquantus"] <= 0.4
    # and by ear's measure: their share of the mix's energy (each device's level, the balance and the turn gains)
    from tensixinet.music.chorus import assign_device, emphasis_of, n_slots, plan_turns, slot_chip, turn_gains
    shares = []
    for seed in (1, 2, 3):
        tel = mock_telemetry(4, seed=seed)
        patch = draw_styled_patch(seed, tel, "enya")
        planned, turn_s = plan_turns(tel, 10.0, patch, None)
        gains = turn_gains(planned, n_slots(4), 10 * 8000, 8000, turn_s, floor=patch.floor)
        energy = {}
        for s in range(n_slots(4)):
            device = assign_device(s)
            voice = device(tel[slot_chip(s, 4)], 10.0, 8000, seed=s, patch=patch)
            energy[device.__name__] = float(np.mean((emphasis_of(patch, device.__name__) * gains[s] * voice) ** 2))
        shares.append((energy["ttcoco"] + energy["ttquantus"]) / sum(energy.values()))
    assert max(shares) < 0.2 and np.mean(shares) < 0.15


def test_enya_keeps_the_low_voices_and_the_room_down_so_the_stereo_field_keeps_its_shape():
    # Measured against Dntel's "Enya Mixes": left/right correlation 0.2 to 0.65. With a send of 0.8 into a 3.5 s room enya was 0.1 to 0.2,
    # a wash with nothing placed in it.
    emphasis = dict(STYLES["enya"].emphasis)
    assert (emphasis["ttdrone"], emphasis["ultrasonic_shimmer"], emphasis["lead"], emphasis["ttchoir"]) == (0.9, 1.0, 1.3, 1.8)
    corr = []
    for seed in (1, 2, 3, 5):
        tel = mock_telemetry(4, seed=seed)
        audio = render_chorus(tel, 24.0, 22050, draw_styled_patch(seed, tel, "enya"), stereo=True)
        corr.append(float(np.corrcoef(audio[:, 0], audio[:, 1])[0, 1]))
    assert np.mean(corr) > 0.25 and min(corr) > 0.1, corr  # the old settings gave a mean of 0.03 on these seeds


def test_each_style_sets_the_choirs_consonant_mix_and_crush():
    assert {n: (s.choir_mix, s.choir_crush) for n, s in STYLES.items()} == {
        "chiptune": ((0.5, 0.3, 0.2), 1.0),
        "enya": ((0.1, 0.3, 0.6), 0.0),
        "blues": ((0.7, 0.15, 0.15), 0.3),
        "aphex": ((0.4, 0.4, 0.2), 0.8),
    }
    enya, blues = STYLES["enya"].choir_mix, STYLES["blues"].choir_mix
    assert enya[0] < enya[2] and blues[0] > max(blues[1:])  # enya leans soft, blues plosive
    for name in STYLES:
        p = draw_styled_patch(3, CHIPS, name)
        assert (p.choir_mix, p.choir_crush) == (STYLES[name].choir_mix, STYLES[name].choir_crush)
        plain = draw_patch(3, CHIPS)
        # the sources still come from the seed
        assert (p.choir_onset, p.choir_coda, p.choir_howl, p.choir_howl_odds, p.choir_drift) == (
            plain.choir_onset, plain.choir_coda, plain.choir_howl, plain.choir_howl_odds, plain.choir_drift)
