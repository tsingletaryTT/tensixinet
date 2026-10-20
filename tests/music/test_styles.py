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
    assert set(p.steam_routes) == set(s.routes) and dict(p.emphasis) == dict(s.emphasis)
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
