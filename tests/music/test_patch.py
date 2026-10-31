# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""A patch is every drawn choice of one run: replayable, valid, and partly decided by the system."""
import dataclasses
import json

import pytest

from tensixinet.music.lead import MOTIF
from tensixinet.music.patch import (
    DEFAULT_PATCH,
    GONG_STRETCH,
    GONG_TUNINGS,
    MODE_COUNT_MAX,
    RAX_BAR_COUNTS,
    SCALES,
    rotate_mode,
    draw_patch,
    load_patch,
    save_patch,
    seed_from_telemetry,
)
from tensixinet.telemetry import FALLBACK


def chips(gddr=(30.0,) * 8, asic=48.5, n=2):
    return [dataclasses.replace(FALLBACK, gddr_temps=gddr, asic_temperature=asic) for _ in range(n)]


FLAT = chips()
UNEVEN = chips(gddr=(30.0, 54.0) * 4)


# --- the default patch is today's sound ----------------------------------------

def test_default_patch_is_the_pre_patch_sound():
    p = DEFAULT_PATCH
    assert p.scale == (0, 2, 4, 7, 9) and p.motif == MOTIF
    assert p.gong_partials == (1.0, 2.76, 5.40, 8.45) and p.gong_amps == (1.0, 0.6, 0.35, 0.2)
    assert p.gong_beat_hz == 0.7 and p.shimmer_band == (12000.0, 20000.0)
    assert p.bars_per_turn == 2 and p.floor == 0.35
    assert p.rax_bars == 7  # the Sidrax, today's sound before this field existed


# --- drawing -------------------------------------------------------------------

def test_the_same_seed_and_chips_give_the_same_patch():
    assert draw_patch(7, UNEVEN) == draw_patch(7, UNEVEN)


def test_different_seeds_give_different_patches():
    drawn = {draw_patch(s, UNEVEN) for s in range(60)}
    assert len(drawn) >= 55


@pytest.mark.parametrize("seed", range(200))
def test_every_drawn_patch_is_valid_and_playable(seed):
    p = draw_patch(seed, UNEVEN)
    steps = [b - a for a, b in zip(p.motif, p.motif[1:])]
    assert 6 <= len(p.motif) <= 12 and p.motif[0] == 0
    assert all(abs(s) <= 3 for s in steps) and -5 <= min(p.motif) and max(p.motif) <= 8
    assert len(set(p.motif)) >= 3
    assert p.gong_partials[0] == 1.0 and list(p.gong_partials) == sorted(p.gong_partials)
    assert all(abs(r - round(r)) > 0.1 for r in p.gong_partials[1:])
    assert len(p.gong_partials) <= 5  # the gong kernel has five partial slots
    assert len(p.gong_amps) == len(p.gong_partials)
    lo, hi = p.shimmer_band
    assert 8000 <= lo < hi <= 20500 and hi - lo >= 3000
    assert p.bars_per_turn in (1, 2, 3, 4, 5, 6) and 0.2 <= p.floor <= 0.5
    assert 0.25 <= p.gong_beat_hz <= 2.0
    assert 0 <= p.mode < len(SCALES[p.scale_name])
    assert p.scale == rotate_mode(SCALES[p.scale_name], p.mode)
    assert p.rax_bars in RAX_BAR_COUNTS
    assert p.seed == seed


def test_the_seed_moves_every_drawn_field_somewhere():
    ps = [draw_patch(s, UNEVEN) for s in range(80)]
    for field in ("motif", "gong_beat_hz", "shimmer_band", "bars_per_turn", "floor", "rax_bars",
                  "mode", "gong_partials", "gong_amps"):
        assert len({getattr(p, field) for p in ps}) > 1, field


# --- the system decides the scale and the gong tuning ---------------------------

def test_even_memory_temperatures_play_the_major_pentatonic():
    assert draw_patch(1, FLAT).scale_name == "major_pentatonic"


def test_uneven_memory_temperatures_move_the_scale():
    assert draw_patch(1, UNEVEN).scale_name != "major_pentatonic"
    q = draw_patch(1, UNEVEN)
    assert q.scale == rotate_mode(SCALES[q.scale_name], q.mode)


def test_the_scale_follows_the_system_not_the_seed():
    assert len({draw_patch(s, UNEVEN).scale_name for s in range(30)}) == 1
    names = {draw_patch(1, chips(gddr=(30.0, 30.0 + d) * 4)).scale_name for d in (0.5, 4, 8, 12, 16, 20)}
    assert len(names) >= 4


def test_the_gong_tuning_follows_the_memory_to_asic_temperature_ratio():
    cool_asic = chips(gddr=(40.0,) * 8, asic=30.0)    # ratio 1.33
    hot_asic = chips(gddr=(40.0,) * 8, asic=44.0)     # ratio 0.91
    a, b = draw_patch(3, cool_asic), draw_patch(3, hot_asic)
    assert a.gong_tuning != b.gong_tuning
    base = GONG_TUNINGS[a.gong_tuning][0]
    assert a.gong_partials[0] == 1.0 and len(a.gong_partials) == len(base)
    assert all(abs(x / y - 1) <= GONG_STRETCH + 1e-3 for x, y in zip(a.gong_partials[1:], base[1:]))
    assert len({draw_patch(s, cool_asic).gong_tuning for s in range(30)}) == 1


# --- ttrax: Tetrax, Sidrax or Quatrax ------------------------------------------

def test_the_rax_bar_counts_are_tetrax_sidrax_and_quatrax():
    assert RAX_BAR_COUNTS == (4, 7, 14)


def test_every_bar_count_is_drawn_somewhere_in_a_run_of_seeds():
    assert {draw_patch(s, UNEVEN).rax_bars for s in range(60)} == set(RAX_BAR_COUNTS)


# --- seeds from the chips -------------------------------------------------------

def test_the_seed_is_stable_for_the_same_readings_and_entropy():
    assert seed_from_telemetry(UNEVEN, 5) == seed_from_telemetry(UNEVEN, 5)


def test_the_seed_changes_with_any_single_reading_or_with_entropy():
    base = seed_from_telemetry(UNEVEN, 5)
    nudged = [dataclasses.replace(UNEVEN[0], asic_temperature=UNEVEN[0].asic_temperature + 0.1), UNEVEN[1]]
    assert seed_from_telemetry(nudged, 5) != base
    assert seed_from_telemetry(UNEVEN, 6) != base
    assert 0 <= base < 2 ** 63


# --- JSON -----------------------------------------------------------------------

def test_a_patch_survives_a_round_trip_through_json_text(tmp_path):
    p = draw_patch(42, UNEVEN)
    path = tmp_path / "p.json"
    save_patch(p, path)
    assert json.loads(path.read_text())["motif"] == list(p.motif)  # plain lists, readable by anything
    assert load_patch(path) == p


def test_the_default_patch_round_trips_too(tmp_path):
    save_patch(DEFAULT_PATCH, tmp_path / "d.json")
    assert load_patch(tmp_path / "d.json") == DEFAULT_PATCH


def test_a_patch_saved_before_rax_bars_existed_still_loads_as_the_sidrax(tmp_path):
    d = json.loads(json.dumps(dataclasses.asdict(DEFAULT_PATCH)))
    del d["rax_bars"]
    path = tmp_path / "old.json"
    path.write_text(json.dumps(d))
    assert load_patch(path).rax_bars == 7


@pytest.mark.parametrize("change", [
    {"motif": []},
    {"scale": []},
    {"gong_partials": [2.0, 3.0]},                 # must start at 1.0
    {"gong_amps": [1.0]},                           # one amplitude per partial
    {"shimmer_band": [9000, 9000]},                 # needs low < high
    {"bars_per_turn": 0},
    {"floor": 0.0},
    {"floor": 1.5},
    {"rax_bars": 5},
])
def test_a_broken_patch_file_is_refused_with_a_reason(tmp_path, change):
    d = json.loads(json.dumps(dataclasses.asdict(DEFAULT_PATCH)))
    d.update(change)
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(d))
    with pytest.raises(ValueError):
        load_patch(path)


def test_a_missing_or_garbled_patch_file_is_a_value_error(tmp_path):
    with pytest.raises(ValueError):
        load_patch(tmp_path / "nope.json")
    (tmp_path / "junk.json").write_text("{not json")
    with pytest.raises(ValueError):
        load_patch(tmp_path / "junk.json")


@pytest.mark.parametrize("field, value", [
    ("asic_temperature", 49.0), ("aiclk", 801.0), ("power", 33.0), ("current", 46.0),
    ("voltage", 0.72), ("gddr_temps", (31.0,) * 8), ("ddr_status", (1, 1, 1, 1)),
])
def test_every_single_reading_feeds_the_seed(field, value):
    base = [dataclasses.replace(FALLBACK, ddr_status=(0, 0, 0, 0))]
    changed = [dataclasses.replace(base[0], **{field: value})]
    assert seed_from_telemetry(changed, 1) != seed_from_telemetry(base, 1), field


def test_every_chip_feeds_the_seed():
    a = chips(n=2)
    b = [a[0], dataclasses.replace(a[1], power=a[1].power + 1)]
    assert seed_from_telemetry(a, 1) != seed_from_telemetry(b, 1)


def test_the_saved_file_is_one_readable_line_per_field(tmp_path):
    save_patch(DEFAULT_PATCH, tmp_path / "p.json")
    lines = (tmp_path / "p.json").read_text().splitlines()
    assert lines[0] == "{" and lines[-1] == "}"
    assert len(lines) == len(dataclasses.fields(DEFAULT_PATCH)) + 2
    assert '"motif": [0, 2, 1, 3, 2, 4, 3, 1]' in lines[4]


# --- a bigger library, so a run has a lot to choose from ------------------------

def test_the_library_is_bigger_and_keeps_its_first_entries_in_place():
    assert len(SCALES) >= 18 and len(GONG_TUNINGS) >= 8
    assert tuple(SCALES)[:6] == ("major_pentatonic", "minor_pentatonic", "hirajoshi", "dorian",
                                 "whole_tone", "phrygian_dominant")
    assert tuple(GONG_TUNINGS)[:3] == ("bell", "tamtam", "singing_bowl")


def test_every_scale_starts_at_zero_rises_and_stays_inside_the_octave():
    for name, sc in SCALES.items():
        assert sc[0] == 0 and list(sc) == sorted(set(sc)) and sc[-1] < 12, name
        assert len(sc) >= 5, name
    assert len({tuple(sc) for sc in SCALES.values()}) == len(SCALES)  # no duplicates under two names


def test_every_gong_tuning_is_inharmonic_even_stretched_to_the_edge_of_the_draw():
    for name, (partials, amps) in GONG_TUNINGS.items():
        assert partials[0] == 1.0 and len(partials) == len(amps) <= 5, name
        assert list(partials) == sorted(partials) and amps[0] == 1.0, name
        for stretch in (1 - GONG_STRETCH, 1 + GONG_STRETCH):
            assert all(abs(r * stretch - round(r * stretch)) > 0.1 for r in partials[1:]), (name, stretch)


def test_a_mode_is_the_scale_started_from_another_of_its_notes():
    assert rotate_mode((0, 2, 4, 7, 9), 0) == (0, 2, 4, 7, 9)
    assert rotate_mode((0, 2, 4, 7, 9), 1) == (0, 2, 5, 7, 10)   # minor-pentatonic-like shape: 2 4 7 9 12 minus 2
    assert rotate_mode((0, 2, 3, 5, 7, 8, 11), 5) == (0, 3, 4, 6, 7, 9, 11)
    assert MODE_COUNT_MAX >= 8


def test_every_mode_of_every_scale_is_drawn_somewhere_and_the_mode_comes_from_the_seed():
    drawn = {draw_patch(s, UNEVEN).mode for s in range(200)}
    assert drawn == set(range(len(SCALES[draw_patch(1, UNEVEN).scale_name])))
    assert len({draw_patch(s, UNEVEN).scale for s in range(200)}) > 1  # same scale name, different shapes


def test_the_motif_length_varies_and_every_length_is_drawn():
    assert {len(draw_patch(s, UNEVEN).motif) for s in range(300)} >= {6, 8, 10, 12}


def test_a_hundred_runs_on_one_quiet_box_still_differ_in_most_of_their_fields():
    ps = [draw_patch(s, UNEVEN) for s in range(100)]
    for field in ("motif", "gong_partials", "gong_amps", "gong_beat_hz", "shimmer_band", "floor"):
        assert len({getattr(p, field) for p in ps}) >= 50, field
    assert len({p.scale for p in ps}) >= 5  # at most one per mode of the one scale the box lands on


def test_a_patch_saved_before_modes_existed_replays_unrotated(tmp_path):
    d = json.loads(json.dumps(dataclasses.asdict(DEFAULT_PATCH)))
    del d["mode"]
    path = tmp_path / "old.json"
    path.write_text(json.dumps(d))
    assert load_patch(path).mode == 0 and load_patch(path).scale == DEFAULT_PATCH.scale


@pytest.mark.parametrize("change", [{"mode": -1}, {"mode": 5}])  # the default scale has five notes
def test_a_mode_outside_the_scale_is_refused(tmp_path, change):
    d = json.loads(json.dumps(dataclasses.asdict(DEFAULT_PATCH)))
    d.update(change)
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(d))
    with pytest.raises(ValueError):
        load_patch(path)


def test_the_widened_ranges_are_reached_past_the_old_ones():
    """Each range used to be narrower (beat 0.4-1.2, band edge 10-14 kHz, 1-4 bars, floor 25-45%); the draw must go beyond it."""
    ps = [draw_patch(s, UNEVEN) for s in range(600)]
    assert {p.bars_per_turn for p in ps} == {1, 2, 3, 4, 5, 6}
    assert min(p.gong_beat_hz for p in ps) < 0.4 and max(p.gong_beat_hz for p in ps) > 1.2
    assert min(p.shimmer_band[0] for p in ps) < 10000 and max(p.shimmer_band[0] for p in ps) > 14000
    assert min(p.floor for p in ps) < 0.25 and max(p.floor for p in ps) > 0.45


# --- steam routes: what the shift register and its noise are patched into -----------------------

def test_the_four_routes_and_the_default_patch_having_none():
    from tensixinet.music.patch import STEAM_ROUTES
    assert STEAM_ROUTES == ("strikes", "avdog", "accent", "pitch")
    assert DEFAULT_PATCH.steam_routes == ()  # the sound from before routes existed


def test_every_route_is_drawn_somewhere_no_patch_has_more_than_three_and_they_come_in_order():
    from tensixinet.music.patch import STEAM_ROUTES
    ps = [draw_patch(s, UNEVEN) for s in range(400)]
    assert {r for p in ps for r in p.steam_routes} == set(STEAM_ROUTES)
    assert max(len(p.steam_routes) for p in ps) == 3 and min(len(p.steam_routes) for p in ps) == 0
    assert all(list(p.steam_routes) == [r for r in STEAM_ROUTES if r in p.steam_routes] for p in ps)
    assert len({p.steam_routes for p in ps}) >= 10


def test_the_register_is_patched_into_something_in_most_patches_and_the_gong_strikes_route_is_the_commonest():
    """The snare is quiet now (see `snare`), so the register's own job is to drive the rest: the gong, shimmer, drone and AVDog follow it
    on the `strikes` route, and a patch has more routes than it did."""
    import collections
    from tensixinet.music.patch import _draw_steam_routes
    drawn = [_draw_steam_routes(s) for s in range(3000)]
    counts = collections.Counter(len(r) for r in drawn)
    for n, want in enumerate((0.10, 0.35, 0.35, 0.20)):
        assert abs(counts[n] / len(drawn) - want) < 0.04, (n, counts)
    on = collections.Counter(r for d in drawn for r in d)
    assert on["strikes"] > 1.2 * max(on["avdog"], on["accent"], on["pitch"])
    assert on["strikes"] / len(drawn) > 0.45


# --- the snare: how often ttsteam's snare plays at all ------------------------------------------------

def test_the_default_patch_keeps_the_snare_as_it_was():
    assert DEFAULT_PATCH.snare == 1.0


def test_a_drawn_patch_mostly_has_a_silent_or_sparse_snare():
    """The user's ear: every patch had the snare. Now 40% have none, 30% one hit in five, 20% one in two, and 10% all of them."""
    import collections
    from tensixinet.music.patch import SNARE_LEVELS
    levels = collections.Counter(draw_patch(s, UNEVEN).snare for s in range(3000))
    assert set(levels) == set(SNARE_LEVELS) == {0.0, 0.2, 0.5, 1.0}
    for level, want in ((0.0, 0.4), (0.2, 0.3), (0.5, 0.2), (1.0, 0.1)):
        assert abs(levels[level] / 3000 - want) < 0.04, (level, levels)


def test_the_snare_level_is_drawn_independently_of_the_steam_routes():
    ps = [draw_patch(s, UNEVEN) for s in range(3000)]
    no_route = [p for p in ps if not p.steam_routes]
    three = [p for p in ps if len(p.steam_routes) == 3]
    for group in (no_route, three):
        assert abs(sum(p.snare == 0.0 for p in group) / len(group) - 0.4) < 0.1


def test_drawing_the_snare_leaves_every_other_drawn_field_where_it_was():
    a = draw_patch(11, UNEVEN)
    assert dataclasses.replace(a, snare=1.0) == dataclasses.replace(draw_patch(11, UNEVEN), snare=1.0)
    assert any(draw_patch(s, UNEVEN).snare != a.snare for s in range(12, 40))  # and it does vary


def test_a_patch_saved_before_the_snare_level_played_every_hit(tmp_path):
    path = tmp_path / "p.json"
    save_patch(DEFAULT_PATCH, path)
    d = json.loads(path.read_text())
    del d["snare"]
    path.write_text(json.dumps(d))
    assert load_patch(path).snare == 1.0


def test_the_snare_level_round_trips_and_one_outside_zero_to_one_is_refused(tmp_path):
    path = tmp_path / "p.json"
    save_patch(dataclasses.replace(DEFAULT_PATCH, snare=0.2), path)
    assert load_patch(path).snare == 0.2
    for bad in (-0.1, 1.5, float("nan")):
        d = json.loads(path.read_text())
        d["snare"] = bad
        path.write_text(json.dumps(d))
        with pytest.raises(ValueError, match="snare"):
            load_patch(path)


def test_drawing_routes_leaves_every_other_drawn_field_where_it_was():
    """The routes use their own random stream, so adding them moved nothing else for a given seed."""
    a = draw_patch(11, UNEVEN)
    assert dataclasses.replace(a, steam_routes=()) == dataclasses.replace(draw_patch(11, UNEVEN), steam_routes=())
    assert a.steam_routes == draw_patch(11, UNEVEN).steam_routes


def test_a_patch_saved_before_routes_existed_has_none(tmp_path):
    d = json.loads(json.dumps(dataclasses.asdict(DEFAULT_PATCH)))
    del d["steam_routes"]
    path = tmp_path / "old.json"
    path.write_text(json.dumps(d))
    assert load_patch(path).steam_routes == ()


def test_routes_round_trip_and_an_unknown_one_is_refused(tmp_path):
    p = dataclasses.replace(DEFAULT_PATCH, steam_routes=("avdog", "pitch"))
    save_patch(p, tmp_path / "r.json")
    assert load_patch(tmp_path / "r.json") == p
    d = json.loads(json.dumps(dataclasses.asdict(DEFAULT_PATCH)))
    d["steam_routes"] = ["strikes", "bogus"]
    (tmp_path / "bad.json").write_text(json.dumps(d))
    with pytest.raises(ValueError):
        load_patch(tmp_path / "bad.json")


# --- the coco bus: how much of the band is sent through the Cocoquantus's delay, and its room ------------

def test_the_default_patch_has_no_send_so_it_sounds_as_it_did():
    assert DEFAULT_PATCH.coco_send == 0.0 and DEFAULT_PATCH.coco_reverb == 0.0 and DEFAULT_PATCH.reverb_s == 1.5


def test_a_drawn_patch_always_sends_some_of_the_band_through_the_coco_and_has_a_room():
    ps = [draw_patch(s, UNEVEN) for s in range(400)]
    assert all(0.25 <= p.coco_send <= 0.8 and 0.15 <= p.coco_reverb <= 0.5 and 0.8 <= p.reverb_s <= 3.5 for p in ps)
    assert min(p.coco_send for p in ps) < 0.3 and max(p.coco_send for p in ps) > 0.75
    assert min(p.reverb_s for p in ps) < 1.1 and max(p.reverb_s for p in ps) > 3.2
    assert len({p.coco_reverb for p in ps}) > 100


def test_drawing_the_bus_leaves_every_other_drawn_field_where_it_was():
    a, b = draw_patch(11, UNEVEN), draw_patch(11, UNEVEN)
    z = dict(coco_send=0.0, coco_reverb=0.0, reverb_s=1.5)
    assert dataclasses.replace(a, **z) == dataclasses.replace(b, **z)
    assert dataclasses.replace(a, **z) == dataclasses.replace(draw_patch(11, UNEVEN), **z)
    assert any(draw_patch(s, UNEVEN).coco_send != a.coco_send for s in range(12, 30))


def test_a_patch_saved_before_the_bus_existed_has_no_send(tmp_path):
    path = tmp_path / "p.json"
    save_patch(draw_patch(5, UNEVEN), path)
    d = json.loads(path.read_text())
    for k in ("coco_send", "coco_reverb", "reverb_s"):
        del d[k]
    path.write_text(json.dumps(d))
    p = load_patch(path)
    assert (p.coco_send, p.coco_reverb, p.reverb_s) == (0.0, 0.0, 1.5)


def test_the_bus_fields_round_trip_and_values_out_of_range_are_refused(tmp_path):
    path = tmp_path / "p.json"
    save_patch(dataclasses.replace(DEFAULT_PATCH, coco_send=0.4, coco_reverb=0.2, reverb_s=2.5), path)
    p = load_patch(path)
    assert (p.coco_send, p.coco_reverb, p.reverb_s) == (0.4, 0.2, 2.5)
    for key, bad in (("coco_send", -0.1), ("coco_send", 1.1), ("coco_reverb", -0.1), ("coco_reverb", 1.1), ("reverb_s", 0.0),
                     ("reverb_s", -1.0), ("reverb_s", float("nan")), ("coco_send", float("nan"))):
        d = json.loads(path.read_text())
        d[key] = bad
        path.write_text(json.dumps(d))
        with pytest.raises(ValueError, match=key.split("_")[0]):
            load_patch(path)
        save_patch(p, path)


def test_the_bus_is_drawn_independently_of_the_routes_and_the_snare():
    ps = [draw_patch(s, UNEVEN) for s in range(3000)]
    mean = lambda group: sum(p.coco_send for p in group) / len(group)
    assert abs(mean([p for p in ps if not p.steam_routes]) - mean([p for p in ps if len(p.steam_routes) == 3])) < 0.04
    assert abs(mean([p for p in ps if p.snare == 0.0]) - mean([p for p in ps if p.snare == 1.0])) < 0.04


# --- rests: the share of the coco's held notes that are silent --------------------------------------------------

def test_the_default_patch_has_no_rests_and_a_drawn_one_always_has_some():
    ps = [draw_patch(s, UNEVEN) for s in range(400)]
    assert DEFAULT_PATCH.rests == 0.0
    assert all(0.15 <= p.rests <= 0.45 for p in ps) and min(p.rests for p in ps) < 0.2 and max(p.rests for p in ps) > 0.4
    assert len({p.rests for p in ps}) > 100


def test_drawing_the_rests_leaves_every_other_drawn_field_where_it_was_and_they_are_independent_of_the_bus():
    ps = [draw_patch(s, UNEVEN) for s in range(3000)]
    assert dataclasses.replace(ps[11], rests=0.0) == dataclasses.replace(draw_patch(11, UNEVEN), rests=0.0)
    mean = lambda group: sum(p.rests for p in group) / len(group)
    assert abs(mean([p for p in ps if p.coco_send < 0.4]) - mean([p for p in ps if p.coco_send > 0.65])) < 0.03
    assert abs(mean([p for p in ps if p.snare == 0.0]) - mean([p for p in ps if p.snare == 1.0])) < 0.03
    assert abs(mean([p for p in ps if not p.steam_routes]) - mean([p for p in ps if len(p.steam_routes) == 3])) < 0.04


def test_a_patch_saved_before_rests_existed_has_none_and_the_value_round_trips_and_is_checked(tmp_path):
    path = tmp_path / "p.json"
    save_patch(dataclasses.replace(DEFAULT_PATCH, rests=0.3), path)
    assert load_patch(path).rests == 0.3
    d = json.loads(path.read_text())
    del d["rests"]
    path.write_text(json.dumps(d))
    assert load_patch(path).rests == 0.0
    for bad in (-0.1, 1.0, 1.5, float("nan")):
        d["rests"] = bad
        path.write_text(json.dumps(d))
        with pytest.raises(ValueError, match="rests"):
            load_patch(path)


# --- lead_rests and roll_rests: the share of the lead's notes and of ttrollz's pulses that are left out ----------

def test_the_default_patch_leaves_the_lead_and_rolls_alone_and_a_drawn_one_rests_some_of_each():
    ps = [draw_patch(s, UNEVEN) for s in range(400)]
    assert DEFAULT_PATCH.lead_rests == 0.0 and DEFAULT_PATCH.roll_rests == 0.0
    for field in ("lead_rests", "roll_rests"):
        values = [getattr(p, field) for p in ps]
        assert all(0.1 <= v <= 0.3 for v in values) and min(values) < 0.13 and max(values) > 0.27 and len(set(values)) > 100


def test_the_lead_and_roll_rests_are_drawn_from_their_own_streams():
    ps = [draw_patch(s, UNEVEN) for s in range(3000)]
    bare = lambda p: dataclasses.replace(p, lead_rests=0.0, roll_rests=0.0)
    assert bare(ps[11]) == bare(draw_patch(11, UNEVEN))
    mean = lambda group, f: sum(getattr(p, f) for p in group) / len(group)
    assert abs(mean(ps, "lead_rests") - mean(ps, "roll_rests")) < 0.02
    assert any(p.lead_rests != p.roll_rests for p in ps) and abs(mean([p for p in ps if p.rests < 0.25], "lead_rests") - mean([p for p in ps if p.rests > 0.35], "lead_rests")) < 0.02
    assert abs(mean([p for p in ps if p.rests < 0.25], "roll_rests") - mean([p for p in ps if p.rests > 0.35], "roll_rests")) < 0.02


def test_a_patch_saved_before_these_rests_existed_has_none_and_the_values_round_trip_and_are_checked(tmp_path):
    path = tmp_path / "p.json"
    save_patch(dataclasses.replace(DEFAULT_PATCH, lead_rests=0.2, roll_rests=0.15), path)
    loaded = load_patch(path)
    assert loaded.lead_rests == 0.2 and loaded.roll_rests == 0.15
    d = json.loads(path.read_text())
    for field in ("lead_rests", "roll_rests"):
        saved = dict(d)
        del saved[field]
        path.write_text(json.dumps(saved))
        assert getattr(load_patch(path), field) == 0.0
        for bad in (-0.1, 1.0, 1.5, float("nan")):
            saved[field] = bad
            path.write_text(json.dumps(saved))
            with pytest.raises(ValueError, match=field):
                load_patch(path)


# --- quantus_mod: how much the looping line's pitch is moved ----------------------------------------------------------------

def test_the_default_patch_does_not_move_the_looping_line_and_a_drawn_one_always_does():
    ps = [draw_patch(s, UNEVEN) for s in range(400)]
    assert DEFAULT_PATCH.quantus_mod == 0.0
    assert all(0.35 <= p.quantus_mod <= 0.9 for p in ps) and min(p.quantus_mod for p in ps) < 0.4 and max(p.quantus_mod for p in ps) > 0.85
    assert len({p.quantus_mod for p in ps}) > 100


def test_drawing_the_modulation_leaves_every_other_drawn_field_where_it_was_and_is_independent_of_them():
    ps = [draw_patch(s, UNEVEN) for s in range(3000)]
    assert dataclasses.replace(ps[11], quantus_mod=0.0) == dataclasses.replace(draw_patch(11, UNEVEN), quantus_mod=0.0)
    mean = lambda group: sum(p.quantus_mod for p in group) / len(group)
    assert abs(mean([p for p in ps if p.rests < 0.25]) - mean([p for p in ps if p.rests > 0.35])) < 0.03
    assert abs(mean([p for p in ps if p.coco_send < 0.4]) - mean([p for p in ps if p.coco_send > 0.65])) < 0.03
    assert abs(mean([p for p in ps if p.lead_rests < 0.15]) - mean([p for p in ps if p.lead_rests > 0.25])) < 0.03


def test_a_patch_saved_before_the_modulation_existed_has_none_and_the_value_round_trips_and_is_checked(tmp_path):
    path = tmp_path / "p.json"
    save_patch(dataclasses.replace(DEFAULT_PATCH, quantus_mod=0.6), path)
    assert load_patch(path).quantus_mod == 0.6
    d = json.loads(path.read_text())
    del d["quantus_mod"]
    path.write_text(json.dumps(d))
    assert load_patch(path).quantus_mod == 0.0
    for bad in (-0.1, 1.5, float("nan")):
        d["quantus_mod"] = bad
        path.write_text(json.dumps(d))
        with pytest.raises(ValueError, match="quantus_mod"):
            load_patch(path)
    d["quantus_mod"] = 1.0                                              # full modulation is allowed
    path.write_text(json.dumps(d))
    assert load_patch(path).quantus_mod == 1.0


# --- the choir's consonant fields ------------------------------------------------

from tensixinet.music.patch import CHOIR_FIELDS, CHOIR_ONSET_ODDS, CHOIR_SOURCES, patch_from_dict  # noqa: E402
from tensixinet.telemetry.mock import mock_telemetry  # noqa: E402

CHOIR_CHIPS = mock_telemetry(4, 2)

# every field of draw_patch(11, CHOIR_CHIPS) as taken before the choir fields existed
OLD_DRAW_11 = dict(
    seed=11, scale_name="in_sen", scale=(0, 2, 5, 7, 8), motif=(0, 1, -2, -4, -2, -4, -2, 0, 1, 0), gong_tuning="tamtam",
    gong_partials=(1.0, 1.471, 2.311, 3.641, 5.572), gong_amps=(1.0, 0.653, 0.446, 0.38, 0.21), gong_beat_hz=1.211,
    shimmer_band=(11586.7, 19487.1), bars_per_turn=6, floor=0.261, rax_bars=14, steam_routes=("strikes", "avdog"), style="",
    emphasis=(), mode=2, snare=0.0, coco_send=0.602, coco_reverb=0.388, reverb_s=2.186, rests=0.397, lead_rests=0.232,
    roll_rests=0.151, quantus_mod=0.47,
)


def test_a_drawn_patch_always_has_an_onset_source_and_the_choir_fields_in_range():
    for seed in range(40):
        p = draw_patch(seed, CHOIR_CHIPS)
        assert p.choir_onset in CHOIR_SOURCES and p.choir_howl in CHOIR_SOURCES
        assert p.choir_coda in ("",) + CHOIR_SOURCES and 0.02 <= p.choir_howl_odds <= 0.08
        assert all(0.2 <= w <= 1.0 for w in p.choir_mix) and len(p.choir_mix) == 3
        assert 0.3 <= p.choir_crush <= 1.0 and 0.3 <= p.choir_drift <= 1.0


def test_the_choir_draw_has_its_own_stream_so_no_other_field_moved():
    p = draw_patch(11, CHOIR_CHIPS)
    old = dataclasses.replace(p, **{f: getattr(DEFAULT_PATCH, f) for f in CHOIR_FIELDS})
    assert dataclasses.asdict(old) == dataclasses.asdict(dataclasses.replace(DEFAULT_PATCH, **OLD_DRAW_11))


def test_the_choir_fields_are_the_seven_listed():
    assert CHOIR_FIELDS == ("choir_onset", "choir_coda", "choir_howl", "choir_howl_odds", "choir_mix", "choir_crush", "choir_drift")


def test_the_coda_odds_are_none_same_and_other():
    patches = [draw_patch(s, CHOIR_CHIPS) for s in range(2000)]
    none = sum(p.choir_coda == "" for p in patches) / 2000
    same = sum(p.choir_coda == p.choir_onset for p in patches) / 2000
    other = 1.0 - none - same
    assert abs(none - 0.40) < 0.03 and abs(same - 0.30) < 0.03 and abs(other - 0.30) < 0.03, (none, same, other)
    # "another source" is spread evenly over the five others
    others = [p.choir_coda for p in patches if p.choir_coda not in ("", p.choir_onset)]
    for src in CHOIR_SOURCES:
        share = sum(c == src for c in others) / len(others)
        assert abs(share - 1 / 6) < 0.06, (src, share)  # 1/6 overall: each is the "other" for five of six onsets


def test_onset_sources_follow_their_odds():
    patches = [draw_patch(s, CHOIR_CHIPS) for s in range(4000)]
    assert CHOIR_ONSET_ODDS == (0.25, 0.2, 0.15, 0.15, 0.15, 0.1)
    for src, odds in zip(CHOIR_SOURCES, CHOIR_ONSET_ODDS):
        assert abs(sum(p.choir_onset == src for p in patches) / 4000 - odds) < 0.03, src


def test_the_howl_source_is_chosen_evenly_and_its_odds_reach_the_ends_of_the_range():
    patches = [draw_patch(s, CHOIR_CHIPS) for s in range(3000)]
    for src in CHOIR_SOURCES:
        assert abs(sum(p.choir_howl == src for p in patches) / 3000 - 1 / 6) < 0.04, src
    odds = [p.choir_howl_odds for p in patches]
    assert min(odds) < 0.025 and max(odds) > 0.075
    mixes = [w for p in patches for w in p.choir_mix]
    assert min(mixes) < 0.22 and max(mixes) > 0.98
    for field in ("choir_crush", "choir_drift"):
        vals = [getattr(p, field) for p in patches]
        assert min(vals) < 0.32 and max(vals) > 0.98, field


def test_an_old_saved_patch_has_no_choir_source_and_round_trips(tmp_path):
    old = {k: v for k, v in dataclasses.asdict(DEFAULT_PATCH).items() if k not in CHOIR_FIELDS}
    p = patch_from_dict(old)
    assert p.choir_onset == "" and p.choir_coda == "" and p.choir_howl == "" and p.choir_howl_odds == 0.0
    assert p.choir_mix == (1.0, 1.0, 1.0) and p.choir_crush == 0.0 and p.choir_drift == 0.0
    drawn = draw_patch(5, CHOIR_CHIPS)
    save_patch(drawn, tmp_path / "p.json")
    assert load_patch(tmp_path / "p.json") == drawn


@pytest.mark.parametrize("bad", [
    {"choir_onset": "snare"}, {"choir_coda": "gong"}, {"choir_howl": "gong"},
    {"choir_onset": "gong", "choir_coda": "snare"}, {"choir_onset": "gong", "choir_howl": "snare"},
    {"choir_onset": "gong", "choir_howl_odds": 1.5}, {"choir_onset": "gong", "choir_howl_odds": -0.1},
    {"choir_onset": "gong", "choir_howl_odds": float("nan")},
    {"choir_onset": "gong", "choir_mix": [0, 0, 0]}, {"choir_onset": "gong", "choir_mix": [1, 1]},
    {"choir_onset": "gong", "choir_mix": [1, -1, 1]}, {"choir_onset": "gong", "choir_mix": [1, float("nan"), 1]},
    {"choir_onset": "gong", "choir_mix": [1, float("inf"), 1]},
    {"choir_onset": "gong", "choir_crush": float("nan")}, {"choir_onset": "gong", "choir_crush": 1.1},
    {"choir_onset": "gong", "choir_crush": -0.1},
    {"choir_onset": "gong", "choir_drift": -0.1}, {"choir_onset": "gong", "choir_drift": 1.1},
    {"choir_onset": "gong", "choir_drift": float("nan")},
])
def test_a_broken_choir_field_is_refused(bad):
    d = dataclasses.asdict(DEFAULT_PATCH) | bad
    with pytest.raises(ValueError):
        patch_from_dict(d)


def test_a_coda_or_howl_without_an_onset_is_refused_with_a_reason():
    for bad in ({"choir_coda": "gong"}, {"choir_howl": "rolls"}):
        with pytest.raises(ValueError, match="needs an onset source"):
            patch_from_dict(dataclasses.asdict(DEFAULT_PATCH) | bad)


def test_edge_values_of_the_choir_fields_are_allowed():
    d = dataclasses.asdict(DEFAULT_PATCH) | {"choir_onset": "gong", "choir_howl_odds": 1.0, "choir_crush": 1.0,
                                             "choir_drift": 1.0, "choir_mix": [0.0, 0.0, 1.0]}
    p = patch_from_dict(d)
    assert p.choir_mix == (0.0, 0.0, 1.0) and p.choir_howl_odds == 1.0


def test_a_bad_choir_pick_is_refused_and_an_empty_or_half_pick_is_accepted():
    base = dataclasses.asdict(DEFAULT_PATCH)
    for bad in (("x", "a"), ("s", "y"), ("s", "ae"), ("sh", "A"), ("s",), ("s", "a", "e")):
        with pytest.raises(ValueError):
            patch_from_dict(base | {"choir_pick": bad})
    for good in (("", ""), ("sh", ""), ("", "i"), ("t", "u")):
        assert patch_from_dict(base | {"choir_pick": list(good)}).choir_pick == good


def test_a_saved_patch_with_no_pick_loads_with_none_and_a_pick_survives_a_round_trip():
    assert DEFAULT_PATCH.choir_pick == ("", "")
    old = {k: v for k, v in dataclasses.asdict(DEFAULT_PATCH).items() if k != "choir_pick"}
    assert patch_from_dict(old).choir_pick == ("", "")
    p = dataclasses.replace(DEFAULT_PATCH, choir_pick=("k", "o"))
    assert patch_from_dict(json.loads(json.dumps(dataclasses.asdict(p)))).choir_pick == ("k", "o")


def test_a_drawn_patch_has_no_pick():
    assert all(draw_patch(s, CHOIR_CHIPS).choir_pick == ("", "") for s in range(10))
