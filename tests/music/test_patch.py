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


def test_every_route_is_drawn_somewhere_no_patch_has_more_than_two_and_they_come_in_order():
    from tensixinet.music.patch import STEAM_ROUTES
    ps = [draw_patch(s, UNEVEN) for s in range(400)]
    assert {r for p in ps for r in p.steam_routes} == set(STEAM_ROUTES)
    assert max(len(p.steam_routes) for p in ps) == 2 and min(len(p.steam_routes) for p in ps) == 0
    assert all(list(p.steam_routes) == [r for r in STEAM_ROUTES if r in p.steam_routes] for p in ps)
    assert len({p.steam_routes for p in ps}) >= 7


def test_the_snare_is_kept_off_most_of_the_band_most_of_the_time():
    """The user's ear said the snare was still everywhere: with 60% odds each, 63% of patches had three or four routes and the strikes
    route, which makes the gong, shimmer, drone and AVDog follow the snare, was on in 60%."""
    import collections
    from tensixinet.music.patch import _draw_steam_routes
    seeds = range(3000)
    drawn = [_draw_steam_routes(s) for s in seeds]
    counts = collections.Counter(len(r) for r in drawn)
    assert 0.20 < counts[0] / len(drawn) < 0.30 and 0.45 < counts[1] / len(drawn) < 0.55 and 0.20 < counts[2] / len(drawn) < 0.30
    on = collections.Counter(r for d in drawn for r in d)
    assert on["strikes"] < 0.7 * min(on["avdog"], on["accent"], on["pitch"])  # strikes is the rarest, by a clear margin
    assert on["strikes"] / len(drawn) < 0.25


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
