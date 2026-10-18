"""A patch is every drawn choice of one run: replayable, valid, and partly decided by the system."""
import dataclasses
import json

import pytest

from tensixinet.music.lead import MOTIF
from tensixinet.music.patch import (
    DEFAULT_PATCH,
    GONG_TUNINGS,
    SCALES,
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
    assert len(p.motif) == 8 and p.motif[0] == 0
    assert all(abs(s) <= 2 for s in steps) and -3 <= min(p.motif) and max(p.motif) <= 6
    assert len(set(p.motif)) >= 3
    assert p.gong_partials[0] == 1.0 and list(p.gong_partials) == sorted(p.gong_partials)
    assert all(abs(r - round(r)) > 0.1 for r in p.gong_partials[1:])
    assert len(p.gong_amps) == len(p.gong_partials)
    lo, hi = p.shimmer_band
    assert 10000 <= lo < hi <= 20500 and hi - lo >= 3000
    assert p.bars_per_turn in (1, 2, 3, 4) and 0.25 <= p.floor <= 0.45
    assert 0.4 <= p.gong_beat_hz <= 1.2
    assert p.seed == seed


def test_the_seed_moves_every_drawn_field_somewhere():
    ps = [draw_patch(s, UNEVEN) for s in range(80)]
    for field in ("motif", "gong_beat_hz", "shimmer_band", "bars_per_turn", "floor"):
        assert len({getattr(p, field) for p in ps}) > 1, field


# --- the system decides the scale and the gong tuning ---------------------------

def test_even_memory_temperatures_play_the_major_pentatonic():
    assert draw_patch(1, FLAT).scale_name == "major_pentatonic"


def test_uneven_memory_temperatures_move_the_scale():
    assert draw_patch(1, UNEVEN).scale_name != "major_pentatonic"
    assert draw_patch(1, UNEVEN).scale == SCALES[draw_patch(1, UNEVEN).scale_name]


def test_the_scale_follows_the_system_not_the_seed():
    assert len({draw_patch(s, UNEVEN).scale_name for s in range(30)}) == 1
    names = {draw_patch(1, chips(gddr=(30.0, 30.0 + d) * 4)).scale_name for d in (0.5, 4, 8, 12, 16, 20)}
    assert len(names) >= 4


def test_the_gong_tuning_follows_the_memory_to_asic_temperature_ratio():
    cool_asic = chips(gddr=(40.0,) * 8, asic=30.0)    # ratio 1.33
    hot_asic = chips(gddr=(40.0,) * 8, asic=44.0)     # ratio 0.91
    a, b = draw_patch(3, cool_asic), draw_patch(3, hot_asic)
    assert a.gong_tuning != b.gong_tuning
    assert a.gong_partials == GONG_TUNINGS[a.gong_tuning][0]
    assert len({draw_patch(s, cool_asic).gong_tuning for s in range(30)}) == 1


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


@pytest.mark.parametrize("change", [
    {"motif": []},
    {"scale": []},
    {"gong_partials": [2.0, 3.0]},                 # must start at 1.0
    {"gong_amps": [1.0]},                           # one amplitude per partial
    {"shimmer_band": [9000, 9000]},                 # needs low < high
    {"bars_per_turn": 0},
    {"floor": 0.0},
    {"floor": 1.5},
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
