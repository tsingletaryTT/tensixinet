# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The patch and style library: named files a listener can write by hand or save from a run.

A patch file is the exact patch of a run; a style file narrows a future draw. Both live in a
library folder, so `--patch NAME`, `--style NAME` and the TUI's `p` and `s` keys all work on
the same files.
"""
import dataclasses
import json

import pytest

from tensixinet.music.library import (
    DEFAULT_LIBRARY, library_styles, resolve_patch, save_named_patch, save_named_style,
)
from tensixinet.music.patch import SCALES, draw_patch, load_patch
from tensixinet.music.styles import (
    STYLES, draw_styled_patch, load_style, save_style, style_from_dict, style_from_patch,
)
from tensixinet.telemetry.mock import mock_telemetry

CHIPS = mock_telemetry(4, 5)
PATCH = draw_patch(5, CHIPS)


def test_the_library_defaults_to_a_folder_beside_where_you_run():
    assert DEFAULT_LIBRARY == "library"


def test_a_named_patch_is_saved_under_patches_and_loads_back_exactly(tmp_path):
    path = save_named_patch(PATCH, "my patch", root=tmp_path)
    assert path == tmp_path / "patches" / "my-patch.json"
    assert load_patch(path) == PATCH
    assert json.loads(path.read_text())["seed"] == PATCH.seed  # still an ordinary patch file


def test_saving_the_same_name_twice_does_not_overwrite_the_first(tmp_path):
    first = save_named_patch(PATCH, "take", root=tmp_path)
    second = save_named_patch(PATCH, "take", root=tmp_path)
    assert (first.name, second.name) == ("take.json", "take-2.json")
    assert load_patch(first) == load_patch(second)


def test_names_cannot_escape_the_library(tmp_path):
    for bad in ("", ".", "..", "../oops", "/tmp/oops", "sub/oops", "a\\b"):
        with pytest.raises(ValueError, match="name"):
            save_named_patch(PATCH, bad, root=tmp_path)


def test_a_bare_patch_name_is_resolved_inside_the_library(tmp_path):
    saved = save_named_patch(PATCH, "favourite", root=tmp_path)
    assert resolve_patch("favourite", root=tmp_path) == saved
    other = tmp_path / "elsewhere.json"
    save_named_patch(PATCH, "elsewhere", root=tmp_path / "other")
    other.write_text(saved.read_text())
    assert resolve_patch(str(other), root=tmp_path) == other
    with pytest.raises(ValueError, match="patches"):
        resolve_patch("missing", root=tmp_path)


def test_a_snapshot_style_is_saved_under_styles_and_replays_the_hearable_choices(tmp_path):
    style = style_from_patch(PATCH, "my patch")
    path = save_named_style(style, "my patch", root=tmp_path)
    assert path == tmp_path / "styles" / "my-patch.json"
    styles = library_styles(root=tmp_path)
    assert list(styles) == ["my-patch"]
    assert styles["my-patch"].name == "my-patch"
    drawn = draw_styled_patch(PATCH.seed, CHIPS, "my-patch", styles=styles)
    assert drawn.style == "my-patch"
    assert (drawn.scale, drawn.motif, drawn.bars_per_turn, drawn.floor, drawn.gong_tuning) == (
        PATCH.scale, PATCH.motif, PATCH.bars_per_turn, PATCH.floor, PATCH.gong_tuning)
    assert (drawn.steam_routes, drawn.emphasis, drawn.snare) == (PATCH.steam_routes, PATCH.emphasis, PATCH.snare)
    assert (drawn.coco_send, drawn.coco_reverb, drawn.reverb_s, drawn.rests) == (
        PATCH.coco_send, PATCH.coco_reverb, PATCH.reverb_s, PATCH.rests)
    assert (drawn.choir_mix, drawn.choir_crush) == (PATCH.choir_mix, PATCH.choir_crush)


def test_saving_a_snapshot_style_twice_gives_it_a_new_name(tmp_path):
    first = save_named_style(style_from_patch(PATCH, "snapshot"), "snapshot", root=tmp_path)
    second = save_named_style(style_from_patch(PATCH, "snapshot"), "snapshot", root=tmp_path)
    assert (first.name, second.name) == ("snapshot.json", "snapshot-2.json")
    assert [s.name for s in library_styles(root=tmp_path).values()] == ["snapshot", "snapshot-2"]


def test_a_style_file_round_trips_and_a_broken_one_names_its_file(tmp_path):
    path = tmp_path / "styles" / "hand.json"
    save_style(STYLES["blues"], path)
    assert load_style(path) == STYLES["blues"]
    assert list(library_styles(root=tmp_path)) == ["hand"]
    path.write_text("{not json")
    with pytest.raises(ValueError, match="hand.json"):
        library_styles(root=tmp_path)


def test_a_style_file_is_checked_before_it_can_be_drawn(tmp_path):
    good = dataclasses.asdict(STYLES["blues"])
    for field, bad in (("scales", ["polka"]), ("modes", "sideways"), ("emphasis", [["gong", -1.0]]),
                       ("motif_steps", [0.1] * 3), ("snare", 2.0)):
        d = dict(good)
        d[field] = bad
        with pytest.raises(ValueError, match=field):
            style_from_dict(d)
    assert style_from_dict(good) == STYLES["blues"]


def test_builtin_styles_are_not_changed_by_saving_a_custom_one(tmp_path):
    save_named_style(style_from_patch(PATCH, "blues"), "blues", root=tmp_path)
    assert STYLES["blues"].scales == ("blues", "mixolydian", "minor_pentatonic", "dorian")
    assert SCALES["blues"] == (0, 3, 5, 6, 7, 10)  # and the scale table is untouched
