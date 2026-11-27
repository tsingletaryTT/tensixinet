# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""--view-map: a JSON map from tt-toplike view names to saved patches."""
import json
from pathlib import Path

import pytest

from tensixinet.music.patch import DEFAULT_PATCH, save_patch
from tensixinet.viewmap import load_view_map


def library_with_defrag(tmp_path):
    patches = tmp_path / "patches"
    patches.mkdir()
    save_patch(DEFAULT_PATCH, patches / "defrag.json")
    return tmp_path


def test_a_map_resolves_each_view_name_to_a_saved_patch(tmp_path):
    root = library_with_defrag(tmp_path)
    (tmp_path / "views.json").write_text(json.dumps({"defrag": "defrag"}))
    mapping = load_view_map(tmp_path / "views.json", root=root)
    assert list(mapping) == ["defrag"]
    assert mapping["defrag"] == DEFAULT_PATCH
    assert mapping.get("arcade") is None  # an unmapped view changes nothing


def test_an_unknown_patch_name_names_the_view_and_the_patch(tmp_path):
    root = library_with_defrag(tmp_path)
    (tmp_path / "views.json").write_text(json.dumps({"defrag": "no-such-patch"}))
    with pytest.raises(ValueError, match="cannot find patch 'no-such-patch'"):
        load_view_map(tmp_path / "views.json", root=root)


@pytest.mark.parametrize("text,message", [
    ("[]", "must be a JSON object"),
    ('{"defrag": 3}', "must name a saved patch"),
    ("{bad", "cannot read"),
])
def test_a_bad_map_is_a_clear_error(tmp_path, text, message):
    root = library_with_defrag(tmp_path)
    (tmp_path / "views.json").write_text(text)
    with pytest.raises(ValueError, match=message):
        load_view_map(tmp_path / "views.json", root=root)


def test_a_missing_file_is_a_clear_error(tmp_path):
    with pytest.raises(ValueError, match="cannot read view map"):
        load_view_map(tmp_path / "absent.json", root=tmp_path)


def test_the_shipped_defrag_patch_resolves_by_name(tmp_path):
    """The documented ``{\"defrag\": \"defrag\"}`` works against the library shipped with the repo."""
    root = Path(__file__).resolve().parents[1] / "library"
    path = tmp_path / "views.json"
    path.write_text(json.dumps({"defrag": "defrag"}))
    mapping = load_view_map(path, root=root)
    assert list(mapping) == ["defrag"]
    assert mapping["defrag"].seed == 11025

