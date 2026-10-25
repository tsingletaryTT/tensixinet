# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Tests for the MIDI route table and its JSON loader."""
import json

import pytest

from tensixinet.score.routing import DEFAULT_ROUTES, Routes, load_routes


def test_default_table_values_are_pinned_as_literals():
    r = DEFAULT_ROUTES
    assert dict(r.note_channels) == {"lead": 0, "ttquantus": 1}
    assert dict(r.trigger_notes) == {"gong": 49, "ttsteam": 38, "ttrollz_clock": 36}
    assert r.roll_lane_notes == (42, 44, 46, 54, 56, 60, 61, 62, 63, 64)
    assert dict(r.control_ccs) == {"temperature": 20, "power": 21, "pressure": 22, "chaos": 23}
    assert r.control_channel == 15
    assert r.trigger_channel == 9
    assert r.pulse_s == 0.05
    assert r.bend_range == 2.0


def test_routes_are_frozen():
    with pytest.raises(Exception):
        DEFAULT_ROUTES.pulse_s = 1.0


def _write(tmp_path, data):
    p = tmp_path / "routes.json"
    p.write_text(json.dumps(data))
    return p


def test_load_merges_a_partial_file_over_the_defaults(tmp_path):
    r = load_routes(_write(tmp_path, {"pulse_s": 0.2, "control_channel": 3}))
    assert r.pulse_s == 0.2 and r.control_channel == 3
    assert r.bend_range == 2.0
    assert dict(r.note_channels) == {"lead": 0, "ttquantus": 1}
    assert r.roll_lane_notes == DEFAULT_ROUTES.roll_lane_notes


def test_load_merges_inside_a_mapping(tmp_path):
    r = load_routes(_write(tmp_path, {"note_channels": {"lead": 5}}))
    assert r.note_channels["lead"] == 5
    assert r.note_channels["ttquantus"] == 1


def test_load_roll_lane_notes_becomes_a_tuple(tmp_path):
    r = load_routes(_write(tmp_path, {"roll_lane_notes": [40, 41]}))
    assert r.roll_lane_notes == (40, 41)


@pytest.mark.parametrize("data,key", [
    ({"bogus": 1}, "bogus"),
    ({"control_channel": 16}, "control_channel"),
    ({"control_channel": -1}, "control_channel"),
    ({"trigger_channel": 16}, "trigger_channel"),
    ({"note_channels": {"lead": 16}}, "note_channels"),
    ({"trigger_notes": {"gong": 128}}, "trigger_notes"),
    ({"roll_lane_notes": [10, 200]}, "roll_lane_notes"),
    ({"control_ccs": {"power": 128}}, "control_ccs"),
    ({"control_ccs": {"power": -1}}, "control_ccs"),
])
def test_load_refuses_bad_values_naming_the_key(tmp_path, data, key):
    with pytest.raises(ValueError) as e:
        load_routes(_write(tmp_path, data))
    assert key in str(e.value)


def test_load_accepts_the_edges(tmp_path):
    r = load_routes(_write(tmp_path, {"control_channel": 15, "trigger_notes": {"gong": 0},
                                      "control_ccs": {"power": 127}}))
    assert r.control_channel == 15 and r.trigger_notes["gong"] == 0 and r.control_ccs["power"] == 127


def test_load_refuses_an_empty_lane_list(tmp_path):
    with pytest.raises(ValueError) as e:
        load_routes(_write(tmp_path, {"roll_lane_notes": []}))
    assert "roll_lane_notes" in str(e.value)


@pytest.mark.parametrize("data,key", [
    ({"control_channel": True}, "control_channel"),
    ({"trigger_channel": -1}, "trigger_channel"),
    ({"trigger_channel": False}, "trigger_channel"),
    ({"control_channel": 1.5}, "control_channel"),
    ({"pulse_s": 0}, "pulse_s"),
    ({"pulse_s": -0.1}, "pulse_s"),
    ({"pulse_s": "fast"}, "pulse_s"),
    ({"pulse_s": True}, "pulse_s"),
    ({"bend_range": 0}, "bend_range"),
    ({"bend_range": -2}, "bend_range"),
    ({"bend_range": None}, "bend_range"),
    ({"note_channels": [1, 2]}, "note_channels"),
    ({"trigger_notes": 5}, "trigger_notes"),
    ({"control_ccs": "x"}, "control_ccs"),
    ({"roll_lane_notes": 5}, "roll_lane_notes"),
    ({"roll_lane_notes": "ab"}, "roll_lane_notes"),
    ({"roll_lane_notes": [True]}, "roll_lane_notes"),
    ({"note_channels": {"lead": True}}, "note_channels"),
])
def test_load_refuses_wrong_types_with_value_error_naming_the_key(tmp_path, data, key):
    with pytest.raises(ValueError) as e:
        load_routes(_write(tmp_path, data))
    assert key in str(e.value)


def test_load_refuses_a_top_level_list(tmp_path):
    with pytest.raises(ValueError) as e:
        load_routes(_write(tmp_path, [1, 2]))
    assert "JSON object" in str(e.value)


def test_load_accepts_an_integer_pulse_and_bend(tmp_path):
    r = load_routes(_write(tmp_path, {"pulse_s": 1, "bend_range": 12}))
    assert r.pulse_s == 1 and r.bend_range == 12
