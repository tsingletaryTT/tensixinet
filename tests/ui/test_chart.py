# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
import dataclasses
import re
import xml.etree.ElementTree as ET

import pytest

from tensixinet.music.chorus import device_names
from tensixinet.music.evict import PRESENCE
from tensixinet.music.patch import DEFAULT_PATCH
from tensixinet.telemetry import mock_telemetry
from tensixinet.ui.chart import field_xy, flow_seconds, flow_svg
from tensixinet.ui.flow import build_flow

NS = {"s": "http://www.w3.org/2000/svg"}


def _svg(n=4, soloist=2, **kw):
    state = build_flow(mock_telemetry(n), DEFAULT_PATCH, 1, soloist, 4.0, **kw)
    return state, flow_svg(state)


def _tree(svg):
    return ET.fromstring(svg)


def _by_class(root, cls):
    return [e for e in root.iter() if cls in (e.get("class") or "").split()]


def test_the_chart_is_well_formed_svg_with_no_script():
    _, svg = _svg()
    root = _tree(svg)
    assert root.tag.endswith("svg") and "<script" not in svg and "viewBox" in root.attrib


def test_every_chip_and_every_device_is_drawn_once_in_the_flow_and_once_in_the_field():
    state, svg = _svg()
    root = _tree(svg)
    assert sorted(e.get("data-chip") for e in _by_class(root, "chip")) == ["0", "1", "2", "3"]
    assert sorted(e.get("data-device") for e in _by_class(root, "device")) == sorted(device_names())
    assert sorted(e.get("data-device") for e in _by_class(root, "voice")) == sorted(device_names())


def test_only_the_soloist_is_marked_in_both_places():
    state, svg = _svg(soloist=5)
    root = _tree(svg)
    assert [e.get("data-device") for e in _by_class(root, "device") if "solo" in e.get("class").split()] == [state.soloist_device]
    assert [e.get("data-device") for e in _by_class(root, "voice") if "solo" in e.get("class").split()] == [state.soloist_device]


def test_a_chips_phase_is_its_class_and_a_dissolving_chips_voices_fade_by_the_presence_table():
    state, svg = _svg(phases={1: "evicting", 3: "idle"})
    root = _tree(svg)
    phase = {e.get("data-chip"): e.get("class").split() for e in _by_class(root, "chip")}
    assert "evicting" in phase["1"] and "idle" in phase["3"] and "running" in phase["0"]
    for v in _by_class(root, "voice"):
        chip = next(d.chip for d in state.devices if d.name == v.get("data-device"))
        want = PRESENCE[state.chips[chip].phase]
        assert float(v.get("opacity")) == pytest.approx(want, abs=0.01)


def test_one_cable_is_drawn_for_each_cable_in_the_state_with_its_kind_and_a_width_that_grows_with_its_push():
    state, svg = _svg()
    root = _tree(svg)
    drawn = _by_class(root, "cable")
    assert len(drawn) == len(state.cables) > 0
    for e, c in zip(drawn, state.cables):
        assert (e.get("data-source"), e.get("data-target")) == (c.source, c.target) and c.kind in e.get("class").split()
    widths = {(e.get("data-source"), e.get("data-target")): float(e.get("stroke-width")) for e in drawn}
    strong, weak = max(state.cables, key=lambda c: c.push), min(state.cables, key=lambda c: c.push)
    if strong.push > weak.push:
        assert widths[(strong.source, strong.target)] > widths[(weak.source, weak.target)]


def test_no_cables_draws_none():
    _, svg = _svg(cables=False)
    assert _by_class(_tree(svg), "cable") == []


def test_the_field_places_left_to_right_by_azimuth_and_near_to_far_by_depth():
    (xl, yn), (xr, yf) = field_xy(-1.0, 0.0), field_xy(1.0, 1.0)
    assert xl < field_xy(0.0, 0.0)[0] < xr and yn > yf  # a near voice is lower on the page, a far one higher


def test_a_hotter_chip_flows_faster():
    assert flow_seconds(100.0) < flow_seconds(40.0) < flow_seconds(14.0)
    assert flow_seconds(1000.0) > 0  # never stops or runs backwards


def test_the_header_names_the_patch_the_segment_and_who_has_the_floor():
    state, svg = _svg()
    for text in (DEFAULT_PATCH.scale_name, DEFAULT_PATCH.gong_tuning, state.soloist_device, "segment 1"):
        assert text in svg, text


def test_text_from_a_patch_is_escaped():
    state = build_flow(mock_telemetry(4), dataclasses.replace(DEFAULT_PATCH, scale_name="a<b>&c"), 0, 0, 4.0)
    svg = flow_svg(state)
    assert "a<b>" not in svg and "a&lt;b&gt;&amp;c" in svg and _tree(svg) is not None
