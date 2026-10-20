# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
import dataclasses

import pytest

from tensixinet.music.cables import draw_cables, strength
from tensixinet.music.chorus import assign_device, device_names, n_slots, slot_chip, slot_weights
from tensixinet.music.mapping import chip_chaos, chip_pressure
from tensixinet.music.patch import DEFAULT_PATCH
from tensixinet.music.space import chip_azimuth, device_space
from tensixinet.telemetry import mock_telemetry
from tensixinet.ui.flow import build_flow, phases_at


def _state(n=4, soloist=2, **kw):
    chips = mock_telemetry(n)
    return chips, build_flow(chips, DEFAULT_PATCH, segment=3, soloist_slot=soloist, seconds=4.0, **kw)


def test_every_device_is_on_the_chip_the_chorus_deals_it_to_and_exactly_one_has_the_floor():
    chips, s = _state()
    assert [d.name for d in s.devices] == device_names()
    for d in s.devices:
        assert d.chip == slot_chip(d.slot, 4) and d.name == assign_device(d.slot).__name__
    assert [d.slot for d in s.devices if d.soloist] == [2]
    assert s.soloist_device == assign_device(2).__name__ and s.soloist_chip == slot_chip(2, 4)


def test_a_device_sits_at_its_chips_home_plus_its_own_offset_and_depth_from_the_space_tables():
    chips, s = _state()
    for d in s.devices:
        offset, _, depth = device_space(d.name)
        home = chip_azimuth(d.chip, 4, chips[d.chip])
        assert d.azimuth == pytest.approx(max(-1.0, min(1.0, home + offset)))
        assert d.depth == depth


def test_the_cables_are_the_ones_the_chorus_draws_pushed_by_the_chaos_of_the_chip_that_plays_the_source():
    chips, s = _state()
    wiring = draw_cables(DEFAULT_PATCH.seed, device_names())
    assert [(c.source, c.target, c.kind, c.depth) for c in s.cables] == [(w.source, w.target, w.kind, w.depth) for w in wiring]
    chip_of = {d.name: d.chip for d in s.devices}
    for view, w in zip(s.cables, wiring):
        assert view.push == pytest.approx(strength(w, chip_chaos(chips[chip_of[w.source]])))


def test_cables_off_gives_no_cables():
    _, s = _state(cables=False)
    assert s.cables == ()


def test_the_chip_views_carry_the_readings_pressure_chaos_and_home():
    chips, s = _state()
    for i, c in enumerate(s.chips):
        t = chips[i]
        assert (c.index, c.temp_c, c.power_w) == (i, t.asic_temperature, t.power)
        assert c.pressure == pytest.approx(chip_pressure(t)) and c.chaos == pytest.approx(chip_chaos(t))
        assert c.azimuth == pytest.approx(chip_azimuth(i, 4, t)) and c.phase == "running"


def test_a_devices_share_is_its_turn_weight_over_all_the_weights():
    chips, s = _state()
    w = slot_weights(chips, DEFAULT_PATCH)
    assert sum(d.share for d in s.devices) == pytest.approx(1.0)
    for d in s.devices:
        assert d.share == pytest.approx(w[d.slot] / sum(w))


def test_phases_name_each_chips_state_and_a_dissolving_chip_is_passed_through():
    chips, s = _state(phases={1: "evicting", 3: "idle"})
    assert [c.phase for c in s.chips] == ["running", "evicting", "running", "idle"]


def test_a_one_chip_box_still_makes_a_state():
    chips, s = _state(n=1, soloist=0)
    assert len(s.chips) == 1 and all(d.chip == 0 for d in s.devices) and s.chips[0].azimuth == pytest.approx(chip_azimuth(0, 1, chips[0]))


def test_the_patch_summary_names_what_the_patch_decided():
    _, s = _state()
    p = s.patch
    assert (p["scale"], p["gong"], p["rax_bars"], p["seed"]) == (DEFAULT_PATCH.scale_name, DEFAULT_PATCH.gong_tuning, DEFAULT_PATCH.rax_bars, DEFAULT_PATCH.seed)
    assert p["style"] == "" and p["routes"] == () and s.segment == 3 and s.seconds == 4.0


def test_phases_at_follows_evict_and_fill_events_by_the_clock():
    from tensixinet.music.evict import EVICT_SECONDS, FILL_SECONDS
    events = [(10.0, "evict", 1), (30.0, "fill", 1), (31.0, "evict", 2)]
    assert phases_at(events, 5.0, 3) == {}
    assert phases_at(events, 10.0 + EVICT_SECONDS - 0.1, 3) == {1: "evicting"}
    assert phases_at(events, 10.0 + EVICT_SECONDS + 0.1, 3) == {1: "idle"}
    assert phases_at(events, 30.0 + FILL_SECONDS - 0.1, 3) == {1: "filling", 2: "evicting"}
    assert phases_at(events, 30.0 + FILL_SECONDS + 0.1, 3) == {2: "evicting"}
    assert phases_at(events, 31.0 + EVICT_SECONDS + 0.1, 3) == {2: "idle"}


def test_a_device_never_sits_outside_the_field_even_when_its_chip_is_at_the_edge(monkeypatch):
    import tensixinet.ui.flow as flow
    for home in (0.95, -0.95):
        monkeypatch.setattr(flow, "chip_azimuth", lambda i, n, t, home=home: home)
        s = build_flow(mock_telemetry(4), DEFAULT_PATCH, 0, 0, 4.0)
        assert all(-1.0 <= d.azimuth <= 1.0 for d in s.devices)
        assert any(abs(d.azimuth) == 1.0 for d in s.devices)  # some offset really does push past the edge, so the clamp is what holds it
