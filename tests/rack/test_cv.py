# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Tests for the Rack CV bridge: override the readings, publish outgoing control."""
import numpy as np
import pytest

from tensixinet.rack.cv import NV, Route, RackBridge, compose_readings
from tensixinet.telemetry.mock import mock_telemetry


def test_nv_maps_volts_to_the_documented_ranges():
    assert NV.temp(0.0) == 25.0 and NV.temp(10.0) == 100.0 and NV.temp(5.0) == 62.5
    assert NV.power(0.0) == 0.0 and NV.power(10.0) == 200.0
    assert NV.clk(0.0) == 400.0 and NV.clk(10.0) == 1350.0
    assert NV.voltage(0.0) == 0.0 and NV.voltage(10.0) == 1.5
    assert NV.gddr(0.0) == 25.0 and NV.gddr(10.0) == 100.0
    assert NV.unitlevel(0.0) == 0.0 and NV.unitlevel(10.0) == 1.0
    assert NV.unitlevel(7.0) == 1.0  # patched above the top holds there


def test_override_gate_bypasses_a_patched_cv():
    chip = list(mock_telemetry(1, seed=1))[0]
    off = compose_readings(chip, {'temperature': 10.0}, 0.0, {'temperature': 1.0})
    on = compose_readings(chip, {'temperature': 10.0}, 1.0, {'temperature': 1.0})
    assert off == chip
    assert on.asic_temperature == pytest.approx(100.0)


def test_compose_readings_honours_override_and_its_knob():
    chip = list(mock_telemetry(1, seed=1))[0]
    r = compose_readings(
        chip,
        {'temperature': 7.0, 'power': 3.0, 'aiclk': 10.0,
         'voltage': 5.0, 'gddr0': 0.0, 'gddr1': 10.0,
         'gddr2': 0.0, 'gddr3': 0.0, 'gddr4': 0.0, 'gddr5': 0.0,
         'gddr6': 0.0, 'gddr7': 0.0, 'ddr_status': 8.0, 'eth_links': 10.0,
         'tensix': 5.0, 'pressure': 10.0, 'heartbeat': 4.0},
        1.0,
        {'temperature': 0.4, 'power': 0.6, 'aiclk': 0.0,
         'voltage': 1.0, 'gddr': 1.0, 'ddr': 1.0, 'eth': 1.0,
         'pressure': 0.5, 'heartbeat': 1.0},
    )
    # An attenuator of 0.4 blends 40% of the patched 7 V (77.5 C) into the hardware reading.
    assert r.asic_temperature == pytest.approx(chip.asic_temperature * 0.6 + 77.5 * 0.4)
    assert r.power == pytest.approx(chip.power * 0.4 + 60.0 * 0.6)
    assert r.aiclk == pytest.approx(chip.aiclk)  # knob 0 leaves the clock alone
    assert r.voltage == pytest.approx(0.75)
    assert r.gddr_temps[0] == 25.0 and r.gddr_temps[1] == 100.0
    assert r.ddr_status[1] >= 1  # ddr_status 8 V is three or four enabled lanes
    assert r.eth_mask != 0
    assert r.tensix_mask != 0
    assert r.thm_limit_throttle >= r.asic_temperature  # pressure still a valid range
    assert r.tdp_limit >= r.power


def test_compose_readings_keeps_the_hardware_reading_where_there_is_no_cable():
    chip = list(mock_telemetry(1, seed=1))[0]
    r = compose_readings(chip, {}, 1.0, {})
    assert r == chip
    r = compose_readings(chip, {'temperature': 5.0}, 0.0, {})
    assert r.asic_temperature == chip.asic_temperature
    assert r.power == chip.power


def test_route_maps_the_six_module_outputs():
    r = Route()
    assert r.OUTS == ('temperature', 'power', 'aiclk', 'pressure', 'chaos', 'gong')
    assert r.control_port('cpu', 'temperature') == 0
    assert r.control_port('cpu', 'chaos') == 3
    assert r.drum_port('gong') == 8
    assert r.drum_port('ttrollz') == 9
    assert r.note_port('cpu', 'lead') == 4
    assert r.note_port('cpu', 'ttquantus') == 5
    assert r.note_port('chip', 'lead') == 6
    assert r.note_port('chip', 'ttquantus') == 7


def test_bridge_renders_real_audio_and_exposes_the_same_event_stream():
    b = RackBridge(chips=2, sample_rate=44100)
    frame, controls, notes, gates = b.render(0.25, segment=0, soloists=[0])
    assert frame.shape == (11025, 2)
    assert np.isfinite(frame).all() and np.abs(frame).max() > 0
    assert controls.shape == (2, 4)
    assert set(notes) <= {'cpu', 'chip'}
    assert gates and all(n == 'gong' or n == 'ttsteam' or n.startswith('ttrollz') for n in gates)
    assert all(0.0 <= v <= 1.0 for v in controls.flatten())


def test_bridge_uses_the_cv_inputs():
    fields = {'temperature': 10.0, 'power': 10.0, 'aiclk': 10.0}
    knobs = {'temperature': 1.0, 'power': 1.0, 'aiclk': 1.0}
    a = RackBridge(chips=2, sample_rate=44100, inputs=fields, knobs=knobs)
    b = RackBridge(chips=2, sample_rate=44100)
    x = a.render(0.25, segment=0)[0]
    y = b.render(0.25, segment=0)[0]
    assert not np.array_equal(x, y)  # the patched CV reaches the audio


def test_bridge_aligns_the_control_readings_with_the_events():
    b = RackBridge(chips=2, sample_rate=44100)
    controls, scores = b.render_events(0.25, segment=1)
    assert len(controls) == 2 and len(scores) == 2
    for chip, score in enumerate(scores):
        ctrls = score.controls()
        assert [e.name for e in ctrls[:4]] == ['temperature', 'power', 'pressure', 'chaos']
        for e in ctrls[:4]:
            assert e.chip == chip and e.t == 0.0
            assert e.value == pytest.approx(float(controls[chip][('temperature', 'power', 'pressure', 'chaos').index(e.name)]))
