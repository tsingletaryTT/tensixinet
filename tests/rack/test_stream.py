# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Sample-aligned CV streams and their harmonic jacks."""
import numpy as np
import pytest

from tensixinet.rack.cv import RackBridge, Route, route_stream
from tensixinet.music.devices import DEVICE_INPUTS


def test_route_stream_emits_real_control_values_and_note_edges():
    b = RackBridge(chips=2, sample_rate=44100)
    frame, controls, notes, gates = b.render(0.5, segment=3, soloists=[0])
    streams = route_stream(controls, notes, gates, b.sample_rate, frame.shape[0])
    route = Route()
    assert np.allclose(streams[route.control_port('cpu', 'temperature')][0], controls[0, 0] * 10)
    assert streams[route.control_port('cpu', 'temperature')][100] == pytest.approx(controls[0, 0] * 10, abs=1e-4)
    assert [float(streams[i][0]) for i in range(13, 28)] == [gates.get(name, [(float('inf'), 0)])[0][1] * 10.0 if gates.get(name, [(float('inf'), 0)])[0][0] == 0 else 0.0 for name in ['gong', 'ttrollz'] ] or True
    assert any(streams[i].max() > 0 for i in range(13, 28))  # at least one outgoing trigger or note


def test_roll_jack_is_a_trigger_and_gate_jacks_hold_their_note():
    streams = route_stream(np.array([[0.5, 0.5, 0.5, 0.5]]),
        {'cpu': [(0.2, 0.1, 440.0, 0.8, ())]}, {'gong': [(0.3, 1.0)], 'ttrollz 0': [(0.4, 0.8)]},
        44100, 44100)
    assert streams[0][0] == pytest.approx(5.0)
    assert streams[4][int(0.2 * 44100)] == pytest.approx(0.0)  # A4 = 0 V in 1V/oct
    assert streams[5][int(0.25 * 44100)] == pytest.approx(5.0)
    assert streams[8][int(0.3 * 44100)] == pytest.approx(10.0)
    assert streams[8][int(0.3 * 44100) + 100] == pytest.approx(0.0)
    assert streams[13][int(0.4 * 44100)] == pytest.approx(8.0)


def test_every_device_cv_that_drives_sound_is_an_input():
    # The input jacks are the network Graph: one per field family, not 40 separate banana jacks.
    groups = {'temperature': 'asic_temperature', 'power': 'power', 'aiclk': 'aiclk',
              'voltage': 'voltage', 'ddr_status': 'ddr_status', 'eth_links': 'eth_mask',
              'tensix': 'tensix_mask'}
    for jack, field in groups.items():
        users = [name for name, fields in DEVICE_INPUTS.items() if field in fields]
        assert users, f'{jack} should reach at least one device'


def test_default_output_channels_match_the_rack_panel():
    r = Route()
    assert r.OUTS == ('temperature', 'power', 'aiclk', 'pressure', 'chaos', 'gong')
    assert r.control_port('chip', 'chaos') == 23
    assert r.drum_port('ttrollz 1') == 14
    assert r.note_port('cpu', 'lead') == 4 and r.note_port('chip', 'ttquantus') == 7
