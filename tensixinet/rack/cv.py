# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The Python side of the VCV Rack bridge: telemetry in, sound and events out.

Patched CV inputs override individual readings of each chip before the chorus is rendered;
the composite jack maps 0-10 V onto the ranges the hardware reports. Outgoing controls,
notes and gates are read from the same Score the sinks use, so what the module shows and
what it hears cannot drift apart.
"""
from __future__ import annotations

import dataclasses
from dataclasses import replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..music.chorus import render_chorus
from ..music.mapping import PRESSURE_TEMP_FLOOR_C
from ..music.patch import Patch
from ..score.collect import chorus_score
from ..telemetry import Telemetry


class NV:
    """The composite jack: 0-10 V becomes one of the hardware's reading ranges, clamped."""

    @staticmethod
    def volts(v: float) -> float:
        return min(10.0, max(0.0, float(v)))

    @staticmethod
    def temp(v: float) -> float:
        return 25.0 + NV.volts(v) / 10.0 * 75.0

    @staticmethod
    def power(v: float) -> float:
        return NV.volts(v) / 10.0 * 200.0

    @staticmethod
    def clk(v: float) -> float:
        return 400.0 + NV.volts(v) / 10.0 * 950.0

    @staticmethod
    def voltage(v: float) -> float:
        return NV.volts(v) / 10.0 * 1.5

    @staticmethod
    def gddr(v: float) -> float:
        return 25.0 + NV.volts(v) / 10.0 * 75.0

    @staticmethod
    def unitlevel(v: float) -> float:
        return min(1.0, NV.volts(v) / 7.0)


def _blend(hardware: float, mapped: float, knob: float) -> float:
    """A knob of 0 leaves the hardware reading, 1 takes the patched CV completely."""
    k = min(1.0, max(0.0, float(knob)))
    return hardware * (1.0 - k) + mapped * k


def _mapped(inputs: dict, key: str) -> Optional[float]:
    return float(inputs[key]) if key in inputs else None


def _level(inputs: dict, key: str, knob: float, bits: int) -> Optional[int]:
    """An unitlevel CV scaled by a knob, into a low-bit decimal field."""
    v = _mapped(inputs, key)
    if v is None:
        return None
    return min(bits, int(round(NV.unitlevel(v) * min(1.0, max(0.0, knob)) * bits)))


def compose_readings(chip: Telemetry, inputs: dict, gate: float, knob_levels: dict) -> Telemetry:
    """Apply the patched CV to one chip's readings. Fields with no cable are kept as they are.

    `gate` is the override switch: 0 leaves every hardware reading untouched. Each patched
    field's attenuator blends the mapped CV into the hardware value, so 0 is a bypass and 1
    takes the CV completely. Weekly fields like GDDR channels blend individually. Masks and
    pressure limits are kept consistent so the mapping functions still see a valid chip.
    """
    fields: Dict = {
        'asic_temperature': chip.asic_temperature, 'aiclk': chip.aiclk, 'power': chip.power,
        'current': chip.current, 'voltage': chip.voltage, 'gddr_temps': chip.gddr_temps,
        'ddr_status': chip.ddr_status, 'board_power': chip.board_power, 'axiclk': chip.axiclk,
        'arcclk': chip.arcclk, 'heartbeat': chip.heartbeat, 'tensix_mask': chip.tensix_mask,
        'eth_mask': chip.eth_mask, 'pcie_usage': chip.pcie_usage,
    }
    if not gate:
        return replace(chip, **fields)

    def knob(name: str) -> float:
        return min(1.0, max(0.0, float(knob_levels.get(name, 1.0))))

    v = _mapped(inputs, 'temperature')
    if v is not None:
        fields['asic_temperature'] = _blend(chip.asic_temperature, NV.temp(v), knob('temperature'))
    v = _mapped(inputs, 'power')
    if v is not None:
        fields['power'] = _blend(chip.power, NV.power(v), knob('power'))

    v = _mapped(inputs, 'aiclk')
    if v is not None:
        fields['aiclk'] = _blend(chip.aiclk, NV.clk(v), knob('aiclk'))

    v = _mapped(inputs, 'voltage')
    if v is not None:
        fields['voltage'] = _blend(chip.voltage, NV.voltage(v), knob('voltage'))

    v = _mapped(inputs, 'gddr')
    if v is not None:
        fields['gddr_temps'] = tuple(
            _blend(old, NV.gddr(v), knob('gddr')) for old in chip.gddr_temps
        )

    v = _mapped(inputs, 'gddr0')
    if v is not None:
        temps = list(fields['gddr_temps'])
        temps[0] = NV.gddr(v)
        fields['gddr_temps'] = tuple(temps)

    v = _mapped(inputs, 'gddr1')
    if v is not None:
        temps = list(fields['gddr_temps'])
        temps[1] = NV.gddr(v)
        fields['gddr_temps'] = tuple(temps)

    if 'gddr2' in inputs or 'gddr3' in inputs or 'gddr4' in inputs or 'gddr5' in inputs or 'gddr6' in inputs or 'gddr7' in inputs:
        temps = list(fields['gddr_temps'])
        for i in range(2, 8):
            named = _mapped(inputs, f'gddr{i}')
            if named is not None:
                temps[i] = NV.gddr(named)
        fields['gddr_temps'] = tuple(temps)

    if 'ddr_status' in inputs:
        v = float(inputs['ddr_status'])
        count = min(4, int(round(NV.unitlevel(v) * knob('ddr') * 4)))
        fields['ddr_status'] = tuple(min(3, count) if i < count else 0 for i in range(4))

    if 'eth_links' in inputs:
        fields['eth_mask'] = _level(inputs, 'eth_links', knob('eth'), 16) or 0

    if 'tensix' in inputs:
        fields['tensix_mask'] = _level(inputs, 'tensix', knob('tensix'), 14) or 0

    thm, tdp = chip.thm_limit_throttle, chip.tdp_limit
    v = _mapped(inputs, 'pressure')
    if v is not None:
        want = NV.unitlevel(v) * knob('pressure')
        tdp = max(fields['power'], 1.0)
        if want <= 0.0:
            thm = PRESSURE_TEMP_FLOOR_C
        else:
            thm = PRESSURE_TEMP_FLOOR_C + (fields['asic_temperature'] - PRESSURE_TEMP_FLOOR_C) / min(1.0, want)
        thm = max(thm, fields['asic_temperature'], PRESSURE_TEMP_FLOOR_C)
    fields['thm_limit_throttle'] = thm
    fields['tdp_limit'] = tdp

    if 'heartbeat' in inputs:
        fields['heartbeat'] = _level(inputs, 'heartbeat', knob('heartbeat'), 16) or 0

    return replace(chip, **fields)


class Route:
    """The Rack module's jack numbers, chosen so the banks do not collide.

    General controls: cpu temperature 0, power 1, aiclk 2, chaos 3; chip bank offset 20,
    other bank offset 40. Notes: cpu lead 4, cpu ttquantus 5, chip lead 6, chip ttquantus 7.
    Drums: gong 8, ttsteam 10, ttrollz clock 12, ttrollz lane n 13+n (up to lane 9 at 22).
    """

    OUTS = ('temperature', 'power', 'aiclk', 'pressure', 'chaos', 'gong')
    GENERAL = {name: i for i, name in enumerate(('temperature', 'power', 'aiclk', 'chaos'))}
    CONTROL_OFFSET = {'cpu': 0, 'chip': 20, 'other': 40}
    NOTES = {('cpu', 'lead'): 4, ('cpu', 'ttquantus'): 5, ('chip', 'lead'): 6, ('chip', 'ttquantus'): 7}
    DRUMS = {'gong': 8, 'ttrollz': 9, 'ttsteam': 10, 'ttrollz clock': 12}
    ROLL_BASE = 13
    ROLL_COUNT = 10

    def control_port(self, source: str, name: str) -> int:
        return self.CONTROL_OFFSET.get(source, 40) + self.GENERAL[name]

    def drum_port(self, name: str) -> int:
        if name.startswith('ttrollz '):
            suffix = name.split(' ', 1)[1]
            if suffix == 'clock':
                return self.DRUMS['ttrollz clock']
            return self.ROLL_BASE + int(suffix)
        return self.DRUMS[name]

    def note_port(self, source: str, name: str) -> int:
        return self.NOTES[(source, name)]


class RackBridge:
    """Holds the readings, composes CV into them and publishes audio and events.

    `source='mock'` uses deterministic synthetic chips; `source='hardware'` reads `tt-smi`
    lazily and falls back to the mock if there is no full snapshot. `set_readings` replaces
    the chips from a live loop; `set_params` swaps the CV and knob maps atomically.

    `render` gives a stereo float32 frame, per-chip controls at t=0, notes by source and
    gates by drum name. `render_events` gives the same controls with the Scores, for a
    caller that wants to route the events itself.
    """

    def __init__(self, chips: int = 4, sample_rate: int = 44100, inputs: Optional[dict] = None,
                 knobs: Optional[dict] = None, source: str = 'mock') -> None:
        self.sample_rate = sample_rate
        self.source = source
        self.inputs: dict = dict(inputs or {})
        self.knobs: dict = dict(knobs or {})
        readings = self._hardware(chips) if source == 'hardware' else None
        self.readings = readings if readings is not None else self._mock(chips)

    @staticmethod
    def _mock(chips: int) -> List[Telemetry]:
        from ..telemetry.mock import mock_telemetry
        return list(mock_telemetry(chips, seed=5))

    @staticmethod
    def _hardware(chips: int) -> Optional[List[Telemetry]]:
        try:
            from ..telemetry.snapshot import fetch_snapshot, parse_snapshot
            readings = parse_snapshot(fetch_snapshot())
            if len(readings) == chips:
                return list(readings)
        except (ImportError, OSError, ValueError, TypeError):
            pass
        return None

    def set_readings(self, readings: Sequence[Telemetry]) -> None:
        values = list(readings)
        if not values:
            raise ValueError('set_readings needs at least one chip')
        self.readings = values

    def set_params(self, inputs: Optional[dict] = None, knobs: Optional[dict] = None) -> None:
        self.inputs, self.knobs = dict(inputs or {}), dict(knobs or {})

    def _render(self, duration: float, segment: int, soloists, only, patch) -> Tuple[np.ndarray, np.ndarray, dict, dict, List]:
        telemetry = [compose_readings(t, self.inputs, 1.0, self.knobs) for t in self.readings]
        kw: Dict = dict(sample_rate=self.sample_rate)
        if patch is not None:
            kw['patch'] = patch
        frame = render_chorus(telemetry, duration, stereo=True, segment=segment,
                              soloists=soloists, only=only, **kw)
        score = chorus_score(telemetry, duration, segment=segment,
                             soloists=soloists, only=only, **kw)
        scores = [replace(score, events=tuple(e for e in score.events if e.chip == chip))
                  for chip in range(len(telemetry))]
        return frame, self._controls(score, len(telemetry)), self._notes(score), self._gates(score), scores

    @staticmethod
    def _controls(score, n_chips: int) -> np.ndarray:
        names = ('temperature', 'power', 'pressure', 'chaos')
        out = np.zeros((n_chips, 4))
        for e in score.controls():
            if e.name in names and e.t == 0.0:
                out[e.chip, names.index(e.name)] = float(e.value)
        return out

    @staticmethod
    def _notes(score) -> dict:
        groups: dict = {}
        for e in score.notes():
            source = 'cpu' if e.chip % 2 == 0 else 'chip'
            groups.setdefault(source, []).append(
                (float(e.t), float(e.dur), float(e.pitch_hz), float(e.velocity), e.bend)
            )
        return groups

    @staticmethod
    def _gates(score) -> dict:
        groups: dict = {}
        for e in score.triggers():
            name = e.source
            if e.source == 'ttrollz':
                if e.lane < 0:
                    name = 'ttrollz clock'
                elif e.lane >= 2:
                    continue  # the module has two roll lane jacks
                else:
                    name = f'ttrollz {e.lane}'
            groups.setdefault(name, []).append((float(e.t), float(e.strength)))
        return groups

    def render(self, duration: float, segment: int = 0, soloists=None, only=None, patch: Patch = None):
        frame, controls, notes, gates, _ = self._render(duration, segment, soloists, only, patch)
        self.last_streams = route_stream(controls, notes, gates, self.sample_rate, frame.shape[0])
        return frame, controls, notes, gates

    def cv_streams(self):
        """The outgoing jacks as raw arrays, in the same order as the Rack panel."""
        return [self.last_streams[i] for i in sorted(self.last_streams)]

    def render_events(self, duration: float, segment: int = 0, soloists=None, only=None, patch: Patch = None):
        _, controls, _, _, scores = self._render(duration, segment, soloists, only, patch)
        return controls, scores


def route_stream(controls: np.ndarray, notes: dict, gates: dict, sample_rate: int,
                 n_samples: int, chip_of_source: Optional[dict] = None) -> dict:
    """Every outgoing jack as one sample-aligned voltage stream.

    Control jacks hold each chip's temperature, power, clock and chaos at 0-10 V. Pitched
    jacks hold 1 V per octave (A4 = 440 Hz), a 5 V gate while a note sounds and an 0-10 V
    velocity envelope. Drum jacks give a 10 V, 1 ms pulse at each hit. The Stream object
    can be read at the audio block's own rate by the C++ host.
    """
    route = Route()

    def place(port: int, value: float, start: int) -> None:
        stream[port][min(start, n_samples - 1)] = float(value)

    ports = 4 * 3 + 4 + 12  # three control banks, four pitched pairs, twelve jacks beyond them
    stream = np.zeros((ports, n_samples), dtype=np.float32)
    names = ('temperature', 'power', 'aiclk', 'chaos')
    for chip in range(controls.shape[0]):
        bank = 'cpu' if chip == 0 else 'chip'
        for i, v in enumerate(controls[chip]):
            port = route.control_port(bank, names[i])
            stream[port][:] = float(v) * 10.0
    sources = {'cpu': 0, 'chip': 1} if chip_of_source is None else chip_of_source
    for source, events in notes.items():
        device = 'ttquantus' if source == 'chip' else 'lead'
        port = route.note_port(source, device)
        p, g, e = port, port + 1, port + 10
        for t, dur, hz, velocity, bend in events:
            start = min(int(float(t) * sample_rate), n_samples - 1)
            end = min(int(float(t + dur) * sample_rate), n_samples)
            stream[p][start:end] = np.log2(np.maximum(hz, 1.0) / 440.0)
            stream[g][start:end] = 5.0
            stream[e][start:end] = velocity * 10.0
    for name, events in gates.items():
        port = route.drum_port(name)
        for t, strength in events:
            start = min(int(float(t) * sample_rate), n_samples - 1)
            width = max(1, min(int(0.001 * sample_rate), n_samples - start))
            stream[port][start:start + width] = 10.0 * float(strength)
    return {i: stream[i] for i in range(ports)}
