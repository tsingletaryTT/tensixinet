# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Per-device extractors: the notes, triggers and controls a render is made of, read from the same functions the audio reads.

Every extractor takes one chip's `Telemetry`, the chip's number, the render's `duration` and `sample_rate`, the
device's `seed` and the `patch`, exactly as the device itself does. The chorus passes each device
`seed_base + slot`, with `seed_base = segment * SEGMENT_SEED_STRIDE`, so an extractor reads its segment out of the
seed (`seed // SEGMENT_SEED_STRIDE`) as `ttrollz` and the gong do. A score and the audio cannot disagree about it.

Velocity and strength are 1.0 (0.6 for an ordinary roll pulse); the caller scales them. Times are seconds from the
start of the render.
"""
from __future__ import annotations

from typing import List

import numpy as np

from ..music import devices
from ..music.lead import lead_notes
from ..music.mapping import POWER_MAX, chip_chaos, chip_pressure, norm_temp, temp_to_pitch_hz
from ..music.patch import DEFAULT_PATCH, Patch
from ..music.ttrolzer import SEGMENT_SEED_STRIDE
from ..music.ttsteam import steam_hits
from ..telemetry import Telemetry
from .events import Control, Note, Trigger

BEND_STEP_S = 0.02  # a note's pitch curve is sampled this often
GATE_THRESHOLD = 0.5  # a held note plays when the gate, read in the middle of the note, is above this
ROLL_PULSE_STRENGTH = 0.6  # an ordinary roll pulse; a pulse that starts a cycle is 1.0


def lead_events(t: Telemetry, chip: int, duration: float, sample_rate: int, seed: int, patch: Patch = DEFAULT_PATCH) -> List[Note]:
    """The lead's notes. Mirrors `lead.lead`, which renders from `lead_notes`."""
    return [Note(s, d, hz, 1.0, "lead", chip) for s, d, hz in lead_notes(t, duration, sample_rate, seed, patch)]


def quantus_events(t: Telemetry, chip: int, duration: float, sample_rate: int, seed: int, patch: Patch = DEFAULT_PATCH) -> List[Note]:
    """One note per held note of ttquantus's line that plays. Mirrors `devices.quantussy_tone`.

    A note lasts `1 / _note_rate` seconds (an eighth at the reference AXI clock). It plays when `ttquantus_gate`, read at the note's
    middle, is above 0.5. Its pitch is the chip's pitch moved by the semitone value at the note's first sample. With the patch's
    `quantus_mod` above 0 the glide and vibrato are kept as `bend`, the semitone value minus the first one every 20 ms."""
    n = int(duration * sample_rate)
    semis = devices.ttquantus_steps(t, duration, sample_rate, seed, patch)
    gate = devices.ttquantus_gate(t, duration, sample_rate, seed, patch)
    note_len = 1.0 / devices._note_rate(t)
    base = temp_to_pitch_hz(t.asic_temperature)
    notes: List[Note] = []
    k = 0
    while True:
        start = k * note_len
        first = int(round(start * sample_rate))
        if first >= n:
            break
        k += 1
        dur = min(note_len, duration - start)
        if gate[min(n - 1, int(round((start + dur / 2) * sample_rate)))] <= GATE_THRESHOLD:
            continue
        bend = ()
        if patch.quantus_mod > 0.0:
            steps = int(np.ceil(dur / BEND_STEP_S - 1e-9))
            bend = tuple(
                (j * BEND_STEP_S, float(semis[min(n - 1, first + int(round(j * BEND_STEP_S * sample_rate)))] - semis[first]))
                for j in range(steps)
            )
        notes.append(Note(start, dur, float(base * 2 ** (semis[first] / 12)), 1.0, "ttquantus", chip, bend))
    return notes


def gong_events(t: Telemetry, chip: int, duration: float, sample_rate: int, seed: int, patch: Patch = DEFAULT_PATCH) -> List[Trigger]:
    """One trigger per gong strike. Mirrors `devices.gong`, which strikes at `strike_times`."""
    segment = seed // SEGMENT_SEED_STRIDE
    return [Trigger(float(s), 1.0, "gong", chip) for s in devices.strike_times(t, duration, patch, segment)]


def roll_events(t: Telemetry, chip: int, duration: float, sample_rate: int, seed: int, patch: Patch = DEFAULT_PATCH) -> List[Trigger]:
    """The pulses ttrollz plays: the clock (lane -1) and each lane that is up (lane index). Mirrors `devices.ttrollz`.

    A pulse that starts a roll's cycle has strength 1.0 and any other 0.6; the clock has no cycle starts."""
    segment = seed // SEGMENT_SEED_STRIDE
    out: List[Trigger] = []
    for _, lane, times, starts in devices.ttrollz_pulse_sets(t, duration, seed, patch, segment):
        for time, is_start in zip(times, starts):
            out.append(Trigger(float(time), 1.0 if is_start else ROLL_PULSE_STRENGTH, "ttrollz", chip, lane))
    return out


def steam_events(t: Telemetry, chip: int, duration: float, sample_rate: int, seed: int, patch: Patch = DEFAULT_PATCH) -> List[Trigger]:
    """One trigger per register hit the snare plays. Mirrors `ttsteam.ttsteam`, which renders from `steam_hits`."""
    return [Trigger(float(s), 1.0, "ttsteam", chip) for s in steam_hits(t, duration, seed, patch)]


def control_events(t: Telemetry, chip: int, segment_seconds: float) -> List[Control]:
    """The chip's temperature, power, pressure and chaos at the start of the segment, each 0 to 1.

    Temperature and power use the ranges in `mapping.py` (25 to 100 C, 0 to `POWER_MAX` W), clamped.
    `segment_seconds` is accepted so a caller can pass the segment's length; the values are read once, at t = 0."""
    power = min(1.0, max(0.0, t.power / POWER_MAX))
    values = (("temperature", norm_temp(t.asic_temperature)), ("power", power), ("pressure", chip_pressure(t)), ("chaos", chip_chaos(t)))
    return [Control(0.0, name, float(v), chip) for name, v in values]
