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

import dataclasses
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from ..music import devices
from ..music.chorus import assign_device, device_names, emphasis_of, n_slots, plan_turns, slot_chip, voice_order
from ..music.evict import ChipDefrag, drain_progress
from ..music.turns import turn_gains
from ..music.lead import lead_notes
from ..music.mapping import POWER_MAX, aiclk_to_bpm, chip_chaos, chip_pressure, norm_temp, temp_to_pitch_hz
from ..music.patch import DEFAULT_PATCH, Patch
from ..music.ttrolzer import SEGMENT_SEED_STRIDE
from ..music.ttsteam import steam_hits
from ..telemetry import Telemetry
from .events import Control, Note, Score, Trigger

BEND_STEP_S = 0.02  # a note's pitch curve is sampled this often
GATE_THRESHOLD = 0.5  # a held note plays when the gate, read in the middle of the note, is above this
MIN_VELOCITY = 0.05  # a quiet voice still sounds
DISSOLVED_KEEP = 0.02  # an evicting or filling chip's voice whose fade factor is at or below this is gone: it has no events
ROLL_PULSE_STRENGTH = 0.6  # an ordinary roll pulse; a pulse that starts a cycle is 1.0


def lead_events(t: Telemetry, chip: int, duration: float, sample_rate: int, seed: int, patch: Patch = DEFAULT_PATCH) -> List[Note]:
    """The lead's notes. Mirrors `lead.lead`, which renders from `lead_notes`."""
    return [Note(s, d, hz, 1.0, "lead", chip) for s, d, hz in lead_notes(t, duration, sample_rate, seed, patch)]


def quantus_events(t: Telemetry, chip: int, duration: float, sample_rate: int, seed: int, patch: Patch = DEFAULT_PATCH) -> List[Note]:
    """One note per held note of ttquantus's line that plays. Mirrors `devices.quantussy_tone`.

    A note lasts `1 / _note_rate` seconds (an eighth at the reference AXI clock). It plays when `ttquantus_gate`, read at the note's
    middle, is above 0.5. Its pitch is the chip's pitch moved by the held scale step (the castle's degree plus the register shift,
    `devices.ttquantus_held`) at the note's middle sample. The first sample of a note can still hold the previous step, or sit
    part-way through a glide, so it is not used. With the patch's `quantus_mod` above 0 the glide and vibrato are kept as `bend`,
    the audio's semitone value minus the held one every 20 ms. A glide wider than the bend range is clamped by the MIDI writer."""
    n = int(duration * sample_rate)
    semis = devices.ttquantus_steps(t, duration, sample_rate, seed, patch)
    held = devices.ttquantus_held(t, duration, sample_rate, seed, patch)
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
        middle = min(n - 1, int(round((start + dur / 2) * sample_rate)))
        if gate[middle] <= GATE_THRESHOLD:
            continue
        step = float(held[middle])
        bend = ()
        if patch.quantus_mod > 0.0:
            steps = int(np.ceil(dur / BEND_STEP_S - 1e-9))
            bend = tuple(
                (j * BEND_STEP_S, float(semis[min(n - 1, first + int(round(j * BEND_STEP_S * sample_rate)))] - step))
                for j in range(steps)
            )
        notes.append(Note(start, dur, float(base * 2 ** (step / 12)), 1.0, "ttquantus", chip, bend))
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


# the extractor for each device the chorus deals; every other device contributes only its chip's controls
EXTRACTORS: Dict[str, Callable] = {
    "lead": lead_events, "ttquantus": quantus_events, "gong": gong_events, "ttrollz": roll_events, "ttsteam": steam_events,
}


def chorus_score(telemetry, duration: float, patch: Patch = DEFAULT_PATCH, segment: int = 0,
                 soloists: Optional[Sequence[int]] = None, only: Optional[Sequence[str]] = None,
                 defrag: Optional[Sequence[ChipDefrag]] = None, sample_rate: int = 44100) -> Score:
    """The Score of one render, built the way `render_chorus` deals and seeds its devices.

    Slot `s` plays `assign_device(s)` on chip `slot_chip(s, n)` with seed `segment * SEGMENT_SEED_STRIDE + s`. Each event's velocity
    (or strength) is the extractor's value times the device's balance and the slot's turn gain at the event's time, clamped to
    0.05 to 1. `only` plays the named devices at full level with no turns. A chip that is idle in `defrag` has no notes or triggers;
    a chip that is evicting or filling scales them by `(1 - dissolve) ** 2`, the render's own fade, read at the event's time, and
    drops those whose fade is at or below `DISSOLVED_KEEP` (0.02): a voice that has dissolved is silent in MIDI as in the audio.
    Controls are emitted once per chip."""
    if not telemetry:
        raise ValueError("chorus_score needs at least one chip's telemetry")
    if only is not None:
        if not list(only):
            raise ValueError("only needs at least one device name")
        unknown = [name for name in only if name not in device_names()]
        if unknown:
            raise ValueError(f"unknown device {', '.join(unknown)}; the devices are {', '.join(device_names())}")
    n = len(telemetry)
    n_samples = int(duration * sample_rate)
    if defrag is not None and len(defrag) != n:
        raise ValueError(f"defrag needs one state for each of the {n} chips, got {len(defrag)}")
    planned, turn_s = plan_turns(telemetry, duration, patch, defrag)
    solo_slots = planned if soloists is None else list(soloists)
    seed_base = segment * SEGMENT_SEED_STRIDE
    gains = turn_gains(solo_slots, n_slots(n), n_samples, sample_rate, turn_s, floor=patch.floor)
    events: list = []
    for chip, t in enumerate(telemetry):
        events.extend(control_events(t, chip, duration))
    for s in range(n_slots(n)):
        name = assign_device(s).__name__
        extractor = EXTRACTORS.get(name)
        if extractor is None or (only is not None and name not in only):
            continue
        chip = slot_chip(s, n)
        phase = "running" if defrag is None or only is not None else defrag[chip].phase
        if phase == "idle":
            continue
        balance = 1.0 if only is not None else emphasis_of(patch, name)
        gain = np.ones(n_samples) if only is not None else gains[s]
        keep = np.ones(n_samples)
        if phase != "running":
            rank, count, prime = voice_order(s, n)
            keep = (1.0 - drain_progress(phase, defrag[chip].elapsed, n_samples, sample_rate, rank, count, prime)) ** 2
        for e in extractor(telemetry[chip], chip, duration, sample_rate, seed_base + s, patch):
            i = min(max(int(e.t * sample_rate), 0), n_samples - 1) if n_samples else 0
            if n_samples and keep[i] <= DISSOLVED_KEEP:
                continue  # the voice has dissolved in the audio, so it is silent in MIDI too (other quiet voices keep the 0.05 floor)
            scale = balance * (gain[i] * keep[i] if n_samples else 1.0)
            if isinstance(e, Note):
                events.append(dataclasses.replace(e, velocity=min(1.0, max(MIN_VELOCITY, e.velocity * scale))))
            else:
                events.append(dataclasses.replace(e, strength=min(1.0, max(MIN_VELOCITY, e.strength * scale))))
    return Score(tuple(sorted(events, key=lambda e: e.t)), duration, aiclk_to_bpm(telemetry[0].aiclk))
