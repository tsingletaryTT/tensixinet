# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Ciat-Lonbarde-inspired devices. Each renders one chip's telemetry as sound.

A device is `f(telemetry, duration, sample_rate, seed) -> float32 array` with
|x| <= 1. Output is deterministic for a given seed. Volume (power) is applied
by the chorus, not here.
"""
from __future__ import annotations

import math
import functools
from typing import List, NamedTuple, Optional, Sequence, Tuple

import numpy as np

from .mapping import (
    aiclk_to_bpm,
    chip_pressure,
    ddr_lane_sides,
    drone_voices,
    eth_lane_levels,
    heartbeat_phase,
    norm_temp,
    personality,
    temp_to_pitch_hz,
    voltage_to_cents,
)
from .patch import DEFAULT_PATCH, SCALES, Patch
from .register import step_times
from .ttrax import FALLBACK_ARCCLK, barre_bounds, bounce_wave, ttrax  # noqa: F401  (re-exported: the chorus and the log import them from here)
from .ttquantussy import castle_offset, castle_output, crunch, petal_phases, petal_ratios, rest_offset
from .ttrolzer import (
    BARRE_GEOMETRY, DDR_LANE_BARRES, DOWNBEAT_GAIN, ETH_LANE_BARRES, ETH_LANE_SIDES, SEGMENT_SEED_STRIDE, Link, Roll,
    barre_bpm, draw_gong_trigger, draw_links, roll_drift, roll_ensemble,
)
from .synth import build_voices, render
from ..telemetry import Telemetry

PENTATONIC = SCALES["major_pentatonic"]

# Which Telemetry fields each device reads. The run log's "why" lines are built
# from this table, and tests/test_explain.py proves it matches the audio: change
# a field a device does not list and its output must not move, and vice versa.
# The chorus applies power as a chip's volume and current as its share of the mix, and
# events.py plays the error counts, so those are not listed. A device reads power only to
# work out how close its chip is to a limit (see mapping.pressure).
PRESSURE_FIELDS = frozenset({"asic_temperature", "power", "thm_limit_throttle", "tdp_limit"})
IDENTITY_FIELDS = frozenset({"tensix_mask", "eth_mask"})
DEVICE_INPUTS = {
    "ttrollz": frozenset({"asic_temperature", "aiclk", "ddr_status", "eth_mask"}) | PRESSURE_FIELDS | IDENTITY_FIELDS,
    "ttrax": frozenset({"asic_temperature", "gddr_temps", "arcclk", "aiclk", "temp_rate", "power_rate"}) | PRESSURE_FIELDS,
    "ttquantus": frozenset({"asic_temperature", "aiclk", "axiclk"}) | IDENTITY_FIELDS,
    "ttdrone": frozenset({"asic_temperature", "voltage", "pcie_usage", "aiclk", "heartbeat", "ddr_status"}) | PRESSURE_FIELDS | IDENTITY_FIELDS,
    "lead": frozenset({"aiclk", "gddr_temps"}) | PRESSURE_FIELDS | IDENTITY_FIELDS,
    "gong": frozenset({"asic_temperature", "aiclk", "heartbeat", "ddr_status"}) | PRESSURE_FIELDS | IDENTITY_FIELDS,
    "ttsteam": frozenset({"aiclk", "tensix_mask", "pcie_usage"}) | PRESSURE_FIELDS,
    "ttavdog": frozenset({"asic_temperature", "aiclk", "heartbeat", "ddr_status"}) | PRESSURE_FIELDS | IDENTITY_FIELDS,
    "ttchoir": frozenset({"asic_temperature", "gddr_temps", "aiclk"}) | PRESSURE_FIELDS | IDENTITY_FIELDS,
    "ultrasonic_shimmer": frozenset({"gddr_temps", "aiclk", "heartbeat", "ddr_status"}) | PRESSURE_FIELDS | IDENTITY_FIELDS,
    "ttcoco": frozenset({"aiclk", "axiclk", "gddr_temps"}) | PRESSURE_FIELDS | IDENTITY_FIELDS,
}

# What each steam route (patch.STEAM_ROUTES) makes a device read, on top of DEVICE_INPUTS. All four read the register,
# so all of them read the chip's tempo and its Tensix mask. tests/music/test_steam_routes.py proves each entry.
ROUTE_INPUTS = {
    "strikes": {name: frozenset({"aiclk", "tensix_mask"}) for name in ("gong", "ttdrone", "ultrasonic_shimmer", "ttavdog")},
    "avdog": {"ttavdog": frozenset({"aiclk", "tensix_mask"})},
    "accent": {"ttrollz": frozenset({"aiclk", "tensix_mask"})},
    "pitch": {"ttavdog": frozenset({"aiclk", "tensix_mask"})},
}


# --- ttrollz: organic asymmetric pulses ------------------------------------------

def ttrollz_pulse_times(bpm: float, duration: float, asymmetry: float = 0.25) -> np.ndarray:
    """Pulse onsets (s). Intervals alternate long/short around 60/bpm.

    A long step of base*(1+a) followed by a short one of base*(1-a) keeps the
    mean at the BPM while giving the lopsided feel of a charging capacitor.
    """
    base = 60.0 / bpm
    times, t, i = [], 0.0, 0
    while t < duration:
        times.append(t)
        t += base * (1 + asymmetry if i % 2 == 0 else 1 - asymmetry)
        i += 1
    return np.array(times)


def _blips(times: np.ndarray, pitch: float, n: int, sr: int, decay_s: float = 0.08) -> np.ndarray:
    out = np.zeros(n)
    length = int(decay_s * 6 * sr)
    k = np.arange(length) / sr
    blip = np.sin(2 * math.pi * pitch * k) * np.exp(-k / decay_s)
    for t in times:
        s = int(t * sr)
        if s >= n:
            break
        seg = blip[: n - s]
        out[s : s + len(seg)] += seg
    return out


ACCENT_WINDOW_S = 0.005  # a clock pulse this close to a register hit counts as landing on it
ACCENT_GAIN = 0.7  # the accent adds this much of a clock blip on top of the blip itself


def accented_pulses(clock: np.ndarray, hits: np.ndarray) -> np.ndarray:
    """The clock pulses that land on a register hit, to within ACCENT_WINDOW_S."""
    if len(clock) == 0 or len(hits) == 0:
        return clock[:0]
    nearest = np.minimum(np.searchsorted(hits, clock), len(hits) - 1)
    before = np.maximum(nearest - 1, 0)
    gap = np.minimum(np.abs(hits[nearest] - clock), np.abs(hits[before] - clock))
    return clock[gap <= ACCENT_WINDOW_S]


class Lane(NamedTuple):
    """One roll that ttrollz sounds: its geometry and tempo, and how its blips sound."""

    roll: Roll
    pitch: float  # multiple of the chip's pitch
    level: float
    decay_s: float
    index: int  # which roll of the chip's network this is


class Network(NamedTuple):
    """Every roll a chip runs, and which of them are heard."""

    rolls: List[Roll]
    lanes: List[Lane]


def ttrollz_network(t: Telemetry) -> Network:
    """The chip's bank of rolls, like the Rolzer's. A lane that is up (a DDR channel with a non-zero status, which a
    zero status would ground, or a group of four ethernet links with any link up) brings the rolls of one barre: a
    triangle, a square, a pentagon and two hexagons, the second hexagon pulsing out of phase with the first. A chip with
    everything up runs 40 rolls (the real Rolzer has 35). Only each lane's own roll is heard; the others are not
    sounded, but they are linked into the network and a gong can be patched to any of them.

    A DDR lane brings all five. An ethernet lane brings as many as its share of links up (rounded up), and always
    its own roll, as its barre's rolls come in order of their geometry."""
    bpm = aiclk_to_bpm(t.aiclk)
    rolls: List[Roll] = []
    lanes: List[Lane] = []

    def add_barre(barre: int, own_sides: int, count: int, pitch: float, level: float, decay_s: float) -> None:
        own = BARRE_GEOMETRY.index(own_sides)
        for g in range(max(count, own + 1)):
            rolls.append(Roll(BARRE_GEOMETRY[g], barre_bpm(bpm, barre), phase=0.5 if g == 4 else 0.0))
        start = len(rolls) - max(count, own + 1)
        lanes.append(Lane(rolls[start + own], pitch, level, decay_s, start + own))

    for lane, sides in enumerate(ddr_lane_sides(t.ddr_status)):
        if sides:
            add_barre(DDR_LANE_BARRES[lane], sides, len(BARRE_GEOMETRY), 1.5 + 0.5 * lane, 0.5, 0.05)
    for lane, level in enumerate(eth_lane_levels(t.eth_mask)):
        if level:
            add_barre(ETH_LANE_BARRES[lane], ETH_LANE_SIDES[lane], math.ceil(level * len(BARRE_GEOMETRY)),
                      1.25 + 0.25 * lane, 0.35 * level, 0.04)
    return Network(rolls, lanes)


def ttrollz_lanes(t: Telemetry) -> List[Lane]:
    """The rolls this chip plays aloud, one per lane that is up (see `ttrollz_network`)."""
    return ttrollz_network(t).lanes


@functools.lru_cache(maxsize=32)
def _run_network(rolls: Tuple[Roll, ...], links: Tuple[Link, ...], duration: float, seed: int, drift: float, cap_seed: int):
    return roll_ensemble(list(rolls), list(links), duration, seed, drift=drift, cap_seed=cap_seed)


def roll_pulses(t: Telemetry, duration: float, patch: Patch = DEFAULT_PATCH, segment: int = 0):
    """For each roll of the chip's network, (the times of its pulses, which of them begin a cycle). Every device that
    reads the rolls (ttrollz, and the gong with everything that follows its strikes) calls this with the same chip, patch
    and segment, so they all hear the same pulses; live mode moves `segment` on so the drift and slide differ from one
    segment to the next. The result is cached for the render and must not be changed."""
    net = ttrollz_network(t)
    identity = personality(t)
    links = draw_links(len(net.rolls), patch.seed, identity, chip_pressure(t), [lane.index for lane in net.lanes])
    return _run_network(
        tuple(net.rolls), tuple(links), float(duration), patch.seed * 7919 + segment,
        roll_drift(t.asic_temperature), patch.seed * 1_000_003 + identity,
    )


def gong_trigger_roll(t: Telemetry, patch: Patch = DEFAULT_PATCH) -> Optional[int]:
    """The roll of the chip's network that its gong is patched to, or None when the gong follows the chip's own clock."""
    return draw_gong_trigger(len(ttrollz_network(t).rolls), patch.seed, personality(t))


def ttrollz(t: Telemetry, duration: float, sample_rate: int, seed: int, patch: Patch = DEFAULT_PATCH) -> np.ndarray:
    """The chip's clock, plus one roll per DDR channel and per group of ethernet links.

    The rolls run together as coupled circuits (`ttrolzer.roll_ensemble`): each lane that is up brings a whole barre of
    them (`ttrollz_network`), and only each lane's own roll is heard. The patch and the chip's identity decide which Browns
    are patched to which, so a chip is wired the same way for a whole run; the chip's pressure sets how hard the links
    pull; its temperature sets how fast the capacitors drift; the same two decide the capacitors' mismatch, which stays
    with the chip. The first step of each lane's cycle is accented an octave up, so an odd roll's 3 or 5 can be heard.
    The segment (read out of `seed`, as the gong reads it) sets the drift, so the gong follows the rolls that are heard.

    With the patch's `accent` route on, the clock pulses that land on a hit of the ttsteam register sound louder.

    Every lane is a roll on its own ttrolzer barre (see ttrolzer.py), so the lanes drift against each other
    and against the bar and never lock into a polyrhythm."""
    n = int(duration * sample_rate)
    bpm = aiclk_to_bpm(t.aiclk)
    pitch = temp_to_pitch_hz(t.asic_temperature)
    clock = ttrollz_pulse_times(bpm, duration)
    out = _blips(clock, pitch, n, sample_rate)
    if "accent" in patch.steam_routes:
        out += ACCENT_GAIN * _blips(accented_pulses(clock, step_times(t, duration)), pitch, n, sample_rate)
    pulses = roll_pulses(t, duration, patch, seed // SEGMENT_SEED_STRIDE)
    for lane in ttrollz_lanes(t):
        times, downbeats = pulses[lane.index]
        out += lane.level * _blips(times, pitch * lane.pitch, n, sample_rate, lane.decay_s)
        out += DOWNBEAT_GAIN * lane.level * _blips(times[downbeats], pitch * lane.pitch * 2, n, sample_rate, lane.decay_s)
    return (np.tanh(out)).astype(np.float32)


# --- ttrax: seven barres bouncing between each other's oscillators (see ttrax.py) ---------------


# --- ttquantus: the ttquantussy's castles, through an 8-bit buffer ---------------------------------------

def quantize_to_scale(values: np.ndarray, scale: Sequence[int], octaves: int = 2) -> np.ndarray:
    """Map [0, 1] values onto semitone offsets of `scale` across `octaves`."""
    notes = [12 * o + d for o in range(octaves) for d in scale]
    idx = np.minimum((np.asarray(values) * len(notes)).astype(int), len(notes) - 1)
    return np.array(notes)[idx]


REFERENCE_AXICLK = 960.0  # MHz: at this AXI clock a note is held for an eighth
CRUNCH_FULL_AXICLK = 1920.0  # MHz: at this AXI clock the buffer runs at the audio sample rate
CRUNCH_MIN_RATE = 0.25  # the buffer clock never drops below this share of the sample rate


def ttquantus_steps(t: Telemetry, duration: float, sample_rate: int, seed: int, patch: Patch = DEFAULT_PATCH) -> np.ndarray:
    """Per-sample semitone offsets: the first petal's castle on the patch's scale.

    The ttquantussy's first petal turns round twice per eighth note, so a note lasts an eighth. The other four
    petals run at low prime ratios to it (the seed nudges them), so the notes make a pattern that repeats
    and slowly drifts. The chip's identity picks which petal the castle samples."""
    n = int(duration * sample_rate)
    rate = aiclk_to_bpm(t.aiclk) / 60.0 * 2 * t.axiclk / REFERENCE_AXICLK
    freqs = [rate / 2 * r for r in petal_ratios(seed)]
    castle = castle_output(0, freqs, petal_phases(seed), np.arange(n) / sample_rate, offset=castle_offset(personality(t)))
    return quantize_to_scale(castle, patch.scale)


REST_FADE_S = 0.005  # a note's edges are glided over this long, so a rest does not click


def ttquantus_gate(t: Telemetry, duration: float, sample_rate: int, seed: int, patch: Patch = DEFAULT_PATCH) -> np.ndarray:
    """1 where a held note plays and 0 where it rests. Sample-and-hold also makes silences: a second castle of the first petal, looking at a
    different petal than the note castle does, is held for each note, and a note whose value falls below the patch's `rests` does not play.
    The values are spread evenly from 0 to 1, so about `rests` of the notes rest, in a pattern that repeats and drifts as the notes do."""
    n = int(duration * sample_rate)
    if patch.rests <= 0.0:
        return np.ones(n)
    rate = aiclk_to_bpm(t.aiclk) / 60.0 * 2 * t.axiclk / REFERENCE_AXICLK
    freqs = [rate / 2 * r for r in petal_ratios(seed)]
    castle = castle_output(0, freqs, petal_phases(seed), np.arange(n) / sample_rate, offset=rest_offset(personality(t)))
    gate = (castle >= patch.rests).astype(float)
    width = max(1, int(REST_FADE_S * sample_rate))
    if width == 1:
        return gate
    c = np.concatenate([[0.0], np.cumsum(gate)])
    hi, lo = np.minimum(np.arange(n) + width // 2 + 1, n), np.maximum(np.arange(n) - width // 2, 0)
    return (c[hi] - c[lo]) / (hi - lo)


def quantussy_tone(t: Telemetry, duration: float, sample_rate: int, seed: int, patch: Patch = DEFAULT_PATCH) -> np.ndarray:
    """The castle's notes as a plain sine (with the rests the patch asks for), before the coco's 8-bit buffer. ttcoco patches this same tone into its
    delay taps, as the real instrument's Quantussy patches into its Coco."""
    semis = ttquantus_steps(t, duration, sample_rate, seed, patch)
    freq = temp_to_pitch_hz(t.asic_temperature) * 2 ** (semis / 12)
    phase = 2 * math.pi * np.cumsum(freq) / sample_rate
    return 0.8 * np.sin(phase) * ttquantus_gate(t, duration, sample_rate, seed, patch)


def ttquantus(t: Telemetry, duration: float, sample_rate: int, seed: int, patch: Patch = DEFAULT_PATCH) -> np.ndarray:
    """The castle's notes as a sine, through the coco's 8-bit buffer. A slower AXI clock slows the buffer's
    sample clock, which holds each sample longer and aliases."""
    rate = min(1.0, max(CRUNCH_MIN_RATE, t.axiclk / CRUNCH_FULL_AXICLK))
    return crunch(quantussy_tone(t, duration, sample_rate, seed, patch), rate).astype(np.float32)


# --- Plumbutter: drone + current-based mixing ----------------------------------

DRONE_SETTLED = 0.6  # the drone's level when no shock wave is passing; each strike swells it to full


def ttdrone(t: Telemetry, duration: float, sample_rate: int, seed: int, patch: Patch = DEFAULT_PATCH) -> np.ndarray:
    """A chord drone whose level ripples after every gong strike, as an AVDog rings in sympathy with a Gongue."""
    since, rate_hz, decay_s = drone_ripple_inputs(t, duration, sample_rate, patch, seed // SEGMENT_SEED_STRIDE)
    return drone_with_ripple(drone_chord(t, duration, sample_rate), shock_wave(since, rate_hz, decay_s))


def drone_chord(t: Telemetry, duration: float, sample_rate: int) -> np.ndarray:
    """The drone's steady chord, before the ripple."""
    f_low = temp_to_pitch_hz(t.asic_temperature) / 2 * 2 ** (voltage_to_cents(t.voltage) / 1200)
    voices = build_voices(1, voices_per_chip=drone_voices(t.pcie_usage), f_low=f_low)
    return render(voices, [t], duration, sample_rate)


def drone_ripple_inputs(t: Telemetry, duration: float, sample_rate: int, patch: Patch, segment: int):
    """(seconds since the latest gong strike for each sample, the ripple's rate in Hz, its decay time in s)."""
    beat = 60.0 / aiclk_to_bpm(t.aiclk)
    since = seconds_since(strike_times(t, duration, patch, segment), int(duration * sample_rate), sample_rate)
    return since, SHOCK_RATE_BEATS / beat, SHOCK_DECAY_BEATS * beat


def drone_with_ripple(chord: np.ndarray, ripple: np.ndarray) -> np.ndarray:
    """The chord's level follows the ripple, settling to DRONE_SETTLED between strikes."""
    return (chord * (DRONE_SETTLED + (1 - DRONE_SETTLED) * ripple)).astype(np.float32)


# --- Gong: inharmonic, long-ringing strikes ---------------------------------------

# Partial ratios of a struck metal bar or bell. None after the first is a whole
# multiple, which is what makes it sound like a gong or gamelan and not a note.
# The default tuning lives in the default patch; a patch brings its own partials, amplitudes and beat.
GONG_PARTIALS = DEFAULT_PATCH.gong_partials
GONG_AMPS = DEFAULT_PATCH.gong_amps
GONG_BEAT_HZ = DEFAULT_PATCH.gong_beat_hz  # each partial is doubled and detuned, like a pair of gamelan gongs


def gong_detune_cents(identity: int) -> int:
    """-10 to +10 cents from a chip's identity, so two gongs on different chips beat against each other."""
    return (identity >> 8) % 21 - 10


TRIGGER_RESPONSE_STEPS = 7  # a chip's identity picks one of seven windows
TRIGGER_RESPONSE_SPREAD = 1 / 16  # each step lengthens the window by this share of a strike interval


def gate_triggers(pulses: np.ndarray, window: float, start: float) -> np.ndarray:
    """The Gongue's trigger response: a strike on the first pulse at or after `start`, then a blocking
    window of `window` seconds in which pulses are ignored, then a strike on the next pulse.

    The window is a time, so a faster clock is divided by more (Plumbutter guide: 100 bpm halves, 250 bpm
    divides by five, with the knob unchanged)."""
    strikes, earliest = [], start
    for pulse in pulses:
        if pulse >= earliest:
            strikes.append(pulse)
            earliest = pulse + window
    return np.array(strikes)


def strike_times(t: Telemetry, duration: float, patch: Patch = DEFAULT_PATCH, segment: int = 0) -> np.ndarray:
    """When this chip's gong is struck. The same moments set off the shimmer's snap and the drone's shock wave,
    as a Plumbutter patch sends one clock to the Ultrasound, the Gongue and the AVDog together.

    The trigger is one of the chip's rolls (the Gongue is patched to a roll's orange), or the chip's own ttrollz clock;
    `gong_trigger_roll` says which, and `segment` says which stretch of the rolls' wander (see `roll_pulses`). Two bars is the shortest gap, and the chip's identity
    lengthens the blocking window by up to six sixteenths of that, so chips divide the clock differently.
    The heartbeat sets how far into the first window the chip starts.

    With the patch's `strikes` route on, the hits of the chip's ttsteam register join the clock pulses, so a strike can
    land on a snare hit as well as on a clock pulse; the blocking window still spaces the strikes."""
    bpm = aiclk_to_bpm(t.aiclk)
    interval = 2 * 4 * 60.0 / bpm
    window = interval * (1 + TRIGGER_RESPONSE_SPREAD * (personality(t) % TRIGGER_RESPONSE_STEPS))
    first = heartbeat_phase(t.heartbeat) * interval / 2
    which = gong_trigger_roll(t, patch)
    pulses = ttrollz_pulse_times(bpm, duration) if which is None else roll_pulses(t, duration, patch, segment)[which][0]
    if "strikes" in patch.steam_routes:
        pulses = np.sort(np.concatenate([pulses, step_times(t, duration)]))
    return gate_triggers(pulses, window, first)


def seconds_since(strikes: np.ndarray, n: int, sample_rate: int) -> np.ndarray:
    """For each of `n` samples, the seconds since the latest strike; infinity before the first."""
    if len(strikes) == 0:
        return np.full(n, np.inf)
    t = np.arange(n) / sample_rate
    idx = np.searchsorted(strikes, t, side="right") - 1
    return np.where(idx >= 0, t - strikes[np.maximum(idx, 0)], np.inf)


SHOCK_RATE_BEATS = 1.5  # undulations per beat
SHOCK_DECAY_BEATS = 2.0  # the ripple dies to a third in this many beats


def shock_wave(since: np.ndarray, rate_hz: float, decay_s: float) -> np.ndarray:
    """An AVDog's undulation after a trigger: a damped swing that changes direction every half cycle.

    The AVDog sends its white voice out on one half and its red voice on the other, so in mono the
    level follows the rectified swing, 0 to 1, at twice the undulation rate."""
    live = np.isfinite(since)
    tau = np.where(live, since, 0.0)
    return np.where(live, np.abs(np.sin(2 * math.pi * rate_hz * tau)) * np.exp(-tau / decay_s), 0.0)


class GongStrike(NamedTuple):
    """Everything a strike is built from. The kernel's host side reads the same numbers."""

    f0: float  # Hz, the fundamental
    interval: float  # s, the shortest gap between strikes
    length: int  # samples a strike rings for
    partials: List[Tuple[float, float, float]]  # (frequency in Hz held below Nyquist, decay time in s, amplitude)
    beat_hz: float  # each partial is doubled and the copy detuned by this


GONG_SAG_DEPTH = 0.004  # the strike's pitch sags by this share as it rings
GONG_SAG_S = 0.4  # with this time constant
GONG_ATTACK_S = 0.005


def gong_parameters(t: Telemetry, sample_rate: int, patch: Patch = DEFAULT_PATCH) -> GongStrike:
    bpm = aiclk_to_bpm(t.aiclk)
    f0 = temp_to_pitch_hz(t.asic_temperature) / 2 * 2 ** (gong_detune_cents(personality(t)) / 1200)
    interval = 2 * 4 * 60.0 / bpm
    partials = [
        (min(f0 * ratio, 0.45 * sample_rate), 0.3 * interval / (1 + 0.25 * (ratio - 1)), amp)
        for ratio, amp in zip(patch.gong_partials, patch.gong_amps)
    ]
    return GongStrike(f0, interval, int(2.5 * interval * sample_rate), partials, patch.gong_beat_hz)


def gong(t: Telemetry, duration: float, sample_rate: int, seed: int, patch: Patch = DEFAULT_PATCH) -> np.ndarray:
    """A strike whenever the chip's clock gets through the trigger response, an octave below the chip's pitch.

    The ring lasts about a third of the shortest strike interval, so a faster clock rings shorter.
    Higher partials die faster, and the strike's pitch sags slightly as it rings.
    """
    n = int(duration * sample_rate)
    g = gong_parameters(t, sample_rate, patch)
    out = np.zeros(n)
    length = g.length
    k = np.arange(length) / sample_rate
    sag = 1 - GONG_SAG_DEPTH * (1 - np.exp(-k / GONG_SAG_S))
    strike = np.zeros(length)
    for base, decay, amp in g.partials:
        for detune in (0.0, g.beat_hz):
            phase = 2 * math.pi * np.cumsum((base + detune) * sag) / sample_rate
            strike += amp * 0.5 * np.sin(phase) * np.exp(-k / decay)
    strike *= np.minimum(1.0, k / GONG_ATTACK_S)  # a 5 ms attack avoids a click
    for onset in strike_times(t, duration, patch, seed // SEGMENT_SEED_STRIDE):
        s = int(onset * sample_rate)
        seg = strike[: n - s]
        out[s : s + len(seg)] += seg
    return (0.6 * np.tanh(out)).astype(np.float32)


# --- Ultrasonic shimmer: a beating pair at the top of the spectrum ----------------

SHIMMER_BAND_HZ = DEFAULT_PATCH.shimmer_band  # a patch brings its own band
SHIMMER_BAND_OF_SAMPLE_RATE = (0.27, 0.45)  # at low sample rates the band shrinks to fit under Nyquist
SHIMMER_BEAT_HZ_PER_DEG = 0.2  # beat rate per degree C of GDDR spread
SNAP_DECAY_S = 0.012  # the Ultrasound's transient on a strike: immediate and short
SNAP_DEPTH = 0.3  # share of the level that the snap adds on top of the steady pair


def shimmer_parameters(t: Telemetry, sample_rate: int, patch: Patch = DEFAULT_PATCH) -> Tuple[float, float, float]:
    """(the shimmer's two tone frequencies in Hz, its level). The kernel's host side reads the same numbers."""
    lo = min(patch.shimmer_band[0], SHIMMER_BAND_OF_SAMPLE_RATE[0] * sample_rate)
    hi = min(patch.shimmer_band[1], SHIMMER_BAND_OF_SAMPLE_RATE[1] * sample_rate)
    f1 = lo + (hi - lo) * norm_temp(t.asic_temperature)
    beat = SHIMMER_BEAT_HZ_PER_DEG * (max(t.gddr_temps) - min(t.gddr_temps))
    return f1, f1 + beat, 0.5 * (1 + chip_pressure(t))  # pressure doubles the level, so at most 1


def ultrasonic_shimmer(t: Telemetry, duration: float, sample_rate: int, seed: int, patch: Patch = DEFAULT_PATCH) -> np.ndarray:
    """Two close high tones. A hotter chip shimmers higher; a wider GDDR spread beats faster.

    Equal GDDR temperatures give one steady tone. The band is 12-20 kHz and shrinks at
    low sample rates so it stays under 0.45 of the sample rate and never aliases.
    """
    n = int(duration * sample_rate)
    f1, f2, level = shimmer_parameters(t, sample_rate, patch)
    k = np.arange(n) / sample_rate
    out = 0.5 * (np.sin(2 * math.pi * f1 * k) + np.sin(2 * math.pi * f2 * k))
    snap = np.exp(-seconds_since(strike_times(t, duration, patch, seed // SEGMENT_SEED_STRIDE), n, sample_rate) / SNAP_DECAY_S)  # 1 on a strike, 0 long after it
    out *= 1 - SNAP_DEPTH + SNAP_DEPTH * snap
    return (level * out).astype(np.float32)


def current_mix(signals: Sequence[np.ndarray], currents: Sequence[float]) -> np.ndarray:
    """Mix sources the way patched banana plugs do: by current, not voltage.

    Each source's weight is its share of the total current (scaled so the
    average weight is 1), and the sum passes through a saturating stage, so
    piling on sources compresses instead of clipping.
    """
    cur = np.maximum(np.asarray(currents, dtype=float), 0.0)
    total = cur.sum()
    weights = np.ones(len(cur)) if total == 0 else cur / total * len(cur)
    mixed = sum(w * np.asarray(s, dtype=float) for w, s in zip(weights, signals))
    return np.tanh(mixed / math.sqrt(len(cur)))
