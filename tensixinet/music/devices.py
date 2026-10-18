# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Ciat-Lonbarde-inspired devices. Each renders one chip's telemetry as sound.

A device is `f(telemetry, duration, sample_rate, seed) -> float32 array` with
|x| <= 1. Output is deterministic for a given seed. Volume (power) is applied
by the chorus, not here.
"""
from __future__ import annotations

import math
from typing import List, NamedTuple, Sequence, Tuple

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
from .ttrax import FALLBACK_ARCCLK, barre_bounds, bounce_wave, ttrax  # noqa: F401  (re-exported: the chorus and the log import them from here)
from .ttquantussy import castle_offset, castle_output, crunch, petal_phases, petal_ratios
from .ttrolzer import DDR_LANE_BARRES, ETH_LANE_BARRES, ETH_LANE_SIDES, barre_bpm, roll_pulse_times
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
    "ttrollz": frozenset({"asic_temperature", "aiclk", "ddr_status", "eth_mask"}),
    "ttrax": frozenset({"asic_temperature", "gddr_temps", "arcclk", "aiclk", "temp_rate", "power_rate"}) | PRESSURE_FIELDS,
    "ttquantus": frozenset({"asic_temperature", "aiclk", "axiclk"}) | IDENTITY_FIELDS,
    "ttdrone": frozenset({"asic_temperature", "voltage", "pcie_usage", "aiclk", "heartbeat"}) | IDENTITY_FIELDS,
    "lead": frozenset({"aiclk", "gddr_temps"}) | PRESSURE_FIELDS | IDENTITY_FIELDS,
    "gong": frozenset({"asic_temperature", "aiclk", "heartbeat"}) | IDENTITY_FIELDS,
    "ttsteam": frozenset({"aiclk", "tensix_mask", "pcie_usage"}) | PRESSURE_FIELDS,
    "ttavdog": frozenset({"asic_temperature", "aiclk", "heartbeat"}) | PRESSURE_FIELDS | IDENTITY_FIELDS,
    "ttchoir": frozenset({"asic_temperature", "gddr_temps", "aiclk"}) | PRESSURE_FIELDS | IDENTITY_FIELDS,
    "ultrasonic_shimmer": frozenset({"gddr_temps", "aiclk", "heartbeat"}) | PRESSURE_FIELDS | IDENTITY_FIELDS,
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


def ttrollz(t: Telemetry, duration: float, sample_rate: int, seed: int, patch: Patch = DEFAULT_PATCH) -> np.ndarray:
    """The chip's clock, plus one roll per DDR channel and per group of ethernet links.

    Every lane is a roll on its own ttrolzer barre (see ttrolzer.py), so the lanes drift against each other
    and against the bar and never lock into a polyrhythm."""
    n = int(duration * sample_rate)
    bpm = aiclk_to_bpm(t.aiclk)
    pitch = temp_to_pitch_hz(t.asic_temperature)
    out = _blips(ttrollz_pulse_times(bpm, duration), pitch, n, sample_rate)
    for lane, sides in enumerate(ddr_lane_sides(t.ddr_status)):
        if sides:
            times = roll_pulse_times(sides, barre_bpm(bpm, DDR_LANE_BARRES[lane]), duration, seed * 8 + lane)
            out += 0.5 * _blips(times, pitch * (1.5 + 0.5 * lane), n, sample_rate, 0.05)
    # Ethernet lanes: four more, one per group of four links, as loud as the share of that group that is up.
    for lane, level in enumerate(eth_lane_levels(t.eth_mask)):
        if level:
            times = roll_pulse_times(ETH_LANE_SIDES[lane], barre_bpm(bpm, ETH_LANE_BARRES[lane]), duration, seed * 8 + 4 + lane)
            out += 0.35 * level * _blips(times, pitch * (1.25 + 0.25 * lane), n, sample_rate, 0.04)
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


def ttquantus(t: Telemetry, duration: float, sample_rate: int, seed: int, patch: Patch = DEFAULT_PATCH) -> np.ndarray:
    """The castle's notes as a sine, through the coco's 8-bit buffer. A slower AXI clock slows the buffer's
    sample clock, which holds each sample longer and aliases."""
    semis = ttquantus_steps(t, duration, sample_rate, seed, patch)
    freq = temp_to_pitch_hz(t.asic_temperature) * 2 ** (semis / 12)
    phase = 2 * math.pi * np.cumsum(freq) / sample_rate
    rate = min(1.0, max(CRUNCH_MIN_RATE, t.axiclk / CRUNCH_FULL_AXICLK))
    return crunch(0.8 * np.sin(phase), rate).astype(np.float32)


# --- Plumbutter: drone + current-based mixing ----------------------------------

DRONE_SETTLED = 0.6  # the drone's level when no shock wave is passing; each strike swells it to full


def ttdrone(t: Telemetry, duration: float, sample_rate: int, seed: int, patch: Patch = DEFAULT_PATCH) -> np.ndarray:
    """A chord drone whose level ripples after every gong strike, as an AVDog rings in sympathy with a Gongue."""
    f_low = temp_to_pitch_hz(t.asic_temperature) / 2 * 2 ** (voltage_to_cents(t.voltage) / 1200)
    voices = build_voices(1, voices_per_chip=drone_voices(t.pcie_usage), f_low=f_low)
    beat = 60.0 / aiclk_to_bpm(t.aiclk)
    since = seconds_since(strike_times(t, duration), int(duration * sample_rate), sample_rate)
    ripple = shock_wave(since, SHOCK_RATE_BEATS / beat, SHOCK_DECAY_BEATS * beat)
    return (render(voices, [t], duration, sample_rate) * (DRONE_SETTLED + (1 - DRONE_SETTLED) * ripple)).astype(np.float32)


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


def strike_times(t: Telemetry, duration: float) -> np.ndarray:
    """When this chip's gong is struck. The same moments set off the shimmer's snap and the drone's shock wave,
    as a Plumbutter patch sends one clock to the Ultrasound, the Gongue and the AVDog together.

    The clock is the chip's ttrollz pulse train. Two bars is the shortest gap, and the chip's identity
    lengthens the blocking window by up to six sixteenths of that, so chips divide the clock differently.
    The heartbeat sets how far into the first window the chip starts."""
    bpm = aiclk_to_bpm(t.aiclk)
    interval = 2 * 4 * 60.0 / bpm
    window = interval * (1 + TRIGGER_RESPONSE_SPREAD * (personality(t) % TRIGGER_RESPONSE_STEPS))
    first = heartbeat_phase(t.heartbeat) * interval / 2
    return gate_triggers(ttrollz_pulse_times(bpm, duration), window, first)


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
    for onset in strike_times(t, duration):
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
    snap = np.exp(-seconds_since(strike_times(t, duration), n, sample_rate) / SNAP_DECAY_S)  # 1 on a strike, 0 long after it
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
