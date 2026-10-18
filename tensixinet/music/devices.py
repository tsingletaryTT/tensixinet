"""Ciat-Lonbarde-inspired devices. Each renders one chip's telemetry as sound.

A device is `f(telemetry, duration, sample_rate, seed) -> float32 array` with
|x| <= 1. Output is deterministic for a given seed. Volume (power) is applied
by the chorus, not here.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from .mapping import (
    aiclk_to_bpm,
    ddr_lane_divisions,
    norm_temp,
    temp_to_pitch_hz,
)
from .patch import DEFAULT_PATCH, SCALES, Patch
from .synth import build_voices, render
from ..telemetry import Telemetry

PENTATONIC = SCALES["major_pentatonic"]

# Which Telemetry fields each device reads. The run log's "why" lines are built
# from this table, and tests/test_explain.py proves it matches the audio: change
# a field a device does not list and its output must not move, and vice versa.
# power and current are applied by the chorus, not by devices.
DEVICE_INPUTS = {
    "rollz": frozenset({"asic_temperature", "aiclk", "ddr_status"}),
    "sidrax": frozenset({"asic_temperature", "gddr_temps"}),
    "cocoquantus": frozenset({"asic_temperature", "aiclk"}),
    "plumbutter_drone": frozenset({"asic_temperature"}),
    "lead": frozenset({"asic_temperature", "aiclk", "gddr_temps"}),
    "gong": frozenset({"asic_temperature", "aiclk"}),
    "ultrasonic_shimmer": frozenset({"asic_temperature", "gddr_temps"}),
}


# --- Rollz: organic asymmetric pulses ------------------------------------------

def rollz_pulse_times(bpm: float, duration: float, asymmetry: float = 0.25) -> np.ndarray:
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


def rollz(t: Telemetry, duration: float, sample_rate: int, seed: int, patch: Patch = DEFAULT_PATCH) -> np.ndarray:
    n = int(duration * sample_rate)
    bpm = aiclk_to_bpm(t.aiclk)
    pitch = temp_to_pitch_hz(t.asic_temperature)
    out = _blips(rollz_pulse_times(bpm, duration), pitch, n, sample_rate)
    bar = 4 * 60.0 / bpm
    for lane, div in enumerate(ddr_lane_divisions(t.ddr_status)):
        if div:
            times = np.arange(0, duration, bar / div)
            out += 0.5 * _blips(times, pitch * (1.5 + 0.5 * lane), n, sample_rate, 0.05)
    return (np.tanh(out)).astype(np.float32)


# --- Sidrax: oscillator bouncing between bounds --------------------------------

def bounce_wave(freq_hz: float, lower: float, upper: float, n: int, sample_rate: int) -> np.ndarray:
    """Triangle wave that reflects between `lower` and `upper`."""
    ph = 2 * math.pi * freq_hz * np.arange(n) / sample_rate
    unit = (2 / math.pi) * np.arcsin(np.sin(ph))  # -1..1
    return lower + (unit + 1) / 2 * (upper - lower)


def sidrax_bounds(gddr_temps: Sequence[float]) -> tuple:
    """(lower, upper) from the GDDR temperatures: the coolest channel sets the
    lower bound and the hottest the upper, so a lopsided memory system swings
    asymmetrically."""
    return (
        -(0.2 + 0.7 * norm_temp(min(gddr_temps))),
        0.2 + 0.7 * norm_temp(max(gddr_temps)),
    )


def sidrax(t: Telemetry, duration: float, sample_rate: int, seed: int, patch: Patch = DEFAULT_PATCH) -> np.ndarray:
    n = int(duration * sample_rate)
    lo, hi = sidrax_bounds(t.gddr_temps)
    return bounce_wave(temp_to_pitch_hz(t.asic_temperature), lo, hi, n, sample_rate).astype(np.float32)


# --- Cocoquantus: sample and hold ----------------------------------------------

def sample_hold(n: int, sample_rate: int, rate_hz: float, rng: np.random.Generator) -> np.ndarray:
    """Random values in [0, 1), each held for 1/rate_hz seconds."""
    steps = int(math.ceil(n * rate_hz / sample_rate)) or 1
    values = rng.random(steps)
    idx = np.minimum((np.arange(n) * rate_hz / sample_rate).astype(int), steps - 1)
    return values[idx]


def quantize_to_scale(values: np.ndarray, scale: Sequence[int], octaves: int = 2) -> np.ndarray:
    """Map [0, 1) values onto semitone offsets of `scale` across `octaves`."""
    notes = [12 * o + d for o in range(octaves) for d in scale]
    idx = np.minimum((np.asarray(values) * len(notes)).astype(int), len(notes) - 1)
    return np.array(notes)[idx]


def cocoquantus_steps(t: Telemetry, duration: float, sample_rate: int, seed: int, patch: Patch = DEFAULT_PATCH) -> np.ndarray:
    """Per-sample semitone offsets: one held note of the patch's scale per eighth note."""
    n = int(duration * sample_rate)
    rate = aiclk_to_bpm(t.aiclk) / 60.0 * 2
    return quantize_to_scale(sample_hold(n, sample_rate, rate, np.random.default_rng(seed)), patch.scale)


def cocoquantus(t: Telemetry, duration: float, sample_rate: int, seed: int, patch: Patch = DEFAULT_PATCH) -> np.ndarray:
    semis = cocoquantus_steps(t, duration, sample_rate, seed, patch)
    freq = temp_to_pitch_hz(t.asic_temperature) * 2 ** (semis / 12)
    phase = 2 * math.pi * np.cumsum(freq) / sample_rate
    return (0.8 * np.sin(phase)).astype(np.float32)


# --- Plumbutter: drone + current-based mixing ----------------------------------

def plumbutter_drone(t: Telemetry, duration: float, sample_rate: int, seed: int, patch: Patch = DEFAULT_PATCH) -> np.ndarray:
    voices = build_voices(1, voices_per_chip=3, f_low=temp_to_pitch_hz(t.asic_temperature) / 2)
    return render(voices, [t], duration, sample_rate)


# --- Gong: inharmonic, long-ringing strikes ---------------------------------------

# Partial ratios of a struck metal bar or bell. None after the first is a whole
# multiple, which is what makes it sound like a gong or gamelan and not a note.
# The default tuning lives in the default patch; a patch brings its own partials, amplitudes and beat.
GONG_PARTIALS = DEFAULT_PATCH.gong_partials
GONG_AMPS = DEFAULT_PATCH.gong_amps
GONG_BEAT_HZ = DEFAULT_PATCH.gong_beat_hz  # each partial is doubled and detuned, like a pair of gamelan gongs


def gong(t: Telemetry, duration: float, sample_rate: int, seed: int, patch: Patch = DEFAULT_PATCH) -> np.ndarray:
    """One strike every two bars at the aiclk tempo, an octave below the chip's pitch.

    The ring lasts about a third of the strike interval, so a faster clock rings shorter.
    Higher partials die faster, and the strike's pitch sags slightly as it rings.
    """
    n = int(duration * sample_rate)
    bpm = aiclk_to_bpm(t.aiclk)
    f0 = temp_to_pitch_hz(t.asic_temperature) / 2
    interval = 2 * 4 * 60.0 / bpm
    ring_s = 2.5 * interval
    out = np.zeros(n)
    length = int(ring_s * sample_rate)
    k = np.arange(length) / sample_rate
    sag = 1 - 0.004 * (1 - np.exp(-k / 0.4))
    strike = np.zeros(length)
    for ratio, amp in zip(patch.gong_partials, patch.gong_amps):
        decay = 0.3 * interval / (1 + 0.25 * (ratio - 1))
        for detune in (0.0, patch.gong_beat_hz):
            freq = min(f0 * ratio, 0.45 * sample_rate) + detune
            phase = 2 * math.pi * np.cumsum(freq * sag) / sample_rate
            strike += amp * 0.5 * np.sin(phase) * np.exp(-k / decay)
    strike *= np.minimum(1.0, k / 0.005)  # a 5 ms attack avoids a click
    for onset in np.arange(0, duration, interval):
        s = int(onset * sample_rate)
        seg = strike[: n - s]
        out[s : s + len(seg)] += seg
    return (0.6 * np.tanh(out)).astype(np.float32)


# --- Ultrasonic shimmer: a beating pair at the top of the spectrum ----------------

SHIMMER_BAND_HZ = DEFAULT_PATCH.shimmer_band  # a patch brings its own band
SHIMMER_BAND_OF_SAMPLE_RATE = (0.27, 0.45)  # at low sample rates the band shrinks to fit under Nyquist
SHIMMER_BEAT_HZ_PER_DEG = 0.2  # beat rate per degree C of GDDR spread


def ultrasonic_shimmer(t: Telemetry, duration: float, sample_rate: int, seed: int, patch: Patch = DEFAULT_PATCH) -> np.ndarray:
    """Two close high tones. A hotter chip shimmers higher; a wider GDDR spread beats faster.

    Equal GDDR temperatures give one steady tone. The band is 12-20 kHz and shrinks at
    low sample rates so it stays under 0.45 of the sample rate and never aliases.
    """
    n = int(duration * sample_rate)
    lo = min(patch.shimmer_band[0], SHIMMER_BAND_OF_SAMPLE_RATE[0] * sample_rate)
    hi = min(patch.shimmer_band[1], SHIMMER_BAND_OF_SAMPLE_RATE[1] * sample_rate)
    f1 = lo + (hi - lo) * norm_temp(t.asic_temperature)
    beat = SHIMMER_BEAT_HZ_PER_DEG * (max(t.gddr_temps) - min(t.gddr_temps))
    f2 = f1 + beat
    k = np.arange(n) / sample_rate
    out = 0.5 * (np.sin(2 * math.pi * f1 * k) + np.sin(2 * math.pi * f2 * k))
    return (0.5 * out).astype(np.float32)


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
