# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""ttavdog, after the AVDog in the Plumbutter guide: a droning oscillator that a ringing filter opens.

The AVDog has a reference oscillator, the dogtone, that drones all the time and is silent because a VCA holds it shut.
A trigger pings a slow, resonant filter (0.4 to 12 Hz) and the filter rings: an undulation wave that swings one way, then
the other, and dies away. On every change of direction the audio switches channel. The envelope of the undulation opens
the VCA, so the white voice sounds on one half of the swing and the red voice answers on the other half, at the same
pitch with the opposite slope: a call and response. In the guide's figures the red out is the second trace.

Here the trigger is the chip's gong strikes (the same ones that set off the shimmer's snap and the ttdrone's ripple), the
undulation rate and decay are the ttdrone's, the dogtone is at the chip's pitch, and pressure skews its wave from a
triangle toward a saw. Output is mono, so the two voices are added, the red a little quieter as it is in the guide's mixer.
The stereo panning, the FM and wave-shape inputs, the Trad/Gonz switch and the AVDog's audio line input are not modelled.
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from .devices import SHOCK_DECAY_BEATS, SHOCK_RATE_BEATS, seconds_since, strike_times
from .mapping import aiclk_to_bpm, chip_pressure, temp_to_pitch_hz
from .patch import DEFAULT_PATCH, Patch
from .register import held_semitones, step_times
from .ttrolzer import SEGMENT_SEED_STRIDE
from ..telemetry import Telemetry

AVDOG_LEVEL = 0.5
RED_GAIN = 0.8  # the guide's mixer has the red channel a little quieter than the white
SHAPE_LIMIT = 0.98  # a saw never quite closes: its rise or fall is at least 1% of a cycle


def undulation(since: np.ndarray, rate_hz: float, decay_s: float) -> np.ndarray:
    """The ringing filter's wave `tau` seconds after the latest strike: a damped sine that swings up first and then
    reverses every half cycle. `since` is infinite before the first strike, where the filter has not been pinged."""
    live = np.isfinite(since)
    tau = np.where(live, since, 0.0)
    return np.where(live, np.sin(2 * math.pi * rate_hz * tau) * np.exp(-tau / decay_s), 0.0)


def skewed_triangle(phase: np.ndarray, shape: float) -> np.ndarray:
    """The dogtone: a triangle at shape 0 that skews toward a saw. Shape +1 rises through most of the cycle and falls fast
    (an up-sloped saw), -1 is the same wave run backwards (a down-sloped saw). `phase` is in cycles."""
    peak = (1 + float(np.clip(shape, -SHAPE_LIMIT, SHAPE_LIMIT))) / 2  # where in the cycle the wave is highest
    x = np.mod(phase, 1.0)
    return 2 * np.minimum(x / peak, (1 - x) / (1 - peak)) - 1


def avdog_parameters(t: Telemetry) -> Tuple[float, float, float, float]:
    """(the dogtone's pitch in Hz, its wave shape, the undulation's rate in Hz, its decay time in s). The kernel's host side
    reads the same numbers."""
    beat = 60.0 / aiclk_to_bpm(t.aiclk)
    # the shape is a triangle at rest and skews toward a saw near a limit
    return temp_to_pitch_hz(t.asic_temperature), chip_pressure(t), SHOCK_RATE_BEATS / beat, SHOCK_DECAY_BEATS * beat


def avdog_pings(t: Telemetry, duration: float, patch: Patch = DEFAULT_PATCH, segment: int = 0) -> np.ndarray:
    """When the AVDog's filter is pinged: every gong strike, and with the patch's `avdog` route every hit of the chip's
    ttsteam register as well."""
    pings = strike_times(t, duration, patch, segment)
    if "avdog" in patch.steam_routes:
        pings = np.sort(np.concatenate([pings, step_times(t, duration)]))
    return pings


def avdog_phase(t: Telemetry, n: int, sample_rate: int, patch: Patch = DEFAULT_PATCH) -> np.ndarray:
    """The dogtone's phase in cycles for each of `n` samples (not reduced to one turn). It runs at the chip's pitch, or with
    the patch's `pitch` route at that pitch moved by the note the ttsteam register holds."""
    pitch = avdog_parameters(t)[0]
    if "pitch" in patch.steam_routes:
        freq = pitch * 2.0 ** (held_semitones(t, n, sample_rate, patch) / 12.0)
        return (np.cumsum(freq) - freq) / sample_rate  # a sample's phase is the sum of the frequencies before it
    return pitch * np.arange(n) / sample_rate


def avdog_voices(t: Telemetry, duration: float, sample_rate: int, patch: Patch = DEFAULT_PATCH, segment: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """(the white voice, the red voice). Each is the dogtone under the envelope of one half of the undulation, so they
    never sound together. The red voice has the inverse wave shape.

    Two of the patch's steam routes reach in here (see `avdog_pings` and `avdog_phase`)."""
    n = int(duration * sample_rate)
    _, shape, rate_hz, decay_s = avdog_parameters(t)
    u = undulation(seconds_since(avdog_pings(t, duration, patch, segment), n, sample_rate), rate_hz, decay_s)
    phase = avdog_phase(t, n, sample_rate, patch)
    white = skewed_triangle(phase, shape) * np.maximum(u, 0.0)
    red = skewed_triangle(phase, -shape) * np.maximum(-u, 0.0)
    return white, red


def ttavdog(t: Telemetry, duration: float, sample_rate: int, seed: int, patch: Patch = DEFAULT_PATCH) -> np.ndarray:
    white, red = avdog_voices(t, duration, sample_rate, patch, seed // SEGMENT_SEED_STRIDE)
    return (AVDOG_LEVEL * (white + RED_GAIN * red)).astype(np.float32)
