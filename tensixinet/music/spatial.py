# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""How a device's own internals move it through the field (see `space.py` for the placing itself).

A pan curve is the device's offset from its place, in azimuth, for each sample of a render. Most devices sit where `space.py`
puts them and orbit with the chip's signals. A few have something inside them that is already a movement, and they use it:

* **ttavdog** - the AVDog puts its white voice and its red voice on two channels. The white sounds on the up half of the undulation,
  so it is left; the red answers on the down half, so it is right.
* **ultrasonic_shimmer** - two tones beat at the GDDR spread (0.2 Hz a degree); the sound pans at that beat. Equal temperatures give
  one steady tone and no movement.
* **gong** - each strike lands at its own place (spread by the golden ratio, so no two neighbours are close) and drifts back to the
  middle as it rings.
* **ttquantus** - the note is the position: low notes left, high notes right.
* **ttrollz** - each lane's blip comes from the lane's own place, spread across the field; the clock's blips are in the middle.
* **ttsteam** - the snare's hits alternate sides.

`depth_curve` is the same for depth: the drone moves toward the listener while its ripple swells.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, Optional

import numpy as np

from .devices import (
    drone_ripple_inputs, gong_parameters, roll_pulses, seconds_since, shimmer_parameters, shock_wave, strike_times,
    ttquantus_steps, ttrollz_network, ttrollz_pulse_times,
)
from .mapping import aiclk_to_bpm, personality
from .patch import Patch
from .register import step_times
from .ttavdog import avdog_parameters, avdog_pings, undulation
from .ttrolzer import SEGMENT_SEED_STRIDE
from ..telemetry import Telemetry

AVDOG_SWING = 0.8  # how far the white and red voices go to the left and right of their place
SHIMMER_SWING = 0.8
GONG_SWING = 0.8
GONG_SETTLE = 0.35  # a struck gong has settled back to the middle after about this share of a strike interval
QUANTUS_SWING = 0.7
ROLL_SPREAD = 0.8  # the outermost lanes are this far from the middle
SNARE_SWING = 0.5
DRONE_FORWARD = 0.3  # a drone at the peak of its ripple is this much nearer
GOLDEN = 0.6180339887498949


def _hold(times: np.ndarray, values: np.ndarray, n: int, sample_rate: int) -> np.ndarray:
    """For each of `n` samples, the value of the latest event at or before it, and 0 before the first."""
    if len(times) == 0:
        return np.zeros(n)
    index = np.searchsorted(times, np.arange(n) / sample_rate, side="right") - 1
    return np.where(index >= 0, np.asarray(values, dtype=np.float64)[np.maximum(index, 0)], 0.0)


def _avdog(t, duration, sample_rate, seed, patch, segment) -> np.ndarray:
    n = int(duration * sample_rate)
    _, _, rate_hz, decay_s = avdog_parameters(t)
    u = undulation(seconds_since(avdog_pings(t, duration, patch, segment), n, sample_rate), rate_hz, decay_s)
    return -AVDOG_SWING * u


def _shimmer(t, duration, sample_rate, seed, patch, segment) -> np.ndarray:
    f1, f2, _ = shimmer_parameters(t, sample_rate, patch)
    return SHIMMER_SWING * np.sin(2 * math.pi * (f2 - f1) * np.arange(int(duration * sample_rate)) / sample_rate)


def _gong(t, duration, sample_rate, seed, patch, segment) -> np.ndarray:
    n = int(duration * sample_rate)
    strikes = strike_times(t, duration, patch, segment)
    if len(strikes) == 0:
        return np.zeros(n)
    phase = (personality(t) % 1000) / 1000.0
    first = segment * 5 + 1  # a new segment starts further along the sequence, so a live run does not repeat its landings
    landing = GONG_SWING * (2 * (((np.arange(len(strikes)) + first) * GOLDEN + phase) % 1.0) - 1)  # each strike a golden step further on
    index = np.searchsorted(strikes, np.arange(n) / sample_rate, side="right") - 1
    since = np.arange(n) / sample_rate - strikes[np.maximum(index, 0)]
    interval = gong_parameters(t, sample_rate, patch).interval
    return np.where(index >= 0, landing[np.maximum(index, 0)] * np.exp(-since / (GONG_SETTLE * interval)), 0.0)


def _quantus(t, duration, sample_rate, seed, patch, segment) -> np.ndarray:
    steps = ttquantus_steps(t, duration, sample_rate, seed, patch)  # semitones above the bottom of two octaves
    return np.clip((steps - 12.0) / 12.0, -1.0, 1.0) * QUANTUS_SWING


def _rollz(t, duration, sample_rate, seed, patch, segment) -> Optional[np.ndarray]:
    net = ttrollz_network(t)
    if not net.lanes:
        return None
    pulses = roll_pulses(t, duration, patch, segment)
    count = len(net.lanes)
    clock = ttrollz_pulse_times(aiclk_to_bpm(t.aiclk), duration)  # the chip's own clock sits in the middle
    times, places = [clock], [np.zeros(len(clock))]
    for position, lane in enumerate(net.lanes):
        lane_times = pulses[lane.index][0]
        times.append(lane_times)
        places.append(np.full(len(lane_times), 0.0 if count == 1 else -ROLL_SPREAD + 2 * ROLL_SPREAD * position / (count - 1)))
    all_times, all_places = np.concatenate(times), np.concatenate(places)
    order = np.argsort(all_times, kind="stable")
    return _hold(all_times[order], all_places[order], int(duration * sample_rate), sample_rate)


def _steam(t, duration, sample_rate, seed, patch, segment) -> Optional[np.ndarray]:
    hits = step_times(t, duration)
    if len(hits) == 0:
        return None
    sides = np.where((np.arange(len(hits)) + segment) % 2 == 0, -SNARE_SWING, SNARE_SWING)
    return _hold(hits, sides, int(duration * sample_rate), sample_rate)


PAN_CURVES: Dict[str, Callable] = {
    "ttavdog": _avdog, "ultrasonic_shimmer": _shimmer, "gong": _gong, "ttquantus": _quantus, "ttrollz": _rollz, "ttsteam": _steam,
}


def pan_curve(name: str, t: Telemetry, duration: float, sample_rate: int, seed: int, patch: Patch) -> Optional[np.ndarray]:
    """Device `name`'s offset from its place in azimuth for each sample, or None when it has no movement of its own. `seed` carries the
    segment of a live run, as it does for the devices themselves."""
    make = PAN_CURVES.get(name)
    return None if make is None else make(t, duration, sample_rate, seed, patch, seed // SEGMENT_SEED_STRIDE)


def depth_curve(name: str, t: Telemetry, duration: float, sample_rate: int, seed: int, patch: Patch) -> Optional[np.ndarray]:
    """Device `name`'s offset from its depth for each sample (negative is nearer), or None. Only the drone has one: it comes toward the
    listener while its ripple swells after a gong strike."""
    if name != "ttdrone":
        return None
    since, rate_hz, decay_s = drone_ripple_inputs(t, duration, sample_rate, patch, seed // SEGMENT_SEED_STRIDE)
    return -DRONE_FORWARD * shock_wave(since, rate_hz, decay_s)
