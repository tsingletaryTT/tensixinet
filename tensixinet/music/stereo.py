# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Placing one device's voice in the field: its chip's home, its own offset, and everything that moves it.

`space.py` has the maths of a place and `spatial.py` the movements that come from inside a device. This joins them for one voice:

* the **home** of the chip that plays it (`space.chip_azimuth`) plus the device's own **offset** (`space.device_space`), which is fixed;
* an **orbit** around that place whose size comes from the chip's chaos (pressure plus how fast its readings move) and whose speed comes
  from its clock (`space.motion_azimuth`), so a calm chip sits and a chaotic one travels;
* the device's **own movement** where it has one (`spatial.pan_curve`: the AVDog's swing, the shimmer's beat, each gong strike);
* **depth**: the device's own distance, the drone's approach while it ripples (`spatial.depth_curve`), and the **turn**: the voice that has
  the floor comes forward and the rest fall back, in step with the level changes the turn gains already make.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .patch import Patch
from .space import chip_azimuth, device_space, motion_azimuth, smooth, spatialize
from .spatial import depth_curve, pan_curve
from ..telemetry import Telemetry

TURN_DEPTH = 0.35  # a voice that does not have the floor is this much farther back than one that does
DISSOLVE_DEPTH = 0.4  # a dissolving voice also falls back this far
PAN_SMOOTHING_MS = 15.0  # a hop from one place to the next takes about this long, so it does not click


def turn_depth(gains: np.ndarray, floor: float) -> np.ndarray:
    """How far back the turn puts a voice, for each sample: nothing at full level (it has the floor), `TURN_DEPTH` at the floor level,
    and in between as it fades. The turn gains already fade a voice in and out, so the voice moves as smoothly."""
    forward = np.clip((np.asarray(gains, dtype=np.float64) - floor) / max(1e-9, 1.0 - floor), 0.0, 1.0)
    return TURN_DEPTH * (1.0 - forward)


def place_voice(
    name: str, chip: int, n_chips: int, t: Telemetry, voice: np.ndarray, duration: float, sample_rate: int, seed: int,
    patch: Patch, chaos: float, depth_offset: np.ndarray, dissolve: Optional[np.ndarray] = None,
) -> np.ndarray:
    """`voice` (a mono render of device `name`, played by chip `chip` of `n_chips`) in the field: float32, shape (n, 2). `depth_offset`
    is the turn's push back (`turn_depth`) for each sample. `dissolve` (0 whole, 1 gone, for each sample) is an eviction in progress
    (`evict.drain_progress`): the voice drifts to the hard left edge and a little farther back as it goes."""
    n = len(voice)
    offset, width, depth = device_space(name)
    azimuth = chip_azimuth(chip, n_chips, t) + offset + motion_azimuth(t, name, chip, n, sample_rate, chaos)
    pan = pan_curve(name, t, duration, sample_rate, seed, patch)
    if pan is not None:
        azimuth = azimuth + smooth(pan[:n], sample_rate, PAN_SMOOTHING_MS)
    own_depth = depth_curve(name, t, duration, sample_rate, seed, patch)
    far = depth + np.asarray(depth_offset, dtype=np.float64) + (0.0 if own_depth is None else own_depth[:n])
    azimuth = np.clip(azimuth, -1.0, 1.0)
    if dissolve is not None:
        d = np.asarray(dissolve, dtype=np.float64)
        azimuth = azimuth * (1.0 - d) - d  # to the left edge
        far = far + DISSOLVE_DEPTH * d
    return spatialize(voice, azimuth, width, np.clip(far, 0.0, 1.0), sample_rate)
