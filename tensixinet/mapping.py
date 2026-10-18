"""Hardware signal -> musical parameter. The README's table, as code.

Ranges are clamped, so a reading outside the documented band holds the nearest
end instead of extrapolating.
"""
from __future__ import annotations

import math
from typing import Tuple

TEMP_RANGE = (25.0, 100.0)  # deg C
AICLK_RANGE = (400.0, 1350.0)  # MHz
BPM_RANGE = (60.0, 180.0)
POWER_MAX = 200.0  # W
VOLUME_RANGE = (0.3, 1.0)
VOLUME_CURVE = 2.0  # exponential steepness
C3_HZ = 130.8128  # equal temperament, A4 = 440 Hz
PITCH_SEMITONES = 24  # C3 -> C5

# One polyrhythm lane per DDR status field: pulses per bar.
DDR_LANE_DIVISIONS = (3, 4, 5, 7)


def _unit(x: float, lo: float, hi: float) -> float:
    return min(1.0, max(0.0, (x - lo) / (hi - lo)))


def norm_temp(temp_c: float) -> float:
    """25-100 C -> 0-1."""
    return _unit(temp_c, *TEMP_RANGE)


def temp_to_pitch_hz(temp_c: float) -> float:
    """25-100 C -> C3-C5, linear in semitones."""
    return C3_HZ * 2 ** (norm_temp(temp_c) * PITCH_SEMITONES / 12)


def aiclk_to_bpm(aiclk_mhz: float) -> float:
    """400-1350 MHz -> 60-180 BPM."""
    lo, hi = BPM_RANGE
    return lo + (hi - lo) * _unit(aiclk_mhz, *AICLK_RANGE)


def power_to_volume(power_w: float) -> float:
    """0-200 W -> 0.3-1.0 on an exponential curve."""
    x = _unit(power_w, 0.0, POWER_MAX)
    shaped = (math.exp(VOLUME_CURVE * x) - 1) / (math.exp(VOLUME_CURVE) - 1)
    lo, hi = VOLUME_RANGE
    return lo + (hi - lo) * shaped


def ddr_lane_divisions(ddr_status: Tuple[int, ...]) -> Tuple[int, ...]:
    """Pulses per bar for each DDR lane; a status of 0 mutes the lane."""
    return tuple(d if s else 0 for d, s in zip(DDR_LANE_DIVISIONS, ddr_status))
