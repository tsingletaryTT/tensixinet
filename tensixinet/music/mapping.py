# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Hardware signal -> musical parameter. The README's table, as code.

Ranges are clamped, so a reading outside the documented band holds the nearest
end and do not extrapolate.
"""
from __future__ import annotations

import hashlib
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

# One ttrolzer roll per DDR status field: a triangle, square, pentagon and hexagon.
DDR_LANE_SIDES = (3, 4, 5, 6)


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


def ddr_lane_sides(ddr_status: Tuple[int, ...]) -> Tuple[int, ...]:
    """Sides of the roll each DDR lane plays; a status of 0 grounds the roll, which silences the lane."""
    return tuple(d if s else 0 for d, s in zip(DDR_LANE_SIDES, ddr_status))


# --- readings beyond temperature, clock and power ---------------------------------------

PRESSURE_TEMP_FLOOR_C = 40.0  # below this a chip is not under thermal pressure
PRESSURE_POWER_FLOOR = 0.25  # fraction of the power limit below which a chip is not under power pressure
VOLTAGE_CENTRE_V = 0.75
CENTS_PER_VOLT = 400.0
MAX_DETUNE_CENTS = 40.0
DRONE_VOICES_RANGE = (1, 6)
HEARTBEAT_STEPS = 16
BUSY_WEIGHT_PER_PROCESS = 0.5
BUSY_WEIGHT_MAX_PROCESSES = 3
PRESSURE_WEIGHT = 2.0
RATE_FULL_TEMP_C_PER_S = 1.0  # a chip heating or cooling this fast is as unsettled as a chip can sound
RATE_FULL_POWER_W_PER_S = 25.0  # and so is one whose power moves this fast


def pressure(temp_c: float, power_w: float, throttle_c: float, tdp_w: float) -> float:
    """0 (cool and idle) to 1 (at the throttle temperature or the power limit), whichever is closer."""
    return max(
        _unit(temp_c, PRESSURE_TEMP_FLOOR_C, throttle_c),
        _unit(power_w, PRESSURE_POWER_FLOOR * tdp_w, tdp_w),
    )


def chip_pressure(t) -> float:
    """`pressure` for one chip's telemetry."""
    return pressure(t.asic_temperature, t.power, t.thm_limit_throttle, t.tdp_limit)


def rate_activity(temp_rate: float, power_rate: float) -> float:
    """0 (readings holding still) to 1 (moving at the full rate), whichever reading is moving faster.
    A reading that falls counts the same as one that rises."""
    return max(
        _unit(abs(temp_rate), 0.0, RATE_FULL_TEMP_C_PER_S),
        _unit(abs(power_rate), 0.0, RATE_FULL_POWER_W_PER_S),
    )


def chip_chaos(t) -> float:
    """The ttsidrax chaos knob: the chip's pressure plus how fast its readings are moving, up to 1.
    A snapshot has no rates, so offline this is the pressure."""
    return min(1.0, chip_pressure(t) + rate_activity(t.temp_rate, t.power_rate))


def personality(t) -> int:
    """`chip_identity` for one chip's telemetry."""
    return chip_identity(t.tensix_mask, t.eth_mask)


def eth_lane_levels(eth_mask: int) -> Tuple[float, ...]:
    """Four lane levels, one per group of four ethernet links: the share of that group that is up."""
    return tuple(bin((eth_mask >> (4 * i)) & 0xF).count("1") / 4 for i in range(4))


def chip_identity(tensix_mask: int, eth_mask: int) -> int:
    """A stable number for a chip from which Tensix columns and ethernet links it has.

    Harvesting leaves every chip with a different set, so this tells chips apart."""
    digest = hashlib.sha256(f"{tensix_mask:x}/{eth_mask:x}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def voltage_to_cents(voltage: float) -> float:
    """A slow detune: 0.75 V is in tune, 400 cents per volt, limited to 40 cents either way."""
    return max(-MAX_DETUNE_CENTS, min(MAX_DETUNE_CENTS, (voltage - VOLTAGE_CENTRE_V) * CENTS_PER_VOLT))


def drone_voices(pcie_usage: int) -> int:
    lo, hi = DRONE_VOICES_RANGE
    return max(lo, min(hi, int(pcie_usage)))


def heartbeat_phase(heartbeat: int) -> float:
    """Where in its cycle a chip's counter is, in sixteenths: 0 to just under 1."""
    return (int(heartbeat) % HEARTBEAT_STEPS) / HEARTBEAT_STEPS


def turn_weight(power: float, board_power: float, mean_board_power: float, pressure: float, busy) -> float:
    """How often a chip's slots solo: its power, scaled by board load against the average board,
    by pressure (up to 3x) and by the processes using it (half again for each, up to three).
    `busy` of None means unknown and adds nothing."""
    load = board_power / mean_board_power if mean_board_power > 0 else 1.0
    processes = 0 if busy is None else min(busy, BUSY_WEIGHT_MAX_PROCESSES)
    return power * load * (1 + PRESSURE_WEIGHT * pressure) * (1 + BUSY_WEIGHT_PER_PROCESS * processes)
