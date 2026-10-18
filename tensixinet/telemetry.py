"""Hardware telemetry: tt-smi JSON -> typed, immutable readings.

Everything here is a pure function of its input except `fetch_snapshot`, which
shells out to `tt-smi -s`. `tt-smi -s` only reads telemetry; it does not open
the device, so it needs no gozer lease.
"""
from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from typing import Callable, Optional, Tuple


@dataclass(frozen=True)
class Telemetry:
    """One device's readings at one instant."""

    asic_temperature: float  # deg C
    aiclk: float  # MHz
    power: float  # W
    current: float  # A
    voltage: float  # V
    gddr_temps: Tuple[float, ...]  # 8 channels, deg C (hotter die of each)
    ddr_status: Tuple[int, ...]  # 4 two-bit fields


# Neutral values used for any field the device does not report.
FALLBACK = Telemetry(
    asic_temperature=48.5,
    aiclk=800.0,
    power=32.0,
    current=45.0,
    voltage=0.71,
    gddr_temps=(30.0,) * 8,
    ddr_status=(0,) * 4,
)

# Packed GDDR_x_y_TEMP words. How many exist depends on the board, so they are
# discovered from the keys present rather than assumed.
_GDDR_TEMP_KEY = re.compile(r"^GDDR_(\d+)_(\d+)_TEMP$")


def decode_gddr_channel_temps(word: int) -> Tuple[float, float]:
    """Decode one GDDR_x_y_TEMP word into the temperatures of channels (x, y).

    Layout, from tt-smi's `decode_gddr_pair_temps`:
      [7:0] x bottom die, [15:8] x top die, [23:16] y bottom die, [31:24] y top die.
    A channel's temperature is the hotter of its two dies.
    """
    x = max(word & 0xFF, (word >> 8) & 0xFF)
    y = max((word >> 16) & 0xFF, (word >> 24) & 0xFF)
    return float(x), float(y)


def decode_ddr_status(word: int) -> Tuple[int, ...]:
    """Split the low byte of DDR_STATUS into four 2-bit status fields."""
    return tuple((word >> (i * 2)) & 3 for i in range(4))


def _hex(smbus: dict, key: str) -> Optional[int]:
    val = smbus.get(key)
    if isinstance(val, str):
        try:
            return int(val, 16)
        except ValueError:
            return None
    return None


def _float(tel: dict, key: str, default: float) -> float:
    try:
        return float(tel[key])  # float() tolerates tt-smi's padding, e.g. " 26.0"
    except (KeyError, TypeError, ValueError):
        return default


def _gddr_temps(smbus: dict) -> Tuple[float, ...]:
    pairs = []
    for key in smbus:
        m = _GDDR_TEMP_KEY.match(key)
        word = _hex(smbus, key) if m else None
        if word is not None:
            pairs.append((int(m.group(1)), word))
    if not pairs:
        return FALLBACK.gddr_temps
    temps = []
    for _, word in sorted(pairs):  # by first channel number, not dict order
        temps.extend(decode_gddr_channel_temps(word))
    return tuple(temps)


def _parse_device(dev: dict) -> Telemetry:
    tel = dev.get("telemetry") or {}
    smbus = dev.get("smbus_telem") or {}
    ddr = _hex(smbus, "DDR_STATUS")
    return Telemetry(
        asic_temperature=_float(tel, "asic_temperature", FALLBACK.asic_temperature),
        aiclk=_float(tel, "aiclk", FALLBACK.aiclk),
        power=_float(tel, "power", FALLBACK.power),
        current=_float(tel, "current", FALLBACK.current),
        voltage=_float(tel, "voltage", FALLBACK.voltage),
        gddr_temps=_gddr_temps(smbus),
        ddr_status=FALLBACK.ddr_status if ddr is None else decode_ddr_status(ddr),
    )


def parse_snapshot(snapshot: dict) -> list:
    """Parse a `tt-smi -s` JSON document into one Telemetry per device."""
    return [_parse_device(d) for d in snapshot.get("device_info") or []]


def fetch_snapshot(run: Callable = subprocess.run) -> dict:
    """Run `tt-smi -s` and return its parsed JSON, or {} on any failure.

    `run` is injectable so callers can substitute a canned response.
    """
    try:
        result = run(["tt-smi", "-s"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return json.loads(result.stdout)
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    return {}
