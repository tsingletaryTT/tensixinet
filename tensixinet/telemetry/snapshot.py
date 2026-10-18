# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Hardware telemetry: tt-smi JSON -> typed, immutable readings.

Everything here is a pure function of its input except `fetch_snapshot`, which
shells out to `tt-smi -s`. `tt-smi -s` only reads telemetry; it does not open
the device, so it needs no gozer lease.
"""
from __future__ import annotations

import hashlib
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
    # Everything below has a neutral default, so a chip that does not report it plays as before.
    board_power: float = 120.0  # W drawn by the whole board
    axiclk: float = 960.0  # MHz
    arcclk: float = 800.0  # MHz
    heartbeat: int = 0  # the chip firmware's counter
    tensix_mask: int = 0x3FFF  # which Tensix columns are enabled (harvesting differs per chip)
    eth_mask: int = 0x3FFF  # which ethernet links are live (low 16 bits of ETH_LIVE_STATUS)
    pcie_usage: int = 3
    corr_errs: int = 0  # correctable GDDR errors, all channels
    uncorr_errs: int = 0  # uncorrectable GDDR errors
    thermal_trips: int = 0
    thm_limit_throttle: float = 90.0  # deg C where the chip starts to throttle
    tdp_limit: float = 125.0  # W
    busy: Optional[int] = None  # processes using this chip, or None when the snapshot does not say


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


def _hex_or(smbus: dict, key: str, default: int) -> int:
    val = _hex(smbus, key)
    return default if val is None else val


_CORR_ERR_KEY = re.compile(r"^GDDR_\d+_\d+_CORR_ERRS$")


def _error_count(smbus: dict, pattern: re.Pattern) -> int:
    return sum(_hex_or(smbus, k, 0) for k in smbus if pattern.match(k))


def _busy(snapshot: dict, index: int) -> Optional[int]:
    """Distinct processes using chip `index`, not counting tt-smi, which lists itself on every chip."""
    if "processes" not in snapshot:
        return None
    pids = {p.get("pid") for p in snapshot["processes"]
            if p.get("device") == index and "tt-smi" not in str(p.get("cmdline", ""))}
    return len(pids)


def _parse_device(dev: dict, busy: Optional[int] = None) -> Telemetry:
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
        board_power=_float(tel, "board_power", FALLBACK.board_power),
        axiclk=float(_hex_or(smbus, "AXICLK", int(FALLBACK.axiclk))),
        arcclk=float(_hex_or(smbus, "ARCCLK", int(FALLBACK.arcclk))),
        heartbeat=int(_float(tel, "heartbeat", FALLBACK.heartbeat)),
        tensix_mask=_hex_or(smbus, "ENABLED_TENSIX_COL", FALLBACK.tensix_mask),
        eth_mask=_hex_or(smbus, "ETH_LIVE_STATUS", FALLBACK.eth_mask) & 0xFFFF,
        pcie_usage=_hex_or(smbus, "PCIE_USAGE", FALLBACK.pcie_usage),
        corr_errs=_error_count(smbus, _CORR_ERR_KEY),
        uncorr_errs=_hex_or(smbus, "GDDR_UNCORR_ERRS", 0),
        thermal_trips=_hex_or(smbus, "THERM_TRIP_COUNT", 0),
        thm_limit_throttle=float(_hex_or(smbus, "THM_LIMIT_THROTTLE", int(FALLBACK.thm_limit_throttle))),
        tdp_limit=float(_hex_or(smbus, "TDP_LIMIT_MAX", int(FALLBACK.tdp_limit))),
        busy=busy,
    )


def parse_snapshot(snapshot: dict) -> list:
    """Parse a `tt-smi -s` JSON document into one Telemetry per device."""
    return [_parse_device(d, _busy(snapshot, i)) for i, d in enumerate(snapshot.get("device_info") or [])]


def host_fingerprint(snapshot: dict) -> int:
    """A number from the host's description (name, kernel, driver, memory), or 0 if there is none.

    It feeds the patch seed, so the same chips on a different machine draw a different patch.
    """
    info = snapshot.get("host_info")
    if not info:
        return 0
    text = "|".join(f"{k}={info[k]}" for k in sorted(info))
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:6], "big")


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
