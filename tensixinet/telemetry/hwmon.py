# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Fast chip telemetry from the Linux hwmon files.

The kernel driver exposes four live readings for each chip under
`/sys/class/hwmon/hwmonN/`. Reading a small text file takes microseconds, so
this can be sampled far faster than `tt-smi -s` (see `snapshot.py`). Reading
these files only reads telemetry; it does not open the device, so it needs no
gozer lease.

The files use the standard hwmon units: millidegrees C, microwatts, millivolts
and milliamps. `read_chip` converts them to C, W, V and A.

Nothing is cached: every call reads the files again.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

HWMON_ROOT = Path("/sys/class/hwmon")

# The `name` file of a hwmon directory that belongs to one of our chips.
CHIP_NAMES = ("blackhole", "wormhole", "grayskull")


@dataclass(frozen=True)
class FastReading:
    """The four readings hwmon gives for one chip at one instant."""

    asic_temperature: float  # deg C
    power: float  # W
    voltage: float  # V
    current: float  # A


@dataclass(frozen=True)
class HwmonChip:
    """One chip's hwmon directory and the PCI address it sits at."""

    path: Path
    bus_id: str  # e.g. "0000:01:00.0"


def find_chips(root: Path = HWMON_ROOT) -> List[HwmonChip]:
    """List the chips under `root`, in PCI bus order.

    The hwmon number depends on probe order and can change between boots, so
    the bus address (the name of the directory `device` points to) sets the order.
    A missing `root` gives an empty list.
    """
    chips: List[HwmonChip] = []
    try:
        entries = list(Path(root).iterdir())
    except OSError:
        return chips
    for d in entries:
        try:
            name = (d / "name").read_text().strip()
            if name not in CHIP_NAMES:
                continue
            bus_id = (d / "device").resolve().name
        except OSError:
            continue
        chips.append(HwmonChip(path=d, bus_id=bus_id))
    return sorted(chips, key=lambda c: c.bus_id)


def _read(path: Path, divisor: float) -> float:
    """Read one integer file and divide it down to the unit we report."""
    return int(path.read_text().strip()) / divisor


def read_chip(chip: HwmonChip) -> Optional[FastReading]:
    """Read the chip's four files; None if any is missing or not a number."""
    try:
        return FastReading(
            asic_temperature=_read(chip.path / "temp1_input", 1000),
            power=_read(chip.path / "power1_input", 1e6),
            voltage=_read(chip.path / "in0_input", 1000),
            current=_read(chip.path / "curr1_input", 1000),
        )
    except (OSError, ValueError):
        return None
