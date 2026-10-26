# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The hwmon reader: units, ordering, filtering and failure handling, on a tmp_path tree."""
from pathlib import Path

from tensixinet.telemetry.hwmon import FastReading, find_chips, read_chip


def make_chip(root: Path, hwmon: str, bus_id: str, name: str = "blackhole",
              files=None) -> Path:
    """Build one hwmon directory the way the kernel lays it out."""
    d = root / "hwmon" / hwmon
    d.mkdir(parents=True)
    (d / "name").write_text(name + "\n")
    values = {"temp1_input": "49188", "power1_input": "41000000",
              "in0_input": "818", "curr1_input": "51000"}
    values.update(files or {})
    for fname, text in values.items():
        if text is not None:
            (d / fname).write_text(text + "\n")
    dev = root / "pci" / bus_id
    dev.mkdir(parents=True, exist_ok=True)
    (d / "device").symlink_to(dev)
    return d


def test_units_become_celsius_watts_volts_amps(tmp_path):
    make_chip(tmp_path, "hwmon3", "0000:01:00.0")
    (chip,) = find_chips(tmp_path / "hwmon")
    r = read_chip(chip)
    assert isinstance(r, FastReading)
    assert r.asic_temperature == 49.188
    assert r.power == 41.0
    assert r.voltage == 0.818
    assert r.current == 51.0  # 51000 mA; the brief said 0.051, which is inconsistent with its own divisor of 1000


def test_chips_are_ordered_by_bus_address_not_by_hwmon_number(tmp_path):
    make_chip(tmp_path, "hwmon9", "0000:01:00.0")
    make_chip(tmp_path, "hwmon2", "0000:02:00.0")
    chips = find_chips(tmp_path / "hwmon")
    assert [c.bus_id for c in chips] == ["0000:01:00.0", "0000:02:00.0"]
    assert chips[0].path.name == "hwmon9"


def test_other_hwmon_devices_are_skipped(tmp_path):
    make_chip(tmp_path, "hwmon0", "0000:00:18.3", name="coretemp")
    make_chip(tmp_path, "hwmon1", "0000:01:00.0", name="wormhole")
    assert [c.bus_id for c in find_chips(tmp_path / "hwmon")] == ["0000:01:00.0"]


def test_a_missing_or_garbled_file_gives_none(tmp_path):
    make_chip(tmp_path, "hwmon1", "0000:01:00.0", files={"power1_input": None})
    make_chip(tmp_path, "hwmon2", "0000:02:00.0", files={"power1_input": "n/a"})
    chips = find_chips(tmp_path / "hwmon")
    assert len(chips) == 2
    assert read_chip(chips[0]) is None
    assert read_chip(chips[1]) is None


def test_no_hwmon_directory_gives_an_empty_list(tmp_path):
    assert find_chips(tmp_path / "missing") == []
