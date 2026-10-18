"""Telemetry parsing: tt-smi JSON -> typed readings. Pure functions, no hardware.

The fixture is a real `tt-smi -s` capture from a 4-device box (2026-10-17).
Devices 0-1 and 2-3 are different boards, so aiclk differs (800 vs 1350).
"""
import json
from pathlib import Path

import pytest

from tensixinet.telemetry import (
    FALLBACK,
    Telemetry,
    decode_gddr_channel_temps,
    decode_ddr_status,
    parse_snapshot,
)

FIXTURE = Path(__file__).parent / "fixtures" / "tt_smi_snapshot.json"


@pytest.fixture(scope="module")
def snapshot():
    return json.loads(FIXTURE.read_text())


def test_parse_snapshot_returns_one_reading_per_device(snapshot):
    readings = parse_snapshot(snapshot)
    assert len(readings) == len(snapshot["device_info"]) == 4


def test_parse_snapshot_reads_scalar_fields_with_padding(snapshot):
    # tt-smi pads numbers with spaces: current " 26.0", aiclk " 800".
    dev0 = parse_snapshot(snapshot)[0]
    assert dev0.asic_temperature == 45.9
    assert dev0.aiclk == 800.0
    assert dev0.power == 18.0
    assert dev0.current == 26.0
    assert dev0.voltage == 0.71


def test_parse_snapshot_distinguishes_devices(snapshot):
    readings = parse_snapshot(snapshot)
    assert [r.aiclk for r in readings] == [800.0, 800.0, 1350.0, 1350.0]


def test_gddr_word_decodes_two_channels_hottest_die_each():
    # tt-smi layout: [7:0] x bottom, [15:8] x top, [23:16] y bottom, [31:24] y top.
    # 0x1e1e2c2e: channel x = max(0x2e, 0x2c) = 46, channel y = max(0x1e, 0x1e) = 30.
    assert decode_gddr_channel_temps(0x1E1E2C2E) == (46.0, 30.0)


def test_gddr_temps_cover_all_eight_channels(snapshot):
    # Regression: the old parser kept 4 bytes per word but truncated to 8
    # values, so channels 4-7 (the last two words) never reached the synth.
    dev0 = parse_snapshot(snapshot)[0]
    assert len(dev0.gddr_temps) == 8
    # Real data: GDDR_6_7_TEMP = 0x2c2c2e2c -> channel 6 = max(0x2c, 0x2e) = 46,
    # channel 7 = max(0x2c, 0x2c) = 44.
    assert dev0.gddr_temps[6:] == (46.0, 44.0)
    # GDDR_0_1_TEMP = 0x2c2c2c2e -> 46, 44.
    assert dev0.gddr_temps[:2] == (46.0, 44.0)


def test_gddr_decode_agrees_with_tt_smi_reference_decoder():
    # Oracle independent of this repo: tt-smi's own decoder for the same word.
    tt_smi_utils = pytest.importorskip("tt_smi.utils")
    for word in (0x1E1E2C2E, 0x2C2C2E2C, 0x3638363A, 0x00000000, 0xFF01FE02):
        (xb, xt), (yb, yt) = tt_smi_utils.decode_gddr_pair_temps(word)
        assert decode_gddr_channel_temps(word) == (float(max(xb, xt)), float(max(yb, yt)))


def test_ddr_status_has_one_2bit_field_per_channel_group():
    assert decode_ddr_status(0x55555555) == (1, 1, 1, 1)
    assert decode_ddr_status(0b11_10_01_00) == (0, 1, 2, 3)


def test_missing_smbus_fields_fall_back_to_neutral_values():
    snap = {"device_info": [{"telemetry": {"asic_temperature": "50.0"}, "smbus_telem": {}}]}
    r = parse_snapshot(snap)[0]
    assert r.asic_temperature == 50.0
    assert r.gddr_temps == FALLBACK.gddr_temps
    assert r.ddr_status == FALLBACK.ddr_status
    assert r.power == FALLBACK.power


def test_empty_or_malformed_snapshot_yields_no_readings():
    assert parse_snapshot({}) == []
    assert parse_snapshot({"device_info": []}) == []


def test_telemetry_is_immutable():
    with pytest.raises(Exception):
        FALLBACK.power = 1.0


# --- dynamic channel count: not every board has the same GDDR layout ---------

def _gddr_snapshot(words):
    smbus = {f"GDDR_{2*i}_{2*i+1}_TEMP": hex(w) for i, w in enumerate(words)}
    return {"device_info": [{"telemetry": {}, "smbus_telem": smbus}]}


def test_gddr_channel_count_follows_the_registers_present():
    two_words = parse_snapshot(_gddr_snapshot([0x1E1E2C2E, 0x2C2C2E2C]))[0]
    six_words = parse_snapshot(_gddr_snapshot([0x1E1E2C2E] * 6))[0]
    assert len(two_words.gddr_temps) == 4
    assert len(six_words.gddr_temps) == 12


def test_gddr_registers_are_ordered_by_channel_not_by_dict_order():
    # Deliberately insert the registers out of order.
    smbus = {"GDDR_2_3_TEMP": hex(0x00000303), "GDDR_0_1_TEMP": hex(0x00000101)}
    r = parse_snapshot({"device_info": [{"telemetry": {}, "smbus_telem": smbus}]})[0]
    assert r.gddr_temps[:2] == (1.0, 0.0)
    assert r.gddr_temps[2:] == (3.0, 0.0)
