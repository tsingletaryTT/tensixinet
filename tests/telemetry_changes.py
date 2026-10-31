# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""One changed value for each telemetry field, shared by the tests that prove which fields a device reads."""

CHANGES = {
    "asic_temperature": 90.0,
    "aiclk": 1350.0,
    "gddr_temps": (26.0, 99.0) * 4,
    "ddr_status": (1, 1, 1, 1),
    "voltage": 0.90,
    "power": 125.0,
    "axiclk": 1920.0,
    "arcclk": 1600.0,
    "heartbeat": 8,
    "tensix_mask": 0x1235,  # picked so the lead offset and gong detune both differ from the default chip
    "eth_mask": 0x0F0C,
    "pcie_usage": 5,
    "thm_limit_throttle": 45.0,
    "tdp_limit": 40.0,
    "temp_rate": 1.0,
    "power_rate": 25.0,
    # read by the chorus or by events.py, never by a device: every device must stay the same
    "board_power": 300.0,
    "busy": 3,
    "corr_errs": 5,
    "uncorr_errs": 1,
    "thermal_trips": 1,
}
