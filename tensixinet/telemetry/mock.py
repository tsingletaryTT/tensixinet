# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Synthetic telemetry for N chips, for anyone without N Tenstorrent chips."""
from __future__ import annotations

import numpy as np

from .snapshot import Telemetry


def mock_telemetry(n_chips: int, seed: int = 0) -> list:
    """Deterministic, plausible readings for `n_chips` chips."""
    if n_chips < 1:
        raise ValueError(f"n_chips must be >= 1, got {n_chips}")
    rng = np.random.default_rng(seed)
    chips = []
    for _ in range(n_chips):
        temp = float(rng.uniform(35, 85))
        power = float(rng.uniform(10, 150))
        chips.append(Telemetry(
            asic_temperature=round(temp, 1),
            aiclk=float(rng.choice([800.0, 1000.0, 1350.0])),
            power=round(power, 1),
            current=round(power * 1.3, 1),
            voltage=0.75,
            gddr_temps=tuple(round(float(temp - rng.uniform(0, 15)), 1) for _ in range(8)),
            ddr_status=(1, 1, 1, 1),
        ))
    return chips
