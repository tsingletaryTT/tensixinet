# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Synthetic telemetry for N chips, for anyone without N Tenstorrent chips."""
from __future__ import annotations

import numpy as np

from .snapshot import Telemetry


MASK_PATTERNS = 91 * (1 + 14 + 91 + 364)


def _distinct_masks(rng: np.random.Generator, n: int) -> list:
    """One (tensix_mask, eth_mask) per chip, all different while there are fewer chips than
    patterns: 91 ways to disable 2 of 14 columns, times 470 ways to lose 0 to 3 of 14 links."""
    seen, out = set(), []
    while len(out) < n:
        tensix = 0x3FFF
        for col in rng.choice(14, size=2, replace=False):
            tensix &= ~(1 << int(col))
        eth = 0x3FFF
        for link in rng.choice(14, size=int(rng.integers(0, 4)), replace=False):
            eth &= ~(1 << int(link))
        if (tensix, eth) not in seen or len(seen) >= MASK_PATTERNS:
            seen.add((tensix, eth))
            out.append((tensix, eth))
    return out


def mock_telemetry(n_chips: int, seed: int = 0, faults: bool = False) -> list:
    """Deterministic, plausible readings for `n_chips` chips.

    With `faults`, some chips (at least one, and not all when there are several) report memory
    errors or a thermal trip, so the event sounds can be heard without a failing chip.
    """
    if n_chips < 1:
        raise ValueError(f"n_chips must be >= 1, got {n_chips}")
    rng = np.random.default_rng(seed)
    # The wider readings come from their own generator, so the temperatures and powers above
    # keep the values they had before these readings existed.
    extra = np.random.default_rng(seed + 10_000)
    masks = _distinct_masks(extra, n_chips)
    sick = extra.random(n_chips) < 0.35
    if not sick.any():
        sick[int(extra.integers(n_chips))] = True
    if sick.all() and n_chips > 1:
        sick[int(extra.integers(n_chips))] = False
    chips = []
    for i in range(n_chips):
        temp = float(rng.uniform(35, 85))
        power = float(rng.uniform(10, 150))
        corr = uncorr = trips = 0
        if faults and sick[i]:
            corr, uncorr, trips = 1 + int(extra.integers(0, 40)), int(extra.integers(0, 3)), int(extra.integers(0, 2))
        chips.append(Telemetry(
            asic_temperature=round(temp, 1),
            aiclk=float(rng.choice([800.0, 1000.0, 1350.0])),
            power=round(power, 1),
            current=round(power * 1.3, 1),
            voltage=round(float(extra.uniform(0.70, 0.85)), 2),
            gddr_temps=tuple(round(float(temp - rng.uniform(0, 15)), 1) for _ in range(8)),
            ddr_status=(1, 1, 1, 1),
            board_power=round(power + float(extra.uniform(40, 90)), 1),
            heartbeat=int(extra.integers(0, 100_000)),
            tensix_mask=masks[i][0],
            eth_mask=masks[i][1],
            pcie_usage=int(extra.choice([1, 4])),
            corr_errs=corr,
            uncorr_errs=uncorr,
            thermal_trips=trips,
            busy=int(extra.integers(0, 3)),
        ))
    return chips
