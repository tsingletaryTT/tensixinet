# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The governor for load kernels: kernels whose job is to spend as much power and as many cycles as the box allows,
briefly, so that a burst of real load becomes a real event in the music and in the readings that drive it.

A load kernel is deliberately impolite to the chip it runs on, so two independent limits bound it, and both are
checked against the chip's own telemetry, never a fixed number of degrees or watts:

* A hard wall-clock cap (`MAX_BURST_S`). A burst stops here even on a chip that stays cool the whole time.
* A live pressure ceiling (`ABORT_PRESSURE`). `mapping.pressure` is 0 (cool and idle) to 1 (at the throttle
  temperature or the power limit); a burst polls it during the run and stops the instant it crosses the ceiling,
  which is comfortably short of 1 so the chip never actually reaches its own limit.

A chip already close to the ceiling may not start a burst at all (`may_start`), so a kernel that is already hot
from something else does not get pushed further.

This module reads telemetry and does arithmetic. It opens no device and imports no `ttnn`, so it is tested in the
ordinary suite (`tests/kernels/test_governor.py`), not under the simulator or a lease."""
from __future__ import annotations

from typing import Tuple

from ..music.mapping import chip_pressure

MAX_BURST_S = 2.0  # a burst never runs longer than this, regardless of what the chip reports
ABORT_PRESSURE = 0.85  # stop the instant pressure reaches this, well short of the chip's own throttle point


def may_start(t) -> bool:
    """Whether a chip is cool and idle enough to start a burst at all."""
    return chip_pressure(t) < ABORT_PRESSURE


def should_stop(elapsed_s: float, t) -> Tuple[bool, str]:
    """Whether a running burst must stop now, and why. Checked every poll during the burst."""
    if elapsed_s >= MAX_BURST_S:
        return True, f"hit the {MAX_BURST_S:g}s hard cap"
    if chip_pressure(t) >= ABORT_PRESSURE:
        return True, f"pressure reached the abort ceiling ({ABORT_PRESSURE:g})"
    return False, ""
