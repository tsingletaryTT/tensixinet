# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Which steps of a sequence rest. Sample-and-hold makes silences as well as notes, so the lead and ttrollz leave out a share of their steps.

The coco's rests come from a petal castle instead (`devices.ttquantus_gate`), because its notes are held by that castle's own clock."""
from __future__ import annotations

import numpy as np


def rest_flags(seed: int, tag: int, n: int, share: float) -> np.ndarray:
    """`n` booleans, True where the step rests. Each step has its own value from one random stream, so the first steps are the same whatever `n` is,
    and a bigger `share` rests every step a smaller one does. `tag` tells the lead and each ttrollz lane apart."""
    if share <= 0.0 or n <= 0:
        return np.zeros(max(n, 0), dtype=bool)
    return np.random.default_rng([abs(int(seed)), tag]).random(n) < share
