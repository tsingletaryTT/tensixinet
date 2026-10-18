# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Turn-taking: who solos next, decided by system conditions.

Like players trading solos, a busy voice gets the floor more often, a quiet one
still gets some turns, and nobody takes two turns in a row.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

GOLDEN = 0.6180339887498949  # frac(k * GOLDEN) spreads picks evenly without a random generator


def pick_soloist(weights: Sequence[float], k: int, previous: Optional[int]) -> int:
    """The soloist for turn `k`, given who soloed in the turn before (`None` for the first).

    `soloists` plans a whole run with this. Live mode calls it once per turn, because the weights change
    as the chips do and nobody should solo twice in a row whatever they were before."""
    w = np.maximum(np.asarray(weights, dtype=float), 0.0)
    if w.sum() == 0:
        w = np.ones(len(w))
    live = [i for i, x in enumerate(w) if x > 0]

    def draw(weights: np.ndarray, u: float) -> int:
        cum = np.cumsum(weights) / weights.sum()
        return int(min(np.searchsorted(cum, u, side="right"), len(weights) - 1))

    u = ((k + 1) * GOLDEN) % 1.0
    pick = draw(w, u)
    if previous is not None and pick == previous and len(live) > 1:
        others = w.copy()
        others[pick] = 0.0
        pick = draw(others, (u * 7) % 1.0)  # a second, different draw from everyone else
    return pick


def soloists(weights: Sequence[float], n_turns: int) -> List[int]:
    """Pick one voice per turn, in proportion to `weights`.

    A voice with weight 0 never solos. If every weight is 0 they are treated as
    equal. When the pick would repeat the previous soloist, the floor is drawn again
    from the other voices, still in proportion to their weights.
    """
    out: List[int] = []
    for k in range(n_turns):
        out.append(pick_soloist(weights, k, out[-1] if out else None))
    return out


def turn_gains(
    soloist_per_turn: Sequence[int],
    n_voices: int,
    n_samples: int,
    sample_rate: int,
    turn_s: float,
    floor: float = 0.35,
    fade_s: float = 0.25,
) -> np.ndarray:
    """(n_voices, n_samples) gains: 1.0 for the soloist, `floor` for the rest.

    Hand-overs are smoothed over `fade_s` so voices swell in and out. Past the
    last turn the last soloist keeps the floor.
    """
    target = np.full((n_voices, n_samples), floor)
    turn_len = max(1, int(turn_s * sample_rate))
    for k, voice in enumerate(soloist_per_turn):
        start = k * turn_len
        end = n_samples if k == len(soloist_per_turn) - 1 else (k + 1) * turn_len
        target[voice, start:end] = 1.0
    w = max(1, int(fade_s * sample_rate))
    if w == 1:
        return target
    pad = np.pad(target, ((0, 0), (w // 2, w - 1 - w // 2)), mode="edge")
    # moving average by differencing a running sum: O(n), where np.convolve would be O(n * w)
    csum = np.concatenate([np.zeros((n_voices, 1)), np.cumsum(pad, axis=1)], axis=1)
    return (csum[:, w:] - csum[:, :-w]) / w
