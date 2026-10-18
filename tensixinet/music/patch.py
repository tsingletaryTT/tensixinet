# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""A patch: every drawn choice of one run, so a run can be replayed exactly.

Two things decide a patch. The system decides the scale (from how unevenly the memory
channels are heated) and the gong tuning (from the memory-to-ASIC temperature ratio), so
the sound follows the machine. A seed decides the rest: the lead motif, the gong's beat,
the shimmer band, the turn length and how far non-soloists drop. The seed comes from the
chips' readings plus some entropy, so no two runs match unless you replay a saved patch.

`DEFAULT_PATCH` holds the constants the devices used before patches existed.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np

from ..telemetry import Telemetry

# Semitone offsets. A scale may have any length; degrees wrap by octave.
SCALES = {
    "major_pentatonic": (0, 2, 4, 7, 9),
    "minor_pentatonic": (0, 3, 5, 7, 10),
    "hirajoshi": (0, 2, 3, 7, 8),
    "dorian": (0, 2, 3, 5, 7, 9, 10),
    "whole_tone": (0, 2, 4, 6, 8, 10),
    "phrygian_dominant": (0, 1, 4, 5, 7, 8, 10),
}
SCALE_NAMES = tuple(SCALES)
DEGREES_PER_SCALE_STEP_C = 3.0  # each 3 C of GDDR spread moves one scale along

# name -> (partial ratios, amplitudes). No partial after the first is a whole multiple.
GONG_TUNINGS = {
    "bell": ((1.0, 2.76, 5.40, 8.45), (1.0, 0.6, 0.35, 0.2)),
    "tamtam": ((1.0, 1.47, 2.31, 3.64, 5.57), (1.0, 0.7, 0.5, 0.35, 0.2)),
    "singing_bowl": ((1.0, 2.71, 5.18), (1.0, 0.5, 0.25)),
}
GONG_TUNING_NAMES = tuple(GONG_TUNINGS)

MOTIF_LENGTH = 8
MOTIF_RANGE = (-3, 6)  # scale degrees


@dataclass(frozen=True)
class Patch:
    seed: int
    scale_name: str
    scale: Tuple[int, ...]
    motif: Tuple[int, ...]
    gong_tuning: str
    gong_partials: Tuple[float, ...]
    gong_amps: Tuple[float, ...]
    gong_beat_hz: float
    shimmer_band: Tuple[float, float]
    bars_per_turn: int
    floor: float  # volume of everyone who is not soloing


DEFAULT_PATCH = Patch(
    seed=0,
    scale_name="major_pentatonic",
    scale=SCALES["major_pentatonic"],
    motif=(0, 2, 1, 3, 2, 4, 3, 1),
    gong_tuning="bell",
    gong_partials=GONG_TUNINGS["bell"][0],
    gong_amps=GONG_TUNINGS["bell"][1],
    gong_beat_hz=0.7,
    shimmer_band=(12000.0, 20000.0),
    bars_per_turn=2,
    floor=0.35,
)


def seed_from_telemetry(telemetry: Sequence[Telemetry], entropy: int = 0) -> int:
    """A seed in [0, 2**63) from every reading on every chip, plus `entropy`.

    Any one reading changing by 0.1 gives a different seed, and so does a different entropy.
    """
    text = repr([dataclasses.astuple(t) for t in telemetry]) + f"|{entropy}"
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big") & (2 ** 63 - 1)


def gddr_spread(telemetry: Sequence[Telemetry]) -> float:
    temps = [c for t in telemetry for c in t.gddr_temps]
    return max(temps) - min(temps)


def memory_to_asic_ratio(telemetry: Sequence[Telemetry]) -> float:
    gddr = [c for t in telemetry for c in t.gddr_temps]
    asic = [t.asic_temperature for t in telemetry]
    return (sum(gddr) / len(gddr)) / (sum(asic) / len(asic))


def scale_for(telemetry: Sequence[Telemetry]) -> str:
    return SCALE_NAMES[int(gddr_spread(telemetry) // DEGREES_PER_SCALE_STEP_C) % len(SCALE_NAMES)]


def gong_tuning_for(telemetry: Sequence[Telemetry]) -> str:
    return GONG_TUNING_NAMES[int(round(memory_to_asic_ratio(telemetry) * 10)) % len(GONG_TUNING_NAMES)]


def _draw_motif(rng: np.random.Generator) -> Tuple[int, ...]:
    """Eight scale degrees from 0 in steps of at most 2, staying inside MOTIF_RANGE, with at least 3 different notes."""
    lo, hi = MOTIF_RANGE
    while True:
        motif = [0]
        for _ in range(MOTIF_LENGTH - 1):
            step = int(rng.choice([-2, -1, 0, 1, 2], p=[1 / 7, 2 / 7, 1 / 7, 2 / 7, 1 / 7]))
            if not lo <= motif[-1] + step <= hi:
                step = -step
            motif.append(motif[-1] + step)
        if len(set(motif)) >= 3:
            return tuple(motif)


def draw_patch(seed: int, telemetry: Sequence[Telemetry]) -> Patch:
    rng = np.random.default_rng(seed)
    scale_name = scale_for(telemetry)
    tuning = gong_tuning_for(telemetry)
    motif = _draw_motif(rng)
    lo = float(rng.uniform(10000, 14000))
    width = float(rng.uniform(4000, 7000))
    return Patch(
        seed=seed,
        scale_name=scale_name,
        scale=SCALES[scale_name],
        motif=motif,
        gong_tuning=tuning,
        gong_partials=GONG_TUNINGS[tuning][0],
        gong_amps=GONG_TUNINGS[tuning][1],
        gong_beat_hz=round(float(rng.uniform(0.4, 1.2)), 3),
        shimmer_band=(round(lo, 1), round(min(lo + width, 20500.0), 1)),
        bars_per_turn=int(rng.integers(1, 5)),
        floor=round(float(rng.uniform(0.25, 0.45)), 3),
    )


def _validate(p: Patch) -> Patch:
    if not p.scale or not p.motif:
        raise ValueError("a patch needs a non-empty scale and motif")
    if not p.gong_partials or p.gong_partials[0] != 1.0:
        raise ValueError("gong_partials must start at 1.0")
    if len(p.gong_amps) != len(p.gong_partials):
        raise ValueError("gong_amps needs one amplitude per partial")
    if not p.shimmer_band[0] < p.shimmer_band[1]:
        raise ValueError("shimmer_band needs low < high")
    if p.bars_per_turn < 1:
        raise ValueError("bars_per_turn must be at least 1")
    if not 0.0 < p.floor <= 1.0:
        raise ValueError("floor must be above 0 and at most 1")
    if p.gong_beat_hz < 0:
        raise ValueError("gong_beat_hz cannot be negative")
    return p


def patch_from_dict(d: dict) -> Patch:
    try:
        return _validate(Patch(
            seed=int(d["seed"]),
            scale_name=str(d["scale_name"]),
            scale=tuple(int(x) for x in d["scale"]),
            motif=tuple(int(x) for x in d["motif"]),
            gong_tuning=str(d["gong_tuning"]),
            gong_partials=tuple(float(x) for x in d["gong_partials"]),
            gong_amps=tuple(float(x) for x in d["gong_amps"]),
            gong_beat_hz=float(d["gong_beat_hz"]),
            shimmer_band=tuple(float(x) for x in d["shimmer_band"]),
            bars_per_turn=int(d["bars_per_turn"]),
            floor=float(d["floor"]),
        ))
    except (KeyError, TypeError) as e:
        raise ValueError(f"patch is missing or has a malformed field: {e}") from e


def save_patch(patch: Patch, path) -> None:
    # One line per field keeps lists of numbers readable and the file easy to diff and hand-edit.
    fields = [f"  {json.dumps(k)}: {json.dumps(v)}" for k, v in dataclasses.asdict(patch).items()]
    Path(path).write_text("{\n" + ",\n".join(fields) + "\n}\n")


def load_patch(path) -> Patch:
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise ValueError(f"cannot read patch {path}: {e}") from e
    if not isinstance(data, dict):
        raise ValueError(f"patch {path} must be a JSON object")
    return patch_from_dict(data)
