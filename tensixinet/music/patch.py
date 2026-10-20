# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""A patch: every drawn choice of one run, so a run can be replayed exactly.

Two things decide a patch. The system decides the scale (from how unevenly the memory
channels are heated) and the gong tuning (from the memory-to-ASIC temperature ratio), so
the sound follows the machine. A seed decides the rest: the lead motif, the gong's beat,
the shimmer band, the turn length and how far non-soloists drop. The seed also picks a mode
of the system's scale (the same notes started from another degree), a small stretch of the gong's
partials and a jitter of their levels, so a quiet box that always lands on the same scale and
tuning still plays a different patch each run. The seed comes from the chips' readings plus some
entropy, so no two runs match unless you replay a saved patch.

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
    # Added after the first six, in this order, so the system still reaches the first six in the same places.
    "blues": (0, 3, 5, 6, 7, 10),
    "harmonic_minor": (0, 2, 3, 5, 7, 8, 11),
    "lydian": (0, 2, 4, 6, 7, 9, 11),
    "mixolydian": (0, 2, 4, 5, 7, 9, 10),
    "phrygian": (0, 1, 3, 5, 7, 8, 10),
    "in_sen": (0, 1, 5, 7, 10),
    "iwato": (0, 1, 5, 6, 10),
    "egyptian": (0, 2, 5, 7, 10),
    "double_harmonic": (0, 1, 4, 5, 7, 8, 11),
    "hungarian_minor": (0, 2, 3, 6, 7, 8, 11),
    "prometheus": (0, 2, 4, 6, 9, 10),
    "augmented": (0, 3, 4, 7, 8, 11),
    "octatonic": (0, 2, 3, 5, 6, 8, 9, 11),
}
SCALE_NAMES = tuple(SCALES)
# A scale of n notes has n modes; the seed picks one.
MODE_COUNT_MAX = max(len(s) for s in SCALES.values())
DEGREES_PER_SCALE_STEP_C = 3.0  # each 3 C of GDDR spread moves one scale along

# name -> (partial ratios, amplitudes). No partial after the first is a whole multiple.
GONG_TUNINGS = {
    "bell": ((1.0, 2.76, 5.40, 8.45), (1.0, 0.6, 0.35, 0.2)),
    "tamtam": ((1.0, 1.47, 2.31, 3.64, 5.57), (1.0, 0.7, 0.5, 0.35, 0.2)),
    "singing_bowl": ((1.0, 2.71, 5.18), (1.0, 0.5, 0.25)),
    # Added after the first three (same reason as the scales). Hand-picked inharmonic sets in the
    # manner of each object, not measurements. At most five partials, the gong kernel's slot count.
    "plate": ((1.0, 1.59, 2.27, 2.65, 3.38), (1.0, 0.8, 0.55, 0.4, 0.25)),
    "gamelan": ((1.0, 2.33, 3.61, 5.47), (1.0, 0.65, 0.4, 0.22)),
    "glass": ((1.0, 2.62, 4.55, 6.72), (1.0, 0.45, 0.3, 0.12)),
    "cymbal": ((1.0, 1.35, 1.83, 2.57, 3.41), (1.0, 0.85, 0.65, 0.45, 0.3)),
    "tubular": ((1.0, 2.32, 4.25, 6.63), (1.0, 0.55, 0.3, 0.15)),
}
GONG_TUNING_NAMES = tuple(GONG_TUNINGS)

# ttrax's bar count: Tetrax, Sidrax (the guide this code was built from) and Quatrax, after the real
# instruments' own names. Sidrax is first because it was ttrax's only size before this field existed,
# so an old saved patch with no rax_bars defaults to it.
RAX_BAR_COUNTS = (4, 7, 14)

# What the shift register of ttsteam (Man With The Red Steam) and its noise are patched into. Each is a route
# from ttsteam's register to another device; a patch lists the ones that are on. The default patch has none.
#   strikes  the register's hits join the clock pulses the gong's trigger response waits on, so the gong, the
#            shimmer's snap, the drone's ripple and the AVDog all follow them
#   avdog    the register's hits also ping the AVDog's filter directly, a snappy call and response on the snare
#   accent   the register accents the ttrollz clock pulses it lands on
#   pitch    the register holds one note of the patch's scale per step, drawn from the seed, and steers the
#            AVDog's dogtone with them
STEAM_ROUTES = ("strikes", "avdog", "accent", "pitch")
STEAM_ROUTE_COUNT_ODDS = (0.25, 0.5, 0.25)  # a drawn patch has no route, one or two (the snare was everywhere when each was on 60% of the time)
STEAM_ROUTE_WEIGHTS = (0.5, 1.0, 1.0, 1.0)  # in STEAM_ROUTES order: "strikes" makes the whole band follow the snare, so it is drawn half as often

MOTIF_LENGTHS = (6, 7, 8, 9, 10, 12)  # eighth notes; the lead's bar is as long as its motif
MOTIF_LOW_RANGE = (-5, -2)  # the lowest degree a motif may reach is drawn from here ...
MOTIF_HIGH_RANGE = (4, 8)   # ... and the highest from here (scale degrees)
GONG_STRETCH = 0.015  # partials after the first move by up to this fraction ...
GONG_AMP_JITTER = 0.15  # ... and their levels by up to this fraction


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
    rax_bars: int  # ttrax's bar count: 4 (Tetrax), 7 (Sidrax) or 14 (Quatrax)
    steam_routes: Tuple[str, ...] = ()  # which of STEAM_ROUTES are on
    style: str = ""  # the name of the style (styles.py) this patch was drawn in, or "" for a plain draw
    emphasis: Tuple[Tuple[str, float], ...] = ()  # (device name, factor): its level in the mix and how often it gets the floor
    mode: int = 0  # `scale` is the named scale started from this one of its notes (0 = as listed)


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
    rax_bars=7,
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


def rotate_mode(scale: Sequence[int], mode: int) -> Tuple[int, ...]:
    """`scale` started from its `mode`-th note, as semitones above that note (mode 0 is the scale itself)."""
    n = len(scale)
    notes = [scale[(mode + i) % n] + 12 * ((mode + i) // n) for i in range(n)]
    return tuple(x - notes[0] for x in notes)


MOTIF_STEP_ODDS = (1 / 20, 3 / 20, 5 / 20, 2 / 20, 5 / 20, 3 / 20, 1 / 20)  # for steps of -3 to +3 degrees


def _draw_motif(rng: np.random.Generator, lengths: Sequence[int] = MOTIF_LENGTHS, step_odds: Sequence[float] = MOTIF_STEP_ODDS,
                low: Tuple[int, int] = MOTIF_LOW_RANGE, high: Tuple[int, int] = MOTIF_HIGH_RANGE) -> Tuple[int, ...]:
    """A motif of scale degrees from 0, steps of at most 3 (by default mostly 1 or 2), inside a drawn range, with at least 3
    different notes. A style passes its own lengths, odds for each step size and range."""
    lo = int(rng.integers(low[0], low[1] + 1))
    hi = int(rng.integers(high[0], high[1] + 1))
    length = int(rng.choice(list(lengths)))
    while True:
        motif = [0]
        for _ in range(length - 1):
            step = int(rng.choice([-3, -2, -1, 0, 1, 2, 3], p=list(step_odds)))
            if not lo <= motif[-1] + step <= hi:
                step = -step
            motif.append(motif[-1] + step)
        if len(set(motif)) >= 3:
            return tuple(motif)


def _draw_steam_routes(seed: int) -> Tuple[str, ...]:
    """No route, one or two of STEAM_ROUTES (see STEAM_ROUTE_COUNT_ODDS), in their listed order. It has its own random stream, so
    changing the routes did not move any other drawn field for an existing seed."""
    rng = np.random.default_rng([seed, 0x57EA])
    count = int(rng.choice(len(STEAM_ROUTE_COUNT_ODDS), p=STEAM_ROUTE_COUNT_ODDS))
    weights = np.asarray(STEAM_ROUTE_WEIGHTS, dtype=float)
    picked = set(int(i) for i in rng.choice(len(STEAM_ROUTES), size=count, replace=False, p=weights / weights.sum()))
    return tuple(r for i, r in enumerate(STEAM_ROUTES) if i in picked)


def _draw_gong(rng: np.random.Generator, tuning: str) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """The tuning's partials stretched or squeezed a little, and each level nudged, so no two draws ring alike."""
    partials, amps = GONG_TUNINGS[tuning]
    stretch = 1.0 + float(rng.uniform(-GONG_STRETCH, GONG_STRETCH))
    jitter = rng.uniform(1.0 - GONG_AMP_JITTER, 1.0 + GONG_AMP_JITTER, size=len(amps))
    new_partials = (1.0,) + tuple(round(r * stretch, 3) for r in partials[1:])
    new_amps = (1.0,) + tuple(round(float(a * j), 3) for a, j in zip(amps[1:], jitter[1:]))
    return new_partials, new_amps


def draw_patch(seed: int, telemetry: Sequence[Telemetry]) -> Patch:
    rng = np.random.default_rng(seed)
    scale_name = scale_for(telemetry)
    tuning = gong_tuning_for(telemetry)
    motif = _draw_motif(rng)
    mode = int(rng.integers(0, len(SCALES[scale_name])))
    partials, amps = _draw_gong(rng, tuning)
    lo = float(rng.uniform(8000, 15000))
    width = float(rng.uniform(3000, 9000))
    return Patch(
        seed=seed,
        scale_name=scale_name,
        scale=rotate_mode(SCALES[scale_name], mode),
        motif=motif,
        gong_tuning=tuning,
        gong_partials=partials,
        gong_amps=amps,
        gong_beat_hz=round(float(rng.uniform(0.25, 2.0)), 3),
        shimmer_band=(round(lo, 1), round(min(lo + width, 20500.0), 1)),
        bars_per_turn=int(rng.integers(1, 7)),
        floor=round(float(rng.uniform(0.2, 0.5)), 3),
        rax_bars=int(rng.choice(RAX_BAR_COUNTS)),
        mode=mode,
        steam_routes=_draw_steam_routes(seed),
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
    if not 0 <= p.mode < len(p.scale):
        raise ValueError(f"mode must be between 0 and {len(p.scale) - 1} for a scale of {len(p.scale)} notes")
    if any(not (isinstance(v, float) and v > 0 and v == v) for _, v in p.emphasis):  # positive, and not NaN
        raise ValueError("emphasis needs a positive factor for every device it names")
    unknown = [r for r in p.steam_routes if r not in STEAM_ROUTES]
    if unknown:
        raise ValueError(f"unknown steam route {', '.join(unknown)}; the routes are {', '.join(STEAM_ROUTES)}")
    if p.rax_bars not in RAX_BAR_COUNTS:
        raise ValueError(f"rax_bars must be one of {RAX_BAR_COUNTS}")
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
            # a patch saved before this field existed has no rax_bars; it played as the Sidrax (7 bars)
            rax_bars=int(d.get("rax_bars", 7)),
            # a patch saved before modes existed played its scale as listed
            mode=int(d.get("mode", 0)),
            # a patch saved before routes existed has none
            steam_routes=tuple(str(x) for x in d.get("steam_routes", ())),
            # a patch saved before styles existed has none
            style=str(d.get("style", "")),
            emphasis=tuple((str(name), float(factor)) for name, factor in d.get("emphasis", ())),
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
