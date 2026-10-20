# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Styles: named patches for listeners who want a genre before they want a circuit.

A style narrows what a patch may draw (the scales, the gong tunings, the length and leaps of the lead's motif, how long a turn
is, how loose the band plays, which steam routes are on) and sets a balance for the band (`emphasis`: a factor for each device's
level and for how often it gets the floor). Inside those limits it is an ordinary patch. The system still decides which of the
style's scales and tunings from the chips' readings, the seed still draws the rest, and so no two runs of a style match.

The four styles are a sketch of a sound, not an imitation: the devices are the devices, and a style leans on the ones that
suit it (the 8-bit buffer of ttquantus for chiptune, the choir and the drone for enya, the organ of ttrax for blues, the rolls and
the snare for aphex).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np

from .patch import (
    GONG_TUNING_NAMES, SCALES, STEAM_ROUTES, Patch, _draw_gong, _draw_motif, draw_patch, gddr_spread, memory_to_asic_ratio,
    rotate_mode, DEGREES_PER_SCALE_STEP_C,
)
from ..telemetry import Telemetry

STYLE_STREAM = 0x57714E


@dataclass(frozen=True)
class Style:
    name: str
    blurb: str
    scales: Tuple[str, ...]  # the system picks one from the memory temperatures
    modes: str  # "root": play the scale as listed; "any": the seed may start it from any of its notes
    gong_tunings: Tuple[str, ...]  # the system picks one from the memory-to-ASIC temperature ratio
    beat: Tuple[float, float]  # the gong's beat in Hz
    band_low: Tuple[float, float]  # the shimmer band's low edge in Hz
    band_width: Tuple[float, float]
    bars: Tuple[int, ...]  # bars in a turn
    floor: Tuple[float, float]  # level of everyone who is not soloing
    motif_lengths: Tuple[int, ...]
    motif_steps: Tuple[float, ...]  # odds of a step of -3 to +3 degrees
    routes: Tuple[str, ...]  # steam routes that are on
    rax_bars: Tuple[int, ...]  # 4 (Tetrax), 7 (Sidrax) or 14 (Quatrax)
    emphasis: Tuple[Tuple[str, float], ...]


def _odds(*weights: float) -> Tuple[float, ...]:
    total = sum(weights)
    return tuple(w / total for w in weights)


STYLES: Dict[str, Style] = {
    "chiptune": Style(
        "chiptune",
        "Square-ish and quick: short turns, bright bells, a small four-barre ttrax, and the 8-bit ttquantus and the lead out in front "
        "with the rolls and the snare keeping it snappy.",
        scales=("major_pentatonic", "mixolydian", "minor_pentatonic", "dorian"), modes="root",
        gong_tunings=("bell", "glass"), beat=(0.3, 0.6), band_low=(12000.0, 14500.0), band_width=(3000.0, 4000.0),
        bars=(1, 2), floor=(0.4, 0.5), motif_lengths=(8, 12), motif_steps=_odds(0.5, 3, 4, 1, 4, 3, 0.5),
        routes=("strikes", "avdog", "accent"), rax_bars=(4,),
        emphasis=(("lead", 1.5), ("ttquantus", 1.6), ("ttrollz", 1.3), ("ttsteam", 1.2), ("ttavdog", 0.9), ("ttrax", 1.0),
                  ("ultrasonic_shimmer", 0.5), ("gong", 0.5), ("ttcoco", 0.6), ("ttdrone", 0.4), ("ttchoir", 0.2)),
    ),
    "enya": Style(
        "enya",
        "Slow, modal and spacious: long turns, singing bowls and glass, the choir and the drone in front, the delay taps filling the "
        "room, and hardly any snare.",
        scales=("dorian", "lydian", "major_pentatonic", "mixolydian"), modes="any",
        gong_tunings=("singing_bowl", "glass", "bell"), beat=(0.25, 0.5), band_low=(8000.0, 11000.0), band_width=(6000.0, 9000.0),
        bars=(4, 5, 6), floor=(0.45, 0.5), motif_lengths=(12,), motif_steps=_odds(0.2, 1, 6, 2, 6, 1, 0.2),
        routes=("pitch",), rax_bars=(7, 14),
        emphasis=(("ttchoir", 1.8), ("ttdrone", 1.5), ("ultrasonic_shimmer", 1.3), ("ttcoco", 1.2), ("ttrax", 1.2), ("lead", 1.0),
                  ("gong", 1.0), ("ttquantus", 0.7), ("ttavdog", 0.4), ("ttrollz", 0.3), ("ttsteam", 0.15)),
    ),
    "blues": Style(
        "blues",
        "A twelve-note phrase over the blues scales, four bars to a turn so three turns make twelve bars, the ttrax organ and the lead "
        "up front, and the snare on the back beat.",
        scales=("blues", "mixolydian", "minor_pentatonic", "dorian"), modes="root",
        gong_tunings=("tamtam", "plate", "tubular"), beat=(0.5, 1.0), band_low=(9000.0, 12000.0), band_width=(4000.0, 6000.0),
        bars=(4,), floor=(0.3, 0.4), motif_lengths=(12,), motif_steps=_odds(0.5, 2, 5, 3, 5, 2, 0.5),
        routes=("accent", "avdog"), rax_bars=(7,),
        emphasis=(("lead", 1.5), ("ttrax", 1.4), ("ttrollz", 1.0), ("ttsteam", 0.9), ("ttavdog", 0.8), ("ttcoco", 0.8),
                  ("ttdrone", 0.7), ("gong", 0.6), ("ttquantus", 0.5), ("ultrasonic_shimmer", 0.4), ("ttchoir", 0.3)),
    ),
    "aphex": Style(
        "aphex",
        "Restless and odd-metered: eight-note scales, motifs of seven, nine or ten notes with big leaps, short turns, fast metallic "
        "gongs, every steam route on, and the rolls, the snare, the AVDog and the delay taps doing the talking.",
        scales=("octatonic", "augmented", "whole_tone", "hungarian_minor", "double_harmonic", "prometheus"), modes="any",
        gong_tunings=("glass", "cymbal", "plate", "gamelan"), beat=(1.5, 2.0), band_low=(12000.0, 15000.0), band_width=(5000.0, 9000.0),
        bars=(1, 2), floor=(0.2, 0.3), motif_lengths=(7, 9, 10, 12), motif_steps=_odds(2, 3, 3, 1, 3, 3, 2),
        routes=STEAM_ROUTES, rax_bars=(14,),
        emphasis=(("ttrollz", 1.5), ("ttsteam", 1.5), ("ttavdog", 1.3), ("ttcoco", 1.3), ("ttquantus", 1.1), ("ultrasonic_shimmer", 1.0),
                  ("lead", 0.9), ("gong", 0.8), ("ttrax", 0.8), ("ttchoir", 0.7), ("ttdrone", 0.6)),
    ),
}


def draw_styled_patch(seed: int, telemetry: Sequence[Telemetry], style: str) -> Patch:
    """A patch drawn inside `style`. Everything a plain draw decides that the style does not touch (the cables, which come from the
    seed) is as it would have been."""
    if style not in STYLES:
        raise ValueError(f"unknown style {style}; the styles are {', '.join(STYLES)}")
    s = STYLES[style]
    base = draw_patch(seed, telemetry)
    rng = np.random.default_rng([seed, STYLE_STREAM])
    scale_name = s.scales[int(gddr_spread(telemetry) // DEGREES_PER_SCALE_STEP_C) % len(s.scales)]
    tuning = s.gong_tunings[int(round(memory_to_asic_ratio(telemetry) * 10)) % len(s.gong_tunings)]
    mode = int(rng.integers(0, len(SCALES[scale_name]))) if s.modes == "any" else 0
    partials, amps = _draw_gong(rng, tuning)
    lo = float(rng.uniform(*s.band_low))
    width = float(rng.uniform(*s.band_width))
    return Patch(
        seed=seed,
        scale_name=scale_name,
        scale=rotate_mode(SCALES[scale_name], mode),
        motif=_draw_motif(rng, s.motif_lengths, s.motif_steps),
        gong_tuning=tuning,
        gong_partials=partials,
        gong_amps=amps,
        gong_beat_hz=round(float(rng.uniform(*s.beat)), 3),
        shimmer_band=(round(lo, 1), round(min(lo + width, 20500.0), 1)),
        bars_per_turn=int(rng.choice(list(s.bars))),
        floor=round(float(rng.uniform(*s.floor)), 3),
        rax_bars=int(rng.choice(list(s.rax_bars))),
        mode=mode,
        steam_routes=tuple(r for r in STEAM_ROUTES if r in s.routes),
        style=style,
        emphasis=tuple((name, float(factor)) for name, factor in s.emphasis),
    )
