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

import dataclasses
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, get_origin, get_type_hints

import numpy as np

from .patch import (
    GONG_TUNING_NAMES, RAX_BAR_COUNTS, SCALES, STEAM_ROUTES, Patch, _draw_choir, _draw_gong, _draw_motif, draw_patch,
    gddr_spread, memory_to_asic_ratio, rotate_mode, DEGREES_PER_SCALE_STEP_C,
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
    snare: float  # the share of ttsteam's snare hits that play (patch.snare)
    bus: Tuple[float, float, float]  # (patch.coco_send, patch.coco_reverb, patch.reverb_s)
    rests: float  # the share of the coco's notes that rest (patch.rests)
    pulse_rests: Tuple[float, float]  # (patch.lead_rests, patch.roll_rests)
    mod: float  # how far the looping line's pitch is moved (patch.quantus_mod)
    choir_mix: Tuple[float, float, float] = (1.0, 1.0, 1.0)  # the choir's consonant families (plosive, fricative, nasal) (patch.choir_mix)
    choir_crush: float = 0.0  # the choir's speech-chip crush at full pressure (patch.choir_crush)
    # A style saved from what is playing can pin the motif and the mode of its scale. The built-in styles leave both alone, so
    # their seed still draws a fresh motif and (in "any" mode) a fresh starting note.
    motifs: Tuple[Tuple[int, ...], ...] = ()  # exact motifs to draw from, when set; each starts on the root
    mode: Optional[int] = None  # the mode to play the scale from, when set; otherwise the style's `modes` rule decides


DEFAULT_STYLE = "chiptune"   # the style a run and the app use when none is named; "plain" (the CLI) or "plain draw" (the app) opts out

# JSON has no tuple type: a saved tuple comes back as a list. Restore every field declared as a Tuple (including nested ones like
# `emphasis` and `motifs`) so a loaded style equals the style that was saved and can be drawn the same way.
_TUPLE_FIELDS = frozenset(
    f.name for f in dataclasses.fields(Style)
    if get_origin(get_type_hints(Style)[f.name]) is tuple
)


def _tuples(value):
    """Lists become tuples recursively, so a nested JSON array matches the dataclass field it came from."""
    return tuple(_tuples(v) for v in value) if isinstance(value, list) else value


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
        routes=("accent", "avdog"), rax_bars=(4,),
        emphasis=(("lead", 1.5), ("ttquantus", 1.6), ("ttrollz", 1.3), ("ttsteam", 0.9), ("ttavdog", 0.9), ("ttrax", 1.0),
                  ("ultrasonic_shimmer", 0.5), ("gong", 1.0), ("ttcoco", 0.6), ("ttdrone", 0.4), ("ttchoir", 0.2)),
        snare=0.5,
        bus=(0.4, 0.1, 0.6),
        rests=0.25,
        pulse_rests=(0.1, 0.15),
        mod=0.15,
        choir_mix=(0.5, 0.3, 0.2),
        choir_crush=1.0,
    ),
    "enya": Style(
        "enya",
        "Slow, modal and spacious: long turns, singing bowls and glass, the choir and the lead in front, the drone and a modest room "
        "of delay behind them, and hardly any snare.",
        scales=("dorian", "lydian", "major_pentatonic", "mixolydian"), modes="any",
        gong_tunings=("singing_bowl", "glass", "bell"), beat=(0.25, 0.5), band_low=(8000.0, 11000.0), band_width=(6000.0, 9000.0),
        bars=(4, 5, 6), floor=(0.45, 0.5), motif_lengths=(12,), motif_steps=_odds(0.2, 1, 6, 2, 6, 1, 0.2),
        routes=("pitch",), rax_bars=(7, 14),
        emphasis=(("ttchoir", 1.8), ("ttdrone", 0.9), ("ultrasonic_shimmer", 1.0), ("ttrax", 1.2), ("lead", 1.3),
                  ("gong", 1.0), ("ttcoco", 0.3), ("ttquantus", 0.3), ("ttavdog", 0.4), ("ttrollz", 0.3), ("ttsteam", 0.15)),
        snare=0.1,
        bus=(0.35, 0.15, 1.6),
        rests=0.3,
        pulse_rests=(0.2, 0.2),
        mod=0.8,
        choir_mix=(0.1, 0.3, 0.6),
        choir_crush=0.0,
    ),
    "blues": Style(
        "blues",
        "A twelve-note phrase over the blues scales, four bars to a turn so three turns make twelve bars, the ttrax organ and the lead "
        "up front, and the snare on the back beat.",
        scales=("blues", "mixolydian", "minor_pentatonic", "dorian"), modes="root",
        gong_tunings=("tamtam", "plate", "tubular"), beat=(0.5, 1.0), band_low=(9000.0, 12000.0), band_width=(4000.0, 6000.0),
        bars=(4,), floor=(0.3, 0.4), motif_lengths=(12,), motif_steps=_odds(0.5, 2, 5, 3, 5, 2, 0.5),
        routes=("accent",), rax_bars=(7,),
        emphasis=(("lead", 1.5), ("ttrax", 1.4), ("ttrollz", 1.0), ("ttsteam", 0.7), ("ttavdog", 0.8), ("ttcoco", 0.8),
                  ("ttdrone", 0.7), ("gong", 1.0), ("ttquantus", 0.5), ("ultrasonic_shimmer", 0.4), ("ttchoir", 0.3)),
        snare=0.5,
        bus=(0.3, 0.3, 1.4),
        rests=0.35,
        pulse_rests=(0.25, 0.2),
        mod=0.45,
        choir_mix=(0.7, 0.15, 0.15),
        choir_crush=0.3,
    ),
    "aphex": Style(
        "aphex",
        "Restless and odd-metered: eight-note scales, motifs of seven, nine or ten notes with big leaps, short turns, fast metallic "
        "gongs, two steam routes on, and the rolls, the AVDog and the delay taps doing the talking with the snare in the mix.",
        scales=("octatonic", "augmented", "whole_tone", "hungarian_minor", "double_harmonic", "prometheus"), modes="any",
        gong_tunings=("glass", "cymbal", "plate", "gamelan"), beat=(1.5, 2.0), band_low=(12000.0, 15000.0), band_width=(5000.0, 9000.0),
        bars=(1, 2), floor=(0.2, 0.3), motif_lengths=(7, 9, 10, 12), motif_steps=_odds(2, 3, 3, 1, 3, 3, 2),
        routes=("avdog", "pitch"), rax_bars=(14,),
        emphasis=(("ttrollz", 1.5), ("ttsteam", 1.0), ("ttavdog", 1.3), ("ttcoco", 1.3), ("ttquantus", 1.1), ("ultrasonic_shimmer", 1.0),
                  ("lead", 0.9), ("gong", 1.3), ("ttrax", 0.8), ("ttchoir", 0.7), ("ttdrone", 0.6)),
        snare=0.6,
        bus=(0.7, 0.25, 1.0),
        rests=0.4,
        pulse_rests=(0.15, 0.3),
        mod=0.9,
        choir_mix=(0.4, 0.4, 0.2),
        choir_crush=0.8,
    ),
}


def draw_styled_patch(seed: int, telemetry: Sequence[Telemetry], style: str,
                      styles: Optional[Dict[str, Style]] = None) -> Patch:
    """A patch drawn inside `style`. Everything a plain draw decides that the style does not touch (the cables, which come from the
    seed) is as it would have been. `styles` lets a caller draw from saved style files as well as the built-in table."""
    table = STYLES if styles is None else styles
    if style not in table:
        raise ValueError(f"unknown style {style}; the styles are {', '.join(table)}")
    s = table[style]
    base = draw_patch(seed, telemetry)
    rng = np.random.default_rng([seed, STYLE_STREAM])
    scale_name = s.scales[int(gddr_spread(telemetry) // DEGREES_PER_SCALE_STEP_C) % len(s.scales)]
    tuning = s.gong_tunings[int(round(memory_to_asic_ratio(telemetry) * 10)) % len(s.gong_tunings)]
    mode = s.mode if s.mode is not None else (int(rng.integers(0, len(SCALES[scale_name]))) if s.modes == "any" else 0)
    partials, amps = _draw_gong(rng, tuning)
    lo = float(rng.uniform(*s.band_low))
    width = float(rng.uniform(*s.band_width))
    return Patch(
        seed=seed,
        scale_name=scale_name,
        scale=rotate_mode(SCALES[scale_name], mode),
        motif=(tuple(s.motifs[int(rng.integers(0, len(s.motifs)))]) if s.motifs
               else _draw_motif(rng, s.motif_lengths, s.motif_steps)),
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
        snare=s.snare,
        coco_send=s.bus[0],
        coco_reverb=s.bus[1],
        reverb_s=s.bus[2],
        rests=s.rests,
        lead_rests=s.pulse_rests[0],
        roll_rests=s.pulse_rests[1],
        quantus_mod=s.mod,
        **{**_draw_choir(seed), "choir_mix": s.choir_mix, "choir_crush": s.choir_crush},
    )


# --- writing styles ---------------------------------------------------------------------------------------------------------
# A style is a plain JSON object with the fields of the Style dataclass. `library/styles/*.json` files are read as styles by the
# CLI, so a listener can copy a saved snapshot, edit a few numbers and name a new style without touching the source.

def _validate_style(s: Style) -> Style:
    """Refuse a style that cannot draw a playable patch. Each check names the field, so a hand-written file says what to fix."""
    from .chorus import device_names  # here: styles.py is imported by chorus.py at module load

    def bad(field: str, why: str) -> None:
        raise ValueError(f"style {s.name or '(unnamed)'}: {field} {why}")

    def unit(field: str, value, low: float = 0.0, high: float = 1.0, low_open: bool = False,
             high_open: bool = False) -> None:
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) \
                or not ((low < value if low_open else low <= value) and (value < high if high_open else value <= high)):
            bad(field, f"must be between {low} and {high}"
                + (" (not including the lower bound)" if low_open else "")
                + (" (not including the upper bound)" if high_open else ""))

    if not s.name:
        bad("name", "cannot be empty")
    if not s.blurb:
        bad("blurb", "needs a sentence saying what the style sounds like")
    if not s.scales or any(x not in SCALES for x in s.scales):
        bad("scales", f"needs at least one of {', '.join(SCALES)}")
    if s.modes not in ("root", "any"):
        bad("modes", "must be root or any")
    if not s.gong_tunings or any(x not in GONG_TUNING_NAMES for x in s.gong_tunings):
        bad("gong_tunings", f"needs at least one of {', '.join(GONG_TUNING_NAMES)}")
    if not (0 <= s.beat[0] <= s.beat[1]) or not math.isfinite(s.beat[1]):
        bad("beat", "needs a low and a high beat, both at least 0")
    if not (0 < s.band_low[0] <= s.band_low[1] <= 20500) or not (0 < s.band_width[0] <= s.band_width[1]):
        bad("band_low or band_width", "needs positive, ordered ranges the shimmer can reach")
    if not s.bars or any(not isinstance(x, int) or isinstance(x, bool) or x < 1 for x in s.bars):
        bad("bars", "needs at least one positive number of bars")
    if not (0 < s.floor[0] <= s.floor[1] <= 1):
        bad("floor", "needs an ordered range above 0 and at most 1")
    if not s.motif_lengths or any(not isinstance(x, int) or isinstance(x, bool) or x < 1 for x in s.motif_lengths):
        bad("motif_lengths", "needs at least one positive note count")
    if len(s.motif_steps) != 7 or any(not math.isfinite(w) or w <= 0 for w in s.motif_steps):
        bad("motif_steps", "needs seven positive odds, for steps of -3 to +3 degrees")
    if any(r not in STEAM_ROUTES for r in s.routes):
        bad("routes", f"may only name {', '.join(STEAM_ROUTES)}")
    if not s.rax_bars or any(x not in RAX_BAR_COUNTS for x in s.rax_bars):
        bad("rax_bars", f"needs at least one of {RAX_BAR_COUNTS}")
    if any(name not in device_names() or not math.isfinite(v) or v <= 0 for name, v in s.emphasis):
        bad("emphasis", "needs known devices and positive factors")
    unit("snare", s.snare)
    unit("coco_send", s.bus[0])
    unit("coco_reverb", s.bus[1])
    if not math.isfinite(s.bus[2]) or s.bus[2] <= 0:
        bad("bus", "needs a positive room time in seconds")
    unit("rests", s.rests, high=1.0, high_open=True)
    unit("lead_rests", s.pulse_rests[0], high=1.0, high_open=True)
    unit("roll_rests", s.pulse_rests[1], high=1.0, high_open=True)
    unit("mod", s.mod)
    if len(s.choir_mix) != 3 or any(not math.isfinite(w) or w < 0 for w in s.choir_mix) or not sum(s.choir_mix) > 0:
        bad("choir_mix", "needs three finite, non-negative weights with a positive sum")
    unit("choir_crush", s.choir_crush)
    if any(not m or m[0] != 0 for m in s.motifs):
        bad("motifs", "needs motifs that start on the root note")
    if s.mode is not None and (not isinstance(s.mode, int) or not 0 <= s.mode < len(SCALES[s.scales[0]])):
        bad("mode", f"must be empty or from 0 to {len(SCALES[s.scales[0]]) - 1} for the first scale")
    return s


def style_from_dict(d: dict, name: Optional[str] = None) -> Style:
    """A style from a JSON object. `name` (when given) overrides the one in the object, for files that leave it out."""
    if not isinstance(d, dict):
        raise ValueError("a style file must hold a JSON object")
    fields = {f.name for f in dataclasses.fields(Style)}
    unknown = sorted(set(d) - fields)
    if unknown:
        raise ValueError(f"unknown style fields: {', '.join(unknown)}")
    d = {k: _tuples(v) if k in _TUPLE_FIELDS else v for k, v in d.items()}
    try:
        s = Style(**{**d, **({"name": name} if name is not None else {})})
    except TypeError as e:
        raise ValueError(f"malformed style: {e}") from e
    return _validate_style(s)


def style_to_dict(style: Style) -> dict:
    """The style as a JSON-ready object, one field per line when written by `save_style`."""
    return dataclasses.asdict(style)


def save_style(style: Style, path) -> None:
    """Write a style as readable JSON, in the same one-field-per-line shape as a saved patch."""
    fields = [f"  {json.dumps(k)}: {json.dumps(v)}" for k, v in style_to_dict(style).items()]
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{\n" + ",\n".join(fields) + "\n}\n")


def load_style(path) -> Style:
    """Read a style file, naming the file in any error. A file without a `name` takes its name from the file stem."""
    p = Path(path)
    try:
        d = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise ValueError(f"cannot read style {p}: {e}") from e
    try:
        return style_from_dict({"name": p.stem, **d} if isinstance(d, dict) else d)
    except ValueError as e:
        raise ValueError(f"{p}: {e}") from e


def style_from_patch(patch: Patch, name: str, blurb: str = "") -> Style:
    """A style that holds what was playing: its scale, motif, tuning, beat, band, balance and every drawn level.

    The gong's own stretch and the choir's consonant sources still come from the seed when this style is drawn: a style is a
    sketch of a sound, not a second copy of the patch file. Save the patch as well if you want the exact same notes."""
    lo, hi = patch.shimmer_band
    return Style(
        name=name,
        blurb=blurb or f"What was playing when this style was saved: seed {patch.seed}, held as a sketch a new draw can vary.",
        scales=(patch.scale_name,), modes="root", gong_tunings=(patch.gong_tuning,),
        beat=(patch.gong_beat_hz, patch.gong_beat_hz), band_low=(lo, lo), band_width=(hi - lo, hi - lo),
        bars=(patch.bars_per_turn,), floor=(patch.floor, patch.floor),
        motif_lengths=(len(patch.motif),), motif_steps=_odds(1, 1, 1, 1, 1, 1, 1),
        routes=patch.steam_routes, rax_bars=(patch.rax_bars,), emphasis=patch.emphasis, snare=patch.snare,
        bus=(patch.coco_send, patch.coco_reverb, patch.reverb_s), rests=patch.rests,
        pulse_rests=(patch.lead_rests, patch.roll_rests), mod=patch.quantus_mod,
        choir_mix=patch.choir_mix, choir_crush=patch.choir_crush,
        motifs=(patch.motif,), mode=patch.mode,
    )
