# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Colour for the run log, from the Ciat-Lonbarde jack colours.

The Plumbutter guide colours its banana jacks by the kind of signal they carry: blue modulates the main controlling parameters,
green receives a trigger, purple modulates the innards of a circuit, white is line-level audio, orange is slower control
voltages, red is triggering pulses, yellow is modular-level audio, brown is pulses and grey is dust. Here each kind of log line
takes the colour of the jack that fits it (readings are blue, the turns that trade the floor are green, patch cables are
purple, audio is white), every device keeps one colour, and every chip keeps another, so a line can be read by its colours.

Colour goes to the screen only. The `.txt` log, the JSONL and anything that is not a terminal stay plain, `NO_COLOR` is
respected, and `--color always` or `FORCE_COLOR` turns it on for a recording that is not a terminal. Art has a left edge and a
bottom, never a right border: a right border breaks when the pane is narrower than expected.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List, Mapping, Optional, TextIO

ESC = "\x1b["

# The nine jacks of the guide, then two that are ours: Tenstorrent's teal and a pink for the choir.
JACK_HEX = {
    "blue": "4EA8F0",
    "green": "5FD38D",
    "purple": "B98CFF",
    "white": "E9F1F0",
    "orange": "FF9A3D",
    "red": "FF6B6B",
    "yellow": "F6D04A",
    "brown": "C58B5A",
    "grey": "8C9A9B",
    "teal": "4FD1C5",
    "pink": "FF8FC8",
}
JACK_MEANING = {  # the guide's own words
    "blue": "modulates the main controlling parameters",
    "green": "receives a trigger signal to activate the circuit",
    "purple": "modulates the innards of the circuit",
    "white": "is line level audio",
    "orange": "is slower rate control voltages",
    "red": "is triggering pulses, CV or audio",
    "yellow": "is modular level audio",
    "brown": "is pulses",
    "grey": "is dust",
}
LOG_MEANING = {  # what each colour stands for in this log
    "blue": "readings", "green": "who plays", "purple": "patch + cables", "white": "audio", "orange": "discovery",
    "red": "events", "yellow": "output", "brown": "the LLM", "grey": "dust",
}

EVENT_JACKS = {
    "discover": "orange", "chip": "blue", "lead": "blue", "motion": "grey", "patch": "purple", "cable": "purple",
    "events": "red", "mix": "white", "render": "white", "eq": "white", "loudness": "white", "only": "white",
    "weight": "green", "turn": "green", "live": "green", "backend": "teal", "llm": "brown", "player": "yellow", "done": "yellow",
}
DEVICE_JACKS = {
    "ttrollz": "brown", "ttrax": "yellow", "ttquantus": "purple", "ttdrone": "blue", "lead": "white", "gong": "orange",
    "ultrasonic_shimmer": "teal", "ttsteam": "red", "ttavdog": "green", "ttchoir": "pink", "ttcoco": "purple",
}
CHIP_JACKS = ("teal", "orange", "purple", "green", "blue", "red", "yellow", "pink")


def _rgb(name: str):
    h = JACK_HEX[name]
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _cube(rgb) -> int:
    """The nearest colour of the 6 x 6 x 6 cube of a 256-colour terminal."""
    r, g, b = (round(c / 255 * 5) for c in rgb)
    return 16 + 36 * r + 6 * g + b


@dataclass(frozen=True)
class Palette:
    enabled: bool = False
    truecolor: bool = True

    def paint(self, text: str, jack: str, bold: bool = False) -> str:
        if not self.enabled:
            return text
        colour = f"38;2;{';'.join(map(str, _rgb(jack)))}" if self.truecolor else f"38;5;{_cube(_rgb(jack))}"
        return f"{ESC}{'1;' if bold else ''}{colour}m{text}{ESC}0m"

    def dim(self, text: str) -> str:
        return self.paint(text, "grey")


def detect(stream: Optional[TextIO], mode: str = "auto", env: Optional[Mapping[str, str]] = None) -> Palette:
    """Whether to colour `stream`: 'always' and 'never' decide outright; 'auto' colours a terminal, unless NO_COLOR is set
    (https://no-color.org) or TERM is dumb, and FORCE_COLOR colours a stream that is not a terminal."""
    env = os.environ if env is None else env
    truecolor = env.get("COLORTERM", "").lower() in ("truecolor", "24bit")
    if mode == "never":
        return Palette(False, truecolor)
    if mode == "always":
        return Palette(True, truecolor)
    if env.get("NO_COLOR"):
        return Palette(False, truecolor)
    if env.get("FORCE_COLOR", "0") not in ("", "0"):
        return Palette(True, truecolor)
    tty = bool(stream is not None and getattr(stream, "isatty", lambda: False)())
    return Palette(tty and env.get("TERM", "") != "dumb", truecolor)


def strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


_DEVICES = "|".join(re.escape(n) for n in sorted(DEVICE_JACKS, key=len, reverse=True))
_TOKEN = re.compile(
    rf"(?P<device>(?<!\w)(?:{_DEVICES})(?!\w))"
    r"|(?P<chip>(?<!\w)chip (?P<n>\d+)(?!\w))"
    r"|(?P<number>(?<![\w.])\d+(?:\.\d+)?\s?(?:kHz|MHz|GHz|Hz|BPM|bars?|dB|ms|%|C|W|V|s)(?![\w]))"
)


def decorate(line: str, palette: Palette) -> str:
    """Colour the devices (each its own jack), the chips (each its own) and the numbers with units in `line`."""
    if not palette.enabled:
        return line

    def paint(m):
        if m.group("device"):
            return palette.paint(m.group("device"), DEVICE_JACKS[m.group("device")], bold=True)
        if m.group("chip"):
            return palette.paint(m.group("chip"), CHIP_JACKS[int(m.group("n")) % len(CHIP_JACKS)], bold=True)
        return palette.paint(m.group("number"), "yellow", bold=True)

    return _TOKEN.sub(paint, line)


def bar(fraction: float, width: int) -> str:
    """A row of `width` blocks, filled in proportion to `fraction` (0 to 1)."""
    filled = max(0, min(width, round(fraction * width)))
    return "█" * filled + "░" * (width - filled)


# --- art: a left edge and a bottom, no right border ----------------------------------------------------------------------

GLYPHS = {  # a half-block face, two rows tall
    "T": ("▀█▀", " █ "), "E": ("█▀▀", "██▄"), "N": ("█▄ █", "█ ▀█"), "S": ("█▀▀", "▄▄█"),
    "I": ("█", "█"), "X": ("▀▄▀", "▄▀▄"),
}
WORD = "TENSIXINET"
ART_JACKS = ("blue", "green", "purple", "white", "orange", "red", "yellow", "brown", "teal", "pink")
TAGLINE = "chip telemetry, played as a chorus of Ciat-Lonbarde-style devices"
CREDIT = "inspired by Ciat-Lonbarde and Josh Singer's guides; not a recreation"
INDENT = "  "


def _art_width() -> int:
    return len(INDENT) + sum(len(GLYPHS[c][0]) for c in WORD) + len(WORD) - 1


def _wrap(text: str, width: int) -> List[str]:
    import textwrap
    return textwrap.wrap(text, width=max(10, width - len(INDENT)), break_on_hyphens=False, break_long_words=False)


def banner(width: int, palette: Palette) -> List[str]:
    """The opening art: TENSIXINET in half blocks, each letter on its own jack colour, a patch cable under it, and a
    tagline. A pane too narrow for the art gets the name and the tagline as text."""
    lines: List[str] = []
    if width >= _art_width() + 6:  # a little room beside it, or a pane just wide enough wraps the cable under it
        for row in range(2):
            parts = [palette.paint(GLYPHS[c][row], ART_JACKS[i], bold=True) for i, c in enumerate(WORD)]
            lines.append(INDENT + " ".join(parts))
        cable = "".join(palette.paint("●", ART_JACKS[i % len(ART_JACKS)]) + palette.paint("━━", ART_JACKS[(i + 1) % len(ART_JACKS)])
                        for i in range(8)) + palette.paint("●", ART_JACKS[8])
        lines.append(INDENT + cable)
    else:
        lines.append(INDENT + "".join(palette.paint(c, ART_JACKS[i], bold=True) for i, c in enumerate(WORD)))
    lines += [INDENT + palette.paint(t, "grey") for t in _wrap(TAGLINE, width)]
    lines += [INDENT + palette.paint(t, "grey") for t in _wrap(CREDIT, width)]
    return lines


def legend(width: int, palette: Palette) -> List[str]:
    """The jack colours and what each stands for in the log, packed into lines that fit `width`."""
    items = [(name, f"● {name} {LOG_MEANING[name]}") for name in ("blue", "green", "purple", "white", "orange", "red", "yellow", "brown", "grey")]
    lines, row, used = [], [], len(INDENT)
    for name, text in items:
        if row and used + 2 + len(text) > width:
            lines.append(INDENT + "  ".join(row))
            row, used = [], len(INDENT)
        row.append(palette.paint("●", name) + " " + palette.paint(f"{name} {LOG_MEANING[name]}", "grey"))
        used += len(text) + (2 if len(row) > 1 else 0)
    if row:
        lines.append(INDENT + "  ".join(row))
    return lines
