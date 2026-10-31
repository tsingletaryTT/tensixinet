# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""A screen of the TUI as plain data: rows of styled text spans, and the escape codes that set a span's style.

A `Frame` is what `layout.render` makes and what the driver (`tui/app.py`) writes to the terminal, through `diff`. It holds no escape codes:
`sgr` turns a `Style` into one when, and only when, the terminal is coloured. Keeping the two apart lets the tests read a frame's
text and styles directly, and lets the driver compare two frames cell by cell to redraw only what changed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from ..narrate.color import DEVICE_JACKS, ESC, JACK_HEX, _cube, _rgb

# How much of a jack's brightness each dim level keeps. Level 0 is full brightness; levels 1 to 3 follow the mix's own PRESENCE
# table (music/evict.py) through `dim_level`.
DIM_SCALE = (1.0, 0.7, 0.45, 0.25)


@dataclass(frozen=True)
class Style:
    jack: Optional[str] = None  # a jack colour from narrate/color.JACK_HEX, or a device name (it takes that device's jack)
    bold: bool = False
    dim: int = 0  # 0 (full) to 3 (faintest)


@dataclass(frozen=True)
class Span:
    text: str
    style: Style = Style()


Row = Tuple[Span, ...]


@dataclass(frozen=True)
class Frame:
    rows: Tuple[Row, ...]
    width: int
    height: int

    def text(self) -> List[str]:
        """The plain text of each row that was drawn."""
        return ["".join(span.text for span in row) for row in self.rows]

    def cells(self) -> List[List[Tuple[str, Style]]]:
        """One (character, style) per cell: every row padded with blanks to `width`, and blank rows added up to `height`."""
        out: List[List[Tuple[str, Style]]] = []
        for row in self.rows[:self.height]:
            cells = [(ch, span.style) for span in row for ch in span.text][:self.width]
            cells += [(" ", Style())] * (self.width - len(cells))
            out.append(cells)
        while len(out) < self.height:
            out.append([(" ", Style())] * self.width)
        return out

    def find(self, text: str) -> List[Tuple[int, int, Style]]:
        """The row, the starting column and the style of every span whose text contains `text`."""
        hits = []
        for r, row in enumerate(self.rows):
            col = 0
            for span in row:
                if text in span.text:
                    hits.append((r, col, span.style))
                col += len(span.text)
        return hits


def dim_level(presence: float) -> int:
    """The dim level for a chip's share of the turns (music/evict.PRESENCE): running 0, filling 1, evicting 2, idle 3."""
    if presence >= 0.9:
        return 0
    if presence >= 0.5:
        return 1
    if presence >= 0.2:
        return 2
    return 3


def _jack_rgb(jack: str) -> Tuple[int, int, int]:
    """A jack's colour; a device name takes its device's jack, and anything unknown is grey."""
    if jack not in JACK_HEX:
        jack = DEVICE_JACKS.get(jack, "grey")
    return _rgb(jack)


def sgr(style: Style, colour: bool, truecolor: bool) -> str:
    """The escape that sets `style`: a reset first, then bold, then the jack's colour scaled to the dim level.
    Nothing at all when the terminal is not coloured."""
    if not colour:
        return ""
    reset = f"{ESC}0m"
    if style.jack is None:
        return reset + (f"{ESC}1m" if style.bold else "")
    scale = DIM_SCALE[max(0, min(len(DIM_SCALE) - 1, style.dim))]
    rgb = tuple(round(c * scale) for c in _jack_rgb(style.jack))
    code = f"38;2;{rgb[0]};{rgb[1]};{rgb[2]}" if truecolor else f"38;5;{_cube(rgb)}"
    return f"{reset}{ESC}{'1;' if style.bold else ''}{code}m"


def fit(text: str, width: int) -> str:
    """`text` cut to `width` characters. When it is cut, the last character kept becomes an ellipsis."""
    if width <= 0:
        return ""
    if len(text) <= width:
        return text
    return text[:width - 1] + "…"


Cells = List[List[Tuple[str, Style]]]


def _span_text(cells: List[Tuple[str, Style]], colour: bool, truecolor: bool) -> str:
    """The characters of a run of cells, with the style's escape written before the first cell and again wherever it changes."""
    out: List[str] = []
    current: Optional[Style] = None
    for ch, style in cells:
        if current is None or style != current:
            out.append(sgr(style, colour, truecolor))
            current = style
        out.append(ch)
    return "".join(out)


def diff(old: Optional[Cells], new: Cells, colour: bool, truecolor: bool) -> str:
    """The text that turns a screen showing `old` into one showing `new`.

    With no `old`, or one of a different size, the screen is cleared (`ESC[2J`) and every row of `new` is drawn from its first
    column. Otherwise only the cells that changed are written: each run of changed cells in a row is a cursor move to the run's
    first cell (`ESC[row;colH`, counted from 1), its style and its characters. A final reset follows whenever anything was
    written, so the terminal is never left in a frame's colour. Nothing changed gives the empty string."""
    out: List[str] = []
    same_size = old is not None and len(old) == len(new) and all(len(a) == len(b) for a, b in zip(old, new))
    if not same_size:
        out.append(f"{ESC}2J")
        for r, row in enumerate(new):
            out.append(f"{ESC}{r + 1};1H")
            out.append(_span_text(row, colour, truecolor))
    else:
        for r, (before, after) in enumerate(zip(old, new)):
            c = 0
            while c < len(after):
                if after[c] == before[c]:
                    c += 1
                    continue
                start = c
                while c < len(after) and after[c] != before[c]:
                    c += 1
                out.append(f"{ESC}{r + 1};{start + 1}H")
                out.append(_span_text(after[start:c], colour, truecolor))
    if not out:
        return ""
    out.append(f"{ESC}0m")
    return "".join(out)
