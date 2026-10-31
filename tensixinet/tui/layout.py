# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Lay a `FlowState` and the log pane's rows out as one `Frame`, the size of the terminal. The screen is built around the
instruments: the modules are grouped by the role they play, and a chip is one line of readings.

Top to bottom:

* the heading: which segment (or, in fast mode, which phrase) is playing, who has the floor, and the patch when there is room,
  with a dim `q quit` at the right end when it fits (after `q`, a notice that the run is stopping takes the heading's place);
* the readings strip: one line per chip, with its mark and index, its phase, temperature, power, pressure and the devices it
  drives, under a dim READINGS heading that names the columns when there is a spare row;
* the modules, grouped by role in three columns (RHYTHM, MELODY, AIR), with MOD/DELAY as a second row of groups under them, or
  as a fourth column on a screen at least WIDE_COLS wide. A device that is in no group goes in OTHER. Each module is one row:
  its name, its chip's mark and index, a bar for its share of the floor, and, on a wide screen, `faint` for a fading chip and
  the strongest cable that touches it;
* the strongest cables, at most CABLE_ROWS, each as `source ═kind .depth═▶ target`;
* the stereo line: a two-letter code for each device at its azimuth, left to right (two letters, so none reads as the `L` and
  `R` at the ends), and under it, when there is a spare row, a legend of the codes;
* the run log, newest at the bottom, in every row that is left.

Focus follows the signal. The module that has the floor is bold with a heavy bar `┃` and a `▶`; the readings line of the chip
that plays it is bold with `┃`; the cables that touch it are at full brightness and the others one step dimmer; its letter in
the stereo line is bold. Bold marks focus and nothing else: a cable is never bold. Every other line has the light bar `╎`, and
a chip that is evicting, idle or filling fades, with its modules, by the mix's own PRESENCE table. Without colour the
soloist's module row starts `▶[name]`, and the phase words and marks say the rest. The chip marks of a module column line up,
the soloist's row among them. No line ends in a border character.

The optional rows (the CABLES and STEREO labels, the legend and the READINGS heading) take only rows left over above
LOG_SHORT_MIN_ROWS of log, and go first when rows run short, the READINGS heading before the legend, the legend before the
STEREO label and that before the CABLES label. On a terminal of 24 rows or fewer only SHORT_CABLE_ROWS cables are listed, so at
80x24 the log has room for a few reasons. The plan below still keeps LOG_MIN_ROWS of log when it cuts cables and folds.

When rows run short, these give way in order: the cables down to one, then MOD/DELAY folds into the MELODY column (and OTHER
into AIR), then the readings collapse to four lines (the soloist's chip among them) and a `+N more chips` line.

`render` is a pure function of its arguments: it reads no clock and no environment, so the same state gives the same frame.
Chips are always found by their own `index`, so a list of chips numbered other than 0..n-1 is drawn correctly.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from ..music.evict import PRESENCE
from ..narrate.color import CHIP_JACKS, DEVICE_JACKS, bar
from ..ui.chart import CABLE_COLOURS
from ..ui.flow import CableView, ChipView, DeviceView, FlowState
from .frame import Frame, Row, Span, Style, dim_level, fit

MIN_COLS = 60  # below this many columns, or MIN_ROWS rows, the screen is one line saying it is too small
MIN_ROWS = 16
TARGET = (80, 24)  # the size the layout is made for: everything shows at this size with at least LOG_MIN_ROWS of log
CABLE_ROWS = 5  # the strongest cables listed
WIDE_COLS = 100  # from this width MOD/DELAY is a fourth column
BAR_W = 4  # the width of a module's share bar and a chip's pressure bar
LOG_MIN_ROWS = 3  # the log keeps this many rows whenever the terminal is at least TARGET
SHORT_CABLE_ROWS = 3  # the cables listed on a terminal of TARGET's height or less, so the log has room for its reasons
LOG_SHORT_MIN_ROWS = 6  # the log rows the optional rows always leave (more than LOG_MIN_ROWS, so a few reasons show)
QUIT_HINT = "q quit"  # at the right end of the heading, when it fits
READINGS_KEEP = 4  # readings lines kept when the strip collapses
PHASE_MARK = {"running": "●", "filling": "…", "evicting": "◌", "idle": "◌"}  # the chip's mark, before its index
CABLE_STEPS = ((0.5, 0), (0.25, 1), (0.1, 2))  # a push at or above the bound gets that dim level; below the last, level 3

LABELS = {"ultrasonic_shimmer": "shimmer"}  # short names for the screen
# Two lower-case letters for each device on the stereo line, so none reads as the `L` and `R` that label its ends.
STEREO_CODES = {
    "gong": "go", "ttrollz": "rz", "ttsteam": "st", "lead": "ld", "ttquantus": "qu", "ttrax": "tx",
    "ttchoir": "ch", "ttdrone": "dr", "ultrasonic_shimmer": "sh", "ttavdog": "av", "ttcoco": "co",
}
GROUPS = (  # the modules by the role they play, in the order they are drawn
    ("RHYTHM", ("gong", "ttrollz", "ttsteam")),
    ("MELODY", ("lead", "ttquantus", "ttrax")),
    ("AIR", ("ttchoir", "ttdrone", "ultrasonic_shimmer")),
    ("MOD/DELAY", ("ttavdog", "ttcoco")),
)
OTHER = "OTHER"  # the group of a device that GROUPS does not name

FOCUS_BAR, BAR = "┃", "╎"
LABEL_STYLE = Style("grey", dim=1)  # the optional rows (CABLES, STEREO, the legend, the READINGS heading), drawn only in spare rows
# The READINGS heading: each word over the column it names in a readings line (see `_readings_line`).
READINGS_HEAD = ((0, "READINGS"), (14, "temp"), (19, "power"), (25, "pressure"), (36, "drives"))


def label(name: str) -> str:
    return LABELS.get(name, name)


def _jack(name: str) -> str:
    return DEVICE_JACKS.get(name, "grey")


def _chip_jack(index: int) -> str:
    return CHIP_JACKS[index % len(CHIP_JACKS)]


def _level(chip: Optional[ChipView]) -> int:
    return dim_level(PRESENCE.get(chip.phase, 1.0)) if chip else 0


# --- the heading ---------------------------------------------------------------------------------------------------------

def _patch_parts(state: FlowState) -> List[str]:
    p = state.patch
    return [p.get("style") or "plain patch", f"scale {p.get('scale')}", f"gong {p.get('gong')}"]


def _heading_row(canvas: "_Canvas", state: Optional[FlowState], unit: str, width: int, notice: Optional[str]) -> None:
    """The first row: the notice or the heading, and the dim `q quit` hint at its right end when the heading fits beside it
    whole (only the patch's parts are given up for it)."""
    room = width - len(QUIT_HINT) - 2
    text = fit(notice, room) if notice else heading(state, unit, room)
    hint = notice is None or len(notice) <= room
    if text.endswith("…") or not hint:
        text = fit(notice, width) if notice else heading(state, unit, width)
        hint = False
    row = canvas.add_row()
    canvas.put(row, 0, text, Style("yellow") if notice else Style("white"))
    if hint:
        canvas.put(row, width - len(QUIT_HINT), QUIT_HINT, LABEL_STYLE)


def heading(state: Optional[FlowState], unit: str, width: Optional[int] = None) -> str:
    """One line: the segment or phrase, who has the floor, then the patch. Given a `width`, the patch's parts are left out
    from the end until the line fits, and only then is the line cut."""
    if state is None:
        text = f"waiting for the first {unit} to play"
        return fit(text, width) if width is not None else text
    who = f"{state.soloist_device} on chip {state.soloist_chip} has the floor" if state.soloist_device else "nobody has the floor"
    lead = f"segment {state.segment} · {state.seconds:.1f} s · {who}" if unit == "segment" else f"{unit} {state.segment} · {who}"
    parts = _patch_parts(state)
    if width is None:
        return " · ".join([lead] + parts)
    while parts and len(" · ".join([lead] + parts)) > width:
        parts.pop()
    return fit(" · ".join([lead] + parts), width)


# --- a canvas of cells, turned into spans at the end ---------------------------------------------------------------------

class _Canvas:
    """Rows of cells. Each `put` is one span: cells keep the number of the put that drew them, so `rows()` can give back the
    spans as they were drawn (a later put on the same cells draws on top)."""

    def __init__(self, width: int):
        self.width = width
        self.cells: List[List[Tuple[str, Style, int]]] = []
        self.puts = 0

    def add_row(self) -> int:
        self.cells.append([(" ", Style(), -1) for _ in range(self.width)])
        return len(self.cells) - 1

    def put(self, row: int, col: int, text: str, style: Style) -> int:
        """Draw `text` from `col`; the column after it is returned."""
        self.puts += 1
        for k, ch in enumerate(text):
            if 0 <= col + k < self.width:
                self.cells[row][col + k] = (ch, style, self.puts)
        return col + len(text)

    def put_row(self, row: int, spans: Row) -> None:
        col = 0
        for span in spans:
            col = self.put(row, col, span.text, span.style)

    def rows(self) -> Tuple[Row, ...]:
        out = []
        for cells in self.cells:
            end = len(cells)
            while end and cells[end - 1][0] == " ":
                end -= 1  # no trailing blanks
            spans: List[Span] = []
            group, text, style = None, "", Style()
            for ch, st, g in cells[:end]:
                if g != group:
                    if text:
                        spans.append(Span(text, style))
                    group, text, style = g, "", st
                text += ch
            if text:
                spans.append(Span(text, style))
            out.append(tuple(spans))
        return tuple(out)


# --- the readings strip --------------------------------------------------------------------------------------------------

def _shown_chips(state: FlowState, keep: Optional[int]) -> List[ChipView]:
    """The chips whose readings are drawn: all, or the first `keep`, with the soloist's chip in the last place when it is not
    among them."""
    chips = list(state.chips)
    if keep is None or keep >= len(chips):
        return chips
    shown = chips[:keep]
    if state.soloist_chip >= 0 and all(c.index != state.soloist_chip for c in shown):
        solo = next((c for c in chips if c.index == state.soloist_chip), None)
        if solo is not None:
            shown = shown[:keep - 1] + [solo]
    return shown


def _readings_head(canvas: _Canvas, width: int) -> None:
    row = canvas.add_row()
    for col, word in READINGS_HEAD:
        if col + len(word) <= width:
            canvas.put(row, col, word, LABEL_STYLE)


def _readings_line(canvas: _Canvas, row: int, chip: ChipView, drives: Sequence[str], focus: bool, width: int) -> None:
    level = 0 if focus else _level(chip)
    style = Style(_chip_jack(chip.index), bold=focus, dim=level)
    text = (f"{FOCUS_BAR if focus else BAR}{PHASE_MARK.get(chip.phase, '?')}{chip.index:<2} {chip.phase:<8} "
            f"{chip.temp_c:>3.0f}C {chip.power_w:>4.0f}W {chip.pressure * 100:>3.0f}% {bar(chip.pressure, BAR_W)}  ")
    col = canvas.put(row, 0, fit(text, width), style)
    # The names it drives, each in its own jack. They are not bold, even on the focused line: bold marks the soloist's own row.
    for k, name in enumerate(drives):
        room = width - col - (1 if k else 0)
        if room <= 0:
            break
        piece = (" " if k else "") + fit(label(name), room)
        col = canvas.put(row, col, piece, Style(_jack(name), dim=level))


# --- the modules ---------------------------------------------------------------------------------------------------------

def _grouped(devices: Sequence[DeviceView]) -> Dict[str, List[DeviceView]]:
    """Each group's devices: in the table's order of names, then by slot (a name dealt to two chips has a row for each)."""
    order = {name: (g, k) for g, (_, names) in enumerate(GROUPS) for k, name in enumerate(names)}
    out: Dict[str, List[DeviceView]] = {group: [] for group, _ in GROUPS}
    for d in sorted(devices, key=lambda d: (order.get(d.name, (len(GROUPS), 0)), d.slot)):
        group = GROUPS[order[d.name][0]][0] if d.name in order else OTHER
        out.setdefault(group, []).append(d)
    return out


def _module_blocks(state: FlowState, width: int, fold: bool) -> Tuple[int, List[List[Tuple[Optional[str], List[DeviceView]]]]]:
    """The column width and the blocks of modules. A block is a list of columns; a column is its heading (None for none) and
    its rows. Folding puts MOD/DELAY's rows under MELODY's and OTHER's under AIR's, with no headings."""
    wide = width >= WIDE_COLS
    n_cols = 4 if wide else 3
    groups = _grouped(state.devices)
    top = [[name, list(groups[name])] for name in ("RHYTHM", "MELODY", "AIR")]
    second: List[List] = []
    if wide:
        top.append(["MOD/DELAY", list(groups["MOD/DELAY"])])
    else:
        second.append(["MOD/DELAY", list(groups["MOD/DELAY"])])
    if groups.get(OTHER):
        second.append([OTHER, list(groups[OTHER])])
    if fold and second:
        for name, rows in second:
            target = top[1] if name == "MOD/DELAY" else top[2]
            target[1].extend(rows)
        second = []
    blocks = [[(h, rows) for h, rows in top]]
    if second:
        blocks.append([(h, rows) for h, rows in second])
    return width // n_cols, blocks


def _blocks_height(blocks) -> int:
    return sum(1 + max(len(rows) for _, rows in block) for block in blocks)


def _strongest_touching(name: str, cables: Sequence[CableView]) -> Optional[CableView]:
    touching = [c for c in cables if name in (c.source, c.target)]
    return max(touching, key=lambda c: c.push) if touching else None


def _module_row(canvas: _Canvas, row: int, x: int, inner: int, d: DeviceView, chip: Optional[ChipView],
                cables: Sequence[CableView], colour: bool, name_w: int) -> None:
    """One module: the bar, a marker cell (`▶` for the soloist, blank for the others), the name padded to `name_w`, a blank,
    the chip's mark and index, then as room allows the strongest cable that touches it, its share bar and `faint`, chosen in
    that order and drawn as bar, faint, cable. Nothing is drawn past `x + inner`.

    Every row has the same number of cells before its chip's mark, so the marks of a column line up with the soloist among
    them. Without colour the soloist's row is `▶[name]`: the arrow takes the bar's cell, and the brackets take the marker cell
    and the blank after the name, which are blank on every other row. So the colour and no-colour frames choose the same parts
    and differ only in those three cells. (Padding every other row by three cells for the arrow and brackets would cost the
    cable hint on most rows at 80 columns.)"""
    if d.soloist:
        style = Style(_jack(d.name), bold=True, dim=0)
    else:
        style = Style(_jack(d.name), dim=3 if not d.heard else max(1, _level(chip)))
    brackets = d.soloist and not colour
    end = x + inner
    lead = "▶[" if brackets else (FOCUS_BAR + "▶") if d.soloist else (BAR + " ")
    col = canvas.put(row, x, lead, style)
    field = min(name_w, end - col - 1)  # the name, then one blank cell (the closing bracket's, without colour)
    if field <= 0:
        return
    name = fit(label(d.name), field)
    col = canvas.put(row, col, name, style)
    col = canvas.put(row, col, ("]" if brackets else " ") + " " * (field - len(name)), style)
    dot = f"{PHASE_MARK.get(chip.phase, '?') if chip else '?'}{d.chip}"
    if col + len(dot) > end:
        return
    col = canvas.put(row, col, dot, Style(_chip_jack(d.chip), bold=d.soloist, dim=style.dim))

    # The optional parts, most wanted first. The cable comes before the share bar so that on an 80-column screen a module
    # names what feeds it.
    wanted = []
    c = _strongest_touching(d.name, cables)
    if c is not None:
        hint = f"◂ {c.kind} {label(c.source)}" if c.target == d.name else f"▸ {c.kind} {label(c.target)}"
        wanted.append(("hint", " " + hint, Style(CABLE_COLOURS.get(c.kind, "grey"), dim=style.dim)))
    wanted.append(("share", " " + bar(min(1.0, d.share * 4), BAR_W), style))
    if chip is not None and chip.phase in ("evicting", "idle"):
        wanted.append(("faint", " faint", style))
    room = end - col
    chosen = {}
    for key, text, st in wanted:
        if len(text) <= room:
            chosen[key] = (text, st)
            room -= len(text)
    for key in ("share", "faint", "hint"):
        if key in chosen:
            text, st = chosen[key]
            col = canvas.put(row, col, " ", style)
            col = canvas.put(row, col, text[1:], st)


def _draw_modules(canvas: _Canvas, state: FlowState, width: int, fold: bool, chips: Dict[int, ChipView], colour: bool) -> None:
    cw, blocks = _module_blocks(state, width, fold)
    for block in blocks:
        head = canvas.add_row()
        for k, (title, _) in enumerate(block):
            if title:
                canvas.put(head, k * cw, fit(title, cw - 1), Style("grey"))
        rows = [canvas.add_row() for _ in range(max(len(r) for _, r in block))]
        for k, (_, devices) in enumerate(block):
            name_w = max((len(label(d.name)) for d in devices), default=0)  # the names in a column line up
            for i, d in enumerate(devices):
                _module_row(canvas, rows[i], k * cw, cw - 1, d, chips.get(d.chip), state.cables, colour, name_w)


# --- the cables and the stereo line ----------------------------------------------------------------------------------------

def _cable_level(push: float) -> int:
    return next((level for bound, level in CABLE_STEPS if push >= bound), 3)


def _cable_cut(state: FlowState, n: int) -> List[CableView]:
    """The `n` strongest cables, strongest first. When none of them touches the soloist, its strongest cable takes the last
    place, so the focus always has a cable to light."""
    ranked = sorted(state.cables, key=lambda c: -c.push)
    cut = ranked[:n]
    solo = state.soloist_device
    if n and solo and not any(solo in (c.source, c.target) for c in cut):
        touching = [c for c in ranked if solo in (c.source, c.target)]
        if touching:
            cut = sorted(cut[:n - 1] + [touching[0]], key=lambda c: -c.push)
    return cut


def _cable_row(canvas: _Canvas, row: int, c: CableView, soloist: str, width: int) -> None:
    """`source ═kind .depth═▶ target` in the kind's jack. A cable that touches the soloist is at full brightness; with a
    soloist the others are at least one step dimmer; with nobody on the floor a hotter cable is brighter. Never bold."""
    if soloist and soloist in (c.source, c.target):
        level = 0
    elif soloist:
        level = max(1, _cable_level(c.push))
    else:
        level = _cable_level(c.push)
    depth = f"{c.depth:.2f}".lstrip("0") or "0"
    pieces = (
        (BAR + " ", Style("grey", dim=level)),
        (label(c.source), Style(_jack(c.source), dim=level)),
        (f" ═{c.kind} {depth}═▶ ", Style(CABLE_COLOURS.get(c.kind, "grey"), dim=level)),
        (label(c.target), Style(_jack(c.target), dim=level)),
    )
    col = 0
    for text, style in pieces:
        if col >= width:
            break
        col = canvas.put(row, col, fit(text, width - col), style)


def _stereo(canvas: _Canvas, row: int, state: Optional[FlowState], width: int, chips: Dict[int, ChipView]) -> None:
    """`L ` + the field + ` R`. Each device's two-letter code starts at its azimuth's place in the field (columns 2 to
    width - 3), so a code at either end stops short of the `L` and the `R`."""
    field_w = width - 4
    canvas.put(row, 0, "L ", Style("grey"))
    canvas.put(row, 2, "─" * field_w, Style("grey", dim=2))
    canvas.put(row, width - 2, " R", Style("grey"))
    if state is None:
        return
    # the soloist last, so where two voices sit in the same column the one with the floor is drawn on top
    for d in sorted(state.devices, key=lambda d: d.soloist):
        col = 2 + round((d.azimuth + 1) / 2 * (field_w - 2))
        if d.soloist:
            style = Style(_jack(d.name), bold=True)
        else:
            style = Style(_jack(d.name), dim=3 if not d.heard else max(1, _level(chips.get(d.chip))))
        canvas.put(row, col, STEREO_CODES.get(d.name, "??"), style)


def _legend(canvas: _Canvas, state: FlowState, width: int) -> None:
    """One row: each code on the stereo line and the device it stands for, in the groups' order, as many whole pairs as fit."""
    row = canvas.add_row()
    names = [d.name for d in sorted(state.devices, key=lambda d: (_group_order(d.name), d.slot))]
    col = 0
    for name in dict.fromkeys(names):
        piece = f"{STEREO_CODES.get(name, '??')} {label(name)}"
        start = col + (2 if col else 0)
        if start + len(piece) > width:
            break
        col = canvas.put(row, start, piece, Style(_jack(name), dim=1))


def _group_order(name: str) -> Tuple[int, int]:
    order = {n: (g, k) for g, (_, names) in enumerate(GROUPS) for k, n in enumerate(names)}
    return order.get(name, (len(GROUPS), 0))


# --- the whole screen -----------------------------------------------------------------------------------------------------

def _plan(state: FlowState, width: int, height: int) -> Tuple[Optional[int], bool, int]:
    """How many readings lines to keep (None: all), whether MOD/DELAY folds, and how many cables to list. The cables give way
    first (down to one), then the fold, then the readings collapse; at each step the cables take back what room there is."""
    n = len(state.chips)
    n_cables = min(SHORT_CABLE_ROWS if height <= TARGET[1] else CABLE_ROWS, len(state.cables))
    least = min(1, n_cables)
    budget = height - LOG_MIN_ROWS

    def need(keep: Optional[int], fold: bool, cables: int) -> int:
        readings = n if keep is None else keep + 1
        return 1 + readings + _blocks_height(_module_blocks(state, width, fold)[1]) + cables + 1

    def most_cables(keep: Optional[int], fold: bool) -> Optional[int]:
        for c in range(n_cables, least - 1, -1):
            if need(keep, fold, c) <= budget:
                return c
        return None

    steps = [(None, False), (None, True)] + [(k, True) for k in range(READINGS_KEEP, 0, -1) if k < n]
    for keep, fold in steps:
        c = most_cables(keep, fold)
        if c is not None:
            return keep, fold, c
    keep, fold = steps[-1]
    return keep, fold, 0  # the log gets whatever is left


def render(state: Optional[FlowState], log: Sequence[Row], width: int, height: int, colour: bool, unit: str = "segment",
           notice: Optional[str] = None) -> Frame:
    """The screen for `state` and the log pane's rows (oldest first), `width` columns by `height` rows. A `notice` (the run is
    stopping) takes the heading's place."""
    if width < MIN_COLS or height < MIN_ROWS:
        msg = f"terminal too small for --tui ({width}x{height}, needs {MIN_COLS}x{MIN_ROWS}); the run goes on"
        return Frame(((Span(fit(msg, width), Style("yellow")),),), width, height)

    canvas = _Canvas(width)
    _heading_row(canvas, state, unit, width, notice)
    chips: Dict[int, ChipView] = {c.index: c for c in state.chips} if state else {}
    extras: List[str] = []
    if state is not None:
        keep, fold, n_cables = _plan(state, width, height)
        shown = _shown_chips(state, keep)
        # The optional rows take only rows the plan left over above the log's floor, so they are the first thing to go, the
        # READINGS heading before the legend, the legend before the STEREO label, and that before the CABLES label.
        fixed = (1 + len(shown) + (len(shown) < len(state.chips)) + _blocks_height(_module_blocks(state, width, fold)[1])
                 + n_cables + 1)
        spare = height - fixed - LOG_SHORT_MIN_ROWS
        extras = [name for name in (("CABLES",) if n_cables else ()) + ("STEREO", "legend", "READINGS")][:max(0, spare)]
        if "READINGS" in extras:
            _readings_head(canvas, width)
        for chip in shown:
            drives = [d.name for d in sorted(state.devices, key=lambda d: d.slot) if d.chip == chip.index]
            _readings_line(canvas, canvas.add_row(), chip, drives, chip.index == state.soloist_chip, width)
        if len(shown) < len(state.chips):
            canvas.put(canvas.add_row(), 0, f"+{len(state.chips) - len(shown)} more chips", Style("grey"))
        _draw_modules(canvas, state, width, fold, chips, colour)
        if "CABLES" in extras:
            canvas.put(canvas.add_row(), 0, "CABLES", LABEL_STYLE)
        for c in _cable_cut(state, n_cables):
            _cable_row(canvas, canvas.add_row(), c, state.soloist_device, width)
        if "STEREO" in extras:
            canvas.put(canvas.add_row(), 0, "STEREO", LABEL_STYLE)
    _stereo(canvas, canvas.add_row(), state, width, chips)
    if "legend" in extras:
        _legend(canvas, state, width)

    log_rows = height - len(canvas.cells)
    for spans in (list(log)[-log_rows:] if log_rows > 0 else []):
        canvas.put_row(canvas.add_row(), _clip(spans, width))
    return Frame(canvas.rows()[:height], width, height)


def _clip(spans: Row, width: int) -> Row:
    """A log row cut to `width` columns."""
    out, used = [], 0
    for span in spans:
        if used >= width:
            break
        text = span.text[:width - used]
        out.append(Span(text, span.style))
        used += len(text)
    return tuple(out)
