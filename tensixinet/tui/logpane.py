# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The log pane of the TUI: run-log records as rows of styled spans.

The rows look like `RunLog`'s screen lines (a left bar, a time stamp, the line, then its reason) and use the same colour rules
as `narrate/color.py`, so the pane and the plain log agree. A reason that an earlier record in the list already gave is left
out, as `RunLog(dedupe_why=True)` does. There is no right-hand border.
"""
from __future__ import annotations

import textwrap
from typing import List, Sequence

from ..narrate.color import _TOKEN, CHIP_JACKS, DEVICE_JACKS, EVENT_JACKS
from .frame import Row, Span, Style

_WHY_PREFIX = "║          why: "


def edge_jack(record: dict) -> str:
    """The jack colour of a record's left bar: a live segment takes its soloist's, any other kind takes its own."""
    event = record.get("event", "")
    if event == "live":
        return DEVICE_JACKS.get(record.get("device", ""), EVENT_JACKS.get(event, "grey"))
    return EVENT_JACKS.get(event, "grey")


def spans(text: str, base: Style) -> List[Span]:
    """`text` as spans: devices bold in their own jack, `chip N` bold in the chip's jack, numbers with units bold yellow."""
    out: List[Span] = []
    pos = 0
    for m in _TOKEN.finditer(text):
        if m.start() > pos:
            out.append(Span(text[pos:m.start()], base))
        if m.group("device"):
            style = Style(DEVICE_JACKS[m.group("device")], bold=True)
        elif m.group("chip"):
            style = Style(CHIP_JACKS[int(m.group("n")) % len(CHIP_JACKS)], bold=True)
        else:
            style = Style("yellow", bold=True)
        out.append(Span(m.group(0), style))
        pos = m.end()
    if pos < len(text):
        out.append(Span(text[pos:], base))
    return out


def _wrap(prefix: str, text: str, width: int) -> List[str]:
    """`text` wrapped to `width` columns with `prefix` on the first line and a bar and blanks on the others."""
    return textwrap.wrap(
        text, width=width, initial_indent=prefix, subsequent_indent="║" + " " * (len(prefix) - 1),
        break_long_words=False, break_on_hyphens=False,
    ) or [prefix.rstrip()]


def log_rows(records: Sequence[dict], width: int) -> List[Row]:
    """The rows for `records`, oldest first, each at most `width` columns wide."""
    rows: List[Row] = []
    seen = set()
    for rec in records:
        edge = Style(edge_jack(rec), bold=True)
        grey = Style("grey")
        mins, secs = divmod(rec.get("t", 0.0), 60)
        stamp = f"[{int(mins):02d}:{secs:04.1f}]"
        prefix = "║ " + stamp + " "
        for i, line in enumerate(_wrap(prefix, str(rec.get("what", "")), width)):
            body = line[len(prefix):] if i == 0 else line[1:]
            if i == 0:
                rows.append((Span("║", edge), Span(" ", grey), Span(stamp, grey), Span(" ", grey), *spans(body, Style())))
            else:
                rows.append((Span("║", edge), *spans(body, Style())))
        why = rec.get("why", "")
        if why and why not in seen:
            dim = Style("grey", dim=1)
            for i, line in enumerate(_wrap(_WHY_PREFIX, why, width)):
                if i == 0:
                    rows.append((Span("║", edge), Span(" " * 10, dim), Span("why:", dim), Span(" ", dim), Span(line[len(_WHY_PREFIX):], dim)))
                else:
                    rows.append((Span("║", edge), Span(line[1:], dim)))
        seen.add(why)
    return rows
