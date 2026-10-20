# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The run log as HTML for a side panel: one row per event, a left bar in the Ciat-Lonbarde jack colour of its kind (the colours the
terminal log uses, narrate/color.py), the reason under it in grey, and a reason already shown left out. The newest row is last, and the
container flips its direction so a panel that scrolls stays pinned to the newest row."""
from __future__ import annotations

from html import escape
from typing import Sequence

from ..narrate.color import _TOKEN, CHIP_JACKS, DEVICE_JACKS, EVENT_JACKS, JACK_HEX

MAX_LINES = 150

STYLE = ("display:flex;flex-direction:column-reverse;height:640px;overflow-y:auto;background:#0b1f1e;border-radius:10px;padding:8px;"
         "font:12px ui-monospace,Menlo,Consolas,monospace;color:#cfe4e2;")


def _jack(rec: dict) -> str:
    if rec["event"] == "live":
        return DEVICE_JACKS.get(rec.get("device", ""), EVENT_JACKS["live"])
    return EVENT_JACKS.get(rec["event"], "grey")


def decorate_html(text: str) -> str:
    """`text` escaped, with the devices (each its own jack), the chips (each its own) and the numbers with units coloured, by the same
    rules as the terminal log (narrate/color.py `decorate`)."""
    out, last = [], 0
    for m in _TOKEN.finditer(text):
        out.append(escape(text[last:m.start()]))
        if m.group("device"):
            jack = DEVICE_JACKS[m.group("device")]
        elif m.group("chip"):
            jack = CHIP_JACKS[int(m.group("n")) % len(CHIP_JACKS)]
        else:
            jack = "yellow"
        out.append(f'<span data-token="1" style="color:#{JACK_HEX[jack]};font-weight:600">{escape(m.group(0))}</span>')
        last = m.end()
    out.append(escape(text[last:]))
    return "".join(out)


def log_html(records: Sequence[dict]) -> str:
    if not records:
        return f'<div style="{STYLE}"><div style="color:#7d9a98">waiting for the first reading…</div></div>'
    rows, seen = [], set()
    for rec in records[-MAX_LINES:]:
        mins, secs = divmod(rec["t"], 60)
        why = "" if rec["why"] in seen else f'<div style="color:#7d9a98;margin-left:64px">{escape(rec["why"])}</div>'
        seen.add(rec["why"])
        rows.append(f'<div class="ln" style="border-left:3px solid #{JACK_HEX[_jack(rec)]};padding:2px 6px;margin:2px 0;color:#cfe4e2">'
                    f'<span style="color:#7d9a98">[{int(mins):02d}:{secs:04.1f}]</span> {decorate_html(rec["what"])}{why}</div>')
    return f'<div style="{STYLE}"><div>{"".join(rows)}</div></div>'
