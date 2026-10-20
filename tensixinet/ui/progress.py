# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""What the page shows before the first segment plays.

Starting takes a while and only some of it is under our control: the TT-Lang simulator imports torch and builds its kernels (about 15 s on a
laptop, longer on a small machine), then the first segment is rendered, then the audio buffer is filled so the browser's player has something
in hand. `Progress` is those steps and where each one is. `progress_html` draws them as a checklist with a bar for the active step: filled
to a fraction when the length is known (the buffer: seconds handed over out of the lead) and an animated stripe when it is not.
"""
from __future__ import annotations

from dataclasses import dataclass
from html import escape
from typing import Optional, Tuple

DONE, ACTIVE, TODO = "done", "active", "todo"


@dataclass(frozen=True)
class Step:
    label: str
    state: str  # DONE, ACTIVE or TODO
    seconds: float  # how long it took (DONE) or has taken so far (ACTIVE)


@dataclass(frozen=True)
class Progress:
    steps: Tuple[Step, ...]
    fraction: Optional[float]  # how far through the active step, 0 to 1, or None when its length is not known
    elapsed: float  # seconds since Play


_MARK = {DONE: "✓", ACTIVE: "●", TODO: "○"}
_COLOUR = {DONE: "#5fd38d", ACTIVE: "#4fd1c5", TODO: "#5d7573"}
_STYLE = ("<style>@keyframes tn-stripe{from{background-position:0 0}to{background-position:40px 0}}"
          ".tn-ind{background:repeating-linear-gradient(45deg,#4fd1c5 0 10px,#2b7f78 10px 20px);background-size:40px 40px;"
          "animation:tn-stripe 1s linear infinite}</style>")


def _bar(fraction: Optional[float]) -> str:
    if fraction is None:
        return '<div class="tn-ind indeterminate" style="height:100%;width:100%"></div>'
    pct = round(100 * max(0.0, min(1.0, fraction)))
    return f'<div style="height:100%;width:{pct}%;background:#4fd1c5"></div>'


def progress_html(p: Progress) -> str:
    rows = []
    for step in p.steps:
        secs = f' <span style="color:#7d9a98">{step.seconds:.0f} s</span>' if step.state != TODO else ""
        bar = (f'<div style="margin:6px 0 2px 28px;height:8px;width:320px;background:#1d3a38;border-radius:4px;overflow:hidden">'
               f'{_bar(p.fraction)}</div>') if step.state == ACTIVE else ""
        rows.append(f'<div data-state="{step.state}" style="padding:4px 0;color:{_COLOUR[step.state]}">'
                    f'<span style="display:inline-block;width:24px">{_MARK[step.state]}</span>{escape(step.label)}{secs}{bar}</div>')
    return (f'{_STYLE}<div style="padding:32px 28px;font:14px ui-monospace,monospace;color:#cfe4e2">'
            f'<div style="font-size:16px;margin-bottom:14px">starting: {p.elapsed:.0f} s so far</div>{"".join(rows)}</div>')
