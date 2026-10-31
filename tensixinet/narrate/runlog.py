# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Run log: what the project is doing, and why, as JSONL plus readable lines.

JSONL is for machines (one object per line, flushed immediately so a crash
keeps everything before it). The readable stream is for the person watching
the demo; it uses a left bar only so it survives any terminal width.
"""
from __future__ import annotations

import json
import sys
import textwrap
import time
from typing import Callable, List, Optional, TextIO

from .color import DEVICE_JACKS, EVENT_JACKS, Palette, bar, decorate


class RunLog:
    def __init__(
        self,
        jsonl: TextIO,
        stream: TextIO,
        clock: Callable[[], float] = time.monotonic,
        width: Optional[int] = None,
        dedupe_why: bool = False,
        palette: Optional[Palette] = None,
        plain_stream: Optional[TextIO] = None,
        on_event: Optional[Callable[[dict], None]] = None,
    ):
        """`stream` is the screen. With an enabled `palette` it gets colour (see color.py) and `plain_stream`, when given, gets
        the same lines without it, for a text log. Without a palette `stream` gets plain text, as it always did.
        `on_event`, when given, is called with each record (the dict that went to the JSONL) once it is written; an error in it is
        swallowed so the log and the run go on."""
        self._jsonl = jsonl
        self._stream = stream
        self._plain_stream = plain_stream
        self._palette = palette if palette is not None and palette.enabled else None
        self._clock = clock
        self._width = width or 10**9  # None: never wrap
        self._dedupe = dedupe_why
        self._seen = set()  # reasons already shown on screen
        self._start = clock()
        self._on_event = on_event

    def event(self, event: str, what: str, why: str, **fields) -> None:
        t = round(self._clock() - self._start, 3)
        record = {"t": t, "event": event, "what": what, "why": why, **fields}
        self._jsonl.write(json.dumps(record) + "\n")
        self._jsonl.flush()
        if self._on_event is not None:
            try:
                self._on_event(record)
            except Exception:  # a broken listener must not stop the log or the run
                pass
        mins, secs = divmod(t, 60)
        # The JSONL above always carries the reason. On screen a reason already
        # shown is skipped, so a long run does not scroll its own start away.
        show_why = not (self._dedupe and why in self._seen)
        self._seen.add(why)
        what_lines = self._lines(f"║ [{int(mins):02d}:{secs:04.1f}] ", what)
        why_lines = self._lines("║          why: ", why) if show_why else []
        plain = "".join(line + "\n" for line in what_lines + why_lines)
        self._stream.write(plain if self._palette is None else self._coloured(event, what_lines, why_lines, fields))
        self._stream.flush()
        if self._plain_stream is not None:
            self._plain_stream.write(plain)
            self._plain_stream.flush()

    def raw(self, screen: str, plain: str) -> None:
        """Write something that is not an event (the opening art) to the screen, in colour, and to the text log, plain. It is not
        in the JSONL, which is one object per event."""
        self._stream.write(screen)
        self._stream.flush()
        if self._plain_stream is not None:
            self._plain_stream.write(plain)
            self._plain_stream.flush()

    def _lines(self, prefix: str, text: str) -> List[str]:
        """Wrap `text` to the pane width; continuation lines keep the left bar
        and hang under the text, so a narrow pane stays readable."""
        return textwrap.fill(
            text,
            width=self._width,
            initial_indent=prefix,
            subsequent_indent="║" + " " * (len(prefix) - 1),
            break_long_words=False,
        ).split("\n")

    def _coloured(self, event: str, what_lines: List[str], why_lines: List[str], fields: dict) -> str:
        """The same lines, in colour: the left bar takes the jack of the kind of line (a live segment takes its soloist's), the
        time and the reasons are grey, and the devices, the chips and the numbers with units pick up their own colours."""
        p = self._palette
        jack = DEVICE_JACKS.get(fields.get("device", ""), EVENT_JACKS.get(event, "grey")) if event == "live" else EVENT_JACKS.get(event, "grey")
        edge = p.paint("║", jack, bold=True)
        out = []
        for i, line in enumerate(what_lines):
            body = line[1:]  # the left bar is painted on its own
            if i == 0 and body.startswith(" ["):
                stamp, text = body[1:body.index("]") + 1], body[body.index("]") + 2:]
                out.append(edge + " " + p.dim(stamp) + " " + decorate(text, p))
            else:
                out.append(edge + decorate(body, p))
        if event == "live" and "seconds" in fields and out:
            room = self._width - len(what_lines[-1]) - 2
            length = min(max(1, round(fields["seconds"])), 12, max(0, room))
            if length:  # a bar as long as the segment, under the soloist's colour; on screen only
                out[-1] += "  " + p.paint("━" * length, jack, bold=True)
        for line in why_lines:
            body = line[1:]
            label = "          why: "
            out.append(edge + (p.dim("          why:") + " " + p.dim(body[len(label) - 0:]) if body.startswith(label) else p.dim(body)))
        return "".join(l + "\n" for l in out)

    def close(self) -> None:
        """Close the files this log writes. The terminal is never closed."""
        terminal = (sys.stdout, sys.stderr, sys.__stdout__, sys.__stderr__)
        for f in (self._jsonl, self._stream, self._plain_stream):
            if f is not None and not any(f is t for t in terminal):
                f.close()
