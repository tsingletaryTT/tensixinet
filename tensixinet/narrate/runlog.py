# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Run log: what the project is doing, and why, as JSONL plus readable lines.

JSONL is for machines (one object per line, flushed immediately so a crash
keeps everything before it). The readable stream is for the person watching
the demo; it uses a left bar only so it survives any terminal width.
"""
from __future__ import annotations

import json
import textwrap
import time
from typing import Callable, Optional, TextIO


class RunLog:
    def __init__(
        self,
        jsonl: TextIO,
        stream: TextIO,
        clock: Callable[[], float] = time.monotonic,
        width: Optional[int] = None,
        dedupe_why: bool = False,
    ):
        self._jsonl = jsonl
        self._stream = stream
        self._clock = clock
        self._width = width or 10**9  # None: never wrap
        self._dedupe = dedupe_why
        self._seen = set()  # reasons already shown on screen
        self._start = clock()

    def event(self, event: str, what: str, why: str, **fields) -> None:
        t = round(self._clock() - self._start, 3)
        record = {"t": t, "event": event, "what": what, "why": why, **fields}
        self._jsonl.write(json.dumps(record) + "\n")
        self._jsonl.flush()
        mins, secs = divmod(t, 60)
        # The JSONL above always carries the reason. On screen a reason already
        # shown is skipped, so a long run does not scroll its own start away.
        show_why = not (self._dedupe and why in self._seen)
        self._seen.add(why)
        self._stream.write(
            self._fill(f"║ [{int(mins):02d}:{secs:04.1f}] ", what)
            + (self._fill("║          why: ", why) if show_why else "")
        )
        self._stream.flush()

    def _fill(self, prefix: str, text: str) -> str:
        """Wrap `text` to the pane width; continuation lines keep the left bar
        and hang under the text, so a narrow pane stays readable."""
        return textwrap.fill(
            text,
            width=self._width,
            initial_indent=prefix,
            subsequent_indent="║" + " " * (len(prefix) - 1),
            break_long_words=False,
        ) + "\n"

    def close(self) -> None:
        for f in (self._jsonl, self._stream):
            f.close()
