# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The screen driver for `--tui`: it draws what the playhead shows, on its own thread, in the terminal's alternate screen.

The live loops only touch the `Playhead` (`hand_over`, `note`, `started`), which takes a lock, appends and returns. Everything
here runs on the daemon thread `tensixinet-tui`: every `frame_s` it waits for a key, asks the playhead what the listener has
reached, lays it out with `layout.render`, and writes only the cells that changed (`frame.diff`). If the terminal stops reading
(a paused pane, a slow ssh link), that thread waits in `write` and frames are dropped; the audio never waits for it.

Keys: `q` or `Q` calls `interrupt`, which by default sends this process a real SIGINT (`os.kill`), so `q` ends the run exactly the
way Ctrl-C does. A real signal is needed: `_thread.interrupt_main` only sets a flag, and a main thread blocked in a C-level wait (the
live loop's `room.acquire()`) sees that flag only when the wait ends, seconds later. The drawing thread blocks SIGINT and SIGWINCH
for itself (`signal.pthread_sigmask`), so the main thread is the one the kernel hands them to. The heading then says the run is
stopping at once (`STOPPING`), while the segments already made play out; a second `q` sends another SIGINT, which stops the
play-out as a second Ctrl-C does. Ctrl-C itself still works because cbreak mode keeps the terminal's `ISIG`. Ctrl-L redraws the
whole screen. After `disarm` (the run loop has returned) and once `close` has begun, `q` sends nothing: a flag set under a lock,
and checked under the same lock just before the signal is sent, closes the window between the loop's check and its signal.

A frame costs little when nothing changed: the log rows are kept for the same records and width, and when the state, the records,
the size, the colour flag and the heading's notice are all as they were, the frame is neither laid out nor diffed. The whole screen
is still redrawn every `REDRAW_S` seconds, so something else that wrote to the terminal (a warning on stderr) is put right.

A resize (SIGWINCH) sets a flag; the next frame forgets the cells it last drew, so the screen is cleared and drawn again in full
at the new size.

Exit: `close(code)` stops the thread (waiting at most two frames, because a write may be stuck), puts back the old SIGWINCH
handler, leaves the alternate screen and restores the terminal mode (all inside one try whose `finally` leaves, so an interrupt
that lands inside `close` still restores the terminal), and only then prints, on the normal screen, the last
record and the log directory. For an exit code other than 0 and 130 (Ctrl-C), the last `ERROR_RECORDS` records and their
reasons come first, so the error is on screen after the run, above the "Press Enter" line that `media/toplike.py` adds.
"""
from __future__ import annotations

import os
import select
import shutil
import signal
import sys
import threading
import time
import traceback
from typing import Callable, List, Optional, Protocol, TextIO, Tuple

from ..narrate.color import ESC
from .frame import Cells, Frame, diff
from .layout import render
from .logpane import log_rows
from .playhead import Playhead

FRAME_S = 0.1  # at most 10 frames a second
REDRAW_S = 5.0  # the whole screen is drawn again this often, to repair anything else written over it
REDRAW_KEYS = ("\x0c",)  # Ctrl-L
STOPPING = "stopping: playing out what is already made, Ctrl-C to stop now"  # the heading after q
ERROR_RECORDS = 5  # records printed after the run ends with an error code
NORMAL_CODES = (0, 130)  # a clean end and Ctrl-C (or q): only the last line and the log directory are printed
QUIT_KEYS = ("q", "Q")


class Terminal(Protocol):
    def write(self, text: str) -> None: ...

    def flush(self) -> None: ...

    def size(self) -> Tuple[int, int]: ...  # (columns, rows)

    def enter(self) -> None: ...

    def leave(self) -> None: ...

    def read_key(self, timeout: float) -> Optional[str]: ...


def send_sigint() -> None:
    """A real SIGINT to this process: it wakes a main thread that is blocked in a wait, as a Ctrl-C at the keyboard does."""
    os.kill(os.getpid(), signal.SIGINT)


class TtyTerminal:
    """The real terminal: `out` for the screen, `inp` for keys. When `inp` is not a terminal (a pipe, /dev/null) the mode is
    left alone and no keys are read. `termios_module` is for a test."""

    def __init__(self, out: TextIO = sys.stdout, inp: TextIO = sys.stdin, termios_module=None):
        self._out = out
        self._inp = inp
        self._termios = termios_module
        self._saved = None  # the termios mode found at `enter`, put back at `leave`
        self._entered = False

    def _input_is_tty(self) -> bool:
        try:
            return self._inp.isatty()
        except (AttributeError, ValueError, OSError):
            return False

    def write(self, text: str) -> None:
        self._out.write(text)

    def flush(self) -> None:
        self._out.flush()

    def size(self) -> Tuple[int, int]:
        s = shutil.get_terminal_size((80, 24))
        return s.columns, s.lines

    def _tc(self):
        if self._termios is None:
            import termios

            self._termios = termios
        return self._termios

    def enter(self) -> None:
        if self._input_is_tty():
            termios = self._tc()
            fd = self._inp.fileno()
            self._saved = termios.tcgetattr(fd)
            # cbreak, as `tty.setcbreak` makes it: keys arrive one at a time with no echo, and ISIG stays on, so Ctrl-C still sends
            # SIGINT. IXON is turned off too: with it on, Ctrl-S stops the terminal's output, a write of a frame then blocks, and the
            # exit would wait for it.
            mode = [list(x) if isinstance(x, list) else x for x in self._saved]
            mode[0] &= ~termios.IXON
            mode[3] &= ~(termios.ECHO | termios.ICANON)
            mode[6][termios.VMIN] = 1
            mode[6][termios.VTIME] = 0
            termios.tcsetattr(fd, termios.TCSAFLUSH, mode)
        self._entered = True
        self.write(f"{ESC}?1049h{ESC}?25l")  # the alternate screen, and the cursor hidden
        self.flush()

    def leave(self) -> None:
        """Reset the colours, show the cursor, leave the alternate screen and restore the mode. A second call does nothing."""
        if not self._entered:
            return
        self._entered = False
        try:
            self.write(f"{ESC}0m{ESC}?25h{ESC}?1049l")
            self.flush()
        finally:
            if self._saved is not None:
                termios = self._tc()
                # TCSANOW: TCSADRAIN would wait for the output to drain, and a terminal that is not reading never drains.
                termios.tcsetattr(self._inp.fileno(), termios.TCSANOW, self._saved)
                self._saved = None

    def read_key(self, timeout: float) -> Optional[str]:
        if not self._input_is_tty():
            time.sleep(timeout)
            return None
        fd = self._inp.fileno()
        ready, _, _ = select.select([fd], [], [], timeout)
        if not ready:
            return None
        # os.read goes around the text wrapper's buffer, which `select` cannot see into.
        data = os.read(fd, 1)
        return data.decode("utf-8", errors="replace") if data else None


class TuiApp:
    def __init__(
        self,
        terminal: Terminal,
        playhead: Playhead,
        colour: bool,
        truecolor: bool,
        unit: str,
        log_dir: str,
        frame_s: float = FRAME_S,
        interrupt: Callable[[], None] = send_sigint,
        install_signal: Callable = signal.signal,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._terminal = terminal
        self._playhead = playhead
        self._colour = colour
        self._truecolor = truecolor
        self._unit = unit
        self._log_dir = log_dir
        self._frame_s = frame_s
        self._interrupt = interrupt
        self._install_signal = install_signal
        self._clock = clock

        self._stop = threading.Event()
        self._resized = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._old_winch = None
        self._started = False
        self._irq_lock = threading.Lock()  # held while `q` decides to send and sends; `disarm` and `close` take it to stop it
        self._armed = True  # False once the run loop has returned or `close` has begun: `q` then sends nothing
        self._stopping = False  # q was pressed: the heading says the run is stopping
        self._redraw = threading.Event()  # Ctrl-L: draw the whole screen again
        self._drawn_at: Optional[float] = None  # the clock at the last full redraw
        self._key = None  # what the frame on the screen was made from, so a frame with nothing new is skipped
        self._rows_key = None  # the records and width the cached log rows were made for
        self._rows = None
        self._closed: Optional[List[str]] = None
        self._last_frame: Optional[Frame] = None  # the frame on the screen, so an identical one is skipped without a diff
        self._last_cells: Optional[Cells] = None  # its cells, for the diff; None forces a full redraw
        self._error: Optional[str] = None  # the first exception the drawing thread caught, as "Type: message"
        self._run_error: Optional[str] = None  # the exception that left a `with` block
        self.frames = 0  # passes of the drawing loop so far (the tests wait on it)

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive() and not self._stop.is_set()

    # -- start -------------------------------------------------------------------------------------------------------------------

    def start(self) -> None:
        """Enter the alternate screen and install the resize handler (on the calling thread, which must be the main one for a
        real signal), then start drawing on the daemon thread."""
        if self._started:
            return
        self._started = True
        self._terminal.enter()
        self._old_winch = self._install_signal(signal.SIGWINCH, self._on_resize)
        self._thread = threading.Thread(target=self._loop, name="tensixinet-tui", daemon=True)
        self._thread.start()

    def _on_resize(self, signum, frame) -> None:
        self._resized.set()

    def disarm(self) -> None:
        """The run loop has returned: `q` sends nothing from now on (it would otherwise interrupt what follows, such as a 40 s chip
        release)."""
        with self._irq_lock:
            self._armed = False

    def _quit(self) -> None:
        """`q`: send the interrupt, unless the run has gone past the point where one is wanted. The check and the send are one
        step under the lock, so `close` cannot begin between them."""
        with self._irq_lock:
            if not self._armed:
                return
            self._stopping = True
            self._interrupt()

    # -- the drawing thread ------------------------------------------------------------------------------------------------------

    def _loop(self) -> None:
        # The signals are for the main thread: blocked here, the kernel hands a SIGINT sent to the process to the main thread.
        signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGWINCH})
        while not self._stop.is_set():
            try:
                key = self._terminal.read_key(self._frame_s)
                if self._stop.is_set():
                    break
                if key in QUIT_KEYS:
                    self._quit()
                elif key in REDRAW_KEYS:
                    self._redraw.set()
                self._draw()
            except Exception as exc:  # kept for `close`; the run itself never sees it
                if self._error is None:
                    self._error = "".join(traceback.format_exception_only(type(exc), exc)).strip()
                self._last_frame = self._last_cells = None  # the screen may hold half a frame: redraw it all next time
            self.frames += 1

    def _log_rows(self, records, width: int):
        """The log pane's rows, made again only when the records (their number and the newest) or the width changed."""
        newest = records[-1] if records else None
        key = (len(records), width)
        if self._rows_key is None or self._rows_key[0] != key or self._rows_key[1] is not newest:
            self._rows = log_rows(records, width)
            self._rows_key = (key, newest)
        return self._rows

    def _draw(self) -> None:
        full = False
        if self._resized.is_set():
            self._resized.clear()
            full = True
        if self._redraw.is_set():
            self._redraw.clear()
            full = True
        now = self._clock()
        if self._drawn_at is None or now - self._drawn_at >= REDRAW_S:
            full = True
        if full:
            self._last_frame = self._last_cells = None
            self._drawn_at = now
        width, height = self._terminal.size()
        shown = self._playhead.shown()
        records = shown.records
        newest = records[-1] if records else None
        notice = STOPPING if self._stopping else None
        # The state and the newest record are compared by identity (kept here, so neither id can be reused while compared).
        key = ((width, height), len(records), self._colour, notice)
        if (self._last_cells is not None and self._key is not None and self._key[0] == key
                and self._key[1] is shown.state and self._key[2] is newest):
            return
        frame = render(shown.state, self._log_rows(records, width), width, height, self._colour, self._unit, notice=notice)
        self._key = (key, shown.state, newest)
        if self._last_cells is not None and frame == self._last_frame:
            return
        cells = frame.cells()
        text = diff(self._last_cells, cells, self._colour, self._truecolor)
        self._last_frame, self._last_cells = frame, cells
        if text and not self._stop.is_set():  # once `close` has begun, nothing more is drawn over the restored screen
            self._terminal.write(text)
            self._terminal.flush()

    # -- exit --------------------------------------------------------------------------------------------------------------------

    def close(self, code: int) -> List[str]:
        """Stop drawing, put the terminal back, then print and return the exit lines. A second call returns the same lines and
        does nothing else."""
        if self._closed is not None:
            return self._closed
        if self._started:
            try:
                self.disarm()  # before the stop flag: a q from here on sends nothing
                self._stop.set()
                if self._thread is not None:
                    self._thread.join(2 * self._frame_s)  # a write stuck on the terminal is left behind: the thread is a daemon
            finally:
                # Whatever was raised above (an interrupt that landed here), the handler and the terminal are put back.
                old = self._old_winch if self._old_winch is not None else signal.SIG_DFL
                try:
                    self._install_signal(signal.SIGWINCH, old)
                finally:
                    self._terminal.leave()
        lines = self._exit_lines(code)
        self._closed = lines
        for line in lines:
            self._terminal.write(line + "\n")
        self._terminal.flush()
        return lines

    def _exit_lines(self, code: int) -> List[str]:
        lines: List[str] = []
        if code not in NORMAL_CODES:
            for record in self._playhead.last_records(ERROR_RECORDS):
                lines.append(str(record.get("what", "")))
                why = record.get("why", "")
                if why:
                    lines.append(f"  why: {why}")
        if self._run_error is not None:
            lines.append(f"tensixinet: {self._run_error}")
        if self._error is not None:
            lines.append(f"tensixinet: the screen stopped drawing a frame: {self._error}")
        last = self._playhead.last_records(1)
        what = str(last[0].get("what", "")) if last else "the run ended before its first log line"
        lines.append(f"tensixinet: {what}; the log is in {self._log_dir}")
        return lines

    def __enter__(self) -> "TuiApp":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is not None:
            if issubclass(exc_type, KeyboardInterrupt):
                self.close(130)  # Ctrl-C or q: a normal end
            else:
                self._run_error = "".join(traceback.format_exception_only(exc_type, exc)).strip()
                self.close(1)
        return False  # the exception goes on to the caller
