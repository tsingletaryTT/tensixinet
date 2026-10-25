# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Where a render's `Score` goes: a Standard MIDI File, or several places at once."""
import heapq
import threading
import time
from typing import Callable, List, Optional, Protocol, Tuple

from .events import Score
from .midi import Message, to_messages
from .routing import DEFAULT_ROUTES, Routes
from .smf import write_midi_file


class ScoreSink(Protocol):
    """Takes each render's score and the time (seconds from the start of the run) it begins at."""

    def add(self, score: Score, start_s: float) -> None: ...

    def close(self) -> None: ...


class TeeScoreSink:
    """Hands every score to each sink in turn."""

    def __init__(self, *sinks) -> None:
        self.sinks = sinks

    def add(self, score: Score, start_s: float) -> None:
        for sink in self.sinks:
            sink.add(score, start_s)

    def close(self) -> None:
        """Closes every sink, even when one fails, then raises the first failure."""
        first = None
        for sink in self.sinks:
            try:
                sink.close()
            except Exception as e:
                first = first or e
        if first is not None:
            raise first

    @property
    def error(self) -> Optional[str]:
        """The first error any sink has kept, or `None`."""
        for sink in self.sinks:
            if getattr(sink, "error", None) is not None:
                return sink.error
        return None

    def summary(self) -> Tuple[int, int, int]:
        """The first sink's counts that has any (every sink was handed the same scores), zeros when none has."""
        for sink in self.sinks:
            if hasattr(sink, "summary"):
                return sink.summary()
        return 0, 0, 0

    @property
    def path(self):
        """The first sink's file path, or `None` when no sink writes a file."""
        for sink in self.sinks:
            if getattr(sink, "path", None) is not None:
                return sink.path
        return None


class MidiFileSink:
    """Keeps every score's messages, shifted to its start, and writes one file on `close()`.
    The tempo in the file is the first score's. With no scores the file holds only the tempo track."""

    DEFAULT_BPM = 120.0

    def __init__(self, path, routes: Routes = DEFAULT_ROUTES) -> None:
        self.path = path
        self.routes = routes
        self.bpm = None
        self.messages: List[Message] = []
        self.notes = self.triggers = self.controls = 0

    def add(self, score: Score, start_s: float) -> None:
        if self.bpm is None:
            self.bpm = score.bpm
        self.notes += len(score.notes())
        self.triggers += len(score.triggers())
        self.controls += len(score.controls())
        self.messages.extend((t + start_s, data, chip) for t, data, chip in to_messages(score, self.routes))

    def summary(self) -> Tuple[int, int, int]:
        """How many notes, triggers and controls have been added."""
        return self.notes, self.triggers, self.controls

    def close(self) -> None:
        write_midi_file(self.path, self.messages, self.bpm or self.DEFAULT_BPM)


class MidiPortError(RuntimeError):
    """The MIDI port cannot be used. The message is one plain line."""


class _MidoPort:
    """Wraps a mido output so it takes raw bytes."""

    def __init__(self, mido, port) -> None:
        self._mido, self._port = mido, port

    def send(self, raw: bytes) -> None:
        self._port.send(self._mido.Message.from_bytes(list(raw)))

    def close(self) -> None:
        self._port.close()


def open_mido_output(name: str):
    """Open the named MIDI output, or a virtual one called `tensixinet` for the name `virtual`.
    `mido` is imported here so nothing else needs it."""
    try:
        import mido
        mido.set_backend("mido.backends.rtmidi")
        names = mido.get_output_names()
    except ImportError:
        raise MidiPortError("a live MIDI port needs the mido and python-rtmidi libraries: pip install -e .[midi]") from None
    try:
        if name == "virtual":
            return _MidoPort(mido, mido.open_output("tensixinet", virtual=True))
        if name not in names:
            raise MidiPortError(f"no MIDI output called {name!r}; the outputs are {', '.join(repr(n) for n in names) or 'none'}")
        return _MidoPort(mido, mido.open_output(name))
    except MidiPortError:
        raise
    except Exception as e:  # rtmidi raises its own types
        raise MidiPortError(f"cannot open the MIDI output {name!r}: {e}") from None


class MidiPortSink:
    """Plays each score's messages on a live MIDI port at their time.

    The first `add` fixes `time_base` (the clock then); a message is due at
    `time_base + start_s + t + offset_ms / 1000`. A scheduler thread sends what is due and sleeps until
    the next. `add` never blocks and never raises: a score that cannot be routed, or a port that fails,
    is kept as text in `error` (the first one only, set once) and the sink drops everything after that. The
    sink never logs: its thread must not touch the run log, so the owner reads `error` (a plain attribute
    read that takes no lock) and reports it from its own thread. `on_error`, when given, is also called
    once with the exception. `close()` drops what is
    not yet due, ends every note still sounding, sends all-notes-off (controller 123) on each channel the
    sink used, closes the port and joins the thread.

    `_pump(now)` sends what is due and returns when the next message is due (`None` when none is
    waiting). The thread calls it, and tests call it directly with a fake clock (`threaded=False`)."""

    CLOSE_WAIT_S = 2.0  # the longest `close()` waits for the thread, and for the lock, when a send has hung

    SLICE_S = 0.02  # the longest the thread sleeps before it looks at the heap again (a new add may be due sooner)

    def __init__(self, port_name: str, routes: Routes = DEFAULT_ROUTES, offset_ms: float = 0.0, open_output=None,
                 clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep,
                 on_error: Optional[Callable[[BaseException], None]] = None, threaded: bool = True) -> None:
        self.routes, self.offset_s = routes, offset_ms / 1000.0
        self._clock, self._sleep = clock, sleep
        self._on_error = on_error or (lambda e: None)
        self._threaded = threaded
        self._port = (open_output or open_mido_output)(port_name)  # raises MidiPortError before anything else happens
        self._lock = threading.Condition()
        self._heap: list = []
        self._seq = 0
        self._time_base: Optional[float] = None
        self._sounding: set = set()
        self._channels: set = set()
        self._dead = False
        self._error: Optional[str] = None
        self._closed = False
        self._thread: Optional[threading.Thread] = None
        self.notes = self.triggers = self.controls = 0

    def _fail(self, error: BaseException) -> None:
        if not self._dead:
            self._dead = True
            self._error = str(error) or type(error).__name__
            self._heap.clear()
            try:
                self._on_error(error)
            except Exception:
                pass

    def add(self, score: Score, start_s: float) -> None:
        try:
            with self._lock:
                if self._dead or self._closed:
                    return
                messages = to_messages(score, self.routes)
                if self._time_base is None:
                    self._time_base = self._clock()
                base = self._time_base + start_s + self.offset_s
                for t, data, _chip in messages:
                    heapq.heappush(self._heap, (base + t, self._seq, data))
                    self._seq += 1
                self.notes += len(score.notes())
                self.triggers += len(score.triggers())
                self.controls += len(score.controls())
                if self._threaded and self._thread is None:
                    self._thread = threading.Thread(target=self._run, name="tensixinet-midi", daemon=True)
                    self._thread.start()
                self._lock.notify_all()
        except Exception as e:
            with self._lock:
                self._fail(e)

    def _send(self, raw: bytes) -> None:
        """Send one message and keep track of which notes sound and which channels were used. Caller holds the lock."""
        self._port.send(raw)
        kind, channel = raw[0] & 0xF0, raw[0] & 0x0F
        self._channels.add(channel)
        if kind == 0x90 and raw[2] > 0:
            self._sounding.add((channel, raw[1]))
        elif kind in (0x80, 0x90):
            self._sounding.discard((channel, raw[1]))

    @property
    def error(self) -> Optional[str]:
        """Why the sink stopped, or `None`. Reads one attribute and takes no lock, so it returns at once even
        while the scheduler thread is inside a slow `port.send`."""
        return self._error

    def _pump(self, now: float) -> Optional[float]:
        """Send what is due and return when the next message is due. `port.send` runs under the lock, so an
        `add` that arrives during a send waits for it (rtmidi sends return in microseconds)."""
        with self._lock:
            while self._heap and self._heap[0][0] <= now and not self._dead:
                _, _, raw = heapq.heappop(self._heap)
                try:
                    self._send(raw)
                except Exception as e:
                    self._fail(e)
            return self._heap[0][0] if self._heap and not self._dead else None

    def _run(self) -> None:
        while True:
            with self._lock:
                if self._closed:
                    return
            nxt = self._pump(self._clock())
            if nxt is None:
                with self._lock:
                    if self._closed:
                        return
                    if not self._heap or self._dead:
                        self._lock.wait(self.SLICE_S)
                continue
            self._sleep(max(0.0, min(nxt - self._clock(), self.SLICE_S)))

    def summary(self) -> Tuple[int, int, int]:
        """How many notes, triggers and controls have been added."""
        return self.notes, self.triggers, self.controls

    def close(self) -> None:
        """Stop, end every sounding note, send all-notes-off on each channel used, close the port. Waits at
        most `CLOSE_WAIT_S` for the thread and the lock, so a port whose send never returns cannot hang the
        exit; the port is then left as it is and `error` says so."""
        if not self._lock.acquire(timeout=self.CLOSE_WAIT_S):
            self._closed = True  # a send holds the lock: say so without it
            if self._error is None:
                self._error = "the port did not answer when closing"
            return
        try:
            if self._closed:
                return
            self._closed = True
            self._heap.clear()
            self._lock.notify_all()
        finally:
            self._lock.release()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(self.CLOSE_WAIT_S)
        if not self._lock.acquire(timeout=self.CLOSE_WAIT_S):
            if self._error is None:
                self._error = "the port did not answer when closing"
            return
        try:
            if not self._dead:
                try:
                    for channel, number in sorted(self._sounding):
                        self._send(bytes([0x80 | channel, number, 0]))
                    for channel in sorted(self._channels):
                        self._send(bytes([0xB0 | channel, 123, 0]))
                except Exception as e:
                    self._fail(e)
            try:
                self._port.close()
            except Exception:
                pass
        finally:
            self._lock.release()
