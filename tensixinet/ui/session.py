# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""One listening session for the app: the live loop on a thread, the audio it makes, and what is playing, kept as plain data.

`Session` runs `run_live` (live.py) exactly as the command line does, with a sink that hands each segment to whoever asks for it and
waits for as long as the audio lasts, so the loop stays about one segment ahead of what is heard. The run log's own events say which
device and chip have the floor and when a chip evicts or fills; `build_flow` (flow.py) turns the readings of that moment and those
events into the state a chart draws. The session holds no UI library, so it is tested without one.
"""
from __future__ import annotations

import io
import json
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np

from ..kernels.backend import KERNEL_DEVICES, KernelBackend
from ..live import run_live
from ..media.chip import ChipError
from ..music.chorus import assign_device, n_slots, slot_chip
from ..music.live import Tracker
from ..music.patch import Patch, draw_patch
from ..music.styles import STYLES, draw_styled_patch
from ..narrate.runlog import RunLog
from ..telemetry import Telemetry, fetch_snapshot, mock_telemetry, parse_snapshot
from .flow import FlowState, build_flow, phases_at

LOG_LINES = 400  # how much of the run log a snapshot keeps
PACE_SLICE_S = 0.25  # the sink waits out a segment in slices this long, so a stop is noticed within one


class _Stopped(BaseException):
    """Raised by the sink to end the live loop when the session is stopped."""


class _Events:
    """A file the run log writes its JSON lines to; each line goes to the session as it arrives."""

    def __init__(self, handle: Callable[[dict], None]):
        self._handle, self._partial = handle, ""

    def write(self, text: str) -> int:
        self._partial += text
        *lines, self._partial = self._partial.split("\n")
        for line in lines:
            if line.strip():
                self._handle(json.loads(line))
        return len(text)

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


@dataclass(frozen=True)
class Snapshot:
    state: Optional[FlowState]  # the segment that is playing now, None before the first one starts
    log_lines: tuple  # the plain-text log, one string per event: "[mm:ss.s] what"
    records: tuple  # the same events as dicts (t, event, what, why, and any fields)
    segment: int  # the number of the segment that is playing, -1 before the first
    running: bool
    error: Optional[str]


def make_patch(style: str, seed: int, chips: Sequence[Telemetry]) -> Patch:
    """The patch for a session: drawn in `style` (one of STYLES, "" for a plain draw) from `seed` and the chips' readings."""
    if style and style not in STYLES:
        raise ValueError(f"unknown style {style!r}; the styles are {', '.join(STYLES)}")
    return draw_styled_patch(seed, chips, style) if style else draw_patch(seed, chips)


def mock_sample(n_chips: int, seed: int = 0, load_period: Optional[float] = None, clock: Callable[[], float] = time.monotonic):
    """Synthetic chips. With `load_period` each chip goes round a load cycle of that many seconds (rest, load, rest), the chips staggered,
    so an eviction can be seen without a model."""
    start = clock()
    return lambda: mock_telemetry(n_chips, seed, load_period=load_period, elapsed=clock() - start)


def real_sample():
    """The real chips, read with tt-smi (read-only, opens no device)."""
    return lambda: parse_snapshot(fetch_snapshot())


class _UiSink:
    def __init__(self, session: "Session"):
        self._session = session

    def write(self, audio: np.ndarray) -> None:
        s = self._session
        if s._stop.is_set():
            raise _Stopped()
        s._begin_segment(s._writes)  # each segment is one write (the last write is the tail that rings out)
        s._writes += 1
        s._audio.put(audio)
        remaining = len(audio) / s.sample_rate
        while remaining > 0 and not s._stop.is_set():
            step = min(PACE_SLICE_S, remaining)
            s._sleep(step)
            remaining -= step
        if s._stop.is_set():
            raise _Stopped()

    def close(self) -> None:
        pass


class Session:
    def __init__(
        self,
        sample: Callable[[], Sequence[Telemetry]],
        patch: Patch,
        source: str = "the telemetry source",
        sample_rate: int = 22050,
        stereo: bool = True,
        evict: bool = True,
        cables: bool = True,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        worker=None,
    ):
        self._worker = worker  # a chip worker (see media/chip.py) that has not started: the session starts it, runs the kernels through it and closes it
        self.sample_rate, self._sleep, self._clock = sample_rate, sleep, clock
        self._sample, self._patch, self._source = sample, patch, source
        self._stereo, self._evict, self._cables = stereo, evict, cables
        self._stop = threading.Event()
        self._audio: "queue.Queue[np.ndarray]" = queue.Queue()
        self._lock = threading.Lock()
        self._lines: deque = deque(maxlen=LOG_LINES)
        self._records: deque = deque(maxlen=LOG_LINES)
        self._evictions: list = []  # (t, "evict" or "fill", chip)
        self._chips: Sequence[Telemetry] = []
        self._tracker = Tracker()
        self._states: dict = {}  # segment number -> the state of that segment, kept until it has played
        self._playing: Optional[int] = None  # the segment whose audio the sink began last
        self._writes = 0
        self._error: Optional[str] = None
        self._thread: Optional[threading.Thread] = None

    # -- the live loop -------------------------------------------------------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("a session runs once; make a new one")
        self._thread = threading.Thread(target=self._run, name="tensixinet-session", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 20.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)

    def _watched_sample(self) -> Sequence[Telemetry]:
        """The source's reading, noted with the rates the music is given, and passed on unchanged for `run_live` to track itself."""
        readings = self._sample()
        if readings:
            self._chips = self._tracker.update(readings, self._clock())
        return readings

    def _run(self) -> None:
        log = RunLog(_Events(self._on_event), io.StringIO(), clock=self._clock, dedupe_why=True)
        backend = None
        try:
            if self._worker is not None:
                self._worker.start()
                backend = KernelBackend(self._worker, log=lambda note: log.event(
                    "backend", note, "the kernels' formulas run on the CPU from here on, so the sound does not stop"))
                log.event("backend", "TT-Lang kernels in the simulator (no chip)",
                          f"the {', '.join(KERNEL_DEVICES)} devices and the mixer do their per-sample arithmetic in the kernels, run by the TT-Lang "
                          "simulator on the CPU, and the rest stay on the CPU")
            run_live(self._watched_sample, _UiSink(self), log, self.sample_rate, duration=None, source=self._source, patch=self._patch,
                     patch_source="the session's patch", stereo=self._stereo, evict=self._evict, cables=self._cables, backend=backend)
        except _Stopped:
            pass
        except (ValueError, ChipError) as e:
            self._error = str(e)
        except BaseException as e:  # the thread has nowhere to raise to; the snapshot carries it
            if not isinstance(e.__cause__, _Stopped) and not self._stop.is_set():
                self._error = f"{type(e).__name__}: {e}"
        finally:
            if self._worker is not None:
                self._worker.close()

    def _on_event(self, rec: dict) -> None:
        with self._lock:
            mins, secs = divmod(rec["t"], 60)
            self._lines.append(f"[{int(mins):02d}:{secs:04.1f}] {rec['what']}")
            self._records.append(rec)
            if rec["event"] in ("evict", "fill"):
                self._evictions.append((rec["t"], rec["event"], rec["chip"]))
            if rec["event"] == "live" and "device" in rec and self._chips:
                self._states[rec["segment"]] = self._flow_for(rec)

    def _flow_for(self, rec: dict) -> FlowState:
        n = len(self._chips)
        slot = next(s for s in range(n_slots(n)) if slot_chip(s, n) == rec["chip"] and assign_device(s).__name__ == rec["device"])
        return build_flow(self._chips, self._patch, rec["segment"], slot, rec["seconds"], phases_at(self._evictions, rec["t"], n),
                          cables=self._cables)

    # -- what the app reads --------------------------------------------------------------------------------------------------

    def _begin_segment(self, segment: int) -> None:
        """The sink is about to play `segment`: from now the snapshot shows its state. The loop renders up to two segments ahead of the
        sound, so older states are dropped here and newer ones wait."""
        with self._lock:
            self._playing = segment
            for old in [k for k in self._states if k < segment]:
                del self._states[old]

    def snapshot(self) -> Snapshot:
        with self._lock:
            running = self._thread is not None and self._thread.is_alive() and not self._stop.is_set()
            state = self._states.get(self._playing) if self._playing is not None else None
            segment = self._playing if state is not None else -1
            return Snapshot(state, tuple(self._lines), tuple(self._records), segment, running, self._error)

    def next_audio(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """The next segment's audio, or None if there is none within `timeout` seconds."""
        try:
            return self._audio.get(timeout=timeout)
        except queue.Empty:
            return None
