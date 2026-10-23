# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The engine behind the Play tab: a thread that renders the chorus from the current knobs and plays it out in short blocks.

`run_live` (live.py) reads the chips once a segment of several seconds. The Play tab has to answer a hand, so the knobs are read every block
(`BLOCK_S` seconds) and a block is cut from a *phrase*, one continuous render of the chorus. Rendering a phrase restarts every device's clock, and
the slow devices need room: the gong's first strike can land two seconds in, so a bare 1.2 s render of it would often be silence. So:

* A knob that has moved, an error or trip, or a change in a chip's eviction state starts a new, short phrase (`SHORT_BLOCKS` blocks), so a hand
  is heard within about a block.
* While nothing moves, a phrase that runs out is followed by a long one (`LONG_BLOCKS` blocks), so the slow devices get their time.
* A new phrase takes over with an equal-power crossfade of `XFADE_S`, from the old phrase's own continuation, so there is no cut.

The loop stays at most `lead_s` seconds of audio ahead of the listener. The sound reaches the ear about `lead_s` plus one block after a knob moves,
plus whatever the browser buffers. Known limit, as in live mode: a phrase restarts the devices' clocks, so a ringing gong fades at the join
(`docs/ROADMAP.md`, item 3). The session holds no UI library, so it is tested without one.
"""
from __future__ import annotations

import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from ..music.chorus import render_chorus, slot_weights
from ..music.eq import Equaliser
from ..music.evict import EvictTracker
from ..music.live import Tracker
from ..music.loudness import LiveGain
from ..music.patch import Patch
from ..music.turns import pick_soloist
from .knobs import Knobs, to_telemetry
from .scopes import heard_on, resolve_scope

BLOCK_S = 1.2  # seconds of sound handed over at a time: the knobs are read this often
XFADE_S = 0.2  # a phrase is rendered this much longer than its blocks; the extra is what the next phrase crossfades from
SAMPLE_RATE = 22050  # as the app's other tab
SHORT_BLOCKS = 2  # a phrase started by a change: cheap to render, so the hand is heard quickly
LONG_BLOCKS = 5  # a phrase that follows one that ran out with nothing moving: the slow devices get their time
PACE_SLICE_S = 0.25  # the pacing waits in slices this long, so a stop is noticed within one
BLOCK_SAMPLES = int(BLOCK_S * SAMPLE_RATE)
XFADE_SAMPLES = int(XFADE_S * SAMPLE_RATE)


def phrase_samples(blocks: int) -> int:
    """How many samples a phrase of `blocks` blocks is rendered to: the blocks and the extra the next phrase crossfades from."""
    return blocks * BLOCK_SAMPLES + XFADE_SAMPLES


def _signature(chips, defrag) -> tuple:
    """What a phrase was made from, as far as a knob can change it (rates and running counters are left out: they follow from these). A phrase is
    replaced when this differs."""
    return (tuple((t.asic_temperature, t.aiclk, t.power, t.voltage, t.gddr_temps) for t in chips),
            tuple(d.phase for d in defrag) if defrag is not None else ())


@dataclass(frozen=True)
class InstrumentConfig:
    """What a Play session was started with. A change of any of these starts a new session; the knobs are not among them, they change the next block."""
    scope: str  # "system", "module:<name>" or "device:<name>" (scopes.py)
    chips: int
    style: str
    seed: int
    stereo: bool
    cables: bool
    evict: bool


@dataclass(frozen=True)
class InstrumentSnapshot:
    running: bool
    error: Optional[str]
    block: int  # the block the listener has reached, -1 before the first
    telemetry: tuple  # that block's readings
    playing: tuple  # the devices heard in it
    notes: tuple  # plain lines about it: a device that is not dealt to the chips, an eviction or a fill


class InstrumentSession:
    """One visitor's instrument: `start()` it, take audio with `next_audio()`, read what is playing with `snapshot()`, `stop()` it."""

    def __init__(self, config: InstrumentConfig, knobs: Knobs, patch: Patch, render: Callable = render_chorus,
                 clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep, lead_s: float = 1.5):
        self.config, self.knobs, self.patch = config, knobs, patch
        self.sample_rate = SAMPLE_RATE
        self._only = resolve_scope(config.scope)
        self._render, self._clock, self._sleep, self._lead_s = render, clock, sleep, lead_s
        k = (np.arange(XFADE_SAMPLES) + 0.5) / XFADE_SAMPLES
        self._fade_in, self._fade_out = np.sin(0.5 * np.pi * k), np.cos(0.5 * np.pi * k)  # equal power, as the stitcher's
        self._equaliser = Equaliser(SAMPLE_RATE)
        self._gain = LiveGain()
        self._tracker = Tracker()
        self._evictor = EvictTracker() if config.evict and self._only is None else None  # a scope that plays some devices ignores eviction
        self._soloist: Optional[int] = None
        self._block = 0  # blocks handed over
        self._phrase: Optional[np.ndarray] = None  # the render being played out, and how far into it
        self._pos = 0
        self._sig: Optional[tuple] = None
        self._phrases = 0  # phrases rendered: the next one's segment number
        self._meta: tuple = ((), (), ())  # the playing phrase's readings, devices heard and notes
        self._stop = threading.Event()
        self._audio: "queue.Queue[np.ndarray]" = queue.Queue()
        self._lock = threading.Lock()
        self._delivered = 0.0  # seconds of audio handed over
        self._t0: Optional[float] = None  # the clock at the first hand-over: the listener's time 0
        self._pending: deque = deque()  # (seconds into the audio where a block starts, block, readings, devices heard, notes)
        self._shown: Optional[tuple] = None
        self._error: Optional[str] = None
        self._thread: Optional[threading.Thread] = None

    # -- the loop ------------------------------------------------------------------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("a session runs once; make a new one")
        self._thread = threading.Thread(target=self._run, name="tensixinet-instrument", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                self._pace()  # the knobs are read after the wait, so a block carries the hand as late as it can
                if self._stop.is_set():
                    break
                self.render_block()
        except Exception as e:  # the thread has nowhere to raise to; the snapshot carries it
            self._error = f"{type(e).__name__}: {e}"

    def _ahead(self) -> float:
        """Seconds of audio handed over that the listener has not reached yet."""
        return self._delivered - (self._clock() - self._t0) if self._t0 is not None else 0.0

    def _pace(self) -> None:
        """Wait until the audio handed over is no more than `lead_s` ahead of the listener."""
        while not self._stop.is_set():
            over = self._ahead() - self._lead_s
            if over <= 1e-9:
                return
            self._sleep(min(PACE_SLICE_S, over))

    def _fades(self, n: int, channels: int):
        fade_in, fade_out = self._fade_in[:n], self._fade_out[:n]
        return (fade_in[:, None], fade_out[:, None]) if channels else (fade_in, fade_out)

    def render_block(self) -> None:
        """One pass: read the knobs; start a phrase if something changed or the last one ran out; hand over the next block of the phrase, through the
        EQ and the level. A stop that comes during the render drops the block."""
        now = self._clock()
        chips = to_telemetry(self.knobs, self._tracker, now)
        notes = []
        defrag = None
        if self._evictor is not None:
            defrag = self._evictor.update(chips, now)
            notes += [event.what for event in self._evictor.take_events()]
        sig = _signature(chips, defrag)
        news = any(t.corr_errs or t.uncorr_errs or t.thermal_trips for t in chips)  # an error or a trip is heard, so it starts a phrase
        changed = self._phrase is None or sig != self._sig or news
        exhausted = self._phrase is not None and self._pos + BLOCK_SAMPLES > len(self._phrase) - XFADE_SAMPLES
        if changed or exhausted:
            block = self._start_phrase(chips, defrag, sig, notes, SHORT_BLOCKS if changed else LONG_BLOCKS)
            if block is None:
                return
        else:
            block = self._phrase[self._pos: self._pos + BLOCK_SAMPLES]
            self._pos += BLOCK_SAMPLES
        ready, _ = self._gain.apply(self._equaliser.process(block))
        with self._lock:
            if self._t0 is None:
                self._t0 = self._clock()
            self._pending.append((self._delivered, self._block) + self._meta)
            self._delivered += len(ready) / SAMPLE_RATE
        self._audio.put(np.clip(ready, -1.0, 1.0).astype(np.float32))
        self._block += 1

    def _start_phrase(self, chips, defrag, sig, notes, blocks) -> Optional[np.ndarray]:
        """Render a phrase of `blocks` blocks and return its first block, crossfaded from the old phrase's continuation. None if stopped meanwhile."""
        extra = {}
        if self._only is None:
            self._soloist = pick_soloist(slot_weights(chips, self.patch), self._phrases, self._soloist)
            extra["soloists"] = [self._soloist]
        n = phrase_samples(blocks)
        phrase = self._render(chips, (n + 0.5) / SAMPLE_RATE, SAMPLE_RATE, patch=self.patch, segment=self._phrases, only=self._only,
                              cables=None if self.config.cables else [], stereo=self.config.stereo, defrag=defrag, **extra)
        if self._stop.is_set():
            return None
        block = np.array(phrase[:BLOCK_SAMPLES], dtype=np.float64)
        if self._phrase is not None:  # take over from where the old phrase would have gone on
            fade_in, fade_out = self._fades(XFADE_SAMPLES, block.ndim == 2)
            block[:XFADE_SAMPLES] = block[:XFADE_SAMPLES] * fade_in + self._phrase[self._pos: self._pos + XFADE_SAMPLES] * fade_out
        self._phrase, self._pos, self._sig = phrase, BLOCK_SAMPLES, sig
        self._phrases += 1
        heard = heard_on(self._only, len(chips))
        wanted = list(self._only) if self._only is not None else heard
        notes = [f"{name} is not dealt to any of the {len(chips)} chip{'s' if len(chips) != 1 else ''}, so it is silent"
                 for name in wanted if name not in heard] + notes
        self._meta = (tuple(chips), tuple(heard), tuple(notes))
        return block.astype(np.float32)

    # -- what the app reads --------------------------------------------------------------------------------------------------------

    def _advance(self) -> None:
        """Show every block the listener has reached."""
        if self._t0 is None:
            return
        elapsed = self._clock() - self._t0
        with self._lock:
            while self._pending and self._pending[0][0] <= elapsed + 1e-9:
                self._shown = self._pending.popleft()[1:]

    def snapshot(self) -> InstrumentSnapshot:
        self._advance()
        running = self._thread is not None and self._thread.is_alive() and not self._stop.is_set()
        with self._lock:
            block, chips, heard, notes = self._shown if self._shown is not None else (-1, (), (), ())
        return InstrumentSnapshot(running, self._error, block, chips, heard, notes)

    def next_audio(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """The next block's audio, or None if there is none within `timeout` seconds."""
        try:
            return self._audio.get(timeout=timeout)
        except queue.Empty:
            return None
