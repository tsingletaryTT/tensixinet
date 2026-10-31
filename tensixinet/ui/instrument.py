# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The engine behind the Play tab: a thread that renders the chorus from the current knobs and plays it out in short blocks.

`run_live` (live.py) reads the chips once a segment of several seconds. The Play tab has to answer a hand, so the knobs are read every block
(`BLOCK_S` seconds by default) and a block is cut from a *phrase*, one continuous render of the chorus. Rendering a phrase restarts every device's
clock, and the slow devices need room: the gong's first strike can land two seconds in, so a bare 1.2 s render of it would often be silence. So:

* A knob that has moved, an error or trip, or a change in a chip's eviction state starts a new, short phrase (`SHORT_BLOCKS` blocks), so a hand
  is heard within about a block.
* While nothing moves, a phrase that runs out is followed by a long one (`LONG_BLOCKS` blocks), so the slow devices get their time.
* A new phrase takes over with an equal-power crossfade of `XFADE_S`, from the old phrase's own continuation, so there is no cut.
* A session that reads real chips many times a second can damp the first rule. With `changed` (for example `signature_changed`), a reading starts a phrase
  only when `changed(signature of the playing phrase, signature now)` is true and at least `min_gap_s` has passed since the last phrase started
  (`min_gap_s` is 0 by default and applies with or without `changed`). An error, a trip or an eviction phase change still starts one at once. The comparison is against the playing phrase, so a change held back by
  the gap is heard when the gap ends, and `held_s` says how long it waited. Without `changed`, any difference at all starts a phrase, which the
  Play tab's knobs need.
* The system scope gives the floor to one chip's device per phrase (`pick_soloist`). With `hold_soloist`, the soloist keeps the floor until the
  seconds that function returns (the patch's turn length) have passed since it was chosen, through every phrase a change starts meanwhile.

The constants are the Play tab's defaults. A session takes its input and its timing as parameters: `read` supplies readings that are already
tracked (so a caller can drive it from real chips with no knobs), `block_s`, `short_s`, `long_s`, `xfade_s` and `sample_rate` set the block, the two
phrase lengths, the crossfade and the rate, `only` names the devices to play, and `eq` and `normalise` switch the EQ and the level gain off. A phrase
is `round(short_s / block_s)` or `round(long_s / block_s)` blocks, at least one, and the crossfade is cut down to one block if it is longer.

The loop stays at most `lead_s` seconds of audio ahead of the listener. The sound reaches the ear about `lead_s` plus one block after a knob moves,
plus whatever the browser buffers. Known limit, as in live mode: a phrase restarts the devices' clocks, so a ringing gong fades at the join
(`docs/ROADMAP.md`, item 3). The session holds no UI library, so it is tested without one.

Two options let a session read real chips many times a second without the render holding up the loop (fast mode, `tensixinet/fast.py`). Both are
off by default, and then the session renders every phrase on its own thread as described above.

* `head_s`: a phrase is started from a *head*, a render of only its first `head_s` seconds (rounded to whole blocks), plus the crossfade and
  `HEAD_MARGIN_S` more. A render's last 50 ms or so differ from a longer render of the same phrase (the margin keeps the join out of them). The
  head is rendered at once and its blocks are handed over, while one background thread (`RenderWorker`) renders the *body*: the full phrase
  from the same readings, with the same segment number and soloist. When the head's blocks are played, the body takes over at the same time
  offset, so at that join the gong and the rolls go on and nothing restarts (a new phrase still restarts them). The head and the body are the same music, so that join is a
  straight (equal-gain) crossfade; an equal-power one would be 3 dB loud in its middle. A body that is not ready when the head runs out is
  late: the head's own continuation is faded out over the crossfade, silence follows, each such block counts in `late_blocks`, and the body
  comes in, faded in from the silence at the listener's offset, as soon as it is ready. The loop never waits for it.
* `render_ahead`: in a quiet stretch, once the playing phrase is half played, the next long phrase is rendered in the background from the
  latest readings. When the playing phrase runs out, it takes over with the usual crossfade (a new phrase, so the devices restart). If it is
  still rendering then, a head of it is rendered at once and it becomes that head's body. A change discards it.

The background thread renders one phrase at a time. `stop()` ends it, and a render that finishes after the stop is dropped. A render that raises
there ends the session with its error, as a render on the loop's own thread does. `carry_from` lets a session that replaces another (fast mode
does this when it lengthens its block) keep the eviction tracker, the held soloist, the level gain and the EQ's state.
"""
from __future__ import annotations

import queue
import threading
import time
from collections import deque
import dataclasses
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

import numpy as np

from ..music.chorus import render_chorus, slot_weights
from ..music.eq import Equaliser
from ..music.evict import EvictTracker
from ..music.live import Tracker
from ..music.loudness import LiveGain
from ..music.patch import Patch
from ..telemetry.snapshot import Telemetry
from ..music.turns import pick_soloist
from .flow import FlowState, build_flow
from .knobs import Knobs, to_telemetry
from .scopes import heard_on, resolve_scope

BLOCK_S = 1.2  # seconds of sound handed over at a time: the knobs are read this often
XFADE_S = 0.2  # a phrase is rendered this much longer than its blocks; the extra is what the next phrase crossfades from
SAMPLE_RATE = 22050  # as the app's other tab
SHORT_BLOCKS = 2  # a phrase started by a change: cheap to render, so the hand is heard quickly
LONG_BLOCKS = 5  # a phrase that follows one that ran out with nothing moving: the slow devices get their time
PACE_SLICE_S = 0.25  # the pacing waits in slices this long, so a stop is noticed within one
HEAD_MARGIN_S = 0.15  # a head is rendered this far past its crossfade: the last ~50 ms of a render differ from a longer render of the same phrase
BLOCK_SAMPLES = int(BLOCK_S * SAMPLE_RATE)  # the defaults' sizes; a session works out its own from its parameters
XFADE_SAMPLES = int(XFADE_S * SAMPLE_RATE)


def phrase_samples(blocks: int, block_samples: int = BLOCK_SAMPLES, xfade_samples: int = XFADE_SAMPLES) -> int:
    """How many samples a phrase of `blocks` blocks is rendered to: the blocks and the extra the next phrase crossfades from."""
    return blocks * block_samples + xfade_samples


VOLTAGE_V = 0.01  # signature_changed: a core voltage move larger than this is heard (starting value, to tune by ear)
AICLK_MHZ = 0.5  # signature_changed: the clock is read in whole MHz, so any step of one MHz or more counts
GDDR_C = 0.5  # signature_changed: GDDR temperatures are read in whole degrees, so any step of one degree or more counts


def signature_changed(previous: Optional[tuple], current: tuple, temp_c: float = 0.5, power_w: float = 3.0) -> bool:
    """Whether `current` differs enough from `previous` (both built by `_signature`) to be worth a new phrase, for readings that wobble.

    True when there was no previous signature, the number of chips or GDDR channels differs, any chip's ASIC temperature moved by more than
    `temp_c` degrees, its power by more than `power_w` watts, its voltage by more than `VOLTAGE_V`, its clock by more than `AICLK_MHZ` or a GDDR
    temperature by more than `GDDR_C`, or any chip's eviction phase differs. A move exactly at a threshold does not count. The defaults are
    starting values, chosen so that the noise of a 100 ms hardware read stays under them; they are to be tuned by ear."""
    if previous is None:
        return True
    (old_chips, old_phases), (new_chips, new_phases) = previous, current
    if len(old_chips) != len(new_chips) or old_phases != new_phases:
        return True
    for (t0, clk0, p0, v0, g0), (t1, clk1, p1, v1, g1) in zip(old_chips, new_chips):
        if abs(t1 - t0) > temp_c or abs(p1 - p0) > power_w or abs(v1 - v0) > VOLTAGE_V or abs(clk1 - clk0) > AICLK_MHZ:
            return True
        if len(g0) != len(g1) or any(abs(b - a) > GDDR_C for a, b in zip(g0, g1)):
            return True
    return False


def _signature(chips, defrag) -> tuple:
    """What a phrase was made from, as far as a knob can change it (rates and running counters are left out: they follow from these). A phrase is
    replaced when this differs (or, with a session's `changed`, when that says it differs enough)."""
    return (tuple((t.asic_temperature, t.aiclk, t.power, t.voltage, t.gddr_temps) for t in chips),
            tuple(d.phase for d in defrag) if defrag is not None else ())


class RenderJob:
    """A render handed to the background thread. `done` is set when it has finished, with `result` or `error`. A job that is dropped (by a change
    or a stop) keeps no result."""

    def __init__(self, fn: Callable[[], np.ndarray]):
        self.fn = fn
        self.done = threading.Event()
        self.result: Optional[np.ndarray] = None
        self.error: Optional[BaseException] = None
        self.dropped = False

    def drop(self) -> None:
        self.dropped = True
        self.result = None

    @property
    def ready(self) -> bool:
        return self.done.is_set() and self.result is not None


class RenderWorker:
    """One daemon thread that runs a session's background renders, one at a time, in the order they were handed over. `stop` ends it: the render
    in progress finishes (numpy cannot be interrupted) and its result is dropped, and the jobs still queued are dropped unrun."""

    def __init__(self):
        self._jobs: "queue.Queue[Optional[RenderJob]]" = queue.Queue()
        self._stopped = threading.Event()
        self._thread = threading.Thread(target=self._run, name="tensixinet-render", daemon=True)
        self._thread.start()

    def submit(self, fn: Callable[[], np.ndarray]) -> RenderJob:
        job = RenderJob(fn)
        self._jobs.put(job)
        return job

    def _run(self) -> None:
        while True:
            job = self._jobs.get()
            if job is None:
                return
            if not (job.dropped or self._stopped.is_set()):
                try:
                    result = job.fn()
                except Exception as e:  # the session reads it from the job and ends with it
                    job.error = e
                else:
                    if not (job.dropped or self._stopped.is_set()):  # a render that finishes after a change or a stop is not kept
                        job.result = result
            job.done.set()

    def stop(self, timeout: Optional[float] = None) -> None:
        self._stopped.set()
        self._jobs.put(None)
        self._thread.join(timeout)


@dataclass
class _Plan:
    """What a phrase is rendered from, decided before it is rendered: a phrase rendered in the background is installed later, and a phrase that
    is discarded must leave the session's soloist as it was."""
    chips: list
    defrag: Optional[list]
    sig: tuple
    notes: list
    blocks: int
    soloist: Optional[int]
    until: float  # with hold_soloist: when the soloist's turn ends
    segment: int
    pick: Tuple[str, str] = ("", "")  # the Play tab's choir pick when the phrase was planned


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
    state: Optional[FlowState] = None  # the chart's data for that block (flow.py); None before the first


class InstrumentSession:
    """One visitor's instrument: `start()` it, take audio with `next_audio()`, read what is playing with `snapshot()`, `stop()` it."""

    def __init__(self, config: InstrumentConfig, knobs: Optional[Knobs], patch: Patch, render: Callable = render_chorus,
                 clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep, lead_s: float = 1.5,
                 read: Optional[Callable[[float], Sequence[Telemetry]]] = None, block_s: float = BLOCK_S,
                 short_s: float = SHORT_BLOCKS * BLOCK_S, long_s: float = LONG_BLOCKS * BLOCK_S, xfade_s: float = XFADE_S,
                 sample_rate: int = SAMPLE_RATE, only: Optional[Sequence[str]] = None, eq: bool = True, normalise: bool = True,
                 min_gap_s: float = 0.0, changed: Optional[Callable[[tuple, tuple], bool]] = None,
                 hold_soloist: Optional[Callable[[Sequence[Telemetry]], float]] = None, head_s: Optional[float] = None,
                 render_ahead: bool = False, worker_factory: Callable[[], RenderWorker] = RenderWorker):
        """`read(now)`, when given, supplies each block's readings (already tracked, rates filled in) in place of the knobs, which may then be None.
        `only` replaces the devices the scope names. `eq=False` skips the equaliser and `normalise=False` skips the level gain (the output is still
        clipped to full scale). `changed`, `min_gap_s` and `hold_soloist` damp how often a phrase starts and the floor changes hands, as the module
        docstring says; their defaults keep the Play tab's behaviour. `head_s` and `render_ahead` move the phrase renders onto a background thread,
        as the module docstring says; `worker_factory` makes that thread (a test hands in its own). The other keywords set the timing."""
        self.config, self.knobs, self.patch = config, knobs, patch
        self.sample_rate = sample_rate
        self._read = read
        self._only = list(only) if only is not None else resolve_scope(config.scope)
        self._render, self._clock, self._sleep, self._lead_s = render, clock, sleep, lead_s
        self._block_s = block_s
        self._block_samples = int(block_s * sample_rate)
        self._xfade_samples = min(int(xfade_s * sample_rate), self._block_samples)  # a crossfade cannot reach past the block it starts in
        self._short_blocks = max(1, round(short_s / block_s))
        self._long_blocks = max(1, round(long_s / block_s))
        k = (np.arange(self._xfade_samples) + 0.5) / self._xfade_samples
        self._fade_in, self._fade_out = np.sin(0.5 * np.pi * k), np.cos(0.5 * np.pi * k)  # equal power, as the stitcher's
        self._equaliser = Equaliser(sample_rate) if eq else None
        self._gain = LiveGain() if normalise else None
        self._tracker = Tracker()
        self._evictor = EvictTracker() if config.evict and self._only is None else None  # a scope that plays some devices ignores eviction
        self._changed, self._min_gap_s, self._hold_soloist = changed, min_gap_s, hold_soloist
        self._soloist: Optional[int] = None
        self._soloist_until = 0.0  # with hold_soloist: the session clock at which the held soloist's turn ends
        self._phrase_t: Optional[float] = None  # the session clock when the playing phrase started
        self._crossed_at: Optional[float] = None  # the read time of the first reading that moved past `changed` but waited for the gap
        self._block = 0  # blocks handed over
        self._phrase: Optional[np.ndarray] = None  # the render being played out, and how far into it
        self._pos = 0
        self._sig: Optional[tuple] = None
        self._pick: Tuple[str, str] = ("", "")  # the choir pick the playing phrase was rendered with
        self._wanted_pick: Tuple[str, str] = ("", "")  # the knobs' choir pick as of the block being made
        self._phrases = 0  # phrases rendered: the next one's segment number
        self._meta: tuple = ((), (), ())  # the playing phrase's readings, devices heard and notes
        self._phases: dict = {}  # the playing phrase's chips that are not running, for the chart
        self._stop = threading.Event()
        self._audio: "queue.Queue[np.ndarray]" = queue.Queue()
        self._lock = threading.Lock()
        self._delivered = 0.0  # seconds of audio handed over
        self._t0: Optional[float] = None  # the clock at the first hand-over: the listener's time 0
        self._pending: deque = deque()  # (seconds into the audio where a block starts, block, readings, devices heard, notes)
        self._shown: Optional[tuple] = None
        self._handed: Optional[FlowState] = None  # the state of the latest block handed over (`handed_state`), set with `_pending` under the lock
        self._error: Optional[str] = None
        self._timing: Optional[tuple] = None  # (read_at, written_at, held, rendered) of the latest block, set as one value so a reader sees a matching set
        self._loop_rendered = False  # whether the block being made rendered (or waited for a render) on the loop's own thread
        self._thread: Optional[threading.Thread] = None
        # heads, bodies and renders ahead (all unused when head_s is None and render_ahead is False)
        self._head_blocks = max(1, round(head_s / block_s)) if head_s is not None else None
        self._head_margin = int(HEAD_MARGIN_S * sample_rate)
        self._render_ahead = render_ahead
        self._worker_factory = worker_factory
        self._worker: Optional[RenderWorker] = None  # made on the first background render
        self._ramp = (np.arange(self._xfade_samples) + 0.5) / max(self._xfade_samples, 1)  # the straight crossfade from a head to its body
        self._body: Optional[RenderJob] = None  # the full phrase rendering for the head that plays (or that ran out: see _silent)
        self._body_samples = 0  # how long that full phrase is
        self._head_end: Optional[int] = None  # while a head plays: the sample its blocks end at, where the body takes over
        self._silent = False  # the head ran out before its body was ready: silence plays until the body comes
        self._upcoming: Optional[tuple] = None  # (job, plan) of the next long phrase, rendering in a quiet stretch
        self.late_blocks = 0  # blocks played as silence (or the head fading out) because a body was late

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
        if self._worker is not None:
            self._worker.stop(timeout)

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
        self._loop_rendered = False
        chips = list(self._read(now)) if self._read is not None else to_telemetry(self.knobs, self._tracker, now)
        notes = []
        defrag = None
        if self._evictor is not None:
            defrag = self._evictor.update(chips, now)
            notes += [event.what for event in self._evictor.take_events()]
        sig = _signature(chips, defrag)
        news = any(t.corr_errs or t.uncorr_errs or t.thermal_trips for t in chips)  # an error or a trip is heard, so it starts a phrase
        self._wanted_pick = self.knobs.syllable() if self.knobs is not None else ("", "")
        # a new choir pick is a hand on the instrument as an error is: it starts a phrase at once, with no minimum gap (when it can be heard)
        # a pick can change the sound only when the patch's choir has a trigger source; without one it would only restart every device's clock
        picked = bool(self.patch.choir_onset) and self._wanted_pick != self._pick
        changed = self._phrase is None or news or picked or sig[1] != self._sig[1] or self._moved(sig, now)
        exhausted = self._phrase is not None and self._pos + self._block_samples > len(self._phrase) - self._xfade_samples
        self._raise_render_error()
        held = 0.0
        if changed:
            # a move that waited for the minimum gap is heard in this phrase: how long it waited is part of its delay
            held = now - self._crossed_at if self._crossed_at is not None else 0.0
            self._crossed_at = None
            self._drop_pending()
            block = self._start_phrase(chips, defrag, sig, notes, self._short_blocks, now)
        elif self._head_end is not None and self._pos >= self._head_end:
            block = self._leave_head()
        elif self._silent and not exhausted:
            block = self._wait_for_body()
        elif exhausted:
            block = self._next_phrase(chips, defrag, sig, notes, now)
        else:
            block = self._phrase[self._pos: self._pos + self._block_samples]
            self._pos += self._block_samples
        if block is None:
            return
        self._maybe_render_ahead(chips, defrag, sig, notes, now)
        ready = block
        if self._equaliser is not None:
            ready = self._equaliser.process(ready)
        if self._gain is not None:
            ready, _ = self._gain.apply(ready)
        with self._lock:
            if self._t0 is None:
                self._t0 = self._clock()
            self._handed = self._flow()
            self._pending.append((self._delivered, self._block) + self._meta + (self._handed,))
            self._delivered += len(ready) / self.sample_rate
        self._audio.put(np.clip(ready, -1.0, 1.0).astype(np.float32))
        self._timing = (now, self._clock(), held, self._loop_rendered)  # `now` is when this block's readings were taken
        self._block += 1

    def _moved(self, sig, now) -> bool:
        """Whether the readings have moved enough since the playing phrase was made to start a new one. Without `changed` any difference counts;
        with it, `changed` decides. Either way the last phrase must have started at least `min_gap_s` ago (0 by default).

        A move that has to wait for the gap is remembered from its first reading (`_crossed_at`), so the phrase that finally plays it can say how
        long it was held back (`held_s`). A reading that no longer differs from the playing phrase forgets it: there is nothing left to hear."""
        moved = sig != self._sig if self._changed is None else self._changed(self._sig, sig)
        if not moved:
            self._crossed_at = None
        elif self._crossed_at is None:
            self._crossed_at = now
        return moved and now - self._phrase_t >= self._min_gap_s

    def _plan(self, chips, defrag, sig, notes, blocks, now: float) -> _Plan:
        """Who has the floor in a phrase that starts at `now`, and what it is rendered from. Nothing is changed until the phrase is installed."""
        soloist, until = self._soloist, self._soloist_until
        if self._only is None:
            if self._hold_soloist is None or self._soloist is None or now >= self._soloist_until:
                # a new turn; the phrase number serves as the turn number, as it does without a hold
                soloist = pick_soloist(slot_weights(chips, self.patch), self._phrases, self._soloist)
                if self._hold_soloist is not None:
                    until = now + self._hold_soloist(chips)
        return _Plan(list(chips), defrag, sig, notes, blocks, soloist, until, self._phrases, self._wanted_pick)

    def _render_plan(self, plan: _Plan, n: int) -> np.ndarray:
        """`n` samples of the phrase `plan` describes."""
        extra = {"soloists": [plan.soloist]} if self._only is None else {}
        # the Play tab's pick fills in the choir's onset and vowel; a part it leaves empty keeps the session patch's own
        pick = (plan.pick[0] or self.patch.choir_pick[0], plan.pick[1] or self.patch.choir_pick[1])
        patch = self.patch if pick == self.patch.choir_pick else dataclasses.replace(self.patch, choir_pick=pick)
        return self._render(plan.chips, (n + 0.5) / self.sample_rate, self.sample_rate, patch=patch, segment=plan.segment, only=self._only,
                            cables=None if self.config.cables else [], stereo=self.config.stereo, defrag=plan.defrag, **extra)

    def _start_phrase(self, chips, defrag, sig, notes, blocks, now: Optional[float] = None) -> Optional[np.ndarray]:
        """Render a phrase of `blocks` blocks and return its first block, crossfaded from the old phrase's continuation. None if stopped meanwhile.
        `now` is the session clock the phrase starts at (read from the clock if not given). With `head_s`, only a head is rendered here."""
        now = self._clock() if now is None else now
        self._loop_rendered = True  # a head or a whole phrase is rendered on this thread
        plan = self._plan(chips, defrag, sig, notes, blocks, now)
        if self._head_blocks is not None and self._head_blocks < blocks:
            return self._start_head(plan, now)
        phrase = self._render_plan(plan, phrase_samples(blocks, self._block_samples, self._xfade_samples))
        if self._stop.is_set():
            return None
        return self._open(plan, phrase, now)

    def _start_head(self, plan: _Plan, now: float, body: Optional[RenderJob] = None) -> Optional[np.ndarray]:
        """Render the head of the phrase `plan` describes at once and hand the full phrase to the background thread (or adopt `body`, a render of
        it already under way). The head is rendered first, so it does not share the processor with the body."""
        head = self._render_plan(plan, self._head_blocks * self._block_samples + self._xfade_samples + self._head_margin)
        if self._stop.is_set():
            return None
        n = phrase_samples(plan.blocks, self._block_samples, self._xfade_samples)
        if body is None:
            body = self._submit(lambda: self._render_plan(plan, n))
        block = self._open(plan, head, now)
        self._body, self._body_samples, self._head_end = body, n, self._head_blocks * self._block_samples
        return block

    def _open(self, plan: _Plan, phrase: np.ndarray, now: float) -> np.ndarray:
        """Make `phrase` the one that plays and return its first block, crossfaded (equal power: a new phrase, so the devices restart) from the old
        phrase's continuation."""
        block = np.array(phrase[:self._block_samples], dtype=np.float64)
        if self._phrase is not None:  # take over from where the old phrase would have gone on
            x = self._xfade_samples
            fade_in, fade_out = self._fades(x, block.ndim == 2)
            block[:x] = block[:x] * fade_in + self._phrase[self._pos: self._pos + x] * fade_out
        self._soloist, self._soloist_until = plan.soloist, plan.until
        self._phrase, self._pos, self._sig, self._phrase_t = phrase, self._block_samples, plan.sig, now
        self._pick = plan.pick
        self._phrases = plan.segment + 1
        self._head_end, self._silent = None, False
        chips = plan.chips
        heard = heard_on(self._only, len(chips))
        wanted = list(self._only) if self._only is not None else heard
        notes = [f"{name} is not dealt to any of the {len(chips)} chip{'s' if len(chips) != 1 else ''}, so it is silent"
                 for name in wanted if name not in heard] + plan.notes
        self._meta = (tuple(chips), tuple(heard), tuple(notes))
        self._phases = {k: d.phase for k, d in enumerate(plan.defrag) if d.phase != "running"} if plan.defrag is not None else {}
        return block.astype(np.float32)

    # -- heads, bodies and renders ahead -------------------------------------------------------------------------------------------

    def _submit(self, fn: Callable[[], np.ndarray]) -> RenderJob:
        if self._worker is None:
            self._worker = self._worker_factory()
        return self._worker.submit(fn)

    def _raise_render_error(self) -> None:
        """A background render that raised ends the session with its error, as a render on the loop's thread does."""
        for job in (self._body, self._upcoming[0] if self._upcoming is not None else None):
            if job is not None and job.done.is_set() and job.error is not None and not job.dropped:
                self._error = f"{type(job.error).__name__}: {job.error}"
                raise job.error

    def _drop_pending(self) -> None:
        """A change: the body and the render ahead were made from readings that no longer hold."""
        for job in (self._body, self._upcoming[0] if self._upcoming is not None else None):
            if job is not None:
                job.drop()
        self._body, self._upcoming = None, None

    def _leave_head(self) -> np.ndarray:
        """The head's blocks are played: the body takes over at this offset with a straight crossfade, or, if it is late, the head fades out."""
        self._head_end = None
        p, x, b = self._pos, self._xfade_samples, self._block_samples
        old = self._phrase
        if self._body.ready:
            body = self._body.result
            block = np.array(body[p:p + b], dtype=np.float64)
            ramp = self._ramp[:, None] if block.ndim == 2 else self._ramp
            block[:x] = block[:x] * ramp + old[p:p + x] * (1.0 - ramp)
            self._phrase, self._pos, self._body = body, p + b, None
            return block.astype(np.float32)
        # late: the head's own continuation fades to nothing, then silence of the full phrase's length plays until the body is ready
        self._phrase = np.zeros((self._body_samples,) + old.shape[1:], dtype=np.float32)
        self._silent = True
        block = np.zeros((b,) + old.shape[1:])
        _, fade_out = self._fades(x, block.ndim == 2)
        block[:x] = old[p:p + x] * fade_out
        self._pos += b
        self.late_blocks += 1
        return block.astype(np.float32)

    def _wait_for_body(self) -> np.ndarray:
        """Silence while the body is late; once it is ready it comes in at the listener's offset, faded in from the silence."""
        p, x, b = self._pos, self._xfade_samples, self._block_samples
        if self._body is not None and self._body.ready:
            block = np.array(self._body.result[p:p + b], dtype=np.float64)
            fade_in, _ = self._fades(x, block.ndim == 2)
            block[:x] = block[:x] * fade_in
            self._phrase, self._pos, self._body, self._silent = self._body.result, p + b, None, False
            return block.astype(np.float32)
        self._pos += b
        self.late_blocks += 1
        return np.zeros((b,) + self._phrase.shape[1:], dtype=np.float32)

    def _next_phrase(self, chips, defrag, sig, notes, now: float) -> Optional[np.ndarray]:
        """The playing phrase ran out with nothing moving: the phrase rendered ahead if there is one, otherwise a new long one."""
        if self._body is not None:  # a body still missing when its phrase has run out is no use: dropped, so a queued one is never rendered
            self._body.drop()
        self._body, self._silent = None, False
        if self._upcoming is not None:
            job, plan = self._upcoming
            self._upcoming = None
            if job.ready:
                return self._open(plan, job.result, now)
            if self._head_blocks is not None and self._head_blocks < plan.blocks:
                return self._start_head(plan, now, body=job)  # still rendering: it becomes the body of a head rendered now
            self._loop_rendered = True  # the loop waits for the render, which costs it as much as rendering
            while not job.done.wait(PACE_SLICE_S):  # no head to play meanwhile: wait for it, as a render on this thread would take as long
                if self._stop.is_set():
                    return None
            self._raise_on(job)
            if job.result is None:
                return None
            return self._open(plan, job.result, now)
        return self._start_phrase(chips, defrag, sig, notes, self._long_blocks, now)

    def _raise_on(self, job: RenderJob) -> None:
        if job.error is not None and not job.dropped:
            self._error = f"{type(job.error).__name__}: {job.error}"
            raise job.error

    def _maybe_render_ahead(self, chips, defrag, sig, notes, now: float) -> None:
        """With `render_ahead`: once the playing phrase is half played and nothing else is rendering, render the next long phrase from these readings."""
        if not self._render_ahead or self._upcoming is not None or self._body is not None or self._silent or self._head_end is not None:
            return
        if self._pos < (len(self._phrase) - self._xfade_samples) / 2:
            return
        plan = self._plan(chips, defrag, sig, notes, self._long_blocks, now)
        n = phrase_samples(self._long_blocks, self._block_samples, self._xfade_samples)
        self._upcoming = (self._submit(lambda: self._render_plan(plan, n)), plan)

    def wait_render(self, timeout: float) -> bool:
        """Wait up to `timeout` seconds for the background render in progress (the body, or else the phrase ahead). True when none is left."""
        job = self._body if self._body is not None else self._upcoming[0] if self._upcoming is not None else None
        return job is None or job.done.wait(timeout)

    def carry_from(self, other: "InstrumentSession") -> None:
        """Take over what `other` has learned, for a session that replaces it: the eviction tracker (each chip's rest and load), the held soloist
        and its turn, the phrase number, the level gain and the EQ's state. Fast mode does this when it lengthens its block."""
        if self._evictor is not None and other._evictor is not None:
            self._evictor = other._evictor
        self._soloist, self._soloist_until, self._phrases = other._soloist, other._soloist_until, other._phrases
        if self._gain is not None and other._gain is not None:
            self._gain = other._gain
        if self._equaliser is not None and other._equaliser is not None:
            self._equaliser = other._equaliser

    def _flow(self) -> FlowState:
        """The chart's data for the block being handed over: the playing phrase's readings, who has the floor (nobody, in a scope that plays some devices),
        which devices are heard and which chips are not running."""
        chips, heard, _ = self._meta
        return build_flow(chips, self.patch, self._phrases - 1, self._soloist, self._block * self._block_s, self._phases, cables=self.config.cables, heard=heard)

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
            block, chips, heard, notes, state = self._shown if self._shown is not None else (-1, (), (), (), None)
        return InstrumentSnapshot(running, self._error, block, chips, heard, notes, state)

    def delay_s(self, buffer_s: float, interval_s: float) -> Optional[float]:
        """The worst-case delay from a chip reading to its sound, for the latest block: the sampling interval (a change can happen just after a
        reading), plus the time from taking the readings to queueing the audio, plus the player's buffer. None before the first block. A change that
        waited for the minimum gap adds `held_s` to this."""
        timing = self._timing  # one read of one tuple, so both times belong to the same block
        if timing is None:
            return None
        read_at, written_at, _, _ = timing
        return interval_s + (written_at - read_at) + buffer_s

    @property
    def handed_state(self) -> Optional[FlowState]:
        """The chart's data for the latest block handed over: the same `FlowState` that `snapshot()` shows once the listener reaches that block.
        None before the first block. Fast mode hands it to its `on_segment` hook with the block (`tensixinet/fast.py`), and the terminal view
        holds it until the listener reaches the block (`tui/playhead.py`). This is the newest block made, which is ahead of the one being heard."""
        with self._lock:
            return self._handed

    @property
    def rendered_on_loop(self) -> bool:
        """Whether the latest block rendered a head or a whole phrase on the loop's own thread (or waited there for a background render). A
        block cut from a phrase already rendered did not. Fast mode counts only these blocks when it decides that renders outrun the block."""
        timing = self._timing
        return bool(timing[3]) if timing is not None else False

    @property
    def held_s(self) -> float:
        """For the latest block: how long the change it starts a phrase for waited for the minimum gap (`min_gap_s`), from the first reading that
        moved past `changed` to this block's reading. 0 for a block that starts no phrase on a change, and for a change that did not wait. The
        delay of that change is `delay_s(...)` plus this."""
        timing = self._timing
        return timing[2] if timing is not None else 0.0

    def tail(self) -> np.ndarray:
        """The playing phrase's continuation for one crossfade, faded out, through the EQ and at the level gain the music ended on. A run that
        stops hands this over last, so the sound fades instead of stopping on a cut (fast mode, `tensixinet/fast.py`). Empty before the first block."""
        if self._phrase is None:
            return np.zeros((0, 2) if self.config.stereo else 0, dtype=np.float32)
        x = self._xfade_samples
        rest = np.array(self._phrase[self._pos: self._pos + x], dtype=np.float64)
        _, fade_out = self._fades(len(rest), rest.ndim == 2)
        tail = rest * fade_out
        if self._equaliser is not None:
            tail = self._equaliser.process(tail)
        if self._gain is not None and self._gain.gain is not None:
            tail = tail * self._gain.gain
        return np.clip(tail, -1.0, 1.0).astype(np.float32)

    def next_audio(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """The next block's audio, or None if there is none within `timeout` seconds."""
        try:
            return self._audio.get(timeout=timeout)
        except queue.Empty:
            return None
