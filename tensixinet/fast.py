# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Fast mode (`--live --interval MS`): read the chips every block and play a change soon after it (a change inside the one-second gap between phrases waits for the gap).

`run_live` (live.py) reads the chips once a segment of 2 to 10 s. Fast mode reads them every block, where a block is the interval and at least
`MIN_BLOCK_S`. It drives the Play tab's phrase engine (`ui/instrument.py`, `InstrumentSession`) with real readings in place of the knobs:

* Each block, `source.sample()` gives the latest readings (hwmon's temperature, power, voltage and current laid over the last `tt-smi -s`
  reading, see `telemetry/fast.py`). A `Tracker` with a window of `RATE_WINDOW_S` gives them rates and turns the cumulative error counts into
  counts since the previous reading, which the session needs: it starts a phrase whenever an error or trip count is above zero. The tracker
  holds the last measured rate for the readings between two measurements.
* A reading that moved past `signature_changed`'s thresholds starts a short phrase (`SHORT_PHRASE_S`), at most once every `MIN_GAP_S`. While
  nothing moves, long phrases (`LONG_PHRASE_S`) follow each other, so the gong and the rolls have time to play. A new phrase crossfades in over
  `MAX_XFADE_S`, or half a block when the block is shorter.
* The soloist is held for the patch's turn length (`turn_seconds`), as in live mode.
* A change renders a head of `HEAD_S` on the loop and the full phrase in the background, and the full phrase takes over at the same offset
  when the head's blocks are played (`InstrumentSession`'s `head_s`). In a quiet stretch the next long phrase is rendered in the background
  once the playing one is half played (`render_ahead`). So the loop's own work in a block is a head render at most: on the four real chips in
  stereo, 73 to 88 ms at 22050 Hz and 138 to 150 ms at 44100 Hz (2026-10-23). A phrase is still rendered whole on the loop when a head would
  not be shorter than it, which happens once the block has been lengthened (a 0.6 s head is as long as a 1 s phrase in blocks of 0.4 s).
* The loop hands each block to a second thread, which writes it to the sink (a player's pipe blocks while it is full). The loop then waits
  until no more than the lead (`lead_for`: the time a head render takes at the rate in use, with a margin) is queued ahead of the listener,
  timed by the clock from the first block, so the sound stays close to the readings. The pipe alone holds less than half a second at 44.1 kHz
  in stereo, so writing from the loop itself would make the loop wait on the pipe.

The delay from a chip changing to the change being heard is at worst: the block (a change can land just after a read; when the fast source
has fallen back to `tt-smi` alone, its period takes the block's place), the time from the read to the audio being handed over (a head render
when a change starts a phrase), the audio already queued in front of it (the lead), and the player's own buffer. A typical change lands half
way between two reads, so the typical delay counts half the interval. A change that lands less than `MIN_GAP_S` after the last phrase started
waits for that gap to end before its phrase starts, so its delay also counts that wait (the session's `held_s`): up to about a second more.
Every `REPORT_EVERY_S` (and at the end of the run, Ctrl-C included) the log gives the worst block over the last stretch with its parts, the
typical delay (the middle block's), the gaps in the sound, and, when the worst is over `DELAY_BUDGET_S`, a warning that names the part that
dominates. The loop never changes the interval to meet the budget.

A gap is either the listener running out of queued audio (a block took longer than the lead to make) or a block played as silence because the
full phrase was not ready when its head ran out (the session's `late_blocks`).

When a head render takes longer than the block, the audio queued ahead runs out and the player has a gap. After `OVERRUNS_TO_LENGTHEN` such
renders on the loop in a row (one stall from another process is not enough), the loop lengthens the block (to `LENGTHEN_MARGIN` times the slowest
of them, in steps of `BLOCK_STEP_S`, at most `MAX_BLOCK_S`), says so in the log with the real rate, and goes on with a new session at that block,
which keeps the old one's eviction tracker, soloist, gain and EQ (`carry_from`). The new session's first block fades in over the old one's tail.
After `CALM_BLOCKS_TO_SHORTEN` blocks in a row whose render fits the interval asked for, the block returns to it the same way, and the log says so.
The sample rate is never changed.

Errors, thermal trips and mask changes come with the `tt-smi` reading, about every 1.5 s, so they are outside the budget, and the log says so.
"""
from __future__ import annotations

import dataclasses
import math
import queue
import threading
import time
from typing import Callable, List, Optional, Sequence

import numpy as np

from .live import _log_changes, _plural
from .media.player import PlayerError, player_buffer_ms
from .music.chorus import turn_seconds
from .music.live import Tracker
from .music.patch import Patch
from .narrate.explain import explain_events
from .narrate.runlog import RunLog
from .run import describe_readings, log_cables, log_eq, log_only
from .telemetry import Telemetry
from .ui.flow import FlowState
from .ui.instrument import InstrumentConfig, InstrumentSession, signature_changed

DELAY_BUDGET_S = 0.25  # the owner's budget from a chip changing to the change being heard
MIN_BLOCK_S = 0.1  # a block is the interval, but never shorter than this: below it the per-block overhead takes most of the time
SHORT_PHRASE_S = 1.0  # a phrase started by a change: short, so it is cheap to render
LONG_PHRASE_S = 6.0  # a phrase that follows one that ran out with nothing moving: the gong's first strike can land 2.2 s in
MIN_GAP_S = 1.0  # a change starts a phrase at most once a second, so a reading that keeps moving does not restart the devices ten times a second
MAX_XFADE_S = 0.2  # the crossfade into a new phrase, or half a block when that is shorter
RATE_WINDOW_S = 0.5  # rates are measured over half a second, so a power spike shows as a rate (live mode uses 2 s)
REPORT_EVERY_S = 5.0  # how often the log gives the delay the run calculates from its own timings
PLAYER_LATENCY_MS = 20  # the buffer the player is asked for (paplay accepted --latency-msec=20 on this box)
LENGTHEN_MARGIN = 1.25  # a lengthened block is this many times the render that outran the block
BLOCK_STEP_S = 0.05  # and is rounded up to a multiple of this
MAX_BLOCK_S = 2.0  # and is never longer than this
OVERRUNS_TO_LENGTHEN = 3  # head renders in a row that outrun the block before it is lengthened: one stall from another process is not enough
CALM_BLOCKS_TO_SHORTEN = 60  # blocks in a row whose render fits the interval asked for before a lengthened block returns to that interval
HEAD_S = 0.6  # a change is heard from a head this long while the full phrase renders in the background
# The lead: how much audio the loop keeps queued ahead of the listener. It must cover the loop's longest render, a head (0.6 s plus its
# crossfade and margin), or the sound gaps when a change starts a phrase. On the four real chips in stereo a head render took 73 to 88 ms at
# 22050 Hz and 138 to 150 ms at 44100 Hz (2026-10-23). At 44100 Hz that is longer than a 100 ms block, so the loop lengthens the block and
# then raises the lead to the new block anyway. The run's own delay report counts the gaps, so it shows whether these still hold.
LEAD_S_AT_22050 = 0.10
LEAD_S_AT_44100 = 0.18


def lead_for(sample_rate: int) -> float:
    """The lead at `sample_rate`: a straight line between the two measured rates (a render's cost grows about in step with the rate), and the
    nearer one's value outside them."""
    if sample_rate <= 22050:
        return LEAD_S_AT_22050
    if sample_rate >= 44100:
        return LEAD_S_AT_44100
    return LEAD_S_AT_22050 + (LEAD_S_AT_44100 - LEAD_S_AT_22050) * (sample_rate - 22050) / (44100 - 22050)


class SampledSource:
    """A plain sampler with the parts of `FastSource`'s interface that `run_fast` uses: `start`, `stop`, `sample`, `fast` and `note`.
    The synthetic chips (`--mock-chips`) and a sampler handed in by a test go through this."""

    fast = False

    def __init__(self, sample: Callable[[], Sequence[Telemetry]], note: str):
        self._sample = sample
        self.note = note

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def sample(self) -> List[Telemetry]:
        return list(self._sample())


def player_in_use(player_name: str, sink) -> str:
    """The player that plays the sound: `auto` is the one the sink actually runs ("none" when auto found no player)."""
    if player_name != "auto":
        return player_name
    parts = getattr(sink, "sinks", (sink,))
    return next((part.command[0] for part in parts if getattr(part, "command", None)), "none")


def player_buffer_s(player_name: str, sink, latency_ms: int) -> float:
    """The player's buffer in seconds, for the delay report. `auto` is the player the sink actually runs (none found: no sound, no buffer)."""
    name = player_in_use(player_name, sink)
    if name == "none":
        return 0.0
    return player_buffer_ms(name, latency_ms) / 1000.0


def block_for(interval_s: float) -> float:
    """The block, and so how often the chips are read: the interval, raised to `MIN_BLOCK_S` when it is shorter."""
    return max(interval_s, MIN_BLOCK_S)


def lengthened(render_s: float) -> float:
    """The block that a render of `render_s` seconds fits inside, with a margin."""
    steps = math.ceil(round(render_s * LENGTHEN_MARGIN / BLOCK_STEP_S, 9))
    return min(MAX_BLOCK_S, steps * BLOCK_STEP_S)


def run_fast(
    source,
    sink,
    log: RunLog,
    sample_rate: int,
    interval_s: float,
    duration: Optional[float],
    patch: Patch,
    patch_source: str,
    *,
    normalise: bool,
    eq: bool,
    only: Optional[Sequence[str]],
    cables: bool,
    stereo: bool,
    evict: bool,
    player_name: str,
    latency_ms: int,
    label: str = "the telemetry source",
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    session_factory: Callable = InstrumentSession,
    on_segment: Optional[Callable[[float, Optional[FlowState]], None]] = None,
) -> int:
    """Play the chips for `duration` seconds (until Ctrl-C when None), reading them every block. Returns the number of blocks played.

    `source` has `start()`, `stop()`, `sample()` (the latest readings, [] when there are none), `fast` and `note` (`FastSource` or
    `SampledSource`). `label` names it in the log. `player_name` and `latency_ms` say which player buffer the delay report counts. Raises
    `ValueError` when there has never been a reading and `PlayerError` when the player stops. The sink and the source are always closed.

    `on_segment` (the terminal view's playhead, see tui/playhead.py) is called on this loop's thread just before each block's audio is queued
    for the sink, with the block's real length in seconds and the session's `handed_state`: the `FlowState` of the phrase that block plays (its
    `segment` is the phrase number, and `seconds` counts from the session's first block, so it starts again at 0 after the block is lengthened
    or shortened). A session without that property gives None. The tail after the last block gets no call. The hook must only store what it
    is given and return. The sound never depends on it: if it raises, one "tui" line names the error, the hook is not called again, and the
    run goes on, as in `run_live`."""
    buffer_s = player_buffer_s(player_name, sink, latency_ms)
    lead_s = lead_for(sample_rate)
    tracker = Tracker(window_s=RATE_WINDOW_S, hold=True)  # hold: a reading between two measured rates carries the last one
    state = dict(last=None, lost=False, count=None)  # the last good raw reading, whether the readings are lost, the chip count
    block_index = [0]

    def read(now: float) -> List[Telemetry]:
        """The session's input: the readings tracked, or the last good ones played again with no movement when a reading is empty."""
        readings = source.sample()
        if not readings:
            if state["last"] is None:
                log.event("discover", "no chips found, nothing to play", f"{label} returned zero devices; refusing to invent any")
                raise ValueError("no chips in telemetry")
            if not state["lost"]:
                log.event("live", f"block {block_index[0]}: no chips in this reading",
                          "the last readings that worked are played again with no movement and no new errors until the chips come back",
                          block=block_index[0])
                state["lost"] = True
            return [dataclasses.replace(t, temp_rate=0.0, power_rate=0.0, corr_errs=0, uncorr_errs=0, thermal_trips=0) for t in state["last"]]
        if state["lost"]:
            log.event("live", f"block {block_index[0]}: the readings are back", "the music follows the chips again", block=block_index[0])
            state["lost"] = False
        if state["count"] is not None and state["count"] != len(readings):
            log.event("live", f"the reading changed from {_plural(state['count'], 'chip')} to {len(readings)}",
                      "chips are paired by position, so with a different number the movement since the last reading cannot be told; "
                      "the music follows the new number", block=block_index[0])
        state["count"] = len(readings)
        state["last"] = readings
        chips = tracker.update(readings, now)
        for i, t in enumerate(chips):
            if t.corr_errs or t.uncorr_errs or t.thermal_trips:
                d = explain_events(i, t)
                log.event("events", d.what.replace("reports", "reports a further", 1), d.why, chip=i, block=block_index[0])
        return chips

    audio_queue: "queue.Queue[Optional[np.ndarray]]" = queue.Queue()
    failures: List[BaseException] = []

    def play() -> None:
        while True:
            chunk = audio_queue.get()
            if chunk is None:
                return
            if failures:
                continue
            try:
                sink.write(chunk)
            except BaseException as e:  # reported by the loop, which then stops
                failures.append(e)

    player = threading.Thread(target=play, name="tensixinet-fast-player", daemon=True)
    player.start()

    session = None
    played_samples = 0
    blocks = 0
    report = dict(worst=None, typical=[], gaps=0, gap_s=0.0)  # the stretch since the last delay line
    shape = dict(block_s=block_for(interval_s), lengthened=False, fallback=False)  # what the over-budget advice needs to know
    interrupted = False  # Ctrl-C ended the run: a player that stopped after that does not make it an error
    if hasattr(source, "log"):  # FastSource says when a chip's hwmon reads fail and come back
        source.log = lambda what, why, **fields: log.event("live", what, why, **fields)
    try:
        source.start()
        first = source.sample()
        started = clock()
        if not first:
            log.event("discover", "no chips found, nothing to play", f"{label} returned zero devices; refusing to invent any")
            raise ValueError("no chips in telemetry")
        state["last"], state["count"] = first, len(first)
        tracker.update(first, started)

        block_s = block_for(interval_s)
        describe_readings(first, LONG_PHRASE_S, log, label, patch, patch_source, turns=False, stereo=stereo, only=only)
        log_only(log, only)
        if only and evict:  # the session ignores eviction when it plays some devices (ui/instrument.py), as the Play tab's scopes do
            log.event("live", "--only turns eviction off in fast mode",
                      "the soloed devices play at full level whatever a chip's load does, so no chip's voices dissolve when its model unloads")
        log_eq(log, eq)
        log_cables(log, first, patch, cables)
        raised = f"--interval {interval_s * 1000:.0f} was raised to {block_s * 1000:.0f} ms, the shortest block; " if block_s > interval_s else ""
        log.event("live", f"fast mode: {raised}the chips are read every {block_s * 1000:.0f} ms, in blocks of {block_s:.2f} s",
                  f"a reading that moves starts a {SHORT_PHRASE_S:.0f} s phrase (at most once every {MIN_GAP_S:.0f} s) that crossfades in, and "
                  f"while nothing moves {LONG_PHRASE_S:.0f} s phrases follow each other so the gong and the rolls have time; a change is heard "
                  f"from a {HEAD_S:.1f} s head while the full phrase renders in the background, and the loop keeps {lead_s * 1000:.0f} ms of "
                  f"audio ahead of the listener (a head's render time at {sample_rate} Hz, with a margin)", interval_ms=round(interval_s * 1000),
                  block_s=block_s, lead_s=lead_s)
        log.event("live", source.note, "where the fast readings come from decides how quickly a change can be heard")
        noted = source.note
        if player_in_use(player_name, sink) == "ffplay":
            log.event("live", f"ffplay's buffer is estimated at {player_buffer_ms('ffplay', latency_ms)} ms, which puts the delay over the "
                              f"{DELAY_BUDGET_S * 1000:.0f} ms budget",
                      "the block, the lead and that buffer add up to more than the budget before any render; --player paplay asks for a "
                      f"{latency_ms} ms buffer, and --player auto picks paplay first when it is installed")
        if hasattr(source, "slow_every_s"):
            log.event("live", f"errors, thermal trips and mask changes arrive with the tt-smi reading, about every {source.slow_every_s:g} s",
                      f"only temperature, power, voltage and current are read every block, so those slower fields are outside the "
                      f"{DELAY_BUDGET_S * 1000:.0f} ms budget")
        else:
            log.event("live", "every field of these readings arrives at the interval, errors and trips included",
                      "the readings do not come from tt-smi, so nothing waits on its 1.5 s reading")

        config = InstrumentConfig(scope="system", chips=len(first), style=patch.style or "", seed=patch.seed, stereo=stereo, cables=cables,
                                  evict=evict)

        def make_session(block: float, lead: float):
            return session_factory(config, None, patch, clock=clock, sleep=sleep, lead_s=lead, read=read, block_s=block, short_s=SHORT_PHRASE_S,
                                   long_s=LONG_PHRASE_S, xfade_s=min(MAX_XFADE_S, block / 2), sample_rate=sample_rate, only=only, eq=eq,
                                   normalise=normalise, min_gap_s=MIN_GAP_S, changed=signature_changed,
                                   hold_soloist=lambda chips: turn_seconds(chips, patch), head_s=HEAD_S, render_ahead=True)

        def switch(old, block: float, lead: float):
            """A new session at another block size, taking over `old`'s learned state; returns it and `old`'s tail, which its first block fades
            in over."""
            nonlocal lead_s
            tail = old.tail()
            old.stop(timeout=0)  # its background render, if any, finishes on its own thread and is dropped
            lead_s = lead
            new = make_session(block, lead)
            if hasattr(new, "carry_from"):
                new.carry_from(old)  # the eviction tracker's rest, the held soloist, the gain and the EQ go on
            return new, tail

        session = make_session(block_s, lead_s)
        asked_s = block_s  # the block the interval asks for; a lengthened block returns to it
        overruns, slowest, calm = 0, 0.0, 0  # renders on the loop in a row that outran the block, the slowest of them, blocks in a row that fit asked_s
        join: Optional[np.ndarray] = None  # an old session's tail, which the new session's first block fades in over
        t0: Optional[float] = None  # the clock when the listener began (moved on after a gap)
        delivered = 0.0  # seconds handed to the sink
        next_report = started + REPORT_EVERY_S
        late_seen = 0  # the session's late blocks already counted as gaps
        floor = None
        target = None if duration is None else int(round(duration * sample_rate))

        while target is None or played_samples < target:
            if failures:
                break
            # wait until at most the lead is queued ahead of the listener, so the reading is as fresh as it can be
            if t0 is not None:
                over = delivered - (clock() - t0) - lead_s
                if over > 1e-9:
                    sleep(over)
            block_index[0] = blocks
            session.render_block()
            audio = session.next_audio(timeout=0)
            render_s = session.delay_s(0.0, 0.0) or 0.0  # from the read to the audio being ready
            if join is not None:
                audio = _fade_in_over(audio, join)
                join = None
            now = clock()
            if t0 is None:
                t0 = now
            queued = delivered - (now - t0)  # audio in front of this block that the listener has not reached
            if queued < -1e-9:  # the listener ran out of sound: a gap for a real player; the listener's time starts again from here
                if buffer_s > 0:
                    report["gaps"] += 1
                    report["gap_s"] -= queued
                t0 = now - delivered
                queued = 0.0
            queued = max(queued, 0.0)
            late = getattr(session, "late_blocks", 0)  # blocks of silence because a full phrase was late: a gap whatever the player
            if late > late_seen:
                report["gaps"] += late - late_seen
                report["gap_s"] += (late - late_seen) * block_s
            late_seen = late
            if on_segment is not None:  # before the put, so the playhead knows the block before the sink can start playing it
                try:
                    on_segment(len(audio) / sample_rate, getattr(session, "handed_state", None))
                except Exception as e:  # the picture is a by-product: its trouble never reaches the audio
                    on_segment = None
                    log.event("tui", f"the terminal view stopped: {e}", "the music goes on without it", block=blocks)
            audio_queue.put(audio)
            delivered += len(audio) / sample_rate
            played_samples += len(audio)
            blocks += 1

            # How often a new temperature or power can arrive: every block, or with each tt-smi reading when the fast source fell back.
            fresh_every = block_s
            shape["fallback"] = not getattr(source, "fast", True) and hasattr(source, "slow_every_s")
            if shape["fallback"]:
                fresh_every = max(block_s, source.slow_every_s)
            # the session's own measure, with the audio queued in front of this block counted as buffer: it is heard before this block is
            # a change that waited for the minimum gap between phrases (MIN_GAP_S) was heard that much later than its reading allowed
            held = getattr(session, "held_s", 0.0)
            delay = session.delay_s(buffer_s + queued, fresh_every) + held
            if report["worst"] is None or delay > report["worst"][0]:
                report["worst"] = (delay, render_s, queued, fresh_every, held)
            report["typical"].append(delay - fresh_every / 2)  # a change lands half way between two reads on average

            snap = session.snapshot()
            if snap.state is not None and snap.state.soloist_device and (snap.state.soloist_device, snap.state.soloist_chip) != floor:
                floor = (snap.state.soloist_device, snap.state.soloist_chip)
                log.event("live", f"{floor[0]} on chip {floor[1]} has the floor",
                          f"the soloist is drawn from the chips' weights and holds the floor for the patch's turn "
                          f"({turn_seconds(snap.telemetry or first, patch):.1f} s), through the phrases that changes start meanwhile",
                          block=blocks - 1, chip=floor[1], device=floor[0])

            if source.note != noted:
                noted = source.note
                log.event("live", noted, "the fast source changed what it reads from")

            # Lengthen the block only when OVERRUNS_TO_LENGTHEN renders on the loop in a row outrun it (one stall from another process is
            # not enough), and return to the interval asked for after CALM_BLOCKS_TO_SHORTEN blocks in a row whose render fits that interval.
            if getattr(session, "rendered_on_loop", True):
                if render_s > block_s + 1e-9:
                    overruns += 1
                    slowest = max(slowest, render_s)
                else:
                    overruns, slowest = 0, 0.0
            calm = 0 if render_s > asked_s + 1e-9 else calm + 1
            if overruns >= OVERRUNS_TO_LENGTHEN and block_s < MAX_BLOCK_S:
                longer = max(lengthened(slowest), block_s)
                if longer > block_s:
                    log.event("live", f"blocks lengthened to {longer:.2f} s: the chips are now read every {longer * 1000:.0f} ms",
                              f"{OVERRUNS_TO_LENGTHEN} head renders in a row took longer than the {block_s:.2f} s block (the slowest "
                              f"{slowest * 1000:.0f} ms at {sample_rate} Hz), so the audio queued ahead ran out; the sample rate is left as it is "
                              f"(a lower --sample-rate renders faster), and the block returns to {asked_s:.2f} s after {CALM_BLOCKS_TO_SHORTEN} "
                              "blocks in a row whose render fits it", block_s=longer, render_s=round(slowest, 4))
                    session, join = switch(session, longer, max(lead_for(sample_rate), longer))  # the lead must cover the render that outran
                    block_s = longer
                    shape.update(block_s=block_s, lengthened=True)
                    late_seen, calm = 0, 0
                overruns, slowest = 0, 0.0
            elif block_s > asked_s + 1e-9 and calm >= CALM_BLOCKS_TO_SHORTEN:
                log.event("live", f"blocks back to {asked_s:.2f} s: the chips are read every {asked_s * 1000:.0f} ms again",
                          f"{CALM_BLOCKS_TO_SHORTEN} blocks in a row rendered within {asked_s * 1000:.0f} ms, so the renders that lengthened the "
                          "block have stopped", block_s=asked_s)
                session, join = switch(session, asked_s, lead_for(sample_rate))
                block_s = asked_s
                shape.update(block_s=block_s, lengthened=False)
                late_seen, calm, overruns, slowest = 0, 0, 0, 0.0

            if now >= next_report:
                _report(log, report, buffer_s, blocks, shape)
                _log_changes(log, [dataclasses.replace(t, corr_errs=0, uncorr_errs=0, thermal_trips=0) for t in snap.telemetry or first],
                             blocks, patch)
                report.update(worst=None, typical=[], gaps=0, gap_s=0.0)
                while next_report <= now:
                    next_report += REPORT_EVERY_S
        if not failures:
            _report(log, report, buffer_s, blocks, shape)  # the stretch since the last report, so a short run reports too
            audio_queue.put(session.tail())  # the last phrase fades out
    except KeyboardInterrupt:
        interrupted = True
        log.event("live", "stopped by Ctrl-C", "the blocks already handed over are played, then the last phrase fades out")
        _report(log, report, buffer_s, blocks, shape)  # the stretch since the last report: a run that ends on Ctrl-C reports too
        if session is not None:
            audio_queue.put(session.tail())
    finally:
        audio_queue.put(None)
        player.join()
        if session is not None:
            session.stop()
        source.stop()
        try:
            sink.close()
        except PlayerError as e:
            failures.append(e)
    if failures and interrupted:
        # The person stopped the run. A player that went with it (one sharing the terminal's Ctrl-C) loses only the tail, so the run still ends
        # as stopped and the CLI's exit code stays 0; the log says what happened to the sound.
        log.event("player", f"the audio output stopped after Ctrl-C: {failures[0]}",
                  "the run was ending anyway, so this is noted and not treated as an error; the fading tail may not have been heard")
    elif failures:
        error = failures[0]
        log.event("player", f"the audio output stopped: {error}", "a live run ends when it has nowhere to play")
        raise error
    log.event("done", f"played {played_samples / sample_rate:.1f} s in {_plural(blocks, 'block')}",
              "the chips were read once a block; the last phrase's tail faded out after the duration", blocks=blocks)
    return blocks


def _fade_in_over(audio: np.ndarray, tail: np.ndarray) -> np.ndarray:
    """`audio` with its start faded in (sine) over `tail`, which is already faded out (cosine): the same equal-power join as a phrase's."""
    n = min(len(tail), len(audio))
    if n == 0:
        return audio
    out = np.array(audio, dtype=np.float64)
    k = (np.arange(n) + 0.5) / n
    fade = np.sin(0.5 * np.pi * k)
    if out.ndim == 2:
        fade = fade[:, None]
    out[:n] = out[:n] * fade + tail[:n]
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def _advice(interval_s: float, render_s: float, queued_s: float, buffer_s: float, shape: dict, held_s: float = 0.0) -> str:
    """What would bring an over-budget delay down, from the part that dominates it. `shape` says whether the loop lengthened its block (then the
    block is long because of a render), whether the readings have fallen back to tt-smi's pace, and the block in use. `held_s` is the time a
    change waited for the minimum gap between phrases."""
    parts = {"interval": interval_s, "render": render_s, "queued": queued_s, "buffer": buffer_s, "held": held_s}
    biggest = max(parts, key=parts.get)
    if biggest == "held":
        return (f"the wait for the minimum gap between phrases is the largest part: a change that lands less than {MIN_GAP_S:.0f} s after a phrase "
                "started is heard when that gap ends")
    if biggest == "buffer":
        return "the player's buffer is the largest part: another player, or paplay with its small buffer, would bring it down"
    if biggest == "interval":
        if shape["fallback"]:
            return "the readings come at tt-smi's pace, so the time between new readings is the largest part; hwmon readings would bring it down"
        if not shape["lengthened"] and shape["block_s"] > MIN_BLOCK_S + 1e-9:
            return "the interval is the largest part: a shorter --interval would bring it down"
        if not shape["lengthened"]:
            return "the time between reads is the largest part, and the block is already the shortest"
    # a render (or the lead that covers it, or a block lengthened because of it) dominates
    return "the render is the largest part (or the block was lengthened to fit it): a lower --sample-rate renders faster"


def _report(log: RunLog, report: dict, buffer_s: float, blocks: int, shape: dict) -> None:
    """The delay line: the worst block since the last report, its parts, the typical delay (the middle block's, with half the interval), the gaps,
    and a warning naming the part that dominates when the worst is over the budget. The numbers are the loop's own timings plus the player buffer
    it asked for; the run cannot see the sound server's latency or the audio waiting in the pipe, so the line says the delay is calculated."""
    worst = report["worst"]
    if worst is None:
        return
    delay, render_s, queued, block_s, held = worst  # block_s: how often a new reading could arrive; held: the wait for the minimum gap
    typical = float(np.median(report["typical"]))
    gaps, gap_s = report["gaps"], report["gap_s"]
    over = delay > DELAY_BUDGET_S + 1e-9
    parts = ("calculated by the run from its own timings and the requested player buffer; the delay at the speaker has not been measured; "
             f"a new reading every {block_s * 1000:.0f} ms, {render_s * 1000:.0f} ms from the read to the audio, {queued * 1000:.0f} ms already "
             f"queued in front of it, a {buffer_s * 1000:.0f} ms player buffer"
             + (f", {held * 1000:.0f} ms held back by the {MIN_GAP_S:.0f} s minimum gap between phrases" if held > 0 else "")
             + "; typically a change lands half way between two reads")
    gap_text = f"; {_plural(gaps, 'gap')} in the sound ({gap_s * 1000:.0f} ms in all)" if gaps else ""
    if shape["lengthened"]:
        parts += "; the block was lengthened to fit a render, so the lead is a whole block"
    if over:
        what = (f"calculated delay up to {delay * 1000:.0f} ms (typically {typical * 1000:.0f} ms), over the {DELAY_BUDGET_S * 1000:.0f} ms budget")
        # the interval is named only when it was left as asked and could be shorter: not at the shortest block, nor after a lengthening
        asked = "; the interval is left as it was asked for" if not shape["lengthened"] and shape["block_s"] > MIN_BLOCK_S + 1e-9 else ""
        why = f"{parts}{gap_text}{asked}; {_advice(block_s, render_s, queued, buffer_s, shape, held)}"
    else:
        what = f"calculated delay up to {delay * 1000:.0f} ms (typically {typical * 1000:.0f} ms), inside the {DELAY_BUDGET_S * 1000:.0f} ms budget"
        why = f"{parts}{gap_text}"
    log.event("delay", what, why, delay_s=round(delay, 6), typical_s=round(typical, 6), interval_s=round(block_s, 6), render_s=round(render_s, 6),
              queued_s=round(queued, 6), buffer_s=round(buffer_s, 6), held_s=round(held, 6), over_budget=over, gaps=gaps, block_lengthened=shape["lengthened"],
              block=blocks - 1)
