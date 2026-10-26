# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Fast mode (`--live --interval MS`): the chips are read every block and a change is heard soon after it (one inside the gap between phrases waits for the gap).

The tests drive `run_fast` with a clock that moves only when the loop sleeps (or when a stand-in render says it took time), a fake source and a
sink that records what it is handed, so no test reads hardware or waits in real time."""
import dataclasses
import functools
import json

import numpy as np
import pytest

from tensixinet import fast as fast_module
from tensixinet.fast import (CALM_BLOCKS_TO_SHORTEN, DELAY_BUDGET_S, HEAD_S, OVERRUNS_TO_LENGTHEN, LEAD_S_AT_22050, LEAD_S_AT_44100, LONG_PHRASE_S, MIN_BLOCK_S, MIN_GAP_S, REPORT_EVERY_S,
                             SHORT_PHRASE_S, SampledSource, lead_for, run_fast)
from tensixinet.media.player import PlayerError
from tensixinet.music.patch import DEFAULT_PATCH
from tensixinet.narrate.runlog import RunLog
from tensixinet.telemetry.mock import mock_telemetry
from tensixinet.ui.instrument import InstrumentSession, RenderJob

SR = 8000
START = 100.0


class Clock:
    """Moves only when something sleeps (the loop's pacing) or a stand-in render says it took time. It refuses to run for ever."""

    def __init__(self):
        self.t = START

    def __call__(self):
        return self.t

    def sleep(self, seconds):
        assert seconds >= 0
        self.t += seconds
        if self.t > START + 600:
            raise AssertionError("the loop ran past ten minutes of its own clock: it ignored the duration")


class Source:
    """A FastSource stand-in: `readings(t)` gives the chips at clock time t. Records when it was sampled."""

    def __init__(self, clock, readings=None, note="Readings come from hwmon (2 chips), with tt-smi for the rest."):
        self.clock = clock
        self.readings = readings or (lambda t: mock_telemetry(2))
        self.note, self.fast = note, True
        self.started = self.stopped = 0
        self.calls = []

    def start(self):
        self.started += 1

    def stop(self):
        self.stopped += 1

    def sample(self):
        self.calls.append(self.clock.t)
        return list(self.readings(self.clock.t))


class Sink:
    def __init__(self, clock=None, fail_after=None):
        self.chunks, self.closed, self.clock, self.times, self.fail_after = [], 0, clock, [], fail_after

    def write(self, audio):
        if self.fail_after is not None and len(self.chunks) >= self.fail_after:
            raise PlayerError("paplay stopped before the audio was done")
        self.chunks.append(np.array(audio))
        self.times.append(self.clock.t if self.clock else None)

    def close(self):
        self.closed += 1

    @property
    def audio(self):
        return np.concatenate(self.chunks)


class Render:
    """Stands in for render_chorus: a constant at a thousandth of chip 0's power, so a power step shows in the audio. `cost` is the seconds of
    clock each call takes (a function of the duration, or a number). `segments` records each call's segment: a head and its body share one, so
    the number of phrases is the number of distinct segments."""

    def __init__(self, clock, cost=0.0):
        self.clock, self.cost, self.calls, self.segments = clock, cost, 0, []

    def __call__(self, chips, duration, sample_rate=44100, stereo=False, segment=0, **kw):
        self.calls += 1
        self.segments.append(segment)
        self.clock.sleep(self.cost(duration) if callable(self.cost) else self.cost)
        n = int(duration * sample_rate)
        level = 0.001 * chips[0].power
        return np.full((n, 2) if stereo else n, level, dtype=np.float32)


class InlineWorker:
    """Stands in for the session's background render thread: a job runs as soon as it is handed over, on the clock as it was, so a background
    render takes none of the loop's time (as on a real second thread) and the tests stay deterministic. `never` leaves every job unfinished, as a
    render that is always late."""

    def __init__(self, clock, never=False):
        self.clock, self.never, self.jobs = clock, never, []

    def submit(self, fn):
        job = RenderJob(fn)
        self.jobs.append(job)
        if not self.never:
            t = self.clock.t
            try:
                job.result = fn()
            finally:
                self.clock.t = t
            job.done.set()
        return job

    def stop(self, timeout=None):
        pass


def _interrupt_on(call):
    """Readings that raise KeyboardInterrupt (as Ctrl-C does) on the `call`-th sample and not before."""
    count = [0]

    def readings(t):
        count[0] += 1
        if count[0] == call:
            raise KeyboardInterrupt
        return mock_telemetry(2)

    return readings


def _log(tmp_path):
    return RunLog(jsonl=open(tmp_path / "run.jsonl", "w"), stream=open(tmp_path / "run.txt", "w"))


def _events(tmp_path):
    return [json.loads(line) for line in (tmp_path / "run.jsonl").read_text().splitlines()]


def go(tmp_path, clock=None, source=None, sink=None, duration=2.0, interval_s=0.1, render=None, player_name="none", latency_ms=50, sample_rate=SR,
       **kw):
    clock = clock or Clock()
    source = source or Source(clock)
    sink = sink if sink is not None else Sink(clock)
    render = render if render is not None else Render(clock)
    worker = kw.pop("worker", None) or (lambda: InlineWorker(clock))
    real = {} if render is False else {"render": render}
    factory = kw.pop("session_factory", functools.partial(InstrumentSession, worker_factory=worker, **real))
    options = dict(normalise=False, eq=False, only=None, cables=True, stereo=False, evict=True)
    options.update(kw)
    log = _log(tmp_path)
    try:
        n = run_fast(source, sink, log, sample_rate, interval_s, duration, DEFAULT_PATCH, "the built-in default patch", player_name=player_name,
                     latency_ms=latency_ms, clock=clock, sleep=clock.sleep, session_factory=factory, **options)
    finally:
        log.close()
    return sink, n, _events(tmp_path), source, clock


# --- the constants ---------------------------------------------------------------------------------------------------------------

def test_the_budget_and_the_timing_are_the_ones_chosen():
    assert (DELAY_BUDGET_S, MIN_BLOCK_S, SHORT_PHRASE_S, LONG_PHRASE_S, MIN_GAP_S, REPORT_EVERY_S) == (0.25, 0.1, 1.0, 6.0, 1.0, 5.0)
    assert fast_module.RATE_WINDOW_S == 0.5 and fast_module.MAX_XFADE_S == 0.2 and fast_module.PLAYER_LATENCY_MS == 20
    assert (HEAD_S, LEAD_S_AT_22050, LEAD_S_AT_44100) == (0.6, 0.10, 0.18)


# --- the run as a whole ----------------------------------------------------------------------------------------------------------

def test_run_fast_plays_blocks_and_stops_at_the_duration(tmp_path):
    sink, n, _, source, _ = go(tmp_path, duration=2.0)
    tail = int(0.05 * SR)  # the crossfade is half a block when the block is 0.1 s
    assert n == 20 and [len(c) for c in sink.chunks[:-1]] == [int(0.1 * SR)] * 20
    assert len(sink.chunks[-1]) == tail and len(sink.audio) == 2 * SR + tail
    assert sink.closed == 1 and source.started == 1 and source.stopped == 1


def test_the_sink_is_written_from_its_own_thread_so_a_full_pipe_does_not_hold_up_the_readings(tmp_path):
    import threading

    clock = Clock()
    source = Source(clock)
    reads_when_written = []
    release = threading.Event()

    class Blocking(Sink):
        def write(self, audio):
            reads_when_written.append(len(source.calls))
            release.wait(2.0)  # the first write blocks like a full pipe; the loop must read on meanwhile
            super().write(audio)

    sink = Blocking(clock)
    timer = threading.Timer(0.5, release.set)
    timer.start()
    _, n, _, _, _ = go(tmp_path, clock=clock, source=source, sink=sink, duration=2.0)
    timer.cancel()
    # the loop read every block while the first write was held up (a loop that wrote itself would be at its second read here)
    assert len(reads_when_written) == n + 1 and reads_when_written[1] == n + 1


def test_the_block_is_the_interval_and_never_shorter_than_a_tenth_of_a_second(tmp_path):
    sink, n, _, _, _ = go(tmp_path, duration=1.0, interval_s=0.25)
    assert n == 4 and {len(c) for c in sink.chunks[:-1]} == {int(0.25 * SR)}
    sink, n, _, _, _ = go(tmp_path, duration=1.0, interval_s=0.01)
    assert n == 10 and {len(c) for c in sink.chunks[:-1]} == {int(0.1 * SR)}


def test_the_chips_are_read_once_a_block_at_the_interval(tmp_path):
    _, n, _, source, _ = go(tmp_path, duration=3.0)
    steps = np.diff(source.calls[2:])  # the first read draws the patch and the first block follows at once
    assert len(source.calls) == n + 1 and np.allclose(steps, 0.1)


def test_the_loop_stays_about_one_block_ahead_of_the_listener(tmp_path):
    _, n, _, source, _ = go(tmp_path, duration=3.0)
    reads = source.calls[1:]  # block k is read at reads[k] (the first call is the run's own first reading)
    assert len(reads) == n
    for k, read in enumerate(reads):
        ahead = 0.1 * k - (read - reads[0])  # audio handed over before this block that the listener has not reached
        assert -1e-9 <= ahead <= 0.1 + 1e-9, k


def test_the_audio_is_finite_and_inside_full_scale_with_the_real_chorus(tmp_path):
    sink, _, _, _, _ = go(tmp_path, duration=1.5, render=False, stereo=True, normalise=True, eq=True)
    assert sink.audio.shape[1] == 2 and np.isfinite(sink.audio).all() and np.abs(sink.audio).max() <= 1.0


def test_an_open_ended_run_plays_until_ctrl_c(tmp_path):
    clock = Clock()
    source = Source(clock, _interrupt_on(302))  # the first read, then 300 blocks, then Ctrl-C during the next read
    sink, n, ev, _, _ = go(tmp_path, clock=clock, source=source, duration=None)
    assert n == 300 and sink.closed == 1


# --- a change is heard at once ---------------------------------------------------------------------------------------------------

def test_a_power_step_reaches_the_audio_within_two_blocks(tmp_path):
    step_at = START + 2.55  # between two reads, and more than the minimum gap after the last phrase started

    def run(step):
        clock = Clock()

        def readings(t):
            chips = mock_telemetry(2)
            if step and t >= step_at:
                chips[0] = dataclasses.replace(chips[0], power=chips[0].power + 40.0)
            return chips

        source = Source(clock, readings)
        sink, _, _, _, _ = go(tmp_path, clock=clock, source=source, duration=4.0, render=False)
        return sink, source

    (still, _), (stepped, source) = run(False), run(True)
    after = [k for k, t in enumerate(source.calls[1:]) if t >= step_at]  # blocks whose reading has the step
    first_diff = next(k for k, (a, b) in enumerate(zip(still.chunks, stepped.chunks)) if not np.array_equal(a, b))
    assert after and first_diff >= after[0] and first_diff <= after[0] + 1  # the first or second block read after the step


def test_noise_under_the_thresholds_does_not_start_a_phrase_every_block(tmp_path):
    clock = Clock()
    render = Render(clock)
    rng = np.random.default_rng(1)

    def readings(t):
        chips = mock_telemetry(2)
        # two readings differ by at most 0.4 C and 2.8 W, under the 0.5 C and 3 W thresholds
        return [dataclasses.replace(c, asic_temperature=c.asic_temperature + rng.uniform(-0.2, 0.2), power=c.power + rng.uniform(-1.4, 1.4))
                for c in chips]

    go(tmp_path, clock=clock, source=Source(clock, readings), render=render, duration=8.0)
    assert len(set(render.segments)) <= 3  # the first short phrase and long phrases after it: 80 blocks of noise start nothing


def test_a_steady_change_restarts_a_phrase_at_most_once_a_minimum_gap(tmp_path):
    clock = Clock()
    render = Render(clock)

    def readings(t):  # power climbs 40 W a second: every read is past the threshold
        chips = mock_telemetry(2)
        return [dataclasses.replace(chips[0], power=chips[0].power + 40.0 * (t - START))] + chips[1:]

    go(tmp_path, clock=clock, source=Source(clock, readings), render=render, duration=5.0)
    assert len(set(render.segments)) <= 1 + 5 / MIN_GAP_S + 1


def test_rates_are_measured_over_half_a_second(tmp_path):
    clock = Clock()
    rates = []

    class Spy(Render):
        def __call__(self, chips, duration, sample_rate=44100, **kw):
            rates.append((self.clock.t, chips[0].power_rate))
            return super().__call__(chips, duration, sample_rate, **kw)

    def readings(t):  # power climbs 10 W a second, under the 3 W threshold per read but past it within a second
        chips = mock_telemetry(2)
        return [dataclasses.replace(chips[0], power=chips[0].power + 10.0 * (t - START))] + chips[1:]

    go(tmp_path, clock=clock, source=Source(clock, readings), render=Spy(clock), duration=3.0)
    moving = [t for t, rate in rates if rate == pytest.approx(10.0, rel=0.05)]
    assert moving and moving[0] - START < 2.0  # live mode's 2 s window would give no rate this early


def test_the_soloist_is_held_for_the_turn_through_the_phrases_changes_start(tmp_path):
    clock = Clock()

    def readings(t):  # power climbs 40 W a second, so a phrase starts every second
        chips = mock_telemetry(2)
        return [dataclasses.replace(chips[0], power=chips[0].power + 40.0 * (t - START))] + chips[1:]

    _, _, ev, _, _ = go(tmp_path, clock=clock, source=Source(clock, readings), duration=20.0)
    floors = [e for e in ev if e["event"] == "live" and "has the floor" in e["what"]]
    assert 1 <= len(floors) <= 20 / 4.3 + 2  # the default patch's turn is about 4.3 s; a phrase every second would hand it on twenty times


# --- what the log says -----------------------------------------------------------------------------------------------------------

def test_the_log_names_the_interval_the_fallback_and_the_calculated_delay(tmp_path):
    clock = Clock()
    note = "No hwmon chips found, so readings come at tt-smi's pace."
    _, _, ev, _, _ = go(tmp_path, clock=clock, source=Source(clock, note=note), duration=6.0)
    kinds = [e["event"] for e in ev]
    assert kinds.index("discover") < kinds.index("patch") < next(i for i, e in enumerate(ev) if "100 ms" in e["what"])
    assert any(note in e["what"] for e in ev)
    assert any("1.5 s" in e["what"] + e["why"] and "errors" in e["what"] + e["why"] for e in ev)  # the slow fields are outside the budget
    delay, _ = [e for e in ev if e["event"] == "delay"]  # one report at five seconds and one for the last second
    # a read every 0.1 s, a render that took no time, the lead (0.10 s) queued ahead of this block, no player buffer
    assert delay["delay_s"] == pytest.approx(0.2, abs=1e-6)
    assert (delay["interval_s"], delay["render_s"], delay["queued_s"], delay["buffer_s"]) == pytest.approx((0.1, 0.0, 0.1, 0.0), abs=1e-6)
    assert not delay["over_budget"] and "200 ms" in delay["what"]
    # the run adds up its own timings and the buffer it asked the player for; nothing listens at the speaker
    assert delay["what"].startswith("calculated delay up to 200 ms")
    assert "calculated by the run from its own timings and the requested player buffer; the delay at the speaker has not been measured" in delay["why"]
    assert "measured" not in delay["what"]


def test_the_delay_is_reported_every_five_seconds_and_at_the_end(tmp_path):
    clock = Clock()
    _, _, ev, _, _ = go(tmp_path, clock=clock, render=Render(clock, cost=0.04), duration=16.0)
    delays = [e for e in ev if e["event"] == "delay"]
    assert len(delays) == 4 and [e["t"] for e in delays] == sorted(e["t"] for e in delays)  # at 5, 10 and 15 s, and the last second
    # a render eats into the lead queued in front of it, so the delay is the same whether or not a block renders
    assert all(e["delay_s"] == pytest.approx(0.2, abs=1e-6) for e in delays)
    assert all(e["delay_s"] == pytest.approx(e["interval_s"] + e["render_s"] + e["queued_s"] + e["buffer_s"], abs=1e-6) for e in delays)


def test_the_delay_counts_the_render_when_nothing_is_queued_in_front(tmp_path):
    clock = Clock()
    _, n, ev, _, _ = go(tmp_path, clock=clock, render=Render(clock, cost=0.04), duration=0.1)
    (delay,) = [e for e in ev if e["event"] == "delay"]
    assert n == 1 and delay["delay_s"] == pytest.approx(0.14, abs=1e-6) and delay["render_s"] == pytest.approx(0.04, abs=1e-6)


def test_without_the_fast_readings_the_delay_counts_tt_smis_period(tmp_path):
    clock = Clock()
    source = Source(clock, note="No hwmon chips found, so readings come at tt-smi's pace.")
    source.fast, source.slow_every_s = False, 1.5  # FastSource fell back: a new temperature or power arrives only with the tt-smi reading
    _, _, ev, _, _ = go(tmp_path, clock=clock, source=source, duration=6.0)
    delay = [e for e in ev if e["event"] == "delay"][0]
    assert delay["interval_s"] == pytest.approx(1.5) and delay["delay_s"] == pytest.approx(1.6, abs=1e-6) and delay["over_budget"]


def test_a_delay_over_the_budget_is_warned_not_adjusted(tmp_path):
    sink, _, ev, _, _ = go(tmp_path, duration=11.0, player_name="paplay", latency_ms=100)
    delays = [e for e in ev if e["event"] == "delay"]
    assert len(delays) == 3  # at 5 and 10 s, and the last second
    for e in delays:
        assert e["buffer_s"] == pytest.approx(0.1) and e["delay_s"] == pytest.approx(0.3, abs=1e-6)
        assert e["over_budget"] and "250 ms" in e["what"] and e["interval_s"] == pytest.approx(0.1)
    assert {len(c) for c in sink.chunks[:-1]} == {int(0.1 * SR)}  # the block (and so the interval) was left as asked


def test_the_player_buffer_for_auto_is_the_player_the_sink_runs(tmp_path):
    class Player:
        command = ["aplay", "-q"]

        def write(self, audio):
            pass

        def close(self):
            pass

    class Tee(Sink):
        def __init__(self, clock):
            super().__init__(clock)
            self.sinks = (object(), Player())

    clock = Clock()
    _, _, ev, _, _ = go(tmp_path, clock=clock, sink=Tee(clock), duration=6.0, player_name="auto", latency_ms=50)
    delay = [e for e in ev if e["event"] == "delay"][0]
    assert delay["buffer_s"] == pytest.approx(0.05)
    assert fast_module.player_buffer_s("auto", Sink(), 50) == 0.0  # auto with no player found plays nothing


def _slow_heads(clock):
    """A source whose chip 0 steps at 1.5 s and 3.6 s: with the first block, three head renders on the loop, far enough apart that the long
    phrases between them are rendered ahead in the background, and each more than the minimum gap after the last phrase started."""
    return Source(clock, _stepped((1.5, 3.6)))


def test_three_head_renders_in_a_row_slower_than_the_block_lengthen_it_and_log_the_real_rate(tmp_path):
    clock = Clock()
    sink, _, ev, _, _ = go(tmp_path, clock=clock, source=_slow_heads(clock), render=Render(clock, cost=0.3), duration=6.0)
    longer = [e for e in ev if e["event"] == "live" and "lengthened" in e["what"]]
    assert len(longer) == 1 and "0.40 s" in longer[0]["what"] and "300 ms" in longer[0]["why"]
    assert f"{OVERRUNS_TO_LENGTHEN} head renders in a row" in longer[0]["why"]
    sizes = [len(c) for c in sink.chunks[:-1]]
    k = sizes.index(int(0.4 * SR))  # blocks at the interval until the third slow head, then the real rate
    assert k > 10 and set(sizes[:k]) == {int(0.1 * SR)} and set(sizes[k:]) == {int(0.4 * SR)}
    delay = [e for e in ev if e["event"] == "delay"][-1]  # the stretch after the lengthening
    assert delay["interval_s"] == pytest.approx(0.4)


def test_one_slow_head_render_does_not_lengthen_the_block(tmp_path):
    clock = Clock()
    sink, _, ev, _, _ = go(tmp_path, clock=clock, render=Render(clock, cost=0.3), duration=6.0)  # only the first block renders on the loop
    assert not [e for e in ev if "lengthened" in e["what"]]
    assert {len(c) for c in sink.chunks[:-1]} == {int(0.1 * SR)}


def test_a_head_render_that_fits_the_block_starts_the_count_again(tmp_path):
    clock = Clock()
    head = (6 * 800 + 400 + 1200 + 0.5) / SR  # a head's length at SR, with the crossfade and the margin
    costs = [0.3, 0.3, 0.0, 0.3, 0.3]  # the loop's head renders, in order: two slow, one quick, two slow

    def cost(duration):
        return costs.pop(0) if duration == pytest.approx(head) and costs else 0.0

    source = Source(clock, _stepped((1.5, 3.0, 4.5, 6.0)))
    _, _, ev, _, _ = go(tmp_path, clock=clock, source=source, render=Render(clock, cost=cost), duration=8.0)
    assert not costs  # all five heads were rendered
    assert not [e for e in ev if "lengthened" in e["what"]]


def test_a_lengthened_block_returns_to_the_interval_after_a_calm_stretch_and_carries_the_state(tmp_path):
    clock = Clock()
    render = Render(clock, cost=lambda duration: 0.3 if clock.t < START + 4.5 else 0.0)  # slow heads only at first
    made, carried = [], []

    class Recording(InstrumentSession):
        def carry_from(self, other):
            super().carry_from(other)
            carried.append(((other._soloist, other._soloist_until), (self._soloist, self._soloist_until)))

    def factory(*args, **kw):
        made.append(Recording(*args, render=render, worker_factory=lambda: InlineWorker(clock), **kw))
        return made[-1]

    sink, _, ev, _, _ = go(tmp_path, clock=clock, source=_slow_heads(clock), render=render, duration=40.0, session_factory=factory,
                           normalise=True, eq=True)
    longer = [e for e in ev if e["event"] == "live" and "lengthened" in e["what"]]
    back = [e for e in ev if e["event"] == "live" and "back to" in e["what"]]
    assert len(longer) == 1 and len(back) == 1 and len(made) == 3
    assert "0.10 s" in back[0]["what"] and f"{CALM_BLOCKS_TO_SHORTEN} blocks" in back[0]["why"]
    sizes = [len(c) for c in sink.chunks[:-1]]
    long_run = [k for k, n in enumerate(sizes) if n == int(0.4 * SR)]
    assert len(long_run) >= CALM_BLOCKS_TO_SHORTEN and long_run == list(range(long_run[0], long_run[-1] + 1))  # one stretch of long blocks
    assert set(sizes[long_run[-1] + 1:]) == {int(0.1 * SR)}  # then the interval again, to the end
    old, new = made[1], made[2]
    assert new._evictor is old._evictor and new._gain is old._gain and new._equaliser is old._equaliser
    assert len(carried) == 2 and carried[1][0] == carried[1][1]  # the held soloist and its turn went on when the block returned
    last = [e for e in ev if e["event"] == "delay"][-1]
    assert not last["block_lengthened"] and last["interval_s"] == pytest.approx(0.1) and last["queued_s"] == pytest.approx(LEAD_S_AT_22050)


def test_a_lengthened_block_stays_while_renders_still_outrun_the_interval_asked_for(tmp_path):
    # renders of 0.3 s fit the lengthened 0.4 s block but not the 0.1 s asked for, and a change every 1.5 s keeps rendering: no return
    clock = Clock()
    source = Source(clock, _stepped(tuple(1.5 * k for k in range(1, 27))))
    sink, _, ev, _, _ = go(tmp_path, clock=clock, source=source, render=Render(clock, cost=0.3), duration=40.0)
    assert len([e for e in ev if "lengthened" in e["what"]]) == 1 and not [e for e in ev if "back to" in e["what"]]
    assert sink.chunks[-2].shape[0] == int(0.4 * SR)


def test_the_lengthening_rules_are_the_ones_chosen():
    assert (OVERRUNS_TO_LENGTHEN, CALM_BLOCKS_TO_SHORTEN) == (3, 60)


def test_a_lengthened_block_joins_the_old_sound_without_a_cut(tmp_path):
    clock = Clock()
    sink, _, _, _, _ = go(tmp_path, clock=clock, source=_slow_heads(clock), render=Render(clock, cost=0.3), duration=5.0)
    level = 0.001 * (mock_telemetry(2)[0].power + 80.0)  # both steps have landed by the third head
    k = [len(c) for c in sink.chunks].index(int(0.4 * SR))
    joined = sink.chunks[k][: int(0.05 * SR)]  # the new session's first block fades in over the old phrase's faded-out tail
    k = (np.arange(len(joined)) + 0.5) / len(joined)
    assert np.allclose(joined, level * (np.sin(0.5 * np.pi * k) + np.cos(0.5 * np.pi * k)), atol=1e-5)


def test_the_floor_is_logged_when_it_changes_hands(tmp_path):
    _, _, ev, _, _ = go(tmp_path, duration=20.0)
    floors = [e for e in ev if e["event"] == "live" and "has the floor" in e["what"]]
    assert floors and all("device" in e and "chip" in e for e in floors)
    assert len(floors) < 20  # once a turn, never once a block


# --- readings that fail ----------------------------------------------------------------------------------------------------------

def test_no_reading_at_all_ends_the_run_with_the_no_chips_error_and_closes_everything(tmp_path):
    clock = Clock()
    source = Source(clock, lambda t: [])
    sink = Sink(clock)
    with pytest.raises(ValueError, match="no chips"):
        go(tmp_path, clock=clock, source=source, sink=sink)
    assert sink.closed == 1 and source.stopped == 1 and not sink.chunks
    assert any(e["event"] == "discover" and "no chips" in e["what"] for e in _events(tmp_path))


def test_a_reading_that_fails_mid_run_plays_the_last_one_and_says_so_once(tmp_path):
    clock = Clock()
    source = Source(clock, lambda t: [] if START + 1.0 <= t < START + 2.0 else mock_telemetry(2))
    sink, n, ev, _, _ = go(tmp_path, clock=clock, source=source, duration=3.0)
    lost = [e for e in ev if "no chips in this reading" in e["what"]]
    back = [e for e in ev if "readings are back" in e["what"]]
    assert n == 30 and len(lost) == 1 and len(back) == 1 and np.isfinite(sink.audio).all()


def test_new_errors_are_logged_when_they_arrive_and_only_then(tmp_path):
    clock = Clock()

    def readings(t):
        chips = mock_telemetry(2)
        if t >= START + 1.0:  # a cumulative counter: one error, reported by every reading after it
            chips[1] = dataclasses.replace(chips[1], corr_errs=1)
        return chips

    _, _, ev, _, _ = go(tmp_path, clock=clock, source=Source(clock, readings), duration=3.0)
    assert len([e for e in ev if e["event"] == "events" and e.get("chip") == 1 and "further" in e["what"]]) == 1


# --- stopping --------------------------------------------------------------------------------------------------------------------

def test_ctrl_c_closes_the_sink_and_the_log(tmp_path):
    clock = Clock()
    source = Source(clock, _interrupt_on(12))  # the first read, then ten blocks, then Ctrl-C during the next read
    sink, n, ev, _, _ = go(tmp_path, clock=clock, source=source, duration=None)
    assert n == 10 and sink.closed == 1 and source.stopped == 1
    assert len(sink.chunks[-1]) == int(0.05 * SR)  # the last phrase's tail is handed over
    assert ev[-1]["event"] == "done" and any("Ctrl-C" in e["what"] for e in ev)


def test_a_player_that_stops_ends_the_run_with_a_player_error_and_a_log_line(tmp_path):
    clock = Clock()
    sink = Sink(clock, fail_after=3)
    source = Source(clock)
    with pytest.raises(PlayerError):
        go(tmp_path, clock=clock, sink=sink, source=source)
    assert sink.closed == 1 and source.stopped == 1
    assert any(e["event"] == "player" for e in _events(tmp_path))


def test_a_player_that_stops_after_ctrl_c_is_logged_and_the_run_still_ends_as_stopped(tmp_path):
    # Ctrl-C after ten blocks; the player is gone by the time the tail is written (as paplay was, when it shared Python's process group)
    clock = Clock()
    source = Source(clock, _interrupt_on(12))
    sink = Sink(clock, fail_after=10)
    _, n, ev, _, _ = go(tmp_path, clock=clock, source=source, sink=sink, duration=None)  # no PlayerError: the user stopped the run
    assert n == 10 and sink.closed == 1 and source.stopped == 1
    (player,) = [e for e in ev if e["event"] == "player"]
    assert "after Ctrl-C" in player["what"] and "paplay stopped" in player["what"]
    assert ev[-1]["event"] == "done"


def test_the_fast_sources_own_log_lines_reach_the_run_log(tmp_path):
    clock = Clock()

    class Logging(Source):
        log = None  # FastSource's attribute: run_fast sets it

        def sample(self):
            if len(self.calls) == 3 and self.log is not None:
                self.log("chip 1 (0000:02:00.0): its hwmon read failed", "its last hwmon reading is played", chip=1)
            return super().sample()

    _, _, ev, _, _ = go(tmp_path, clock=clock, source=Logging(clock), duration=1.0)
    (line,) = [e for e in ev if "hwmon read failed" in e["what"]]
    assert line["event"] == "live" and line["chip"] == 1 and line["why"] == "its last hwmon reading is played"


# --- the synthetic source --------------------------------------------------------------------------------------------------------

def test_a_sampled_source_wraps_a_plain_sampler_with_a_note():
    calls = []
    source = SampledSource(lambda: calls.append(1) or mock_telemetry(3), "The readings are synthetic.")
    source.start()
    assert len(source.sample()) == 3 and calls == [1] and source.note == "The readings are synthetic." and source.fast is False
    source.stop()


def test_only_plays_the_named_devices(tmp_path):
    seen = []

    class Spy(Render):
        def __call__(self, chips, duration, sample_rate=44100, **kw):
            seen.append(kw.get("only"))
            return super().__call__(chips, duration, sample_rate, **kw)

    clock = Clock()
    _, _, ev, _, _ = go(tmp_path, clock=clock, render=Spy(clock), only=["gong"], duration=1.0)
    assert seen and all(o == ["gong"] for o in seen)
    assert any(e["event"] == "only" for e in ev)


def test_only_says_once_that_it_turns_eviction_off(tmp_path):
    def lines(**kw):
        _, _, ev, _, _ = go(tmp_path, duration=1.0, **kw)
        return [e for e in ev if e["event"] == "live" and "eviction" in e["what"] and "--only" in e["what"]]

    (line,) = lines(only=["gong"], evict=True)
    assert "turns eviction off" in line["what"]
    assert not lines(only=None, evict=True) and not lines(only=["gong"], evict=False)  # nothing is turned off in either


# --- an interval under the shortest block ----------------------------------------------------------------------------------------

def test_an_interval_under_the_shortest_block_is_logged_at_the_rate_the_loop_runs(tmp_path):
    _, n, ev, source, _ = go(tmp_path, duration=6.0, interval_s=0.01)
    (line,) = [e for e in ev if e["event"] == "live" and e["what"].startswith("fast mode")]
    assert "every 10 ms" not in line["what"] and "every 100 ms" in line["what"]
    assert "--interval 10 was raised to 100 ms, the shortest block" in line["what"]
    assert len(source.calls) == n + 1 and n == 60  # one read a block: every 100 ms


def test_the_delay_at_an_interval_of_10_ms_counts_the_100_ms_block(tmp_path):
    _, _, ev, _, _ = go(tmp_path, duration=6.0, interval_s=0.01)
    delay = [e for e in ev if e["event"] == "delay"][0]
    assert delay["interval_s"] == pytest.approx(0.1) and delay["delay_s"] == pytest.approx(0.2, abs=1e-6)
    assert "every 100 ms" in delay["why"]


def test_an_interval_of_100_ms_or_more_is_logged_as_before(tmp_path):
    for interval_s, text in ((0.1, "fast mode: the chips are read every 100 ms, in blocks of 0.10 s"),
                             (0.25, "fast mode: the chips are read every 250 ms, in blocks of 0.25 s")):
        _, _, ev, _, _ = go(tmp_path, duration=1.0, interval_s=interval_s)
        assert [e["what"] for e in ev if e["event"] == "live" and e["what"].startswith("fast mode")] == [text]


# --- meeting the delay budget: heads, bodies and renders ahead (Task 8b) -------------------------------------------------------------

def test_the_lead_is_the_head_render_time_with_a_margin_at_the_rate_in_use():
    assert lead_for(22050) == pytest.approx(0.10) and lead_for(44100) == pytest.approx(0.18)
    assert lead_for(33075) == pytest.approx(0.14)  # between the two measured rates: a straight line
    assert lead_for(8000) == pytest.approx(0.10) and lead_for(96000) == pytest.approx(0.18)  # outside them: the nearest


def test_the_loop_keeps_the_lead_queued_ahead_of_the_listener(tmp_path):
    for rate, lead in ((8000, 0.10), (44100, 0.18)):
        _, _, ev, _, _ = go(tmp_path, duration=6.0, sample_rate=rate)
        delay = [e for e in ev if e["event"] == "delay"][0]
        assert delay["queued_s"] == pytest.approx(lead, abs=1e-6) and delay["delay_s"] == pytest.approx(0.1 + lead, abs=1e-6)


def test_fast_mode_does_not_lengthen_the_block_for_a_long_phrase(tmp_path):
    clock = Clock()
    render = Render(clock, cost=lambda duration: 0.08 * duration)  # a head (0.8 s) costs 64 ms; a long phrase (6 s) costs 480 ms
    sink, n, ev, _, _ = go(tmp_path, clock=clock, render=render, duration=15.0)
    assert not [e for e in ev if "lengthened" in e["what"]]
    assert n == 150 and {len(c) for c in sink.chunks[:-1]} == {int(0.1 * SR)}  # the block stayed at the interval
    assert len(set(render.segments)) >= 3  # long phrases did play: rendered in the background, not on the loop


def test_a_head_render_slower_than_the_block_still_lengthens_it_and_the_new_session_carries_the_state(tmp_path):
    clock = Clock()
    render = Render(clock, cost=0.3)
    made = []

    def factory(*args, **kw):
        made.append(InstrumentSession(*args, render=render, worker_factory=lambda: InlineWorker(clock), **kw))
        return made[-1]

    _, _, ev, _, _ = go(tmp_path, clock=clock, source=_slow_heads(clock), render=render, duration=5.0, session_factory=factory, normalise=True,
                        eq=True)
    assert len([e for e in ev if "lengthened" in e["what"]]) == 1 and len(made) == 2
    old, new = made
    # the learned rest of the eviction tracker, the held soloist, the gain level and the EQ's state go on in the new session
    assert new._evictor is old._evictor and new._gain is old._gain and new._equaliser is old._equaliser
    assert old._soloist is not None and (new._soloist, new._soloist_until) == (old._soloist, old._soloist_until)


def test_the_report_gives_the_typical_and_the_worst_delay(tmp_path):
    clock = Clock()
    # 0.25 s blocks; a head render of 150 ms outruns the 100 ms lead: that block's delay is 0.25 + 0.15 = 0.40 s, and a block cut from a
    # rendered phrase is 0.25 + 0.10 = 0.35 s
    _, _, ev, _, _ = go(tmp_path, clock=clock, render=Render(clock, cost=0.15), duration=6.0, interval_s=0.25)
    delay = [e for e in ev if e["event"] == "delay"][0]
    assert delay["delay_s"] == pytest.approx(0.40, abs=1e-6)
    # typical: the middle block of the stretch, with half the interval (a change lands half way between two reads on average)
    assert delay["typical_s"] == pytest.approx(0.35 - 0.125, abs=1e-6)
    assert "400 ms" in delay["what"] and "225 ms" in delay["what"]


def test_a_body_that_is_late_is_counted_as_a_gap_in_the_report(tmp_path):
    clock = Clock()
    sink, n, ev, _, _ = go(tmp_path, clock=clock, duration=6.0, worker=lambda: InlineWorker(clock, never=True))
    assert n == 60  # the loop went on at the interval with every body late
    delay = [e for e in ev if e["event"] == "delay"][0]
    assert delay["gaps"] > 0 and "gap" in delay["why"]
    assert not sink.chunks[7].any()  # the head is six blocks; the eighth is silence while the body is missing


def test_the_over_budget_advice_names_the_term_that_dominates(tmp_path):
    def worst(**kw):
        _, _, ev, _, _ = go(tmp_path, duration=6.0, **kw)
        delay = [e for e in ev if e["event"] == "delay"][0]
        assert delay["over_budget"], delay
        return delay["why"]

    buffer = worst(player_name="paplay", latency_ms=200)  # 0.1 + 0.1 + 0.2
    assert "player" in buffer.split("; ")[-1] and "--interval" not in buffer and "--sample-rate" not in buffer
    interval = worst(interval_s=0.25)  # 0.25 + 0.1
    assert "a shorter --interval" in interval and "--sample-rate" not in interval and "the interval is left as it was asked for" in interval
    clock = Clock()
    _, _, ev, _, _ = go(tmp_path, clock=clock, source=_slow_heads(clock), render=Render(clock, cost=0.3), duration=6.0)
    lengthened = [e for e in ev if e["event"] == "delay"][-1]["why"]  # the stretch after slow head renders made the loop lengthen the block
    assert "--sample-rate" in lengthened and "--interval" not in lengthened and "the block was lengthened" in lengthened
    assert "left as it was asked for" not in lengthened  # the loop changed the block, so it was not left as asked
    near = worst(player_name="paplay", latency_ms=60)  # 0.1 + 0.1 + 0.06, with the block already the shortest: nothing about the interval
    assert "--interval" not in near and "the interval" not in near
    clock = Clock()
    source = Source(clock, note="No hwmon chips found, so readings come at tt-smi's pace.")
    source.fast, source.slow_every_s = False, 1.5
    slow = worst(clock=clock, source=source)  # the block is already the shortest: nothing about the interval
    assert "--interval" not in slow and "the interval" not in slow and "tt-smi" in slow


def test_the_advice_follows_the_largest_part_and_what_the_block_is():
    advice = fast_module._advice
    plain = dict(block_s=0.25, lengthened=False, fallback=False)
    assert "a shorter --interval" in advice(0.25, 0.0, 0.06, 0.0, plain)
    assert "--sample-rate" in advice(0.4, 0.3, 0.1, 0.0, dict(plain, block_s=0.4, lengthened=True))
    assert "--interval" not in advice(0.1, 0.0, 0.06, 0.095, dict(plain, block_s=0.1))
    assert "already the shortest" in advice(0.1, 0.0, 0.06, 0.095, dict(plain, block_s=0.1))
    assert "tt-smi" in advice(1.5, 0.0, 0.06, 0.0, dict(plain, block_s=0.1, fallback=True))
    assert "player" in advice(0.1, 0.0, 0.06, 0.2, plain)


def test_the_delay_line_says_when_the_block_was_lengthened(tmp_path):
    clock = Clock()
    _, _, ev, _, _ = go(tmp_path, clock=clock, source=_slow_heads(clock), render=Render(clock, cost=0.3), duration=6.0)
    delays = [e for e in ev if e["event"] == "delay"]
    assert all(e["block_lengthened"] for e in delays)
    # after the lengthening the lead is the new block, so a block with no render has a whole block queued in front of it
    assert delays[-1]["delay_s"] == pytest.approx(0.8, abs=1e-6) and delays[-1]["queued_s"] == pytest.approx(0.4, abs=1e-6)
    _, _, ev, _, _ = go(tmp_path, duration=6.0)
    assert not any(e["block_lengthened"] for e in ev if e["event"] == "delay")


def test_ctrl_c_also_writes_a_last_delay_line(tmp_path):
    clock = Clock()
    source = Source(clock, _interrupt_on(12))  # ten blocks, under the first five-second report
    _, _, ev, _, _ = go(tmp_path, clock=clock, source=source, duration=None)
    kinds = [e["event"] for e in ev]
    assert kinds.count("delay") == 1 and kinds.index("delay") > next(i for i, e in enumerate(ev) if "Ctrl-C" in e["what"])


def test_the_tt_smi_line_gives_the_sources_own_period(tmp_path):
    clock = Clock()
    source = Source(clock)
    source.slow_every_s = 2.5
    _, _, ev, _, _ = go(tmp_path, clock=clock, source=source, duration=1.0)
    assert any("about every 2.5 s" in e["what"] for e in ev)


def test_ffplay_is_said_to_be_over_the_budget_and_auto_to_pick_paplay_first(tmp_path):
    _, _, ev, _, _ = go(tmp_path, duration=1.0, player_name="ffplay")
    (line,) = [e for e in ev if e["event"] == "live" and "ffplay" in e["what"]]
    assert "100 ms" in line["what"] + line["why"] and "over the 250 ms budget" in line["what"] + line["why"]
    assert "--player auto" in line["why"] and "paplay first" in line["why"]
    _, _, ev, _, _ = go(tmp_path, duration=1.0, player_name="paplay", latency_ms=20)
    assert not [e for e in ev if e["event"] == "live" and "ffplay" in e["what"]]


def test_a_quiet_stretch_renders_the_next_long_phrase_ahead_so_it_starts_with_no_head(tmp_path):
    clock = Clock()
    durations = []

    class Spy(Render):
        def __call__(self, chips, duration, sample_rate=44100, **kw):
            durations.append(duration)
            return super().__call__(chips, duration, sample_rate, **kw)

    go(tmp_path, clock=clock, render=Spy(clock), duration=12.0)
    head = (6 * 800 + 400 + 1200 + 0.5) / SR  # six 0.1 s blocks, the crossfade and the margin
    long = (60 * 800 + 400 + 0.5) / SR
    # only the first phrase starts from a head; each long phrase after it was rendered ahead, in the background, while nothing moved: at 0.6 s
    # (the short phrase half played once its body took over), 4 s and 10 s (each long phrase half played)
    assert durations.count(pytest.approx(head)) == 1 and durations.count(pytest.approx(long)) == 3



def _stepped(at):
    """Readings whose chip 0 power steps up by 40 W at each clock time in `at` (seconds after START), so each starts a short phrase."""
    def readings(t):
        chips = mock_telemetry(2)
        steps = sum(1 for a in at if t >= START + a)
        return [dataclasses.replace(chips[0], power=chips[0].power + 40.0 * steps)] + chips[1:]
    return readings


def test_a_head_render_longer_than_the_lead_is_counted_as_a_gap_of_its_overrun_for_a_real_player(tmp_path):
    clock = Clock()
    # 0.25 s blocks, so a 0.15 s head render fits the block (no lengthening) and overruns the 0.10 s lead by 0.05 s at each change
    source = Source(clock, _stepped((2.0, 3.5)))
    _, _, ev, _, _ = go(tmp_path, clock=clock, source=source, render=Render(clock, cost=0.15), duration=6.0, interval_s=0.25,
                        player_name="paplay", latency_ms=50)
    first = [e for e in ev if e["event"] == "delay"][0]  # 0 to 5 s: the two changes; the first block has nothing queued to run out of
    assert first["gaps"] == 2 and "2 gaps in the sound (100 ms in all)" in first["why"]
    clock = Clock()
    _, _, ev, _, _ = go(tmp_path, clock=clock, source=Source(clock, _stepped((2.0, 3.5))), render=Render(clock, cost=0.15), duration=6.0,
                        interval_s=0.25, player_name="none")
    assert all(e["gaps"] == 0 for e in ev if e["event"] == "delay")  # with no player nobody hears the listener run out


def test_late_blocks_are_counted_exactly_once_each_as_gaps(tmp_path):
    clock = Clock()
    _, n, ev, _, _ = go(tmp_path, clock=clock, duration=6.0, worker=lambda: InlineWorker(clock, never=True))
    # every body is late: the short phrase is 6 head blocks and 4 late ones; then a long phrase of 6 head blocks and late blocks to the end
    delays = [e for e in ev if e["event"] == "delay"]
    assert n == 60 and sum(e["gaps"] for e in delays) == 4 + (60 - 16)
    assert sum(float(e["why"].split("gaps in the sound (")[1].split(" ms")[0]) for e in delays) == pytest.approx(48 * 100)


# --- a change held back by the minimum gap between phrases (final review, C1) ---------------------------------------------------------

def test_a_change_inside_the_minimum_gap_is_held_back_and_the_report_counts_the_wait(tmp_path):
    # the step at 2.5 s starts a phrase (the last one began at 1.0 s); the step at 2.7 s lands 0.2 s after it, so it waits for the gap to end at
    # about 3.5 s: held back about 0.8 s, on top of the 0.2 s a change otherwise takes
    clock = Clock()
    _, _, ev, _, _ = go(tmp_path, clock=clock, source=Source(clock, _stepped((2.5, 2.7))), duration=4.0)
    (delay,) = [e for e in ev if e["event"] == "delay"]
    assert delay["held_s"] == pytest.approx(MIN_GAP_S - 0.2, abs=0.11)
    assert delay["delay_s"] == pytest.approx(0.2 + MIN_GAP_S - 0.2, abs=0.11) and delay["over_budget"]
    assert delay["delay_s"] == pytest.approx(delay["interval_s"] + delay["render_s"] + delay["queued_s"] + delay["buffer_s"] + delay["held_s"],
                                             abs=1e-6)
    assert "held back" in delay["why"] and f"{delay['held_s'] * 1000:.0f} ms" in delay["why"]
    assert "the wait for the minimum gap between phrases is the largest part" in delay["why"]  # the advice names it


def test_a_change_after_the_gap_is_not_held_back(tmp_path):
    clock = Clock()
    _, _, ev, _, _ = go(tmp_path, clock=clock, source=Source(clock, _stepped((2.5,))), duration=4.0)
    (delay,) = [e for e in ev if e["event"] == "delay"]
    assert delay["held_s"] == 0.0 and delay["delay_s"] == pytest.approx(0.2, abs=1e-6) and not delay["over_budget"]


def test_a_reading_that_moves_back_inside_the_gap_is_not_counted_as_held(tmp_path):
    # up at 2.7 s, back at 2.8 s (nothing to play any more), up again at 3.3 s: held from 3.3 s to the gap's end at 3.5 s, not from 2.7 s
    clock = Clock()

    def readings(t):
        chips = mock_telemetry(2)
        up = 40.0 if t >= START + 2.5 else 0.0
        if START + 2.7 <= t < START + 2.8 or t >= START + 3.3:
            up += 40.0
        return [dataclasses.replace(chips[0], power=chips[0].power + up)] + chips[1:]

    _, _, ev, _, _ = go(tmp_path, clock=clock, source=Source(clock, readings), duration=4.0)
    (delay,) = [e for e in ev if e["event"] == "delay"]
    assert delay["held_s"] == pytest.approx(0.2, abs=0.11)


def test_a_held_change_is_counted_once_when_errors_start_phrases_back_to_back(tmp_path):
    # a step at 2.7 s waits for the gap; an error at 2.8 s starts a phrase at once and carries it (held 0.1 s); another error at 2.9 s starts the
    # next phrase, which has nothing held back any more
    clock = Clock()
    base = _stepped((2.5, 2.7))

    def readings(t):
        chips = base(t)
        errors = sum(1 for a in (2.8, 2.9) if t >= START + a)  # a cumulative counter, as tt-smi gives it
        return [chips[0], dataclasses.replace(chips[1], corr_errs=errors)]

    _, _, ev, _, _ = go(tmp_path, clock=clock, source=Source(clock, readings), duration=4.0)
    (delay,) = [e for e in ev if e["event"] == "delay"]
    assert delay["held_s"] == pytest.approx(0.1, abs=0.05)
