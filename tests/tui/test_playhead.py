# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The playhead: what the TUI shows is the segment the listener hears, and the log lines made with it."""
import threading
import time

import numpy as np
import pytest

from tensixinet.fast import player_in_use
from tensixinet.tui.playhead import KEEP_RECORDS, Playhead, Shown, TimedSink


class Clock:
    """A clock the test moves by hand."""

    def __init__(self, t=100.0):
        self.t = t

    def __call__(self):
        return self.t


class MemorySink:
    def __init__(self):
        self.chunks, self.closed = [], 0

    def write(self, audio):
        self.chunks.append(np.array(audio))

    def close(self):
        self.closed += 1


def rec(what):
    return {"t": 0.0, "event": "live", "what": what, "why": "a reason"}


def whats(shown):
    return [r["what"] for r in shown.records]


# Stand-ins for three segments' states: the playhead only stores and returns them, so any distinct objects will do.
s0, s1, s2 = object(), object(), object()
a = np.zeros(8, dtype=np.float32)


@pytest.fixture
def clock():
    return Clock()


def test_nothing_is_heard_before_the_first_write(clock):
    p = Playhead(clock)
    p.hand_over(4.0, s0)
    assert p.shown().state is None and p.heard_s() is None
    assert isinstance(p.shown(), Shown) and p.shown().heard_s is None


def test_the_state_shown_is_the_segment_heard_while_two_more_are_rendered(clock):
    p = Playhead(clock)
    p.hand_over(4.0, s0)
    p.started()
    p.hand_over(4.0, s1)
    p.hand_over(4.0, s2)
    clock.t += 2.5
    assert p.shown().state is s0
    clock.t += 2.0
    assert p.shown().state is s1  # 4.5 s in
    clock.t += 4.0
    assert p.shown().state is s2  # 8.5 s in


def test_a_state_shows_from_exactly_its_own_offset(clock):
    p = Playhead(clock)
    p.hand_over(4.0, s0)
    p.started()
    p.hand_over(4.0, s1)
    clock.t += 4.0  # exactly at s1's offset (whole numbers, so no rounding)
    assert p.shown().state is s1 and p.shown().heard_s == pytest.approx(4.0)


def test_the_first_segment_shows_from_the_first_write(clock):
    p = Playhead(clock)
    p.hand_over(4.0, s0)
    p.started()
    assert p.shown().state is s0 and p.heard_s() == 0.0


def test_the_player_buffer_delays_what_is_shown(clock):
    p = Playhead(clock, buffer_s=0.5)
    p.hand_over(4.0, s0)
    p.started()
    p.hand_over(4.0, s1)
    clock.t += 4.2
    assert p.shown().state is s0 and p.heard_s() == pytest.approx(3.7)
    clock.t += 0.4
    assert p.shown().state is s1


def test_a_record_logged_while_a_segment_is_made_shows_when_that_segment_is_heard(clock):
    p = Playhead(clock)
    p.hand_over(4.0, s0)
    p.started()
    p.note(rec("segment 1"))
    p.hand_over(4.0, s1)
    clock.t += 3.9
    assert "segment 1" not in whats(p.shown())
    clock.t += 0.2
    assert "segment 1" in whats(p.shown())


def test_the_opening_lines_show_before_any_audio(clock):
    p = Playhead(clock)
    p.note(rec("discover"))
    assert whats(p.shown()) == ["discover"]


def test_a_gap_restarts_the_listeners_time_from_the_audio_handed_over(clock):
    p = Playhead(clock)
    p.hand_over(1.0, s0)
    p.started()
    clock.t += 3.0
    p.hand_over(1.0, s1)  # ran dry for 2 s
    assert p.heard_s() == pytest.approx(1.0) and p.shown().state is s1


def test_audio_still_queued_is_no_gap(clock):
    p = Playhead(clock)
    p.hand_over(4.0, s0)
    p.started()
    clock.t += 3.0
    p.hand_over(4.0, s1)  # 1 s of s0 still to hear
    assert p.heard_s() == pytest.approx(3.0) and p.shown().state is s0


def test_only_the_newest_records_are_kept(clock):
    p = Playhead(clock)
    for i in range(KEEP_RECORDS + 5):
        p.note(rec(str(i)))
    assert len(p.shown().records) == KEEP_RECORDS and p.shown().records[-1]["what"] == str(KEEP_RECORDS + 4)


def test_started_twice_keeps_the_first_time(clock):
    p = Playhead(clock)
    p.hand_over(4.0, s0)
    p.started()
    clock.t += 2.0
    p.started()
    assert p.heard_s() == pytest.approx(2.0)


def test_the_timed_sink_starts_the_clock_on_its_first_write_and_passes_the_rest_through(clock):
    p = Playhead(clock)
    inner = MemorySink()
    inner.command = ["paplay"]
    t = p.wrap(inner)
    assert isinstance(t, TimedSink)
    clock.t = 7.0
    t.write(a)
    clock.t = 9.0
    t.write(a)
    assert p._t0 == 7.0 and len(inner.chunks) == 2 and t.command == ["paplay"]
    t.close()
    assert inner.closed == 1


def test_the_wrapped_sink_still_names_its_player_for_the_delay_report(clock):
    inner = MemorySink()
    inner.command = ["paplay", "--raw"]
    assert player_in_use("auto", Playhead(clock).wrap(inner)) == "paplay"


def test_what_shown_returns_does_not_change_when_more_arrives(clock):
    p = Playhead(clock)
    p.note(rec("one"))
    shown = p.shown()
    p.note(rec("two"))
    assert whats(shown) == ["one"]


def test_a_slow_consumer_never_holds_up_the_loop():
    """The TUI's thread takes what is shown and then spends a long time drawing it, and asks again as often as it likes. Each of the
    loop's calls is timed on its own: it only takes the lock, appends and returns, so none of them waits on the consumer."""
    p = Playhead()
    p.hand_over(1.0, s0)
    p.started()
    stop = threading.Event()

    def drawer():  # takes a frame, then holds the draw path for 0.5 s
        while not stop.is_set():
            p.shown()
            time.sleep(0.5)

    def poller():  # asks for what is shown every 5 ms, twenty times as often as the TUI draws
        while not stop.is_set():
            p.shown()
            time.sleep(0.005)

    threads = [threading.Thread(target=f, daemon=True) for f in (drawer, poller)]
    for t in threads:
        t.start()
    took = []
    try:
        ends = time.monotonic() + 0.8
        i = 0
        while time.monotonic() < ends:
            began = time.perf_counter()
            p.hand_over(0.001, s1)
            took.append(time.perf_counter() - began)
            began = time.perf_counter()
            p.note(rec(str(i)))
            took.append(time.perf_counter() - began)
            i += 1
            time.sleep(0.001)
    finally:
        stop.set()
        for t in threads:
            t.join(2)
    # Python hands the interpreter between threads every 5 ms (sys.getswitchinterval), so a call can be held that long by a thread
    # that is not touching the playhead at all; 20 ms leaves room for that and still fails if `shown` does slow work under the lock.
    assert len(took) > 100 and max(took) < 0.02


def test_the_opening_lines_stay_shown_while_the_player_buffer_fills(clock):
    """With a player buffer the listener's time is below zero for the first `buffer_s` after the first write. The opening lines,
    shown before any audio, stay shown; the first state waits until the listener reaches it."""
    p = Playhead(clock, buffer_s=0.5)
    p.note(rec("discover"))
    p.hand_over(4.0, s0)
    p.started()
    clock.t += 0.2
    shown = p.shown()
    assert whats(shown) == ["discover"] and shown.state is None and shown.heard_s == pytest.approx(-0.3)
    clock.t += 0.4
    assert p.shown().state is s0


def test_last_records_gives_the_newest_records_whether_or_not_the_listener_has_reached_them():
    # The exit lines print the run's last records. An error logged while a segment is still ahead of the listener must be among them.
    clock = Clock()
    ph = Playhead(clock=clock)
    for i in range(6):
        ph.note({"what": f"line {i}"})  # logged at offset 0
    ph.hand_over(4.0, None)
    ph.started()
    ph.hand_over(4.0, None)
    ph.note({"what": "the audio output stopped"})  # logged at offset 8.0; the listener is at 0
    assert [r["what"] for r in ph.shown().records][-1] == "line 5"
    assert [r["what"] for r in ph.last_records(3)] == ["line 4", "line 5", "the audio output stopped"]
    assert ph.last_records(0) == ()


def test_states_the_listener_has_passed_are_dropped_as_new_ones_arrive_even_when_nothing_draws(clock):
    # A drawing thread that is stuck never calls `shown`; at ten states a second the list must still stay short.
    p = Playhead(clock)
    p.hand_over(0.1, s0)
    p.started()
    for _ in range(1000):
        clock.t += 0.1
        p.hand_over(0.1, s1)
    assert len(p._states) <= 3
    assert p.shown().state is s1  # and what is shown is unchanged by the pruning


def test_pruning_in_hand_over_keeps_the_state_being_heard(clock):
    p = Playhead(clock)
    p.hand_over(4.0, s0)
    p.started()
    clock.t += 1.0  # the listener is inside s0
    p.hand_over(4.0, s1)
    p.hand_over(4.0, s2)
    assert p.shown().state is s0


def test_set_buffer_moves_the_listener_back_by_that_much(clock):
    p = Playhead(clock)
    p.hand_over(4.0, s0)
    p.started()
    clock.t += 1.0
    p.set_buffer(0.25)
    assert p.heard_s() == pytest.approx(0.75)
