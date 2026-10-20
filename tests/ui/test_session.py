# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
import threading
import time

import numpy as np
import pytest

from tensixinet.music.chorus import assign_device, device_names, slot_chip
from tensixinet.music.patch import DEFAULT_PATCH
from tensixinet.telemetry import mock_telemetry
from tensixinet.ui.session import Session, make_patch, mock_sample

SR = 22050


def _session(sample=None, **kw):
    sample = sample or (lambda: mock_telemetry(4, 0))
    return Session(sample, DEFAULT_PATCH, source="four synthetic chips", sample_rate=SR, sleep=lambda s: None, **kw)


def _wait(cond, timeout=40.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.05)
    return False


def test_a_session_plays_and_shows_the_segment_that_just_played():
    s = _session()
    s.start()
    try:
        assert _wait(lambda: s.snapshot().segment >= 1)
        snap = s.snapshot()
        assert snap.running and snap.error is None and snap.state is not None
        assert [d.name for d in snap.state.devices] == device_names()
        solo = next(d for d in snap.state.devices if d.soloist)
        assert (solo.name, solo.chip) == (snap.state.soloist_device, snap.state.soloist_chip)
        assert any("has the floor" in line for line in snap.log_lines)
        chunk = s.next_audio(timeout=5)
        assert chunk is not None and chunk.ndim == 2 and chunk.shape[1] == 2 and chunk.dtype == np.float32
    finally:
        s.stop()


def test_the_state_names_the_device_and_chip_the_log_says_have_the_floor():
    s = _session()
    s.start()
    try:
        assert _wait(lambda: s.snapshot().segment >= 1)
        snap = s.snapshot()
        said = [l for l in snap.log_lines if f"segment {snap.state.segment}:" in l][-1]
        assert snap.state.soloist_device in said and f"on chip {snap.state.soloist_chip}" in said
    finally:
        s.stop()


def test_stop_ends_the_run_promptly_without_an_error():
    s = _session()
    s.start()
    assert _wait(lambda: s.snapshot().segment >= 0 and s.snapshot().state is not None)
    s.stop()
    assert not s.snapshot().running and s.snapshot().error is None


def test_mono_gives_one_channel():
    s = _session(stereo=False)
    s.start()
    try:
        assert _wait(lambda: s.snapshot().state is not None)
        chunk = s.next_audio(timeout=5)
        assert chunk.ndim == 1
    finally:
        s.stop()


def test_a_source_with_no_chips_ends_the_session_with_the_reason():
    s = _session(sample=lambda: [])
    s.start()
    assert _wait(lambda: not s.snapshot().running)
    snap = s.snapshot()
    assert snap.error and "no chips" in snap.error and snap.state is None


def test_eviction_and_fill_events_set_the_phases_the_chart_shows():
    s = _session()
    s._chips = mock_telemetry(4, 0)
    live = lambda t, seg: {"t": t, "event": "live", "segment": seg, "chip": 0, "device": assign_device(0).__name__, "seconds": 5.0,
                           "what": f"segment {seg}", "why": "x"}
    s._on_event({"t": 5.0, "event": "evict", "chip": 1, "what": "chip 1 evicted", "why": "x"})
    s._on_event(live(6.0, 4))
    s._on_event(live(30.0, 9))
    s._begin_segment(4)
    assert [c.phase for c in s.snapshot().state.chips] == ["running", "evicting", "running", "running"]
    s._begin_segment(9)
    assert s.snapshot().state.chips[1].phase == "idle"


def test_the_chart_shows_the_segment_that_is_playing_and_not_the_one_the_loop_has_rendered_ahead():
    s = _session()
    s._chips = mock_telemetry(4, 0)
    for seg in (3, 4, 5):  # the loop is two segments ahead of the sound
        s._on_event({"t": float(seg), "event": "live", "segment": seg, "chip": 0, "device": assign_device(0).__name__, "seconds": 5.0,
                     "what": f"segment {seg}", "why": "x"})
    assert s.snapshot().state is None and s.snapshot().segment == -1  # nothing has started to play
    s._begin_segment(3)
    assert (s.snapshot().state.segment, s.snapshot().segment) == (3, 3)
    s._begin_segment(4)
    assert s.snapshot().state.segment == 4


def test_the_sink_starts_each_segment_in_order_so_the_chart_follows_the_sound():
    s = _session()
    s.start()
    try:
        assert _wait(lambda: s.snapshot().segment >= 2)
        assert s.snapshot().state.segment == s.snapshot().segment
    finally:
        s.stop()


def test_make_patch_draws_in_a_style_or_plain_and_is_repeatable():
    chips = mock_telemetry(4, 0)
    plain, enya = make_patch("", 7, chips), make_patch("enya", 7, chips)
    assert plain.style == "" and enya.style == "enya" and make_patch("enya", 7, chips) == enya
    assert make_patch("", 8, chips) != plain


def test_make_patch_rejects_an_unknown_style():
    with pytest.raises(ValueError):
        make_patch("polka", 1, mock_telemetry(4, 0))


def test_the_mock_source_cycles_load_when_asked_so_an_eviction_can_be_seen():
    t = [0.0]
    sample = mock_sample(4, seed=0, load_period=60.0, clock=lambda: t[0])
    rest = [c.power for c in sample()]
    t[0] = 20.0
    loaded = [c.power for c in sample()]
    assert loaded != rest and min(rest + loaded) == pytest.approx(14.0) and max(rest + loaded) == pytest.approx(62.0)  # the chips are staggered


def test_segments_that_have_played_are_dropped_and_later_ones_wait():
    s = _session()
    s._chips = mock_telemetry(4, 0)
    for seg in range(5):
        s._on_event({"t": float(seg), "event": "live", "segment": seg, "chip": 0, "device": assign_device(0).__name__, "seconds": 5.0,
                     "what": f"segment {seg}", "why": "x"})
    s._begin_segment(2)
    assert sorted(s._states) == [2, 3, 4]


def test_a_fill_after_an_eviction_shows_the_chip_filling_and_then_running():
    s = _session()
    s._chips = mock_telemetry(4, 0)
    s._on_event({"t": 5.0, "event": "evict", "chip": 1, "what": "e", "why": "x"})
    s._on_event({"t": 20.0, "event": "fill", "chip": 1, "what": "f", "why": "x"})
    live = lambda t, seg: s._on_event({"t": t, "event": "live", "segment": seg, "chip": 0, "device": assign_device(0).__name__,
                                       "seconds": 5.0, "what": "s", "why": "x"})
    live(21.0, 1)
    live(40.0, 2)
    s._begin_segment(1)
    assert s.snapshot().state.chips[1].phase == "filling"
    s._begin_segment(2)
    assert s.snapshot().state.chips[1].phase == "running"


def test_a_stopped_session_refuses_more_audio_so_the_loop_ends():
    from tensixinet.ui.session import _Stopped, _UiSink
    s = _session()
    s._stop.set()
    with pytest.raises(_Stopped):
        _UiSink(s).write(np.zeros((SR, 2), dtype=np.float32))
    assert s.next_audio(timeout=0.01) is None  # and the refused chunk was not queued


# --- the lead: audio handed over ahead of real time, so the browser has something to play while the next segment is made ---------------------

class _Time:
    """A clock that only moves when the session sleeps, and a sleep that records what it was asked for."""

    def __init__(self):
        self.now, self.slept = 100.0, []

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.now += seconds


def _led(lead_s, seconds=4.0):
    t = _Time()
    s = Session(lambda: mock_telemetry(4, 0), DEFAULT_PATCH, sample_rate=SR, sleep=t.sleep, clock=t.clock, lead_s=lead_s)
    s._chips = mock_telemetry(4, 0)
    chunk = np.zeros((int(seconds * SR), 2), dtype=np.float32)
    return s, t, chunk


def _event(s, seg):
    s._on_event({"t": float(seg), "event": "live", "segment": seg, "chip": 0, "device": assign_device(0).__name__, "seconds": 4.0,
                 "what": f"segment {seg}", "why": "x"})


def test_with_a_lead_the_sink_hands_over_audio_without_waiting_until_it_is_that_far_ahead():
    from tensixinet.ui.session import _UiSink
    s, t, chunk = _led(10.0)
    sink = _UiSink(s)
    sink.write(chunk), sink.write(chunk)          # 4 s and 8 s ahead of the listener: inside the 10 s lead
    assert t.slept == []
    sink.write(chunk)                             # 12 s ahead: it waits until it is 10 s ahead, no longer
    assert sum(t.slept) == pytest.approx(2.0)
    assert s._audio.qsize() == 3


def test_a_wait_is_in_slices_so_a_stop_is_noticed_within_one():
    from tensixinet.ui.session import PACE_SLICE_S, _UiSink
    s, t, chunk = _led(2.0, seconds=4.0)
    _UiSink(s).write(chunk)                       # 4 s ahead of a 2 s lead: waits 2 s
    assert len(t.slept) >= 2 and max(t.slept) <= PACE_SLICE_S + 1e-9 and sum(t.slept) == pytest.approx(2.0)


def test_the_lead_counts_from_the_clock_so_time_spent_rendering_is_not_slept_twice():
    from tensixinet.ui.session import _UiSink
    s, t, chunk = _led(1.0)
    sink = _UiSink(s)
    sink.write(chunk)                             # 4 s handed over, 1 s of lead allowed: it waits 3 s
    assert sum(t.slept) == pytest.approx(3.0)
    t.now += 3.0                                  # the next segment took 3 s to render, while the listener heard the first go by
    t.slept.clear()
    sink.write(chunk)                             # 8 s handed over and 6 s played: 2 s ahead, so it waits 1 s to be 1 s ahead
    assert sum(t.slept) == pytest.approx(1.0)


def test_with_a_lead_the_chart_stays_on_the_segment_the_listener_is_hearing():
    from tensixinet.ui.session import _UiSink
    s, t, chunk = _led(20.0)
    sink = _UiSink(s)
    for seg in range(4):
        _event(s, seg)
        sink.write(chunk)                         # four 4 s segments, all handed over at once
    assert s.snapshot().segment == 0 and s.snapshot().state.segment == 0     # the first has just begun to play; the rest wait
    t.now += 3.9
    assert s.snapshot().segment == 0
    t.now += 0.2                                  # 4.1 s in: the second segment has begun
    assert (s.snapshot().segment, s.snapshot().state.segment) == (1, 1)
    t.now += 8.0                                  # 12.1 s in: the fourth
    assert (s.snapshot().segment, s.snapshot().state.segment) == (3, 3)


def test_a_session_without_a_lead_still_begins_each_segment_as_it_is_written():
    from tensixinet.ui.session import _UiSink
    s, t, chunk = _led(0.0)
    _event(s, 0)
    _UiSink(s).write(chunk)
    assert s.snapshot().segment == 0


# --- progress: the steps before the first segment plays ---------------------------------------------------------------------------------------

def _states(s):
    p = s.progress()
    return None if p is None else [(st.label, st.state) for st in p.steps]


SIM, RENDER, BUFFER = "start the TT-Lang simulator", "render the first segment", "fill the buffer"


def test_without_a_worker_the_steps_are_the_render_and_the_buffer_and_the_first_is_active_from_play():
    s, t, chunk = _led(10.0)
    assert s.progress() is None   # nothing has started
    s._t_play = t.clock()
    s._enter(0)
    assert _states(s) == [(RENDER, "active"), (BUFFER, "todo")]


def test_the_buffer_step_fills_with_the_audio_handed_over_and_the_progress_ends_when_the_lead_is_reached():
    from tensixinet.ui.session import _UiSink
    s, t, chunk = _led(10.0, seconds=4.0)
    s._t_play = t.clock()
    s._enter(0)
    sink = _UiSink(s)
    sink.write(chunk)                                         # 4 s of the 10 s lead
    assert _states(s) == [(RENDER, "done"), (BUFFER, "active")]
    assert s.progress().fraction == pytest.approx(0.4)
    sink.write(chunk)
    assert s.progress().fraction == pytest.approx(0.8)
    sink.write(chunk)                                         # 12 s: the lead is reached
    assert s.progress() is None


def test_with_no_lead_the_progress_ends_at_the_first_segment():
    from tensixinet.ui.session import _UiSink
    s, t, chunk = _led(0.0)
    s._t_play = t.clock()
    s._enter(0)
    _UiSink(s).write(chunk)
    assert s.progress() is None


def test_each_step_reports_how_long_it_took_and_the_total_since_play():
    from tensixinet.ui.session import _UiSink
    s, t, chunk = _led(10.0, seconds=4.0)
    s._t_play = t.clock()
    s._enter(0)
    t.now += 6.5
    s.progress()
    _UiSink(s).write(chunk)
    t.now += 1.0
    p = s.progress()
    assert p.steps[0].seconds == pytest.approx(6.5) and p.steps[1].seconds == pytest.approx(1.0) and p.elapsed == pytest.approx(7.5)


class _SlowWorker:
    """A chip worker whose start takes as long as the test says."""

    def __init__(self):
        self.release = threading.Event()
        self.closed = False

    def start(self):
        self.release.wait(30)

    def close(self):
        self.closed = True


def test_a_simulator_session_shows_the_simulator_step_while_the_worker_starts_and_logs_it():
    w = _SlowWorker()
    s = _session(worker=w, lead_s=10.0)
    s.start()
    try:
        assert _wait(lambda: _states(s) is not None)
        assert _states(s) == [(SIM, "active"), (RENDER, "todo"), (BUFFER, "todo")]
        assert s.progress().fraction is None                  # no way to know how long the simulator takes
        log = " ".join(s.snapshot().log_lines)
        assert "starting the TT-Lang simulator" in log and "ready" not in log
        w.release.set()
        assert _wait(lambda: "ready" in " ".join(s.snapshot().log_lines))
        assert "TT-Lang simulator is ready" in " ".join(s.snapshot().log_lines)
        assert s.progress() is None or _states(s)[0] == (SIM, "done")
    finally:
        w.release.set()
        s.stop()


def test_the_snapshot_carries_the_progress_until_the_first_segment_plays():
    from tensixinet.ui.session import _UiSink
    s, t, chunk = _led(10.0)
    s._t_play = t.clock()
    s._enter(0)
    assert s.snapshot().progress is not None and s.snapshot().state is None
    sink = _UiSink(s)
    for _ in range(3):
        sink.write(chunk)
    assert s.snapshot().progress is None


def test_with_no_lead_there_is_no_buffer_step_to_show():
    s, t, chunk = _led(0.0)
    s._t_play = t.clock()
    s._enter(0)
    assert _states(s) == [(RENDER, "active")]
