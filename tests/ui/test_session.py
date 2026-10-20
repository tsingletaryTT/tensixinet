# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
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
