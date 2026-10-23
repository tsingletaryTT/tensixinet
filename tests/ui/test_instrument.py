# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The Play tab's engine: a render loop that reads the knobs once a block."""
import dataclasses
import threading
import time

import numpy as np
import pytest

from tensixinet.music.chorus import device_names
from tensixinet.music.patch import DEFAULT_PATCH
from tensixinet.ui import instrument
from tensixinet.ui.instrument import (BLOCK_S, LONG_BLOCKS, SAMPLE_RATE, SHORT_BLOCKS, XFADE_S, InstrumentConfig, InstrumentSession,
                                      phrase_samples)
from tensixinet.ui.knobs import Knobs

SR = SAMPLE_RATE
BLOCK_SAMPLES = int(BLOCK_S * SR)  # what a block hands over
XFADE_SAMPLES = int(XFADE_S * SR)


class Clock:
    """A clock that only moves when something sleeps or the test moves it."""

    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t

    def sleep(self, seconds):
        self.t += seconds


class Render:
    """Stands in for render_chorus: records how it was called and returns a quiet tone of the length asked for."""

    def __init__(self):
        self.calls = []

    def __call__(self, chips, duration, sample_rate=44100, patch=None, segment=0, soloists=None, only=None, cables=None, stereo=False,
                 defrag=None, **extra):
        self.calls.append(dict(chips=list(chips), duration=duration, sample_rate=sample_rate, patch=patch, segment=segment, soloists=soloists,
                               only=only, cables=cables, stereo=stereo, defrag=defrag, extra=extra))
        n = int(duration * sample_rate)
        tone = (0.1 * np.sin(2 * np.pi * 220 * np.arange(n) / sample_rate)).astype(np.float32)
        audio = np.stack([tone, tone], axis=1) if stereo else tone
        self.calls[-1]["audio"] = audio
        return audio


def make(scope="system", chips=2, stereo=True, cables=True, evict=True, lead_s=1.5, render=None, clock=None, knobs=None, patch=DEFAULT_PATCH):
    clock = clock or Clock()
    render = render or Render()
    knobs = knobs or Knobs(chips)
    config = InstrumentConfig(scope=scope, chips=chips, style="", seed=1, stereo=stereo, cables=cables, evict=evict)
    return InstrumentSession(config, knobs, patch, render=render, clock=clock, sleep=clock.sleep, lead_s=lead_s), render, knobs, clock


def drain(session):
    out = []
    while (chunk := session.next_audio(timeout=0)) is not None:
        out.append(chunk)
    return out


def test_the_block_phrase_and_crossfade_sizes_are_the_ones_chosen():
    assert (BLOCK_S, XFADE_S, SAMPLE_RATE, SHORT_BLOCKS, LONG_BLOCKS) == (1.2, 0.2, 22050, 2, 5)
    assert phrase_samples(2) == 2 * 26460 + 4410 and phrase_samples(5) == 5 * 26460 + 4410


def test_the_first_phrase_is_short_and_a_phrase_that_runs_out_with_the_knobs_still_is_long():
    session, render, _, clock = make()
    for _ in range(SHORT_BLOCKS + 1):
        session.render_block()
        clock.sleep(1.0)
    assert [len(c["audio"]) for c in render.calls] == [phrase_samples(SHORT_BLOCKS), phrase_samples(LONG_BLOCKS)]


def test_a_phrase_is_played_out_in_blocks_without_another_render():
    session, render, _, clock = make()
    for _ in range(SHORT_BLOCKS):
        session.render_block()
        clock.sleep(1.0)
    assert len(render.calls) == 1


@pytest.mark.parametrize("scope,only", [("system", None), ("module:melody", ["lead", "ttquantus", "ttrax"]), ("device:gong", ["gong"])])
def test_each_phrase_renders_with_the_scope_the_config_names(scope, only):
    session, render, _, _ = make(scope=scope)
    session.render_block()
    call = render.calls[0]
    assert call["only"] == only
    assert (call["soloists"] is None) == (only is not None)  # the system picks a soloist; a scope plays its devices together
    if only is None:
        assert len(call["soloists"]) == 1


def test_the_soloist_comes_from_the_chips_weights_and_the_last_soloist_is_remembered(monkeypatch):
    picks = []

    def fake(weights, segment, previous):
        picks.append((len(weights), segment, previous))
        return 7

    monkeypatch.setattr(instrument, "pick_soloist", fake)
    session, render, knobs, _ = make(scope="system", chips=2)
    session.render_block()
    session.render_block()  # the same phrase plays on: the block number moves, the phrase number does not
    knobs.set(0, "power", 90.0)
    session.render_block()
    assert [c["soloists"] for c in render.calls] == [[7], [7]]
    assert picks == [(11, 0, None), (11, 1, 7)]  # eleven slots for two chips; the phrase number; who had it before


def test_the_patch_and_the_stereo_and_cable_choices_reach_the_render():
    patch = dataclasses.replace(DEFAULT_PATCH, seed=77)
    session, render, _, _ = make(stereo=False, cables=False, patch=patch)
    session.render_block()
    call = render.calls[0]
    assert call["patch"] is patch and call["stereo"] is False and call["cables"] == [] and call["sample_rate"] == SR
    session, render, _, _ = make(stereo=True, cables=True)
    session.render_block()
    assert render.calls[0]["stereo"] is True and render.calls[0]["cables"] is None


def test_a_new_phrase_has_a_new_segment_number_so_it_is_not_a_copy_of_the_last():
    session, render, knobs, clock = make()
    for v in (40.0, 41.0, 42.0):
        knobs.set(0, "temperature", v)
        session.render_block()
        clock.sleep(1.0)
    assert [c["segment"] for c in render.calls] == [0, 1, 2]


def test_a_knob_moved_between_blocks_starts_a_new_phrase_that_carries_it():
    session, render, knobs, clock = make()
    knobs.set(0, "power", 40.0)
    session.render_block()
    session.render_block()  # nothing moved: the same phrase plays on
    knobs.set(0, "power", 120.0)
    session.render_block()
    assert [c["chips"][0].power for c in render.calls] == [40.0, 120.0]


@pytest.mark.parametrize("name,value", [("temperature", 70.0), ("aiclk", 1200.0), ("voltage", 0.9), ("gddr_spread", 25.0), ("power", 99.0)])
def test_any_knob_starts_a_new_phrase(name, value):
    session, render, knobs, _ = make()
    session.render_block()
    knobs.set(1, name, value)
    session.render_block()
    assert len(render.calls) == 2


def test_a_move_starts_a_short_phrase_so_the_hand_is_heard_quickly():
    session, render, knobs, clock = make()
    for _ in range(SHORT_BLOCKS):
        session.render_block()
    session.render_block()  # the long one
    knobs.set(0, "power", 90.0)
    session.render_block()
    assert [len(c["audio"]) for c in render.calls] == [phrase_samples(SHORT_BLOCKS), phrase_samples(LONG_BLOCKS), phrase_samples(SHORT_BLOCKS)]


def test_ten_moves_inside_one_block_give_the_last_value():
    session, render, knobs, _ = make()
    for v in range(10):
        knobs.set(1, "temperature", 30.0 + v)
    session.render_block()
    assert render.calls[0]["chips"][1].asic_temperature == 39.0


def test_a_memory_error_starts_a_phrase_and_the_counter_going_back_to_zero_does_not():
    session, render, knobs, clock = make()
    session.render_block()
    knobs.trigger(0, "memory_error")
    clock.sleep(1.0)
    session.render_block()
    clock.sleep(1.0)
    session.render_block()  # the error is one block old; the phrase that carries its tick plays on
    assert len(render.calls) == 2 and render.calls[1]["chips"][0].corr_errs == 1


def test_a_thermal_trip_starts_a_phrase_too():
    session, render, knobs, clock = make()
    session.render_block()
    knobs.trigger(1, "thermal_trip")
    clock.sleep(1.0)
    session.render_block()
    assert len(render.calls) == 2 and render.calls[1]["chips"][1].thermal_trips == 1


def test_a_slider_moved_fast_gives_the_phrase_a_rate():
    session, render, knobs, clock = make()
    knobs.set(0, "temperature", 40.0)
    session.render_block()
    knobs.set(0, "temperature", 60.0)
    clock.sleep(2.0)
    session.render_block()
    assert render.calls[1]["chips"][0].temp_rate == pytest.approx(10.0)


def test_every_block_hands_over_exactly_one_block_of_audio():
    session, _, _, _ = make()
    for _ in range(8):
        session.render_block()
    assert [len(c) for c in drain(session)] == [BLOCK_SAMPLES] * 8


def test_the_audio_is_the_phrase_itself_block_after_block_with_nothing_lost_or_repeated():
    session, render, _, _ = make(stereo=False)
    session._equaliser, session._gain = _Identity(), _Unity()  # so what comes out is what the render made
    for _ in range(SHORT_BLOCKS):
        session.render_block()
    first, second = drain(session)
    phrase = render.calls[0]["audio"]
    assert np.array_equal(first, phrase[:BLOCK_SAMPLES])
    assert np.array_equal(second, phrase[BLOCK_SAMPLES: 2 * BLOCK_SAMPLES])


def test_the_blocks_of_a_long_phrase_are_its_consecutive_slices():
    session, render, _, _ = make(stereo=False)
    session._equaliser, session._gain = _Identity(), _Unity()
    for _ in range(SHORT_BLOCKS + LONG_BLOCKS):
        session.render_block()
    chunks = drain(session)[SHORT_BLOCKS:]
    phrase = render.calls[1]["audio"]
    assert len(render.calls) == 2 and len(chunks) == LONG_BLOCKS
    for k in range(1, LONG_BLOCKS):  # block 0 of a phrase starts with the crossfade
        assert np.array_equal(chunks[k], phrase[k * BLOCK_SAMPLES: (k + 1) * BLOCK_SAMPLES]), k


def test_a_phrase_that_replaces_another_is_crossfaded_in_with_equal_power_not_cut():
    class Levels(Render):
        """Each phrase is a constant: 0.2 for the first, 0.6 for the second, so which side fades in can be read."""

        def __call__(self, chips, duration, sample_rate=44100, **kw):
            super().__call__(chips, duration, sample_rate, **kw)
            return np.full(int(duration * sample_rate), 0.2 if len(self.calls) == 1 else 0.6, dtype=np.float32)

    session, _, knobs, _ = make(render=Levels(), stereo=False)
    session._equaliser = _Identity()
    session._gain = _Unity()
    session.render_block()
    knobs.set(0, "power", 90.0)
    session.render_block()
    first, second = drain(session)
    k = (np.arange(XFADE_SAMPLES) + 0.5) / XFADE_SAMPLES
    expected = 0.6 * np.sin(0.5 * np.pi * k) + 0.2 * np.cos(0.5 * np.pi * k)  # the new phrase fades in, the old one's continuation out
    assert np.allclose(first, 0.2, atol=1e-6)  # the first block of the first phrase has nothing to fade from
    assert np.allclose(second[:XFADE_SAMPLES], expected, atol=1e-5)
    assert np.allclose(second[XFADE_SAMPLES:], 0.6, atol=1e-6)


class _Identity:
    def process(self, chunk):
        return chunk


class _Unity:
    def apply(self, chunk):
        return chunk, 1.0


def test_stereo_blocks_have_two_channels_and_mono_blocks_one():
    session, *_ = make(stereo=True)
    session.render_block()
    assert drain(session)[0].ndim == 2
    session, *_ = make(stereo=False)
    session.render_block()
    assert drain(session)[0].ndim == 1


def test_the_audio_is_finite_and_within_full_scale_even_for_a_loud_render():
    class Loud(Render):
        def __call__(self, chips, duration, sample_rate=44100, **kw):
            return super().__call__(chips, duration, sample_rate, **kw) * 40.0

    session, *_ = make(render=Loud())
    for _ in range(3):
        session.render_block()
    for chunk in drain(session):
        assert np.isfinite(chunk).all() and np.abs(chunk).max() <= 1.0


def test_pacing_waits_until_the_session_is_no_more_than_lead_s_ahead():
    session, _, _, clock = make(lead_s=1.5)
    session.render_block()  # 1.2 s ahead: within the lead
    before = clock.t
    session._pace()
    assert clock.t == before
    session.render_block()  # 2.4 s ahead
    session._pace()
    slept = clock.t - before
    assert slept >= 0.9 - 1e-3 and 2 * BLOCK_SAMPLES / SR - slept <= 1.5 + 1e-9


def test_pacing_sleeps_in_slices_so_a_stop_is_noticed():
    slept = []
    session, _, _, clock = make(lead_s=0.0)
    session._sleep = lambda s: (slept.append(s), clock.sleep(s))
    session.render_block()
    session._pace()
    assert instrument.PACE_SLICE_S == 0.25
    assert slept and max(slept) <= 0.25 + 1e-9 and sum(slept) == pytest.approx(BLOCK_S, abs=1e-3)


def test_pacing_ends_at_once_when_the_session_is_stopped():
    session, _, _, clock = make(lead_s=0.0)
    session.render_block()
    session._stop.set()
    before = clock.t
    session._pace()
    assert clock.t == before


def test_the_snapshot_shows_the_block_the_listener_has_reached_not_the_one_just_rendered():
    session, _, knobs, clock = make(lead_s=3.0)
    assert session.snapshot().block == -1 and session.snapshot().telemetry == ()
    knobs.set(0, "power", 40.0)
    session.render_block()
    knobs.set(0, "power", 120.0)
    session.render_block()
    assert session.snapshot().block == 0 and session.snapshot().telemetry[0].power == 40.0
    clock.sleep(BLOCK_S + 0.01)
    assert session.snapshot().block == 1 and session.snapshot().telemetry[0].power == 120.0


def test_the_snapshot_names_the_devices_heard():
    session, *_ = make(scope="module:delay and echo")
    session.render_block()
    assert session.snapshot().playing == ("ttquantus", "ttcoco")
    session, *_ = make(scope="system")
    session.render_block()
    assert session.snapshot().playing == tuple(device_names())


def test_a_device_that_is_not_dealt_to_the_chips_is_said_so_and_the_stream_goes_on(monkeypatch):
    monkeypatch.setattr(instrument, "heard_on", lambda only, n: [])
    session, render, *_ = make(scope="device:ttchoir", chips=1)
    session.render_block()
    snap = session.snapshot()
    assert snap.playing == () and any("ttchoir" in n and "1 chip" in n for n in snap.notes)
    assert len(render.calls) == 1 and len(drain(session)) == 1


def test_eviction_is_followed_in_the_system_scope_only_and_only_when_asked():
    session, render, *_ = make(scope="system", evict=True)
    session.render_block()
    assert render.calls[0]["defrag"] is not None and len(render.calls[0]["defrag"]) == 2
    for scope, evict in (("system", False), ("device:gong", True), ("module:melody", True)):
        session, render, *_ = make(scope=scope, evict=evict)
        session.render_block()
        assert render.calls[0]["defrag"] is None, (scope, evict)


def test_an_eviction_starts_a_phrase_and_shows_in_the_notes_of_its_block():
    session, render, knobs, clock = make(scope="system", chips=1, lead_s=0.0)
    knobs.set(0, "power", 14.0)
    for _ in range(3):
        session.render_block()
        clock.sleep(5.0)
    knobs.set(0, "power", 100.0)
    for _ in range(6):
        session.render_block()
        clock.sleep(5.0)
    knobs.set(0, "power", 14.0)
    seen = []
    for _ in range(25):
        session.render_block()
        clock.sleep(5.0)
        seen += [n for n in session.snapshot().notes if "evicted" in n]
        if seen:
            break
    assert seen, "a chip that ran a load and fell back to rest should evict"
    assert any(c["defrag"] and c["defrag"][0].phase == "evicting" for c in render.calls)


def test_a_render_error_stops_the_session_and_is_reported():
    class Broken(Render):
        def __call__(self, *a, **k):
            raise RuntimeError("no sound today")

    session, *_ = make(render=Broken())
    session._sleep = time.sleep
    session.start()
    session._thread.join(2.0)
    snap = session.snapshot()
    assert not snap.running and snap.error and "no sound today" in snap.error


def test_stop_during_a_render_ends_the_thread_within_one_block():
    entered, release = threading.Event(), threading.Event()

    class Slow(Render):
        def __call__(self, *a, **k):
            entered.set()
            release.wait(5.0)
            return super().__call__(*a, **k)

    session, *_ = make(render=Slow())
    session._sleep = time.sleep
    session.start()
    assert entered.wait(2.0)
    stopper = threading.Thread(target=session.stop)
    stopper.start()
    time.sleep(0.05)
    release.set()
    stopper.join(2.0)
    session._thread.join(2.0)
    assert not session._thread.is_alive() and not session.snapshot().running


def test_a_session_runs_once():
    session, *_ = make()
    session._sleep = time.sleep
    session.start()
    try:
        with pytest.raises(RuntimeError):
            session.start()
    finally:
        session.stop()


def test_the_running_thread_keeps_producing_blocks_until_stopped():
    session, render, *_ = make(lead_s=0.0)
    session._sleep = lambda s: time.sleep(0.001)  # fast: the stub renders instantly
    session._clock = lambda: time.monotonic() * 1000.0  # and time runs fast, so the pacing ends at once
    session.start()
    deadline = time.time() + 3.0
    while len(render.calls) < 3 and time.time() < deadline:
        time.sleep(0.01)
    session.stop()
    assert len(render.calls) >= 3 and not session.snapshot().running


def test_the_stop_flag_makes_next_audio_return_whatever_is_queued_then_none():
    session, *_ = make()
    session.render_block()
    assert session.next_audio(timeout=0) is not None and session.next_audio(timeout=0) is None


# --- the real chorus -----------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("device", device_names())
def test_every_device_makes_sound_on_four_chips(device):
    knobs = Knobs(4)
    config = InstrumentConfig(scope=f"device:{device}", chips=4, style="", seed=1, stereo=True, cables=True, evict=True)
    session = InstrumentSession(config, knobs, DEFAULT_PATCH, lead_s=0.0)
    for _ in range(SHORT_BLOCKS):  # the first phrase: a slow device (the gong's first strike) may start in its second block
        session.render_block()
    chunk = np.concatenate(drain(session))
    assert chunk.shape == (SHORT_BLOCKS * BLOCK_SAMPLES, 2) and np.isfinite(chunk).all()
    assert float(np.sqrt(np.mean(chunk ** 2))) > 1e-4, f"{device} was silent"


def test_the_real_system_phrases_render_faster_than_they_play():
    """The blocks that cost something are the ones that start a phrase: one after a move (short) and one after a still stretch (long)."""
    knobs = Knobs(4)
    config = InstrumentConfig(scope="system", chips=4, style="", seed=1, stereo=True, cables=True, evict=True)
    session = InstrumentSession(config, knobs, DEFAULT_PATCH, lead_s=0.0)
    session.render_block()  # the first phrase pays for imports and caches
    knobs.set(0, "power", 90.0)
    started = time.perf_counter()
    session.render_block()  # a move: a short phrase
    short = time.perf_counter() - started
    for _ in range(SHORT_BLOCKS - 1):
        session.render_block()
    started = time.perf_counter()
    session.render_block()  # nothing moved and the phrase ran out: a long one
    long = time.perf_counter() - started
    print(f"four chips, system, stereo: a short phrase took {short:.2f} s for {SHORT_BLOCKS * BLOCK_S:.1f} s of sound, "
          f"a long one {long:.2f} s for {LONG_BLOCKS * BLOCK_S:.1f} s")
    assert short < SHORT_BLOCKS * BLOCK_S and long < LONG_BLOCKS * BLOCK_S


def test_the_snapshot_carries_the_flow_of_the_block_the_listener_has_reached():
    session, _, knobs, clock = make(lead_s=3.0, chips=3)
    assert session.snapshot().state is None
    knobs.set(0, "power", 40.0)
    session.render_block()
    knobs.set(0, "power", 120.0)
    session.render_block()
    state = session.snapshot().state
    assert [c.power_w for c in state.chips] == [40.0, pytest.approx(session.snapshot().telemetry[1].power), pytest.approx(session.snapshot().telemetry[2].power)]
    clock.sleep(BLOCK_S + 0.01)
    assert session.snapshot().state.chips[0].power_w == 120.0 and len(session.snapshot().state.chips) == 3


def test_the_flow_names_the_soloist_the_engine_chose_and_the_cables_the_patch_draws():
    from tensixinet.music.cables import draw_cables
    session, render, *_ = make(scope="system", chips=4)
    session.render_block()
    state = session.snapshot().state
    assert [d.slot for d in state.devices if d.soloist] == render.calls[0]["soloists"]
    assert {(c.source, c.target, c.kind) for c in state.cables} == {(w.source, w.target, w.kind) for w in draw_cables(DEFAULT_PATCH.seed, device_names())}
    session, *_ = make(scope="system", chips=4, cables=False)
    session.render_block()
    assert session.snapshot().state.cables == ()


@pytest.mark.parametrize("scope,heard", [("device:gong", ("gong",)), ("module:delay and echo", ("ttquantus", "ttcoco"))])
def test_a_scope_that_plays_some_devices_shows_only_those_as_heard_and_nobody_has_the_floor(scope, heard):
    session, *_ = make(scope=scope, chips=4)
    session.render_block()
    state = session.snapshot().state
    assert {d.name for d in state.devices if d.heard} == set(heard)
    assert not any(d.soloist for d in state.devices) and state.soloist_device == ""


def test_the_flow_shows_a_chips_eviction_phase():
    session, render, knobs, clock = make(scope="system", chips=1, lead_s=0.0)
    knobs.set(0, "power", 14.0)
    for _ in range(3):
        session.render_block()
        clock.sleep(5.0)
    knobs.set(0, "power", 100.0)
    for _ in range(6):
        session.render_block()
        clock.sleep(5.0)
    knobs.set(0, "power", 14.0)
    phases = set()
    for _ in range(25):
        session.render_block()
        clock.sleep(5.0)
        phases.add(session.snapshot().state.chips[0].phase)
    assert "evicting" in phases


def test_the_flow_counts_the_phrase_and_the_time_the_block_starts_at():
    session, _, _, clock = make(lead_s=9.0)
    for _ in range(SHORT_BLOCKS + 1):
        session.render_block()
    seen = []
    for _ in range(SHORT_BLOCKS + 1):
        seen.append((session.snapshot().state.segment, session.snapshot().state.seconds))
        clock.sleep(BLOCK_S + 0.01)
    # the first phrase is segment 0 and has SHORT_BLOCKS blocks, the next block is the second phrase's first
    assert seen == [(0, pytest.approx(0.0)), (0, pytest.approx(BLOCK_S)), (1, pytest.approx(2 * BLOCK_S))]


def test_a_chips_eviction_phase_holds_for_every_block_of_the_phrase_it_began(monkeypatch):
    from types import SimpleNamespace

    class Stub:
        def update(self, chips, now):
            return [SimpleNamespace(phase="evicting") for _ in chips]

        def take_events(self):
            return []

    monkeypatch.setattr(instrument, "EvictTracker", Stub)
    session, _, _, clock = make(chips=2, lead_s=3.0)
    for _ in range(3):
        session.render_block()
    phases = []
    for _ in range(3):
        phases.append([c.phase for c in session.snapshot().state.chips])
        clock.sleep(BLOCK_S + 0.01)
    assert phases == [["evicting", "evicting"]] * 3
