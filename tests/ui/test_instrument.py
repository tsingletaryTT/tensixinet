# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The Play tab's engine: a render loop that reads the knobs once a block."""
import dataclasses
import threading
import time

import numpy as np
import pytest

from tensixinet.music.chorus import device_names
from tensixinet.music.evict import ChipDefrag
from tensixinet.music.patch import DEFAULT_PATCH
from tensixinet.ui import instrument
from tensixinet.ui.instrument import (BLOCK_S, LONG_BLOCKS, SAMPLE_RATE, SHORT_BLOCKS, XFADE_S, InstrumentConfig, InstrumentSession,
                                      phrase_samples, signature_changed)
from tensixinet.music.live import Tracker
from tensixinet.ui.knobs import Knobs, to_telemetry

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


def make(scope="system", chips=2, stereo=True, cables=True, evict=True, lead_s=1.5, render=None, clock=None, knobs=None, patch=DEFAULT_PATCH,
         **options):
    clock = clock or Clock()
    render = render or Render()
    knobs = knobs or Knobs(chips)
    config = InstrumentConfig(scope=scope, chips=chips, style="", seed=1, stereo=stereo, cables=cables, evict=evict)
    return InstrumentSession(config, knobs, patch, render=render, clock=clock, sleep=clock.sleep, lead_s=lead_s, **options), render, knobs, clock


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


# -- the input and the timing as parameters (the fast-sampling work drives this engine from real chips at about 0.1 s a block) --------------------


def _readings(n=2):
    """Readings a caller has already tracked: a power no knob would give, so they can be told from the knobs' own."""
    return [dataclasses.replace(t, power=77.0 + k) for k, t in enumerate(to_telemetry(Knobs(n), Tracker(), 0.0))]


def test_a_session_with_a_read_function_renders_from_what_it_returns():
    asked = []
    given = _readings()

    def read(now):
        asked.append(now)
        return given

    session, render, _, clock = make(knobs=None, read=read)
    session.knobs = None  # a caller with a read function has no knobs
    session.render_block()
    assert asked == [clock.t]
    assert [c.power for c in render.calls[0]["chips"]] == [77.0, 78.0]
    assert render.calls[0]["chips"][0] is given[0]


def test_a_read_function_works_when_the_session_is_given_no_knobs():
    config = InstrumentConfig(scope="system", chips=2, style="", seed=1, stereo=False, cables=False, evict=True)
    clock, render = Clock(), Render()
    session = InstrumentSession(config, None, DEFAULT_PATCH, render=render, clock=clock, sleep=clock.sleep, read=lambda now: _readings())
    session.render_block()
    assert len(render.calls) == 1 and len(drain(session)) == 1


def test_without_a_read_function_the_knobs_are_read_as_before():
    session, render, knobs, _ = make()
    knobs.set(0, "power", 55.0)
    session.render_block()
    assert render.calls[0]["chips"][0].power == pytest.approx(55.0)


def test_block_and_phrase_lengths_follow_the_parameters():
    session, render, _, clock = make(block_s=0.1, short_s=1.0, long_s=3.0, xfade_s=0.05)
    session.render_block()
    for _ in range(10):  # the short phrase's ten blocks; the eleventh asks for a long one
        clock.sleep(0.1)
        session.render_block()
    assert [c["duration"] for c in render.calls] == [pytest.approx(1.05, abs=1e-3), pytest.approx(3.05, abs=1e-3)]
    assert [len(c["audio"]) for c in render.calls] == [10 * 2205 + 1102, 30 * 2205 + 1102]
    assert [len(c) for c in drain(session)] == [2205] * 11


def test_a_phrase_is_at_least_one_block_long():
    session, render, _, clock = make(block_s=1.0, short_s=0.1, long_s=0.2, xfade_s=0.1)
    session.render_block()
    session.render_block()
    assert [len(c["audio"]) for c in render.calls] == [22050 + 2205, 22050 + 2205]  # both round to a block; the second is the long one


def test_the_crossfade_is_never_longer_than_a_block():
    session, render, knobs, _ = make(block_s=0.1, short_s=0.2, long_s=0.4, xfade_s=0.5, stereo=False)
    session.render_block()
    knobs.set(0, "power", 90.0)
    session.render_block()  # a new phrase crossfades in: it must not run past the block
    assert [len(c["audio"]) for c in render.calls] == [2 * 2205 + 2205] * 2
    assert [len(c) for c in drain(session)] == [2205, 2205]


def test_the_sample_rate_reaches_the_render_and_the_audio():
    session, render, _, _ = make(sample_rate=8000, stereo=False)
    session.render_block()
    assert session.sample_rate == 8000
    assert render.calls[0]["sample_rate"] == 8000
    assert len(render.calls[0]["audio"]) == 2 * int(BLOCK_S * 8000) + int(XFADE_S * 8000)
    assert [len(c) for c in drain(session)] == [int(BLOCK_S * 8000)]


def test_the_pacing_counts_seconds_at_the_sessions_own_sample_rate():
    session, _, _, _ = make(sample_rate=8000, stereo=False)
    session.render_block()
    assert session._delivered == pytest.approx(BLOCK_S)


def test_the_charts_block_time_uses_the_sessions_block_length():
    session, _, _, clock = make(block_s=0.1, short_s=0.3, long_s=0.5, xfade_s=0.05)
    for _ in range(4):
        session.render_block()
        clock.sleep(0.1)
    clock.sleep(10.0)
    assert session.snapshot().state.seconds == pytest.approx(0.3)  # the fourth block, number 3


def test_only_replaces_the_scopes_devices_and_plays_them_with_nobody_on_the_floor():
    session, render, _, _ = make(scope="system", only=["gong", "lead"])
    session.render_block()
    assert render.calls[0]["only"] == ["gong", "lead"] and render.calls[0]["soloists"] is None
    assert session._evictor is None  # a session that plays some devices ignores eviction
    session2, render2, _, _ = make(scope="device:gong", only=["lead"])
    session2.render_block()
    assert render2.calls[0]["only"] == ["lead"]


def test_without_only_the_scope_decides():
    session, render, _, _ = make(scope="device:gong")
    session.render_block()
    assert render.calls[0]["only"] == ["gong"]


class _Constant(Render):
    def __init__(self, level):
        super().__init__()
        self.level = level

    def __call__(self, chips, duration, sample_rate=44100, **kw):
        super().__call__(chips, duration, sample_rate, **kw)
        return np.full(int(duration * sample_rate), self.level, dtype=np.float32)


def test_eq_false_skips_the_equaliser():
    session, _, _, _ = make(render=_Constant(0.2), stereo=False, eq=False, normalise=False)
    session.render_block()
    assert np.allclose(drain(session)[0], 0.2, atol=1e-6)  # nothing filtered, nothing scaled
    session, _, _, _ = make(render=_Constant(0.2), stereo=False, normalise=False)
    session.render_block()
    assert not np.allclose(drain(session)[0], 0.2, atol=1e-3)  # the default equaliser does change a constant's start


def test_normalise_false_skips_the_level_but_the_output_is_still_clipped():
    session, _, _, _ = make(render=_Constant(0.01), stereo=False, eq=False, normalise=False)
    session.render_block()
    assert np.allclose(drain(session)[0], 0.01, atol=1e-6)  # no gain lifted a quiet render
    session, _, _, _ = make(render=_Constant(0.01), stereo=False, eq=False)
    session.render_block()
    assert np.abs(drain(session)[0]).max() > 0.05  # the default level does lift it
    session, _, _, _ = make(render=_Constant(5.0), stereo=False, eq=False, normalise=False)
    session.render_block()
    assert np.allclose(drain(session)[0], 1.0)  # clipped to full scale
    session, _, _, _ = make(render=_Constant(-5.0), stereo=False, eq=False, normalise=False)
    session.render_block()
    assert np.allclose(drain(session)[0], -1.0)


def test_the_equaliser_is_built_for_the_sessions_sample_rate():
    from tensixinet.music.eq import Equaliser
    session, render, _, _ = make(sample_rate=8000, stereo=False, normalise=False)
    session.render_block()
    block = render.calls[0]["audio"][:int(BLOCK_S * 8000)].astype(np.float64)
    assert np.allclose(drain(session)[0], Equaliser(8000).process(block).astype(np.float32), atol=1e-6)
    assert not np.allclose(Equaliser(8000).process(block), Equaliser(SR).process(block), atol=1e-4)  # the two rates really differ here


# -- change detection with a minimum gap, and a held soloist (fast sampling reads the chips about ten times a second) -------------------------------


_GAP_TIMING = dict(block_s=0.25, short_s=5.0, long_s=5.0, xfade_s=0.05)  # quarter seconds are exact in binary, so the gap's boundary is exact


def _sig(temp=40.0, aiclk=1000.0, power=50.0, voltage=0.8, gddr=(30.0,) * 8, phases=()):
    """A one-chip signature as `_signature` builds it."""
    return (((temp, aiclk, power, voltage, tuple(gddr)),), tuple(phases))


def test_signature_changed_ignores_moves_at_or_under_each_threshold():
    base = _sig(voltage=0.0)
    assert not signature_changed(base, _sig(voltage=0.0))
    assert not signature_changed(base, _sig(temp=40.5, voltage=0.0))  # exactly the temperature threshold
    assert not signature_changed(base, _sig(power=53.0, voltage=0.0))  # exactly the power threshold
    assert not signature_changed(base, _sig(voltage=0.01))  # exactly the voltage threshold
    assert not signature_changed(base, _sig(aiclk=1000.5, voltage=0.0))  # exactly the clock tolerance
    assert not signature_changed(base, _sig(gddr=(30.5,) + (30.0,) * 7, voltage=0.0))  # exactly the GDDR tolerance


def test_signature_changed_sees_each_move_over_its_threshold():
    base = _sig()
    assert signature_changed(base, _sig(temp=40.51))
    assert signature_changed(base, _sig(temp=39.49))  # a fall counts as much as a rise
    assert signature_changed(base, _sig(power=53.01))
    assert signature_changed(base, _sig(power=46.99))
    assert signature_changed(base, _sig(voltage=0.8 + 0.0101))
    assert signature_changed(base, _sig(aiclk=1001.0))
    assert signature_changed(base, _sig(gddr=(30.0,) * 7 + (31.0,)))
    assert signature_changed(base, _sig(phases=("evicting",))) and signature_changed(_sig(phases=("running",)), _sig(phases=("evicting",)))
    assert signature_changed(base, (base[0] * 2, ()))  # a chip more
    assert signature_changed(None, base)  # nothing played yet


def test_signature_changed_takes_its_temperature_and_power_thresholds_as_arguments():
    assert signature_changed(_sig(), _sig(temp=40.3), temp_c=0.2)
    assert not signature_changed(_sig(), _sig(power=55.0), power_w=6.0)


def _fast(changed=signature_changed, evict=False, **options):
    """A session driven by a read function whose readings the test sets, with the gap timing."""
    values = {"power": 50.0, "asic_temperature": 40.0, "voltage": 0.8, "thermal_trips": 0}
    base = to_telemetry(Knobs(2), Tracker(), 0.0)
    timing = dict(_GAP_TIMING)
    timing.update(options)
    session, render, _, clock = make(knobs=None, evict=evict, read=lambda now: [dataclasses.replace(t, **values) for t in base], changed=changed,
                                     **timing)
    session.knobs = None
    return session, render, values, clock


def _blocks(session, clock, n, step=0.25):
    for _ in range(n):
        session.render_block()
        clock.sleep(step)


def test_noise_under_the_thresholds_does_not_start_a_phrase():
    session, render, values, clock = _fast(block_s=0.1, short_s=0.3, long_s=1.2)
    for k in range(10):
        wobble = 1 if k % 2 else -1
        values.update(asic_temperature=40.0 + 0.2 * wobble, power=50.0 + 1.4 * wobble, voltage=0.8 + 0.004 * wobble)  # each swing under its threshold
        session.render_block()
        clock.sleep(0.1)
    assert [len(c["audio"]) // 2205 for c in render.calls] == [3, 12]  # the first short phrase, then the long one when it ran out; no restarts
    session2, render2, values2, clock2 = _fast(changed=None, block_s=0.1, short_s=0.3, long_s=1.2)
    for k in range(4):
        values2["power"] = 50.0 + (1 if k % 2 else -1)
        session2.render_block()
        clock2.sleep(0.1)
    assert len(render2.calls) == 4  # without `changed` any difference starts a phrase, as the Play tab's knobs need


def test_a_step_in_power_over_the_threshold_starts_a_short_phrase_in_the_next_block():
    session, render, values, clock = _fast(short_s=0.5, long_s=2.0)
    _blocks(session, clock, 3)  # a short phrase of two blocks, then a long one of eight
    values["power"] = 53.5
    session.render_block()
    assert len(render.calls) == 3
    assert [c.power for c in render.calls[2]["chips"]] == [53.5, 53.5]
    assert len(render.calls[2]["audio"]) == phrase_samples(2, int(0.25 * SR), int(0.05 * SR))  # short: a change is heard quickly


def test_two_changes_inside_the_gap_make_one_phrase():
    session, render, values, clock = _fast(min_gap_s=1.0)
    session.render_block()  # t = 100.0: the first phrase
    clock.sleep(0.25)
    values["power"] = 60.0
    _blocks(session, clock, 2)  # t = 100.25 and 100.5: held back by the gap
    values["power"] = 70.0
    session.render_block()  # t = 100.75: still held
    assert len(render.calls) == 1
    clock.sleep(0.25)
    session.render_block()  # t = 101.0: the gap has passed and nothing moved this block, but the playing phrase was made from 50 W
    assert len(render.calls) == 2
    assert [c.power for c in render.calls[1]["chips"]] == [70.0, 70.0]  # one phrase, carrying the latest reading
    assert len(render.calls[1]["audio"]) == phrase_samples(20, int(0.25 * SR), int(0.05 * SR))
    clock.sleep(0.25)
    session.render_block()
    assert len(render.calls) == 2  # and the change is not heard twice
    values["power"] = 80.0
    _blocks(session, clock, 3)  # t = 101.25 to 101.75: the gap counts from the phrase that started at 101.0
    assert len(render.calls) == 2
    session.render_block()  # t = 102.0
    assert len(render.calls) == 3 and render.calls[2]["chips"][0].power == 80.0


def test_a_change_that_comes_and_goes_inside_the_gap_starts_nothing():
    session, render, values, clock = _fast(min_gap_s=1.0)
    session.render_block()
    clock.sleep(0.25)
    values["power"] = 60.0
    _blocks(session, clock, 2)
    values["power"] = 50.0
    _blocks(session, clock, 4)
    assert len(render.calls) == 1


def test_a_trip_inside_the_gap_still_starts_a_phrase():
    session, render, values, clock = _fast(min_gap_s=1.0)
    session.render_block()
    clock.sleep(0.25)
    values["thermal_trips"] = 1
    session.render_block()  # t = 100.25, well inside the gap
    assert len(render.calls) == 2 and render.calls[1]["chips"][0].thermal_trips == 1


class _Phases:
    """Stands in for the EvictTracker: hands back the phases the test sets."""

    def __init__(self, phases):
        self.phases = phases

    def update(self, chips, now):
        return [ChipDefrag(p) for p in self.phases]

    def take_events(self):
        return []


def test_an_eviction_phase_change_inside_the_gap_still_starts_a_phrase():
    session, render, values, clock = _fast(min_gap_s=1.0)
    session._evictor = _Phases(["running", "running"])
    session.render_block()
    clock.sleep(0.25)
    session._evictor.phases = ["evicting", "running"]
    session.render_block()
    assert len(render.calls) == 2 and render.calls[1]["defrag"][0].phase == "evicting"


def test_the_soloist_is_held_for_the_turn_length_and_survives_a_change(monkeypatch):
    picks = []

    def fake(weights, segment, previous):
        picks.append((segment, previous))
        return 3 + segment  # a different soloist every time it is asked

    monkeypatch.setattr(instrument, "pick_soloist", fake)
    asked = []
    session, render, values, clock = _fast(hold_soloist=lambda chips: (asked.append(len(chips)), 6.0)[1])
    for k in range(4):  # t = 100.0, 100.5, 101.0, 101.5: each block a change, so each starts a phrase
        values["power"] = 50.0 + 10.0 * k
        session.render_block()
        clock.sleep(0.5)
    assert [c["soloists"] for c in render.calls] == [[3]] * 4
    assert picks == [(0, None)] and asked == [2]
    clock.t = 105.75  # 5.75 s after the soloist was chosen
    values["power"] = 100.0
    session.render_block()
    assert render.calls[-1]["soloists"] == [3]
    clock.t = 106.0  # the turn is over: the next phrase picks again, with the phrase number as the turn
    values["power"] = 110.0
    session.render_block()
    assert render.calls[-1]["soloists"] == [3 + 5] and picks[-1] == (5, 3)


# -- the calculated delay: interval + (read to queued) + the player's buffer ----------------------------------------------------------

class _SlowRender(Render):
    """A render that takes `step` seconds of the fake clock, which the test can change between blocks."""

    def __init__(self, clock, step):
        super().__init__()
        self.clock, self.step = clock, step

    def __call__(self, *args, **kwargs):
        self.clock.sleep(self.step)
        return super().__call__(*args, **kwargs)


def test_the_delay_is_the_interval_plus_the_time_from_reading_to_queueing_plus_the_buffer():
    clock = Clock()
    session, _, _, _ = make(clock=clock, render=_SlowRender(clock, 0.07))
    session.render_block()
    assert session.delay_s(buffer_s=0.05, interval_s=0.1) == pytest.approx(0.22, abs=1e-9)


def test_the_delay_is_none_before_the_first_block():
    session, _, _, _ = make()
    assert session.delay_s(buffer_s=0.05, interval_s=0.1) is None


def test_the_delay_follows_the_latest_block():
    clock = Clock()
    render = _SlowRender(clock, 0.07)
    session, _, _, _ = make(clock=clock, render=render)
    session.render_block()  # starts a phrase: the render takes 0.07 s
    render.step = 0.5  # a second block played out of the same phrase never calls the render, so it is quick
    session.render_block()
    assert session.delay_s(buffer_s=0.05, interval_s=0.1) == pytest.approx(0.15, abs=1e-9)


# -- the tail a run hands over when it stops (fast mode, tensixinet/fast.py) ------------------------------------------------------

def test_the_tail_is_empty_before_the_first_block():
    session, _, _, _ = make(stereo=False)
    assert len(session.tail()) == 0


def test_the_tail_is_the_playing_phrases_continuation_for_one_crossfade_faded_out():
    session, render, _, _ = make(stereo=False)
    session._equaliser, session._gain = _Identity(), None  # no level gain, so the tail is the phrase's own samples
    session.render_block()
    phrase = render.calls[0]["audio"]
    k = (np.arange(XFADE_SAMPLES) + 0.5) / XFADE_SAMPLES
    expected = phrase[BLOCK_SAMPLES: BLOCK_SAMPLES + XFADE_SAMPLES] * np.cos(0.5 * np.pi * k)
    tail = session.tail()
    assert tail.dtype == np.float32 and np.allclose(tail, expected, atol=1e-6)


def test_the_tail_is_stereo_when_the_session_is_and_takes_the_gain_the_music_ended_on():
    session, _, _, _ = make(stereo=True)
    session._equaliser = _Identity()
    session.render_block()
    gain = session._gain.gain
    assert gain is not None and gain != 1.0
    session._gain = None  # the same tail without the gain
    plain = session.tail()
    session._gain = type("G", (), {"gain": gain})()
    assert plain.shape == (XFADE_SAMPLES, 2) and np.allclose(session.tail(), np.clip(plain * gain, -1, 1), atol=1e-6)


# -- heads and bodies: a change is heard from a short head while the full phrase renders in the background (fast mode) -------------
#
# The render below gives every call its own audio (a level for the call plus a slow ramp), so a test can tell which render a block was cut
# from and at which offset. A call can be held on an Event, which is how a test decides when a background render finishes. No test here
# waits on the wall clock except a bounded wait on an Event.

HB = int(0.1 * SR)  # a 0.1 s block
HX = int(0.05 * SR)  # its crossfade
HEAD_BLOCKS = 3  # head_s = 0.3 at a 0.1 s block
HEAD_END = HEAD_BLOCKS * HB


class Calls(Render):
    """A render whose every call is told apart: call i is 0.01 * (i + 1) plus a ramp of 1e-6 a sample. `hold[i]` (an Event) holds call i until it
    is set; `entered[i]` is set when call i begins. `fail` names calls that raise. Calls can come from two threads, so the count is locked."""

    def __init__(self, hold=None, fail=()):
        super().__init__()
        self.hold, self.fail = dict(hold or {}), set(fail)
        self.entered = {i: threading.Event() for i in range(64)}
        self._lock = threading.Lock()

    def __call__(self, chips, duration, sample_rate=44100, patch=None, segment=0, soloists=None, only=None, cables=None, stereo=False,
                 defrag=None, **extra):
        with self._lock:
            i = len(self.calls)
            self.calls.append(dict(chips=list(chips), duration=duration, segment=segment, soloists=soloists, thread=threading.current_thread()))
        self.entered[i].set()
        if i in self.hold:
            assert self.hold[i].wait(5.0), "the test never let this render finish"
        if i in self.fail:
            raise RuntimeError("the body render broke")
        n = int(duration * sample_rate)
        audio = (0.01 * (i + 1) + 1e-6 * np.arange(n)).astype(np.float32)
        self.calls[i]["audio"] = audio
        return audio


def _headed(render=None, **options):
    """A mono session with no EQ and no level gain, 0.1 s blocks, a 0.6 s short phrase, a 1.2 s long one and a 0.3 s head."""
    timing = dict(block_s=0.1, short_s=0.6, long_s=1.2, xfade_s=0.05, head_s=0.3, eq=False, normalise=False)
    timing.update(options)
    session, render, knobs, clock = make(stereo=False, render=render or Calls(), **timing)
    return session, render, knobs, clock


def _ramp_in():
    return (np.arange(HX) + 0.5) / HX


def _returns_promptly(fn):
    """Run `fn` on a thread and require it to return within a second (an Event wait, not a sleep): a loop that waited for a held render would not."""
    done = threading.Event()
    threading.Thread(target=lambda: (fn(), done.set()), daemon=True).start()
    assert done.wait(1.0), "the call waited for the render in the background"


def test_a_change_hands_over_a_head_at_once_and_the_body_takes_over_at_the_same_offset():
    session, render, _, _ = _headed()
    for _ in range(HEAD_BLOCKS):
        session.render_block()
    assert session.wait_render(2.0)
    session.render_block()  # the head's blocks are played: the body takes over
    blocks = drain(session)
    head, body = render.calls[0], render.calls[1]
    # the head is rendered past its crossfade by HEAD_MARGIN_S, because the last samples of a render differ from a longer render's
    assert len(head["audio"]) == HEAD_END + HX + 3307  # 0.15 s at 22050 Hz
    assert len(body["audio"]) == phrase_samples(6, HB, HX)  # the full short phrase
    assert (body["segment"], body["soloists"], body["chips"]) == (head["segment"], head["soloists"], head["chips"])  # the same phrase
    assert body["thread"] is not threading.current_thread()  # rendered in the background
    for k in range(HEAD_BLOCKS):
        assert np.allclose(blocks[k], head["audio"][k * HB:(k + 1) * HB], atol=1e-7)
    h, b, r = head["audio"], body["audio"], _ramp_in()
    # the body joins at the same time offset (its samples from head_s on), so the slow devices keep their timeline; the two are the same
    # music, so the join is a straight (equal-gain) crossfade
    assert np.allclose(blocks[3][:HX], b[HEAD_END:HEAD_END + HX] * r + h[HEAD_END:HEAD_END + HX] * (1 - r), atol=1e-7)
    assert np.allclose(blocks[3][HX:], b[HEAD_END + HX:HEAD_END + HB], atol=1e-7)
    session.render_block()
    assert np.allclose(drain(session)[0], b[HEAD_END + HB:HEAD_END + 2 * HB], atol=1e-7) and len(render.calls) == 2


def test_the_loop_does_not_wait_for_the_body_render():
    gate = threading.Event()
    session, render, _, _ = _headed(render=Calls(hold={1: gate}))
    try:
        for _ in range(HEAD_BLOCKS):
            session.render_block()  # each returns while the body is held
        assert render.entered[1].wait(2.0) and not session.wait_render(0.0)
        assert len(drain(session)) == HEAD_BLOCKS
    finally:
        gate.set()
    assert session.wait_render(2.0)
    session.render_block()
    (joined,) = drain(session)
    assert np.allclose(joined[HX:], render.calls[1]["audio"][HEAD_END + HX:HEAD_END + HB], atol=1e-7) and session.late_blocks == 0


def test_a_body_that_is_late_is_counted_as_a_gap_and_does_not_stall_the_loop():
    gate = threading.Event()
    session, render, _, _ = _headed(render=Calls(hold={1: gate}))
    try:
        for _ in range(HEAD_BLOCKS):
            session.render_block()
        for _ in range(2):  # two blocks with the body still rendering: each comes back without waiting for it
            _returns_promptly(session.render_block)
        blocks = drain(session)
        assert len(blocks) == HEAD_BLOCKS + 2 and session.late_blocks == 2
        h = render.calls[0]["audio"]
        k = (np.arange(HX) + 0.5) / HX
        # the first late block plays the head's own continuation faded out, then silence until the body is ready
        assert np.allclose(blocks[3][:HX], h[HEAD_END:HEAD_END + HX] * np.cos(0.5 * np.pi * k), atol=1e-7)
        assert not blocks[3][HX:].any() and not blocks[4].any()
    finally:
        gate.set()
    assert session.wait_render(2.0)
    session.render_block()  # the body comes in where the listener has got to, faded in from the silence
    (back,) = drain(session)
    b, at = render.calls[1]["audio"], HEAD_END + 2 * HB
    k = (np.arange(HX) + 0.5) / HX
    assert np.allclose(back[:HX], b[at:at + HX] * np.sin(0.5 * np.pi * k), atol=1e-7)
    assert np.allclose(back[HX:], b[at + HX:at + HB], atol=1e-7) and session.late_blocks == 2


def test_a_second_change_discards_the_body_still_rendering():
    gate = threading.Event()
    session, render, knobs, _ = _headed(render=Calls(hold={1: gate}))
    try:
        session.render_block()  # head A; body A is held
        assert render.entered[1].wait(2.0)
        knobs.set(0, "power", 90.0)
        session.render_block()  # head B, crossfaded in; body B waits behind body A
    finally:
        gate.set()
    assert session.wait_render(2.0)
    for _ in range(HEAD_BLOCKS):
        session.render_block()
    blocks = drain(session)
    body_b = render.calls[3]
    assert render.calls[2]["chips"][0].power == 90.0 and body_b["chips"][0].power == 90.0 and len(render.calls) == 4
    # head B's blocks, then body B at head B's offset: body A never reaches the sound
    assert np.allclose(blocks[-1][HX:], body_b["audio"][HEAD_END + HX:HEAD_END + HB], atol=1e-7)
    assert not any(np.allclose(block[HX:], render.calls[1]["audio"][HEAD_END + HX:HEAD_END + HB], atol=1e-7) for block in blocks)


def test_a_quiet_stretch_renders_the_next_long_phrase_ahead_and_a_change_discards_it():
    session, render, knobs, _ = _headed(render_ahead=True)
    for _ in range(HEAD_BLOCKS):
        session.render_block()
    assert session.wait_render(2.0)
    session.render_block()  # the body takes over at 0.3 s of a 0.6 s phrase: half played, so the next long phrase is started
    assert render.entered[2].wait(2.0) and len(render.calls) == 3 and render.calls[2]["duration"] == pytest.approx((phrase_samples(12, HB, HX) + 0.5) / SR)
    assert render.calls[2]["segment"] == 1 and render.calls[2]["thread"] is not threading.current_thread()
    assert session.wait_render(2.0)
    session.render_block()
    session.render_block()  # the short phrase's last block
    drain(session)
    session.render_block()  # it ran out: the long phrase rendered ahead takes over, with no render on the loop
    (taken,) = drain(session)
    assert len(render.calls) == 3
    body, ahead = render.calls[1]["audio"], render.calls[2]["audio"]
    k = (np.arange(HX) + 0.5) / HX
    assert np.allclose(taken[:HX], ahead[:HX] * np.sin(0.5 * np.pi * k) + body[6 * HB:6 * HB + HX] * np.cos(0.5 * np.pi * k), atol=1e-7)
    assert np.allclose(taken[HX:], ahead[HX:HB], atol=1e-7)
    for _ in range(4):  # the long phrase is twelve blocks: nothing more is rendered until half of it is played
        session.render_block()
    assert len(render.calls) == 3
    session.render_block()
    assert render.entered[3].wait(2.0) and render.calls[3]["segment"] == 2

    # a change while the next phrase is rendering ahead throws that render away
    gate = threading.Event()
    session, render, knobs, _ = _headed(render=Calls(hold={2: gate}), render_ahead=True)
    try:
        for _ in range(HEAD_BLOCKS):
            session.render_block()
        assert session.wait_render(2.0)
        session.render_block()  # the ahead render (call 2) starts and is held
        assert render.entered[2].wait(2.0)
        knobs.set(0, "power", 90.0)
        session.render_block()  # a change: a new head (call 3) and its body (call 4)
    finally:
        gate.set()
    assert session.wait_render(2.0)
    for _ in range(12):
        session.render_block()
        session.wait_render(2.0)
    discarded = render.calls[2]["audio"][HX:HB]
    assert not any(np.allclose(block[HX:], discarded, atol=1e-7) for block in drain(session))
    assert len(render.calls) > 5 and all(c["chips"][0].power == 90.0 for c in render.calls[3:])  # nothing later is made from the old readings


def test_stop_ends_the_render_worker():
    gate = threading.Event()
    session, render, _, _ = _headed(render=Calls(hold={1: gate}))
    session.render_block()
    assert render.entered[1].wait(2.0)
    worker = session._worker
    session.stop(timeout=0.05)  # the body is still rendering
    job = session._body
    gate.set()
    worker._thread.join(2.0)
    assert not worker._thread.is_alive()  # the worker thread ends once its render returns
    assert job.done.is_set() and job.result is None  # and the render that finished after the stop is dropped


def test_a_render_that_raises_in_the_worker_ends_the_session_with_the_error():
    session, render, _, _ = _headed(render=Calls(fail={1}), lead_s=0.0)
    session._clock, session._sleep = time.monotonic, time.sleep  # the thread paces itself on the real clock: a block every 0.1 s
    session.start()
    session._thread.join(3.0)
    snap = session.snapshot()
    assert not session._thread.is_alive() and not snap.running and snap.error and "the body render broke" in snap.error
    session.stop()


def test_without_head_and_render_ahead_nothing_changed():
    session, render, knobs, clock = make()  # the Play tab's defaults
    for k in range(8):
        if k == 4:
            knobs.set(0, "power", 90.0)
        session.render_block()
        clock.sleep(1.0)
    assert session._worker is None and session.late_blocks == 0  # no background thread at all
    assert all(c is render.calls[0] or len(c["audio"]) in (phrase_samples(SHORT_BLOCKS), phrase_samples(LONG_BLOCKS)) for c in render.calls)


def test_carry_from_takes_over_the_eviction_tracker_the_soloist_the_gain_and_the_eq():
    old, _, _, _ = make()
    old.render_block()
    new, _, _, _ = make()
    new.carry_from(old)
    assert new._evictor is old._evictor and new._gain is old._gain and new._equaliser is old._equaliser
    assert (new._soloist, new._soloist_until, new._phrases) == (old._soloist, old._soloist_until, old._phrases) and old._soloist is not None


def test_a_phrase_rendered_ahead_that_is_not_ready_when_it_is_due_becomes_the_body_of_a_head_rendered_then():
    gate = threading.Event()
    session, render, _, _ = _headed(render=Calls(hold={2: gate}), render_ahead=True)
    try:
        for _ in range(HEAD_BLOCKS):
            session.render_block()
        assert session.wait_render(2.0)
        session.render_block()  # the body takes over and the phrase ahead (call 2) starts, held
        assert render.entered[2].wait(2.0)
        session.render_block()
        session.render_block()
        session.render_block()  # the short phrase ran out with the phrase ahead still rendering: a head of it now, on the loop
        assert len(render.calls) == 4 and render.calls[3]["thread"] is threading.current_thread()
        assert (render.calls[3]["segment"], render.calls[3]["chips"]) == (render.calls[2]["segment"], render.calls[2]["chips"])
    finally:
        gate.set()
    assert session.wait_render(2.0)
    for _ in range(HEAD_BLOCKS):
        session.render_block()
    last = drain(session)[-1]
    # no second render of that phrase: the one rendered ahead is the head's body, joined at the head's offset
    assert len(render.calls) == 4
    assert np.allclose(last[HX:], render.calls[2]["audio"][HEAD_END + HX:HEAD_END + HB], atol=1e-7)



def test_a_body_still_missing_when_its_phrase_runs_out_is_dropped_so_the_worker_never_renders_it():
    """The worker renders one job at a time. A body forgotten without being dropped stays queued, takes the worker's time and, if it raises, its
    error is lost. Here body 1 is held, so body 2 waits behind it; both phrases run out first, and body 2 must never be rendered."""
    gate = threading.Event()

    class Held(Calls):
        def __call__(self, chips, duration, *a, segment=0, **k):
            if threading.current_thread() is not main:
                background.append(segment)
                if segment == 0:
                    started.set()
                    assert gate.wait(5.0)
                if segment == 1:
                    raise RuntimeError("a forgotten body was rendered")
            return super().__call__(chips, duration, *a, segment=segment, **k)

    main, background, started = threading.current_thread(), [], threading.Event()
    session, render, _, _ = _headed(render=Held())
    try:
        session.render_block()
        assert started.wait(2.0)  # body 0 is on the worker and held there
        for _ in range(5):  # phrase 0: three head blocks and three late ones
            session.render_block()
        for _ in range(12):  # phrase 1 (long), its body queued behind the held one: three head blocks and nine late ones
            session.render_block()
        session.render_block()  # phrase 2 starts: body 1 is forgotten
    finally:
        gate.set()
    assert session.wait_render(2.0)
    session.render_block()  # a background error, if one was kept, would surface here
    assert background[0] == 0 and 1 not in background and session.snapshot().error is None
