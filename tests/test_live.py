# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Live mode: sample the chips every turn, render that turn, play it while the next one is made."""
import dataclasses
import json
import threading
import time

import numpy as np
import pytest

from tensixinet import live as live_module
from tensixinet.live import MAX_SEGMENT_S, MIN_SEGMENT_S, run_live
from tensixinet.media.player import PlayerError
from tensixinet.music.chorus import DEVICES, slot_chip, slot_weights
from tensixinet.music.live import XFADE_S
from tensixinet.music.loudness import TARGET_RMS
from tensixinet.music.patch import DEFAULT_PATCH
from tensixinet.narrate.runlog import RunLog
from tensixinet.telemetry import FALLBACK
from tensixinet.telemetry.mock import mock_telemetry

SR = 8000
ONE_BAR = dataclasses.replace(DEFAULT_PATCH, bars_per_turn=1)  # a turn of one bar is under the 2 s minimum at any tempo here, so segments are 2 s


class MemorySink:
    def __init__(self, fail_after=None):
        self.chunks, self.closed, self.threads, self.fail_after, self.attempts = [], 0, set(), fail_after, 0

    def write(self, audio):
        self.attempts += 1
        self.threads.add(threading.current_thread().name)
        if self.fail_after is not None and len(self.chunks) >= self.fail_after:
            raise PlayerError("the player stopped")
        self.chunks.append(np.array(audio))

    def close(self):
        self.closed += 1

    @property
    def audio(self):
        return np.concatenate(self.chunks) if self.chunks else np.zeros(0, dtype=np.float32)


class Clock:
    """A clock that moves a fixed step each time it is read, so the seconds between readings are known."""

    def __init__(self, step=2.0):
        self.now, self.step = 1000.0, step

    def __call__(self):
        self.now += self.step
        return self.now


def _log(tmp_path):
    return RunLog(jsonl=open(tmp_path / "run.jsonl", "w"), stream=open(tmp_path / "run.txt", "w"))


def _events(tmp_path):
    return [json.loads(l) for l in (tmp_path / "run.jsonl").read_text().splitlines()]


def _go(tmp_path, sample, duration=6.0, sink=None, **kw):
    sink = sink or MemorySink()
    log = _log(tmp_path)
    box = {}
    kw.setdefault("patch", ONE_BAR)
    kw.setdefault("eq", False)  # most tests are about the loop; the EQ has its own tests below

    def target():
        try:
            box["n"] = run_live(sample, sink, log, sample_rate=kw.pop("sample_rate", SR), duration=duration, clock=kw.pop("clock", Clock()), **kw)
        except BaseException as e:  # reported below, in the test's own thread
            box["error"] = e

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(20)
    log.close()
    assert not thread.is_alive(), "run_live did not finish (a deadlock between the loop and the sink's thread?)"
    if "error" in box:
        raise box["error"]
    return sink, box["n"]


def steady(chips=None):
    chips = chips or mock_telemetry(2)
    return lambda: list(chips)


# --- the run as a whole --------------------------------------------------------------------------------------------

def test_a_live_run_plays_the_duration_it_was_asked_for_and_lets_the_last_segment_ring_out(tmp_path):
    sink, n = _go(tmp_path, steady(), duration=7.0)
    assert n == 3 and abs(len(sink.audio) - (7.0 + XFADE_S) * SR) <= 2 * n + 2  # a sample of rounding at each join


def test_the_audio_is_finite_and_inside_full_scale(tmp_path):
    sink, _ = _go(tmp_path, steady())
    assert np.isfinite(sink.audio).all() and np.abs(sink.audio).max() <= 1.0 and sink.audio.dtype == np.float32


def test_the_sink_is_played_from_its_own_thread_and_closed_once(tmp_path):
    sink, _ = _go(tmp_path, steady())
    assert sink.closed == 1 and "MainThread" not in sink.threads and len(sink.threads) == 1


def test_a_segment_is_never_shorter_than_the_minimum_and_a_short_run_plays_one_segment(tmp_path):
    sink, n = _go(tmp_path, steady(), duration=0.5)
    assert n == 1 and len(sink.audio) == pytest.approx((MIN_SEGMENT_S + XFADE_S) * SR, abs=2)  # the segment and its tail


def test_the_last_segment_takes_up_what_is_left_instead_of_leaving_a_sliver(tmp_path):
    one_bar = dataclasses.replace(DEFAULT_PATCH, bars_per_turn=1)
    chips = [dataclasses.replace(FALLBACK, aiclk=1350.0)]  # 180 bpm: a one-bar turn is 1.3 s, so the minimum applies
    sink, n = _go(tmp_path, lambda: chips, duration=2.0 * MIN_SEGMENT_S + 0.5, patch=one_bar)
    lengths = [len(c) / SR for c in sink.chunks[:n]]  # the chunk after the segments is the faded tail
    assert n == 2 and lengths == pytest.approx([MIN_SEGMENT_S, MIN_SEGMENT_S + 0.5], abs=0.01)


def test_segments_are_as_long_as_a_turn_between_the_minimum_and_maximum(tmp_path):
    slow = [dataclasses.replace(FALLBACK, aiclk=400.0)]  # 60 bpm: a 2-bar turn is 8 s
    sink, n = _go(tmp_path, lambda: slow, duration=16.0, clock=Clock(8.0), patch=DEFAULT_PATCH)
    assert MIN_SEGMENT_S < 8.0 < MAX_SEGMENT_S
    assert n == 2 and [len(c) / SR for c in sink.chunks[:n]] == pytest.approx([8.0, 8.0], abs=0.01)


def test_a_segment_longer_than_the_maximum_is_cut_to_it(tmp_path):
    very_slow = dataclasses.replace(DEFAULT_PATCH, bars_per_turn=4)
    chips = [dataclasses.replace(FALLBACK, aiclk=400.0)]  # 60 bpm and 4 bars is 16 s
    sink, n = _go(tmp_path, lambda: chips, duration=2 * MAX_SEGMENT_S, patch=very_slow, clock=Clock(MAX_SEGMENT_S))
    assert n == 2 and len(sink.audio) == pytest.approx((2 * MAX_SEGMENT_S + XFADE_S) * SR, abs=4)
    assert len(sink.chunks[0]) == pytest.approx(MAX_SEGMENT_S * SR, abs=2)  # a 16 s turn is cut to the maximum


def test_the_first_reading_is_described_like_an_offline_run_and_each_segment_is_logged(tmp_path):
    _go(tmp_path, steady(mock_telemetry(2)), duration=6.0)
    ev = _events(tmp_path)
    kinds = [e["event"] for e in ev]
    assert kinds[0] == "discover" and "2 chips" in ev[0]["what"]
    assert {"patch", "chip", "events", "mix", "weight"} <= set(kinds)
    segments = [e for e in ev if e["event"] == "live"]
    assert [e["segment"] for e in segments] == list(range(len(segments))) and len(segments) >= 2
    assert all(e["what"] and e["why"] for e in ev) and kinds[-1] == "done"


# --- every segment follows the chips ----------------------------------------------------------------------------

@pytest.fixture
def renders(monkeypatch):
    """The arguments of every render_chorus call a run makes."""
    calls = []
    real = live_module.render_chorus

    def spy(telemetry, duration, sample_rate, patch, segment, soloists, only=None, cables=None):
        calls.append({"telemetry": list(telemetry), "duration": duration, "segment": segment, "soloists": list(soloists), "patch": patch,
                      "cables": cables})
        return real(telemetry, duration, sample_rate, patch=patch, segment=segment, soloists=soloists, only=only, cables=cables)

    monkeypatch.setattr(live_module, "render_chorus", spy)
    return calls


def test_every_segment_is_rendered_from_a_fresh_reading(tmp_path, renders):
    readings = []

    def sample():
        chips = [dataclasses.replace(FALLBACK, asic_temperature=40.0 + 3 * len(readings))]
        readings.append(chips[0].asic_temperature)
        return chips

    _, n = _go(tmp_path, sample, duration=6.0)
    assert len(readings) == n == len(renders)
    assert [c["telemetry"][0].asic_temperature for c in renders] == readings


def test_the_second_segment_knows_how_fast_the_chip_is_heating(tmp_path, renders):
    temps = iter([50.0, 51.0, 53.0, 53.0])
    _go(tmp_path, lambda: [dataclasses.replace(FALLBACK, asic_temperature=next(temps), power=10.0)], duration=6.0, clock=Clock(2.0))
    assert renders[0]["telemetry"][0].temp_rate == 0.0  # the first reading has nothing before it
    assert renders[1]["telemetry"][0].temp_rate == pytest.approx(0.5)  # one degree in the 2 s the clock moved


def test_a_chip_that_heats_quickly_makes_the_log_say_so(tmp_path):
    temps = iter([40.0, 44.0, 44.0, 44.0])
    _go(tmp_path, lambda: [dataclasses.replace(FALLBACK, asic_temperature=next(temps), power=10.0)], duration=6.0)
    motion = [e for e in _events(tmp_path) if e["event"] == "motion"]
    assert len(motion) == 1 and "+2.0 C/s" in motion[0]["what"] and motion[0]["chip"] == 0 and motion[0]["segment"] == 1


def test_the_log_says_whether_the_chip_plays_ttrax_so_its_chaos_is_what_moved(tmp_path):
    ttrax_slot = [d.__name__ for d in DEVICES].index("ttrax")
    chips = [dataclasses.replace(FALLBACK, asic_temperature=40.0, power=10.0)] * len(DEVICES)  # one device per chip

    def sample(temps=iter(range(100))):
        t = next(temps)
        return [dataclasses.replace(c, asic_temperature=40.0 + 4 * t) for c in chips]

    _go(tmp_path, sample, duration=6.0)
    by_chip = {e["chip"]: e for e in _events(tmp_path) if e["event"] == "motion" and e["segment"] == 1}
    assert "ttrax" in by_chip[ttrax_slot]["why"] and "chaos" in by_chip[ttrax_slot]["why"]
    assert "does not play ttrax" in by_chip[(ttrax_slot + 1) % len(DEVICES)]["why"]


def test_a_still_chip_logs_no_motion(tmp_path):
    _go(tmp_path, steady(), duration=6.0)
    assert not [e for e in _events(tmp_path) if e["event"] == "motion"]


def test_each_segment_gets_its_own_number_and_a_soloist_who_is_not_the_last_one(tmp_path, renders):
    _go(tmp_path, steady(mock_telemetry(3)), duration=10.0, clock=Clock(2.0))
    assert [c["segment"] for c in renders] == list(range(len(renders))) and len(renders) >= 3
    solos = [c["soloists"][0] for c in renders]
    assert all(len(c["soloists"]) == 1 for c in renders) and all(a != b for a, b in zip(solos, solos[1:]))


def test_the_soloist_is_drawn_from_the_chips_current_weights(tmp_path, renders):
    busy = [dataclasses.replace(FALLBACK, power=180.0)] + [dataclasses.replace(FALLBACK, power=1.0)] * 7
    _go(tmp_path, lambda: list(busy), duration=30.0, clock=Clock(2.0))
    weights = slot_weights(busy)
    picks = [c["soloists"][0] for c in renders]
    assert weights[0] > 10 * weights[1]
    assert [slot_chip(p, 8) for p in picks[::2]] == [0] * len(picks[::2])  # the busy chip takes every other turn (it owns two of the nine slots)
    assert all(a != b for a, b in zip(picks, picks[1:]))  # nobody solos twice running, however heavy their weight


def test_a_segment_renders_its_own_length_plus_the_crossfade(tmp_path, renders):
    _go(tmp_path, steady(), duration=7.0)
    assert all(c["duration"] >= MIN_SEGMENT_S + XFADE_S - 1e-9 for c in renders)


def test_only_new_errors_play_after_the_first_reading(tmp_path, renders):
    totals = iter([5, 5, 8, 8])
    _go(tmp_path, lambda: [dataclasses.replace(FALLBACK, corr_errs=next(totals))], duration=8.0)
    assert [c["telemetry"][0].corr_errs for c in renders] == [5, 0, 3, 0][: len(renders)]


def test_new_errors_are_logged_when_they_appear(tmp_path):
    totals = iter([0, 0, 2, 2])
    _go(tmp_path, lambda: [dataclasses.replace(FALLBACK, thermal_trips=next(totals))], duration=8.0)
    news = [e for e in _events(tmp_path) if e["event"] == "events" and e.get("segment") is not None]
    assert len(news) == 1 and "2 thermal trips" in news[0]["what"] and news[0]["segment"] == 2


# --- when the readings go wrong ---------------------------------------------------------------------------------

def test_a_failed_reading_repeats_the_last_one_and_says_so(tmp_path, renders):
    answers = iter([mock_telemetry(2), [], mock_telemetry(2), mock_telemetry(2)])
    _go(tmp_path, lambda: next(answers), duration=8.0)
    lines = [e for e in _events(tmp_path) if e["event"] == "live" and "no chips" in e["what"]]
    assert len(lines) == 1 and lines[0]["segment"] == 1 and "last" in lines[0]["why"]
    assert renders[1]["telemetry"][0].temp_rate == 0.0 and renders[1]["telemetry"][0].asic_temperature == renders[0]["telemetry"][0].asic_temperature


def test_a_run_that_cannot_read_the_chips_at_all_says_so_and_plays_nothing(tmp_path):
    sink = MemorySink()
    with pytest.raises(ValueError):
        _go(tmp_path, lambda: [], sink=sink)
    assert "no chips" in (tmp_path / "run.jsonl").read_text() and not sink.chunks and sink.closed == 1


def test_a_change_in_the_number_of_chips_is_logged_and_the_music_follows(tmp_path, renders):
    answers = iter([mock_telemetry(4), mock_telemetry(4), mock_telemetry(3), mock_telemetry(3)])
    _go(tmp_path, lambda: next(answers), duration=8.0)
    changed = [e for e in _events(tmp_path) if "4 chips to 3" in e["what"]]
    assert len(changed) == 1 and [len(c["telemetry"]) for c in renders][:3] == [4, 4, 3]


# --- stopping ---------------------------------------------------------------------------------------------------

def test_ctrl_c_ends_an_open_ended_run_and_plays_the_tail_of_the_last_segment(tmp_path):
    calls = []

    def sample():
        calls.append(1)
        if len(calls) == 3:
            raise KeyboardInterrupt
        return mock_telemetry(2)

    sink, n = _go(tmp_path, sample, duration=None)
    assert n == 2 and sink.closed == 1
    assert len(sink.chunks) == n + 1 and len(sink.chunks[-1]) == int(XFADE_S * SR)  # both segments, then the tail of the second
    assert any("stopped" in e["what"] for e in _events(tmp_path))


def test_an_error_while_reading_still_closes_the_sink(tmp_path):
    def sample():
        raise RuntimeError("tt-smi fell over")
    sink = MemorySink()
    with pytest.raises(RuntimeError):
        _go(tmp_path, sample, sink=sink)
    assert sink.closed == 1


def test_a_player_that_stops_ends_the_run_with_a_player_error_and_a_log_line(tmp_path):
    sink, reads = MemorySink(fail_after=1), []
    with pytest.raises(PlayerError):
        _go(tmp_path, lambda: reads.append(1) or mock_telemetry(2), duration=40.0, sink=sink)
    assert sink.closed == 1 and any(e["event"] == "player" for e in _events(tmp_path))
    assert len(reads) <= 4  # twenty segments were asked for, and the run stopped soon after the player did
    assert sink.attempts == 2  # the good write and the one that failed: nothing more is sent to a dead player


def test_the_tail_of_the_last_segment_has_the_same_gain_as_the_audio_before_it(tmp_path, monkeypatch):
    monkeypatch.setattr(live_module, "render_chorus", lambda chips, seconds, sr, patch, segment, soloists, only=None, cables=None: np.full(int(seconds * sr), 0.05, dtype=np.float32))
    sink, n = _go(tmp_path, steady(), duration=2.0)
    last_segment, tail = sink.chunks[0], sink.chunks[-1]
    assert last_segment[-1] > 0.1  # the 0.05 render was brought up to the target level
    assert tail[0] == pytest.approx(last_segment[-1], rel=0.01)


# --- the loop stays about one segment ahead of the sound ---------------------------------------------------------

def test_the_loop_reads_the_chips_no_more_than_two_segments_ahead_of_what_is_playing(tmp_path):
    gate, calls = threading.Event(), []

    class Gated(MemorySink):
        def write(self, audio):
            gate.wait(10)
            super().write(audio)

    def sample():
        calls.append(time.monotonic())
        return mock_telemetry(2)

    sink = Gated()
    log = _log(tmp_path)
    thread = threading.Thread(target=lambda: run_live(sample, sink, log, sample_rate=SR, duration=30.0, clock=Clock()), daemon=True)
    thread.start()
    time.sleep(1.0)  # long enough to render a dozen segments if nothing held the loop back
    assert len(calls) == 2  # one is playing (held), one waits behind it; the third reading waits for the first to finish
    gate.set()
    thread.join(30)
    assert not thread.is_alive() and len(calls) > 2
    log.close()


# --- loudness ---------------------------------------------------------------------------------------------------

def test_live_audio_is_brought_to_the_target_level_and_the_log_says_the_gain(tmp_path):
    quiet = mock_telemetry(2)
    sink, _ = _go(tmp_path, steady(quiet), duration=10.0)
    audio = sink.audio[SR:]  # let the gain settle on the first segment's level
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    assert rms == pytest.approx(TARGET_RMS, rel=0.5)
    segments = [e for e in _events(tmp_path) if e["event"] == "live" and "gain" in e["what"]]
    assert segments


def test_without_normalising_the_level_follows_the_chips(tmp_path):
    quiet = [dataclasses.replace(c, power=5.0) for c in mock_telemetry(2)]
    loud, _ = _go(tmp_path, steady(quiet), duration=6.0, normalise=True)
    raw, _ = _go(tmp_path, steady(quiet), duration=6.0, normalise=False)
    assert float(np.sqrt(np.mean(loud.audio ** 2))) > 2 * float(np.sqrt(np.mean(raw.audio ** 2)))
    assert not [e for e in _events(tmp_path) if "gain x" in e["what"]]


def test_two_readings_taken_back_to_back_do_not_make_a_rate_out_of_noise(tmp_path, renders):
    temps = iter([50.0, 50.4, 50.8, 51.2])
    _go(tmp_path, lambda: [dataclasses.replace(FALLBACK, asic_temperature=next(temps), power=10.0)], duration=6.0, clock=Clock(0.3))
    assert renders[1]["telemetry"][0].temp_rate == 0.0  # 0.4 C in 0.3 s is inside the two-second rate window
    assert not [e for e in _events(tmp_path) if e["event"] == "motion"]


# --- the EQ in live mode -----------------------------------------------------------------------------------------

def _share(audio, lo, hi, rate=SR):
    spec = np.abs(np.fft.rfft(audio.astype(np.float64))) ** 2
    f = np.fft.rfftfreq(len(audio), 1 / rate)
    return float(spec[(f >= lo) & (f < hi)].sum() / spec.sum())


def test_live_audio_is_eqd_like_an_offline_render(tmp_path, monkeypatch):
    # noise, so there is something in every band for the EQ to act on
    rng = np.random.default_rng(0)
    monkeypatch.setattr(live_module, "render_chorus",
                        lambda chips, seconds, sr, patch, segment, soloists, only=None, cables=None: (0.1 * rng.standard_normal(int(seconds * sr))).astype(np.float32))
    flat, _ = _go(tmp_path, steady(), duration=6.0, eq=False, normalise=False, sample_rate=44100)
    rng = np.random.default_rng(0)
    (tmp_path / "eq").mkdir()
    shaped, _ = _go(tmp_path / "eq", steady(), duration=6.0, eq=True, normalise=False, sample_rate=44100)
    assert _share(shaped.audio, 10000, 22050, 44100) < 0.3 * _share(flat.audio, 10000, 22050, 44100)  # the shimmer band
    assert _share(shaped.audio, 40, 250, 44100) > 1.3 * _share(flat.audio, 40, 250, 44100)  # the lows


def test_eq_comes_before_the_gain_so_live_audio_still_reaches_the_target_level(tmp_path):
    sink, _ = _go(tmp_path, steady(), duration=10.0, eq=True)
    audio = sink.audio[SR:]
    assert float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))) == pytest.approx(TARGET_RMS, rel=0.5)


def test_the_eq_runs_in_chunks_without_a_seam_at_the_join_of_two_turns(tmp_path, monkeypatch):
    monkeypatch.setattr(live_module, "render_chorus",
                        lambda chips, seconds, sr, patch, segment, soloists, only=None, cables=None: (0.3 * np.sin(2 * np.pi * 440 * np.arange(int(seconds * sr)) / sr)).astype(np.float32))
    sink, n = _go(tmp_path, steady(), duration=8.0, eq=True, normalise=False)
    audio = sink.audio
    jumps = np.abs(np.diff(audio[SR:]))  # after the filter has settled
    assert n >= 3 and float(np.max(jumps)) < 0.3 * 2 * np.pi * 440 / SR * 1.5  # no step bigger than the sine's own steepest slope


def test_without_the_loudness_gain_live_audio_is_kept_inside_full_scale(tmp_path, monkeypatch):
    monkeypatch.setattr(live_module, "render_chorus",
                        lambda chips, seconds, sr, patch, segment, soloists, only=None, cables=None: (0.98 * np.sin(2 * np.pi * 100 * np.arange(int(seconds * sr)) / sr)).astype(np.float32))
    sink, _ = _go(tmp_path, steady(), duration=6.0, eq=True, normalise=False)
    assert float(np.max(np.abs(sink.audio))) <= 1.0


def test_the_eqd_run_is_the_same_length_as_the_plain_one(tmp_path):
    plain, _ = _go(tmp_path, steady(), duration=6.0, eq=False)
    (tmp_path / "eq").mkdir()
    shaped, _ = _go(tmp_path / "eq", steady(), duration=6.0, eq=True)
    assert len(shaped.audio) == len(plain.audio)


def test_the_equaliser_sees_every_turn_and_then_the_tail(tmp_path, monkeypatch):
    seen = []

    class Spy:
        def __init__(self, rate):
            pass

        def process(self, chunk):
            seen.append(len(chunk))
            return chunk

    monkeypatch.setattr(live_module, "Equaliser", Spy)
    sink, n = _go(tmp_path, steady(), duration=6.0, eq=True)
    assert len(seen) == n + 1 and seen[-1] == int(XFADE_S * SR) and seen[:-1] == [len(c) for c in sink.chunks[:n]]


def test_the_log_says_whether_the_eq_is_on(tmp_path):
    _go(tmp_path, steady(), duration=4.0, eq=True)
    on = [e for e in _events(tmp_path) if e["event"] == "eq"]
    assert len(on) == 1 and "35 Hz" in on[0]["what"]
    (tmp_path / "off").mkdir()
    _go(tmp_path / "off", steady(), duration=4.0, eq=False)
    off = [e for e in _events(tmp_path / "off") if e["event"] == "eq"]
    assert len(off) == 1 and "off" in off[0]["what"]



def test_only_reaches_every_turns_render(tmp_path, renders):
    import tensixinet.live as live_mod
    seen = []
    real = live_mod.render_chorus
    live_mod.render_chorus = lambda *a, **k: seen.append(k.get("only")) or real(*a, **k)
    try:
        _go(tmp_path, steady(), duration=4.0, only=["gong"])
    finally:
        live_mod.render_chorus = real
    assert seen and all(o == ["gong"] for o in seen)
    assert any(e["event"] == "only" for e in _events(tmp_path))


def test_cables_are_drawn_from_the_patch_by_default_and_left_out_when_off(tmp_path, renders):
    _go(tmp_path, lambda: [FALLBACK], duration=4.0)
    assert renders and all(c["cables"] is None for c in renders)
    renders.clear()
    (tmp_path / "off").mkdir()
    _go(tmp_path / "off", lambda: [FALLBACK], duration=4.0, cables=False)
    assert renders and all(c["cables"] == [] for c in renders)


# --- the kernel backend -----------------------------------------------------------------------------------------

class _NullBackend:
    """Stands in for a KernelBackend: it changes nothing, so the audio is the CPU's."""
    from tensixinet.music.devices import current_mix as _mix
    devices = {}
    mix = staticmethod(_mix)
    device_for = staticmethod(lambda name, chip=0: None)


def test_live_mode_hands_the_backend_to_every_render_and_only_when_there_is_one(tmp_path, monkeypatch):
    seen = []
    real = live_module.render_chorus

    def spy(*a, **k):
        seen.append(k)
        return real(*a, **k)

    monkeypatch.setattr(live_module, "render_chorus", spy)
    marker = _NullBackend()
    _, n = _go(tmp_path, lambda: [FALLBACK], duration=4.0, backend=marker)
    assert n >= 1 and len(seen) == n and all(k.get("backend") is marker for k in seen)
    seen.clear()
    _go(tmp_path, lambda: [FALLBACK], duration=4.0)
    assert seen and all("backend" not in k for k in seen)


def test_live_mode_gives_the_patch_to_the_turn_weights_so_a_styles_balance_picks_who_has_the_floor(tmp_path, monkeypatch):
    seen = []
    real = live_module.slot_weights
    monkeypatch.setattr(live_module, "slot_weights", lambda chips, patch=None: (seen.append(patch), real(chips, patch))[1])
    patch = dataclasses.replace(ONE_BAR, emphasis=(("ttchoir", 3.0),))
    _go(tmp_path, lambda: [FALLBACK], duration=4.0, patch=patch)
    assert seen and all(p is patch for p in seen)


def test_live_stereo_hands_stereo_segments_to_the_sink_and_asks_the_chorus_for_them(tmp_path, monkeypatch):
    seen = []
    real = live_module.render_chorus

    def spy(*a, **k):
        seen.append(k.get("stereo", False))
        return real(*a, **k)

    monkeypatch.setattr(live_module, "render_chorus", spy)
    sink = MemorySink()
    _go(tmp_path, lambda: [FALLBACK, dataclasses.replace(FALLBACK, asic_temperature=60.0)], duration=4.0, sink=sink, stereo=True)
    assert seen and all(seen) and sink.chunks and all(c.ndim == 2 and c.shape[1] == 2 for c in sink.chunks)
    mono_sink = MemorySink()
    _go(tmp_path, lambda: [FALLBACK], duration=4.0, sink=mono_sink)
    assert all(c.ndim == 1 for c in mono_sink.chunks)


# --- eviction: a chip whose model unloads dissolves out of the band -----------------------------------------------------------------------

def _power_profile(idle=6, load=6, rest=10):
    return [14.0] * idle + [62.0] * load + [14.0] * rest


def test_live_mode_tracks_each_chips_power_and_logs_an_eviction_and_a_fill(tmp_path, monkeypatch):
    seen = []
    real = live_module.render_chorus
    monkeypatch.setattr(live_module, "render_chorus", lambda *a, **k: (seen.append(k.get("defrag")), real(*a, **k))[1])
    powers = iter(_power_profile() + [62.0] * 8)
    ev, _ = _run_with_powers(tmp_path, lambda: [dataclasses.replace(FALLBACK, power=next(powers), tdp_limit=125.0)], duration=2.0 * 30, evict=True)
    kinds = [e["event"] for e in ev]
    assert "evict" in kinds and "fill" in kinds and kinds.index("evict") < kinds.index("fill")
    evict = next(e for e in ev if e["event"] == "evict")
    assert evict["chip"] == 0 and evict["what"].startswith("chip 0 evicted") and "right to left" in evict["why"]
    phases = [d[0].phase for d in seen if d is not None]
    assert phases[0] == "running" and "evicting" in phases and "idle" in phases and "filling" in phases and phases[-1] in ("filling", "running")
    assert phases.index("evicting") < phases.index("idle") < phases.index("filling")


def test_without_evict_the_chorus_is_not_given_a_defrag_and_nothing_is_logged(tmp_path, monkeypatch):
    seen = []
    real = live_module.render_chorus
    monkeypatch.setattr(live_module, "render_chorus", lambda *a, **k: (seen.append("defrag" in k), real(*a, **k))[1])
    powers = iter(_power_profile() + [62.0] * 8)
    ev, _ = _run_with_powers(tmp_path, lambda: [dataclasses.replace(FALLBACK, power=next(powers), tdp_limit=125.0)], duration=2.0 * 30)
    assert seen and not any(seen) and not [e for e in ev if e["event"] in ("evict", "fill")]


def test_an_idle_box_that_never_loads_never_evicts_in_a_live_run(tmp_path):
    ev, _ = _run_with_powers(tmp_path, lambda: [dataclasses.replace(FALLBACK, power=14.0, tdp_limit=125.0)], duration=2.0 * 30, evict=True)
    assert not [e for e in ev if e["event"] in ("evict", "fill")]


def _run_with_powers(tmp_path, sample, duration, **kw):
    sink = MemorySink()
    _go(tmp_path, sample, duration=duration, sink=sink, clock=Clock(3.0), **kw)
    return _events(tmp_path), sink
