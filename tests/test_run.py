# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The end-to-end pipeline and the no-hardware path for N chips."""
import json
import wave
from pathlib import Path

import pytest

from tensixinet.telemetry.mock import mock_telemetry
from tensixinet.run import run
from tensixinet.narrate.runlog import RunLog
from tensixinet.telemetry import parse_snapshot

FIXTURE = Path(__file__).parent / "fixtures" / "tt_smi_snapshot.json"


def _log(tmp_path):
    return RunLog(jsonl=open(tmp_path / "run.jsonl", "w"), stream=open(tmp_path / "run.txt", "w"))


def test_run_writes_a_wav_and_a_log_that_explains_every_chip(tmp_path):
    chips = parse_snapshot(json.loads(FIXTURE.read_text()))  # the real 4-chip capture
    out = tmp_path / "out.wav"
    log = _log(tmp_path)
    audio = run(chips, duration=1.0, out_path=out, log=log, sample_rate=8000)
    log.close()

    with wave.open(str(out)) as w:
        assert w.getframerate() == 8000 and w.getnframes() == 8000
    assert audio.shape == (8000,)

    events = [json.loads(l) for l in (tmp_path / "run.jsonl").read_text().splitlines()]
    kinds = [e["event"] for e in events]
    assert kinds[0] == "discover" and kinds[-1] == "done"
    assert "4 chips" in events[0]["what"]
    assert sum(e["event"] == "chip" for e in events) >= 4 * 3  # several decisions per chip
    assert {e.get("chip") for e in events if e["event"] == "chip"} == {0, 1, 2, 3}
    assert all(e["what"] and e["why"] for e in events)


def test_run_with_no_chips_logs_why_and_raises(tmp_path):
    log = _log(tmp_path)
    with pytest.raises(ValueError):
        run([], duration=1.0, out_path=tmp_path / "x.wav", log=log, sample_rate=8000)
    log.close()
    text = (tmp_path / "run.jsonl").read_text()
    assert "no chips" in text and not (tmp_path / "x.wav").exists()


@pytest.mark.parametrize("n,phrase", [(1, "found 1 chip"), (2, "found 2 chips")])
def test_run_log_pluralises_the_chip_count(tmp_path, n, phrase):
    log = _log(tmp_path)
    run(mock_telemetry(n), duration=0.2, out_path=tmp_path / "o.wav", log=log, sample_rate=8000)
    log.close()
    first = json.loads((tmp_path / "run.jsonl").read_text().splitlines()[0])
    assert first["what"] == phrase


def test_run_log_says_who_solos_and_how_the_lead_changes(tmp_path):
    chips = mock_telemetry(1, seed=3)  # one chip plays every device, so the lead is present
    log = _log(tmp_path)
    run(chips, duration=12.0, out_path=tmp_path / "o.wav", log=log, sample_rate=8000)
    log.close()
    events = [json.loads(l) for l in (tmp_path / "run.jsonl").read_text().splitlines()]
    kinds = [e["event"] for e in events]
    assert "turn" in kinds and "lead" in kinds
    assert kinds.index("turn") < kinds.index("render")
    assert "plays ttrollz" in next(e["what"] for e in events if e["event"] == "chip")


# --- patches in a run --------------------------------------------------------------

from tensixinet.music.patch import DEFAULT_PATCH, draw_patch  # noqa: E402


def _events(tmp_path):
    return [json.loads(l) for l in (tmp_path / "run.jsonl").read_text().splitlines()]


def test_run_logs_the_patch_before_it_describes_any_chip(tmp_path):
    chips = mock_telemetry(2, seed=4)
    log = _log(tmp_path)
    run(chips, 2.0, tmp_path / "o.wav", log, 8000, patch=draw_patch(5, chips), patch_source="a test seed")
    log.close()
    kinds = [e["event"] for e in _events(tmp_path)]
    assert "patch" in kinds and kinds.index("patch") < kinds.index("chip")
    assert "a test seed" in next(e["why"] for e in _events(tmp_path) if e["event"] == "patch")


def test_run_plays_the_patch_it_was_given(tmp_path):
    chips = mock_telemetry(3, seed=4)
    log = _log(tmp_path)
    plain = run(chips, 6.0, tmp_path / "a.wav", log, 8000)
    same = run(chips, 6.0, tmp_path / "b.wav", log, 8000, patch=DEFAULT_PATCH)
    other = run(chips, 6.0, tmp_path / "c.wav", log, 8000, patch=draw_patch(5, chips))
    log.close()
    assert (plain == same).all() and not (plain == other).all()


def test_run_logs_turns_with_the_patch_turn_length(tmp_path):
    import dataclasses
    chips = mock_telemetry(2, seed=4)
    times = {}
    for bars in (1, 4):
        log = RunLog(jsonl=open(tmp_path / f"{bars}.jsonl", "w"), stream=open(tmp_path / f"{bars}.txt", "w"))
        run(chips, 40.0, tmp_path / f"{bars}.wav", log, 8000, patch=dataclasses.replace(DEFAULT_PATCH, bars_per_turn=bars))
        log.close()
        evs = [json.loads(l) for l in (tmp_path / f"{bars}.jsonl").read_text().splitlines()]
        times[bars] = [e["what"] for e in evs if e["event"] == "turn"]
    assert times[1] != times[4]


# --- the wider readings in the log ---------------------------------------------------------

def test_run_logs_every_chips_turn_weight_and_event_status_before_rendering(tmp_path):
    chips = mock_telemetry(3, seed=2, faults=True)
    log = _log(tmp_path)
    run(chips, 2.0, tmp_path / "o.wav", log, 8000)
    log.close()
    evs = _events(tmp_path)
    kinds = [e["event"] for e in evs]
    weights = [e for e in evs if e["event"] == "weight"]
    events = [e for e in evs if e["event"] == "events"]
    assert [e["chip"] for e in weights] == [0, 1, 2] and [e["chip"] for e in events] == [0, 1, 2]
    assert max(kinds.index("weight"), kinds.index("events")) < kinds.index("turn") < kinds.index("render")
    assert any("correctable" in e["what"] or "thermal" in e["what"] or "uncorrectable" in e["what"] for e in events)


def test_run_with_healthy_chips_logs_that_no_event_sounds_are_added(tmp_path):
    log = _log(tmp_path)
    run(mock_telemetry(2, seed=1), 1.0, tmp_path / "o.wav", log, 8000)
    log.close()
    assert all("no memory errors" in e["what"] for e in _events(tmp_path) if e["event"] == "events")


# --- loudness: one gain over the finished render ---------------------------------------------------------------

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from tensixinet.music.chorus import render_chorus  # noqa: E402
from tensixinet.music.loudness import TARGET_RMS, loudness_gain  # noqa: E402


def _events(tmp_path):
    return [json.loads(l) for l in (tmp_path / "run.jsonl").read_text().splitlines()]


def test_the_wav_lands_at_the_target_level_and_the_log_says_what_gain_did_it(tmp_path):
    chips = mock_telemetry(3)
    log = _log(tmp_path)
    audio = run(chips, duration=3.0, out_path=tmp_path / "o.wav", log=log, sample_rate=8000, eq=False)
    log.close()
    raw = render_chorus(chips, 3.0, 8000)
    gain = loudness_gain(raw)
    assert np.allclose(audio, raw * gain, atol=1e-6)
    written, _ = sf.read(str(tmp_path / "o.wav"))
    assert float(np.sqrt(np.mean(written ** 2))) == pytest.approx(TARGET_RMS, rel=0.05)
    (ev,) = [e for e in _events(tmp_path) if e["event"] == "loudness"]
    import math
    assert f"x{gain:.2f}" in ev["what"] and f"({20 * math.log10(gain):+.1f} dB)" in ev["what"] and "dBFS" in ev["what"] and "one gain" in ev["why"]
    kinds = [e["event"] for e in _events(tmp_path)]
    assert kinds.index("render") < kinds.index("loudness") < kinds.index("done")


def test_the_done_line_reports_the_peak_of_the_file_that_was_written(tmp_path):
    log = _log(tmp_path)
    audio = run(mock_telemetry(2), duration=2.0, out_path=tmp_path / "o.wav", log=log, sample_rate=8000)
    log.close()
    done = [e for e in _events(tmp_path) if e["event"] == "done"][0]
    assert f"peak {float(np.max(np.abs(audio))):.2f}" in done["why"]


def test_without_normalising_the_render_is_written_as_it_came_and_the_log_says_so(tmp_path):
    chips = mock_telemetry(3)
    log = _log(tmp_path)
    audio = run(chips, duration=3.0, out_path=tmp_path / "o.wav", log=log, sample_rate=8000, normalise=False, eq=False)
    log.close()
    assert np.array_equal(audio, render_chorus(chips, 3.0, 8000))
    (ev,) = [e for e in _events(tmp_path) if e["event"] == "loudness"]
    assert "left as rendered" in ev["what"] and "power" in ev["why"]


# --- the EQ on the final mix ---------------------------------------------------------------------------

import numpy as np  # noqa: E402

from tensixinet.music.chorus import render_chorus  # noqa: E402
from tensixinet.music.eq import apply_eq  # noqa: E402
from tensixinet.music.loudness import normalise  # noqa: E402


def _band_share(audio, rate, lo, hi):
    spec = np.abs(np.fft.rfft(audio.astype(np.float64))) ** 2
    f = np.fft.rfftfreq(len(audio), 1 / rate)
    return float(spec[(f >= lo) & (f < hi)].sum() / spec.sum())


def _run_audio(tmp_path, **kw):
    log = _log(tmp_path)
    audio = run(mock_telemetry(4), duration=4.0, out_path=tmp_path / "o.wav", log=log, sample_rate=44100, **kw)
    log.close()
    return audio


def test_the_final_mix_is_eqd_by_default_so_the_shimmer_band_is_quieter(tmp_path):
    with_eq, without = _run_audio(tmp_path), _run_audio(tmp_path, eq=False)
    assert _band_share(with_eq, 44100, 10000, 22050) < 0.5 * _band_share(without, 44100, 10000, 22050)
    assert _band_share(with_eq, 44100, 40, 250) > _band_share(without, 44100, 40, 250)


def test_the_eq_goes_before_the_loudness_gain_so_the_level_still_lands_on_target(tmp_path):
    audio = _run_audio(tmp_path)
    assert float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))) == pytest.approx(10 ** (-18 / 20), rel=0.02)


def test_the_eq_is_what_apply_eq_gives_the_render(tmp_path):
    chips = mock_telemetry(2)
    log = _log(tmp_path)
    got = run(chips, duration=1.0, out_path=tmp_path / "o.wav", log=log, sample_rate=8000)
    log.close()
    want, _ = normalise(apply_eq(render_chorus(chips, 1.0, 8000), 8000))
    assert np.allclose(got, want, atol=1e-6)


def test_turning_the_eq_off_gives_the_plain_render(tmp_path):
    chips = mock_telemetry(2)
    log = _log(tmp_path)
    got = run(chips, duration=1.0, out_path=tmp_path / "o.wav", log=log, sample_rate=8000, eq=False)
    log.close()
    want, _ = normalise(render_chorus(chips, 1.0, 8000))
    assert np.allclose(got, want, atol=1e-6)


def test_the_log_says_what_the_eq_does_and_when_it_is_off(tmp_path):
    _run_audio(tmp_path)
    eq = [json.loads(l) for l in (tmp_path / "run.jsonl").read_text().splitlines() if '"event": "eq"' in l]
    assert len(eq) == 1 and "35 Hz" in eq[0]["what"] and "14 dB" in eq[0]["what"] and "shimmer" in eq[0]["why"]
    other = tmp_path / "off"
    other.mkdir()
    _run_audio(other, eq=False)
    off = [json.loads(l) for l in (other / "run.jsonl").read_text().splitlines() if '"event": "eq"' in l]
    assert len(off) == 1 and "off" in off[0]["what"]


def test_without_the_loudness_gain_the_eqs_lift_cannot_push_the_file_past_full_scale(tmp_path, monkeypatch):
    from tensixinet import run as run_module
    t = np.arange(8000 * 3) / 8000
    monkeypatch.setattr(run_module, "render_chorus", lambda *a, **k: (0.95 * np.sin(2 * np.pi * 100 * t)).astype(np.float32))
    log = _log(tmp_path)
    audio = run(mock_telemetry(1), duration=3.0, out_path=tmp_path / "o.wav", log=log, sample_rate=8000, normalise=False)
    log.close()
    assert float(np.max(np.abs(audio))) == pytest.approx(1.0, abs=1e-6)  # scaled down to just fit



# --- auditioning ----------------------------------------------------------------------------------------------------

def test_only_is_passed_to_the_render_and_the_log_says_which_devices_are_soloed(tmp_path, monkeypatch):
    from tensixinet import run as run_module
    seen = {}
    real = run_module.render_chorus
    monkeypatch.setattr(run_module, "render_chorus", lambda *a, **k: seen.update(k) or real(*a, **k))
    log = _log(tmp_path)
    run(mock_telemetry(2), duration=0.5, out_path=tmp_path / "o.wav", log=log, sample_rate=8000, only=["ttchoir", "gong"])
    log.close()
    assert seen["only"] == ["ttchoir", "gong"]
    ev = [json.loads(l) for l in (tmp_path / "run.jsonl").read_text().splitlines()]
    only = [e for e in ev if e["event"] == "only"]
    assert len(only) == 1 and "ttchoir" in only[0]["what"] and "gong" in only[0]["what"] and "turn-taking" in only[0]["why"]


def test_without_only_the_log_has_no_only_line(tmp_path):
    log = _log(tmp_path)
    run(mock_telemetry(2), duration=0.5, out_path=tmp_path / "o.wav", log=log, sample_rate=8000)
    log.close()
    assert '"event": "only"' not in (tmp_path / "run.jsonl").read_text()


def test_a_backend_is_handed_to_the_render_and_the_run_is_the_cpus_when_it_changes_nothing(tmp_path, monkeypatch):
    import numpy as np
    from tensixinet import run as run_module
    from tensixinet.music.devices import current_mix

    class Null:
        devices = {}
        mix = staticmethod(current_mix)
        device_for = staticmethod(lambda name, chip=0: None)

    seen = []
    real = run_module.render_chorus
    monkeypatch.setattr(run_module, "render_chorus", lambda *a, **k: (seen.append(k.get("backend")), real(*a, **k))[1])
    chips = mock_telemetry(2)
    log = _log(tmp_path)
    marker = Null()
    with_backend = run(chips, 1.0, tmp_path / "a.wav", log, 8000, backend=marker)
    without = run(chips, 1.0, tmp_path / "b.wav", log, 8000)
    log.close()
    assert seen == [marker, None] and np.array_equal(with_backend, without)


def test_a_stereo_run_writes_two_channels_and_says_where_each_chip_sits(tmp_path):
    import soundfile as sf
    chips = mock_telemetry(3)
    log = _log(tmp_path)
    audio = run(chips, 2.0, tmp_path / "s.wav", log, 8000, stereo=True)
    log.close()
    assert audio.shape == (16000, 2) and sf.read(str(tmp_path / "s.wav"))[0].shape == (16000, 2)
    events = [json.loads(l) for l in (tmp_path / "run.jsonl").read_text().splitlines()]
    space = [e for e in events if e["event"] == "space"]
    assert len(space) == 1 + 3 and space[0]["what"].startswith("stereo field")


def test_a_mono_run_is_one_channel_and_says_nothing_about_space(tmp_path):
    import soundfile as sf
    log = _log(tmp_path)
    audio = run(mock_telemetry(3), 1.0, tmp_path / "m.wav", log, 8000)
    log.close()
    assert audio.ndim == 1 and sf.read(str(tmp_path / "m.wav"))[0].ndim == 1
    assert not [l for l in (tmp_path / "run.jsonl").read_text().splitlines() if '"space"' in l]


def test_a_score_sink_gets_a_file_with_notes_and_controls_and_the_audio_is_unchanged(tmp_path):
    import numpy as np
    from tensixinet.score.sinks import MidiFileSink
    from tests.score.test_smf import read_smf

    chips = mock_telemetry(2)
    log = _log(tmp_path)
    plain = run(chips, duration=3.0, out_path=tmp_path / "a.wav", log=log, sample_rate=8000)
    mid = tmp_path / "out" / "r.mid"
    with_sink = run(chips, duration=3.0, out_path=tmp_path / "b.wav", log=log, sample_rate=8000, score_sink=MidiFileSink(mid))
    log.close()
    assert np.array_equal(plain, with_sink)
    _, _, _, tracks = read_smf(mid)
    events = [e for track in tracks for _, e in track if e[0] != 0xFF]
    assert any(e[0] & 0xF0 == 0x90 for e in events), "no note-on"
    assert any(e[0] & 0xF0 == 0xB0 for e in events), "no control change"
    lines = [json.loads(l) for l in (tmp_path / "run.jsonl").read_text().splitlines()]
    (line,) = [e for e in lines if e["event"] == "midi file"]
    assert str(mid) in line["what"] and "notes" in line["what"] and "triggers" in line["what"] and "controls" in line["what"]


def test_only_reaches_the_score_so_an_audition_of_the_gong_has_no_notes(tmp_path):
    from tensixinet.score.sinks import MidiFileSink
    sink = MidiFileSink(tmp_path / "g.mid")
    log = _log(tmp_path)
    run(mock_telemetry(2), duration=3.0, out_path=tmp_path / "g.wav", log=log, sample_rate=8000, only=["gong"], score_sink=sink)
    log.close()
    notes, triggers, _ = sink.summary()
    assert notes == 0 and triggers > 0


# --- closing any score sink ---------------------------------------------------------------------------------------------

def test_close_score_sink_copes_with_a_sink_that_has_no_summary_or_path(tmp_path):
    from tensixinet.run import close_score_sink

    class Bare:
        closed = 0

        def close(self):
            self.closed += 1

    sink = Bare()
    close_score_sink(sink, _log(tmp_path))
    assert sink.closed == 1


def test_close_score_sink_logs_the_counts_and_the_path_of_a_tee(tmp_path):
    from tensixinet.run import close_score_sink
    from tensixinet.score.events import Note, Score
    from tensixinet.score.sinks import MidiFileSink, TeeScoreSink
    path = tmp_path / "t.mid"
    tee = TeeScoreSink(MidiFileSink(path))
    tee.add(Score((Note(0.1, 0.1, 440.0, 0.5, "lead", 0),), 1.0, 120.0), 0.0)
    log = _log(tmp_path)
    close_score_sink(tee, log)
    log.close()
    done = [e for e in _events(tmp_path) if e["event"] == "midi file"]
    assert done and "1 notes" in done[0]["what"] and str(path) in done[0]["what"] and path.exists()
