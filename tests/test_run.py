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
    audio = run(chips, duration=3.0, out_path=tmp_path / "o.wav", log=log, sample_rate=8000)
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
    audio = run(chips, duration=3.0, out_path=tmp_path / "o.wav", log=log, sample_rate=8000, normalise=False)
    log.close()
    assert np.array_equal(audio, render_chorus(chips, 3.0, 8000))
    (ev,) = [e for e in _events(tmp_path) if e["event"] == "loudness"]
    assert "left as rendered" in ev["what"] and "power" in ev["why"]
