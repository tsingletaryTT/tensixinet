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
    assert "plays rollz" in next(e["what"] for e in events if e["event"] == "chip")
