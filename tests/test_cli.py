"""Command line: prompt the LLM, then sample chips, then play."""
import json
from pathlib import Path

import pytest

from tensixinet.cli import main
from tensixinet.media.llm import LLMError
from tensixinet.telemetry.mock import mock_telemetry


def read_events(log_dir):
    (jsonl,) = Path(log_dir).glob("*.jsonl")
    return [json.loads(l) for l in jsonl.read_text().splitlines()]


def args(tmp_path, *extra):
    return ["--duration", "0.5", "--sample-rate", "8000", "--out", str(tmp_path / "o.wav"),
            "--log-dir", str(tmp_path / "logs"), *extra]


def test_mock_chips_runs_with_no_hardware_and_writes_wav_and_logs(tmp_path):
    code = main(args(tmp_path, "--mock-chips", "3"))
    assert code == 0 and (tmp_path / "o.wav").exists()
    ev = read_events(tmp_path / "logs")
    assert "3 chips" in ev[0]["what"]
    assert any("mock" in e["why"].lower() for e in ev)  # says where the chips came from


def test_prompt_is_sent_before_telemetry_is_sampled(tmp_path):
    order = []
    def ask(prompt, endpoint): order.append("ask"); return "a reply"
    def sample(): order.append("sample"); return mock_telemetry(2)
    assert main(args(tmp_path, "think", "--endpoint", "http://h:1"), ask=ask, sample=sample) == 0
    assert order == ["ask", "sample"]  # the LLM's load must be on the chips when we read them
    ev = read_events(tmp_path / "logs")
    llm = next(e for e in ev if e["event"] == "llm")
    assert "a reply" in llm["what"]


def test_llm_failure_is_logged_and_the_music_still_plays(tmp_path):
    def ask(prompt, endpoint): raise LLMError("connection refused")
    code = main(args(tmp_path, "think"), ask=ask, sample=lambda: mock_telemetry(2))
    assert code == 0 and (tmp_path / "o.wav").exists()
    ev = read_events(tmp_path / "logs")
    warn = next(e for e in ev if e["event"] == "llm")
    assert "connection refused" in warn["what"] and "continuing" in warn["why"]


def test_no_chips_is_a_nonzero_exit_with_a_logged_reason(tmp_path):
    code = main(args(tmp_path), sample=lambda: [])
    assert code != 0 and not (tmp_path / "o.wav").exists()
    assert "no chips" in read_events(tmp_path / "logs")[0]["what"]


def test_without_a_prompt_the_llm_is_never_called(tmp_path):
    def ask(*a): raise AssertionError("must not be called")
    assert main(args(tmp_path), ask=ask, sample=lambda: mock_telemetry(1)) == 0


def test_mock_chips_must_be_positive(tmp_path):
    with pytest.raises(SystemExit):
        main(args(tmp_path, "--mock-chips", "0"))


def test_log_width_follows_the_terminal_so_it_fits_the_pane(tmp_path, monkeypatch):
    import shutil
    monkeypatch.chdir(tmp_path)  # short relative paths: an unbroken path is never split, by design
    monkeypatch.setattr(shutil, "get_terminal_size", lambda *a, **k: shutil.os.terminal_size((60, 24)))
    main(["--duration", "0.5", "--sample-rate", "8000", "--out", "o.wav",
          "--log-dir", "logs", "--mock-chips", "2"])
    (txt,) = (tmp_path / "logs").glob("*.txt")
    lines = txt.read_text().splitlines()
    assert max(len(l) for l in lines) <= 59
    assert any(l.startswith("║               ") for l in lines)  # and it did wrap


def test_the_screen_log_states_each_reason_once_but_the_jsonl_keeps_them_all(tmp_path):
    main(args(tmp_path, "--mock-chips", "3"))
    (txt,) = (tmp_path / "logs").glob("*.txt")
    (jsonl,) = (tmp_path / "logs").glob("*.jsonl")
    reason = "25-100 C maps onto C3-C5"
    assert " ".join(txt.read_text().split()).count(reason) == 1
    assert jsonl.read_text().count(reason) == 3
