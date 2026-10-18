# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
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


# --- patches ----------------------------------------------------------------------

from tensixinet.music.patch import load_patch  # noqa: E402


def run_cli(tmp_path, name, *extra, entropy=None):
    d = tmp_path / name
    d.mkdir()
    argv = ["--duration", "3", "--sample-rate", "8000", "--out", str(d / "o.wav"),
            "--log-dir", str(d / "logs"), "--mock-chips", "3", *extra]
    code = main(argv, entropy=entropy) if entropy else main(argv)
    return code, d


def test_every_run_saves_its_patch_next_to_its_log_and_says_so(tmp_path):
    code, d = run_cli(tmp_path, "a", entropy=lambda: 1)
    assert code == 0
    (saved,) = (d / "logs").glob("patch_*.json")
    assert load_patch(saved).seed >= 0
    ev = read_events(d / "logs")
    assert any(e["event"] == "patch" and "entropy" in e["why"] for e in ev)


def test_a_fixed_patch_seed_gives_the_same_sound_every_time(tmp_path):
    _, a = run_cli(tmp_path, "a", "--patch-seed", "5", entropy=lambda: 1)
    _, b = run_cli(tmp_path, "b", "--patch-seed", "5", entropy=lambda: 2)
    assert (a / "o.wav").read_bytes() == (b / "o.wav").read_bytes()
    assert load_patch(next((a / "logs").glob("patch_*.json"))).seed == 5


def test_different_entropy_gives_a_different_patch_and_sound(tmp_path):
    _, a = run_cli(tmp_path, "a", entropy=lambda: 1)
    _, b = run_cli(tmp_path, "b", entropy=lambda: 2)
    pa = load_patch(next((a / "logs").glob("patch_*.json")))
    pb = load_patch(next((b / "logs").glob("patch_*.json")))
    assert pa != pb and (a / "o.wav").read_bytes() != (b / "o.wav").read_bytes()


def test_a_saved_patch_replays_the_run_exactly(tmp_path):
    _, a = run_cli(tmp_path, "a", entropy=lambda: 7)
    saved = next((a / "logs").glob("patch_*.json"))
    code, b = run_cli(tmp_path, "b", "--patch", str(saved), entropy=lambda: 99)
    assert code == 0 and (a / "o.wav").read_bytes() == (b / "o.wav").read_bytes()
    assert any(e["event"] == "patch" and str(saved) in e["why"] for e in read_events(b / "logs"))


def test_an_unreadable_patch_stops_the_run_with_a_logged_reason(tmp_path):
    code, d = run_cli(tmp_path, "a", "--patch", str(tmp_path / "missing.json"))
    assert code == 2 and not (d / "o.wav").exists()
    assert any(e["event"] == "patch" and "cannot read" in e["what"] for e in read_events(d / "logs"))


# --- fault injection and the host in the seed ----------------------------------------------

def test_mock_faults_makes_chips_report_errors_and_the_log_says_so(tmp_path):
    assert main(args(tmp_path, "--mock-chips", "6", "--mock-faults", "--seed", "1")) == 0
    ev = [e for e in read_events(tmp_path / "logs") if e["event"] == "events"]
    assert len(ev) == 6 and any("no memory errors" not in e["what"] for e in ev)
    assert any("no memory errors" in e["what"] for e in ev)


def test_mock_faults_needs_mock_chips(tmp_path, capsys):
    with pytest.raises(SystemExit) as e:
        main(args(tmp_path, "--mock-faults"))
    assert e.value.code == 2 and "--mock-chips" in capsys.readouterr().err  # our message, not "unrecognized argument"


def _snapshot(hostname):
    snap = json.loads((Path(__file__).parent / "fixtures" / "tt_smi_snapshot.json").read_text())
    snap["host_info"]["Hostname"] = hostname
    return snap


def _patch_seed_for(tmp_path, monkeypatch, hostname, name):
    monkeypatch.setattr("tensixinet.cli.fetch_snapshot", lambda: _snapshot(hostname))
    out = tmp_path / name
    assert main(["--duration", "0.3", "--sample-rate", "8000", "--out", str(out / "o.wav"),
                 "--log-dir", str(out / "logs")], entropy=lambda: 7) == 0
    (patch,) = (out / "logs").glob("patch_*.json")
    return json.loads(patch.read_text())["seed"]


def test_the_same_chips_on_a_different_host_draw_a_different_patch(tmp_path, monkeypatch):
    a = _patch_seed_for(tmp_path, monkeypatch, "host-a", "a")
    b = _patch_seed_for(tmp_path, monkeypatch, "host-b", "b")
    again = _patch_seed_for(tmp_path, monkeypatch, "host-a", "c")
    assert a != b and a == again
