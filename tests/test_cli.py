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


def test_no_normalise_keeps_the_absolute_level_and_normalising_is_the_default(tmp_path):
    import soundfile as sf
    import numpy as np
    _, a = run_cli(tmp_path, "norm", "--patch-seed", "5")
    _, b = run_cli(tmp_path, "raw", "--patch-seed", "5", "--no-normalise")
    rms = lambda d: float(np.sqrt(np.mean(sf.read(str(d / "o.wav"))[0] ** 2)))
    assert rms(a) > 1.5 * rms(b) and 0.1 < rms(a) < 0.15
    assert any("left as rendered" in e["what"] for e in read_events(b / "logs"))


# --- live mode ---------------------------------------------------------------------------

import numpy as np  # noqa: E402

from tensixinet.media.player import PlayerError  # noqa: E402


class MemorySink:
    def __init__(self):
        self.chunks, self.closed = [], 0

    def write(self, audio):
        self.chunks.append(np.array(audio))

    def close(self):
        self.closed += 1


def live(tmp_path, *extra, sample=None, ask=None, sink=None, **kw):
    """Run the CLI in live mode with a memory sink. Returns (exit code, the sink, what make_sink was asked)."""
    sink = sink or MemorySink()
    asked = []
    tmp_path.mkdir(exist_ok=True)

    def make_sink(player, rate, out):
        asked.append((player, rate, out))
        return sink, "a memory sink"

    argv = ["--live", "--sample-rate", "8000", "--out", str(tmp_path / "o.wav"), "--log-dir", str(tmp_path / "logs"), *extra]
    if "--duration" not in extra:
        argv += ["--duration", "4"]
    if sample is None and "--mock-chips" not in extra:
        argv += ["--mock-chips", "2"]
    code = main(argv, sample=sample, ask=ask, make_sink=kw.pop("make_sink", make_sink), **kw)
    return code, sink, asked


def test_live_mode_plays_into_the_sink_and_logs_every_segment(tmp_path):
    code, sink, _ = live(tmp_path)
    assert code == 0 and sink.closed == 1 and sum(len(c) for c in sink.chunks) > 4 * 8000
    ev = read_events(tmp_path / "logs")
    assert any(e["event"] == "live" and "segment 0" in e["what"] for e in ev)
    assert any(e["event"] == "live" and "a memory sink" in e["what"] for e in ev)  # says where the sound goes


def test_live_mode_asks_for_the_player_the_rate_and_the_wav_path(tmp_path):
    _, _, asked = live(tmp_path, "--player", "aplay")
    assert asked == [("aplay", 8000, tmp_path / "o.wav")]


def test_no_sound_is_a_player_choice_too(tmp_path):
    _, _, asked = live(tmp_path, "--player", "none")
    assert asked[0][0] == "none"


def test_the_player_is_chosen_automatically_by_default(tmp_path):
    _, _, asked = live(tmp_path)
    assert asked[0][0] == "auto"


def test_an_unknown_player_is_refused_by_the_parser(tmp_path):
    with pytest.raises(SystemExit):
        live(tmp_path, "--player", "winamp")


def test_a_player_means_nothing_without_live(tmp_path, capsys):
    with pytest.raises(SystemExit) as e:
        main(args(tmp_path, "--mock-chips", "2", "--player", "aplay"))
    assert e.value.code == 2 and "--live" in capsys.readouterr().err


def test_live_mode_sends_the_prompt_first_then_reads_the_chips(tmp_path):
    order = []
    def ask(prompt, endpoint): order.append("ask"); return "a reply"
    def sample(): order.append("sample"); return mock_telemetry(2)
    code, _, _ = live(tmp_path, "think", sample=sample, ask=ask)
    assert code == 0 and order[:2] == ["ask", "sample"] and order.count("ask") == 1


def test_live_mode_draws_and_saves_a_patch_like_any_run(tmp_path):
    live(tmp_path, "--patch-seed", "5")
    (saved,) = (tmp_path / "logs").glob("patch_*.json")
    assert load_patch(saved).seed == 5


def test_live_mode_with_no_chips_exits_nonzero_and_still_closes_the_sink(tmp_path):
    code, sink, _ = live(tmp_path, sample=lambda: [])
    assert code == 2 and sink.closed == 1 and not sink.chunks
    assert any("no chips" in e["what"] for e in read_events(tmp_path / "logs"))


def test_a_player_that_cannot_start_is_a_logged_error_with_its_own_exit_code(tmp_path):
    def make_sink(player, rate, out):
        raise PlayerError("cannot start paplay: not installed")
    code, _, _ = live(tmp_path, make_sink=make_sink)
    assert code == 3
    assert any(e["event"] == "player" and "cannot start paplay" in e["what"] for e in read_events(tmp_path / "logs"))


def test_a_player_that_stops_midway_is_the_same_exit_code(tmp_path):
    class Dies(MemorySink):
        def write(self, audio):
            raise PlayerError("aplay stopped before the audio was done")
    code, _, _ = live(tmp_path, sink=Dies())
    assert code == 3


def test_live_mode_follows_no_normalise(tmp_path):
    live(tmp_path, "--no-normalise")
    assert not any("gain x" in e["what"] for e in read_events(tmp_path / "logs"))


def test_a_zero_duration_means_play_until_interrupted(tmp_path):
    calls = []

    def sample():
        calls.append(1)
        if len(calls) == 6:  # one read draws the patch, then segments follow until this one is interrupted
            raise KeyboardInterrupt
        return mock_telemetry(2)

    code, sink, _ = live(tmp_path, "--duration", "0", sample=sample)
    assert code == 0 and len(calls) == 6 and len(sink.chunks) >= 4


def test_ctrl_c_before_the_run_starts_exits_quietly(tmp_path):
    def ask(prompt, endpoint): raise KeyboardInterrupt
    code, sink, _ = live(tmp_path, "think", ask=ask)
    assert code == 130 and not sink.chunks


def test_offline_mode_never_builds_a_sink(tmp_path):
    def make_sink(*a): raise AssertionError("offline runs write a file and play nothing")
    assert main(args(tmp_path, "--mock-chips", "2"), make_sink=make_sink) == 0


def test_live_mode_end_to_end_with_the_real_sinks_and_no_sound(tmp_path, monkeypatch):
    import time
    import soundfile as sf
    slept = []
    monkeypatch.setattr(time, "sleep", slept.append)  # the null sink would otherwise take as long as the music
    argv = ["--live", "--player", "none", "--mock-chips", "2", "--duration", "3", "--sample-rate", "8000",
            "--out", str(tmp_path / "o.wav"), "--log-dir", str(tmp_path / "logs")]
    assert main(argv) == 0
    audio, rate = sf.read(str(tmp_path / "o.wav"))
    assert rate == 8000 and len(audio) > 3 * 8000  # the three seconds and the last turn's tail
    assert sum(slept) == pytest.approx(len(audio) / 8000, abs=0.01)  # it kept time with the music it did not play
    assert any(e["event"] == "live" and "no sound" in e["what"] for e in read_events(tmp_path / "logs"))


def test_no_eq_leaves_the_mix_unshaped_and_the_eq_is_the_default(tmp_path):
    import soundfile as sf
    import numpy as np
    _, a = run_cli(tmp_path, "eq", "--patch-seed", "5")
    _, b = run_cli(tmp_path, "flat", "--patch-seed", "5", "--no-eq")
    def high_share(d):
        x, rate = sf.read(str(d / "o.wav"))
        spec = np.abs(np.fft.rfft(x)) ** 2
        f = np.fft.rfftfreq(len(x), 1 / rate)
        return spec[f > 1500].sum() / spec.sum()  # 8 kHz sample rate: the top of the band
    assert high_share(a) < 0.7 * high_share(b)
    assert any(e["event"] == "eq" and "off" in e["what"] for e in read_events(b / "logs"))


def test_live_mode_follows_no_eq(tmp_path):
    live(tmp_path, "--no-eq")  # the run_live call must accept it
    code, sink, _ = live(tmp_path / "second")
    assert code == 0 and sink.chunks


@pytest.mark.parametrize("flag,expected", [([], True), (["--no-eq"], False)])
def test_live_mode_passes_the_eq_choice_to_the_loop(tmp_path, monkeypatch, flag, expected):
    import tensixinet.cli as cli
    seen = {}
    monkeypatch.setattr(cli, "run_live", lambda *a, **k: seen.update(k) or 0)
    live(tmp_path, *flag)
    assert seen["eq"] is expected



# --- --only ---------------------------------------------------------------------------------------------------------

def test_only_plays_just_the_named_devices_and_says_so(tmp_path):
    import soundfile as sf
    import numpy as np
    _, all_devices = run_cli(tmp_path, "all", "--patch-seed", "5", "--no-normalise", "--no-eq")
    _, just_gong = run_cli(tmp_path, "gong", "--patch-seed", "5", "--only", "gong", "--no-normalise", "--no-eq")
    a, b = sf.read(str(all_devices / "o.wav"))[0], sf.read(str(just_gong / "o.wav"))[0]
    assert not np.allclose(a, b)
    assert any(e["event"] == "only" and "gong" in e["what"] for e in read_events(just_gong / "logs"))


def test_only_takes_a_comma_separated_list(tmp_path):
    code, d = run_cli(tmp_path, "two", "--only", "gong,ttchoir")
    assert code == 0
    assert any(e["event"] == "only" and "gong" in e["what"] and "ttchoir" in e["what"] for e in read_events(d / "logs"))


def test_an_unknown_device_is_refused_with_the_list_of_devices(tmp_path, capsys):
    with pytest.raises(SystemExit) as e:
        run_cli(tmp_path, "bad", "--only", "nope")
    assert e.value.code == 2
    err = capsys.readouterr().err
    assert "nope" in err and "ttchoir" in err and "gong" in err


def test_live_mode_passes_only_to_the_loop(tmp_path, monkeypatch):
    import tensixinet.cli as cli
    seen = {}
    monkeypatch.setattr(cli, "run_live", lambda *a, **k: seen.update(k) or 0)
    live(tmp_path, "--only", "ttavdog")
    assert seen["only"] == ["ttavdog"]


def test_spaces_around_the_commas_are_ignored_and_a_list_of_nothing_is_refused(tmp_path, capsys):
    code, d = run_cli(tmp_path, "spaced", "--only", "gong, ttchoir ")
    assert code == 0 and any(e["event"] == "only" and "gong, ttchoir" in e["what"] for e in read_events(d / "logs"))
    with pytest.raises(SystemExit):
        run_cli(tmp_path, "empty", "--only", " , ")
    assert "none given" in capsys.readouterr().err
