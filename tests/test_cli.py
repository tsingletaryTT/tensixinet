# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Command line: prompt the LLM, then sample chips, then play."""
import json
import sys
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
    # the margin is loose on purpose: it only needs to show normalising did something, not pin the raw mix's
    # own level, which moves a little with every device the chorus adds
    assert rms(a) > 1.2 * rms(b) and 0.1 < rms(a) < 0.15
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
    class Asked(list):
        channels = None

    asked = Asked()
    asked.channels = []
    tmp_path.mkdir(exist_ok=True)

    def make_sink(player, rate, out, channels=1):
        asked.append((player, rate, out))
        asked.channels.append(channels)
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
    def make_sink(player, rate, out, channels=1):
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
    def make_sink(*a, **k): raise AssertionError("offline runs write a file and play nothing")
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
        x = x.mean(axis=1) if x.ndim == 2 else x  # the default is stereo
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


# --- patch cables -----------------------------------------------------------------------------------------------

def test_cables_are_logged_and_on_by_default_and_no_cables_turns_them_off(tmp_path):
    import soundfile as sf
    import numpy as np
    _, a = run_cli(tmp_path, "patched", "--patch-seed", "5")
    _, b = run_cli(tmp_path, "bare", "--patch-seed", "5", "--no-cables")
    cable_lines = [e for e in read_events(a / "logs") if e["event"] == "cable"]
    assert 5 <= len(cable_lines) <= 8 and all("->" in e["what"] for e in cable_lines)
    off = [e for e in read_events(b / "logs") if e["event"] == "cable"]
    assert len(off) == 1 and "off" in off[0]["what"]
    x, _ = sf.read(str(a / "o.wav"))
    y, _ = sf.read(str(b / "o.wav"))
    assert not np.allclose(x, y)


@pytest.mark.parametrize("flag,expected", [([], True), (["--no-cables"], False)])
def test_live_mode_passes_the_cable_choice_to_the_loop(tmp_path, monkeypatch, flag, expected):
    import tensixinet.cli as cli
    seen = {}
    monkeypatch.setattr(cli, "run_live", lambda *a, **k: seen.update(k) or 0)
    live(tmp_path, *flag)
    assert seen["cables"] is expected


# --- --kernels: the soundtrack on the TT-Lang kernels ------------------------------------------------------------

class _Worker:
    """A host-mode worker that records whether it was started and closed, so a lease would be released."""

    def __init__(self, devices=1):
        from tensixinet.media.chip import ChipWorker, host_command
        self.devices = devices
        self.inner = ChipWorker(host_command(chunk_tiles=16, devices=devices), devices=devices)
        self.started = self.closed = False
        self.calls = []
        self.where = []

    def start(self):
        self.started = True
        self.inner.start()

    def run(self, op, inputs, n, device=0):
        self.calls.append(op)
        self.where.append((op, device))
        return self.inner.run(op, inputs, n, device=device)

    def close(self):
        self.closed = True
        self.inner.close()


def test_kernels_runs_the_kernel_devices_through_the_worker_and_releases_it(tmp_path):
    import numpy as np
    import soundfile as sf
    worker = _Worker()
    code = main(args(tmp_path, "--mock-chips", "4", "--duration", "3", "--patch-seed", "4", "--kernels", "host"), make_worker=lambda mode, chips: worker)
    assert code == 0 and worker.started and worker.closed
    assert {"ttshimmer", "ttshock", "ttavdog", "ttcoco", "ttmix"} <= set(worker.calls)
    cpu_dir = tmp_path / "cpu"
    cpu_dir.mkdir()
    assert main(["--duration", "3", "--sample-rate", "8000", "--out", str(cpu_dir / "o.wav"), "--log-dir", str(cpu_dir / "logs"),
                 "--mock-chips", "4", "--patch-seed", "4"]) == 0  # the same patch, or the two runs are not comparable
    on_chip, _ = sf.read(str(tmp_path / "o.wav"))
    on_cpu, _ = sf.read(str(cpu_dir / "o.wav"))
    assert np.allclose(on_chip, on_cpu, atol=5e-3)  # the kernels' arithmetic, loudness-normalised, not another sound
    ev = read_events(tmp_path / "logs")
    line = next(e for e in ev if e["event"] == "backend")
    assert "host" in line["what"] and "gong" in line["why"] and "the rest stay on the CPU" in line["why"]


def test_kernels_without_a_mode_means_the_chip_and_asks_for_it(tmp_path):
    modes = []

    def make(mode, chips):
        modes.append((mode, chips))
        return _Worker()

    assert main(args(tmp_path, "--mock-chips", "2", "--kernels"), make_worker=make) == 0
    assert modes == [("chip", 1)]  # one chip, whatever number of chips the run plays


def test_a_worker_that_cannot_start_is_a_logged_failure_and_nothing_is_played(tmp_path):
    from tensixinet.media.chip import ChipError

    class Busy:
        closed = False

        def start(self):
            raise ChipError("the chip worker exited before it was ready (exit 10) (gozer queued the request: the chips are busy)")

        def close(self):
            Busy.closed = True

    code = main(args(tmp_path, "--mock-chips", "2", "--kernels", "chip"), make_worker=lambda mode, chips: Busy())
    assert code == 3 and not (tmp_path / "o.wav").exists() and Busy.closed
    ev = read_events(tmp_path / "logs")
    assert not [e for e in ev if "releasing" in e["what"]]   # nothing was held, so nothing is released
    line = next(e for e in ev if e["event"] == "backend")
    assert "chips are busy" in line["what"] and "nothing was played" in line["why"]


def test_kernels_with_live_mode_hands_the_backend_to_the_live_loop(tmp_path):
    seen = {}

    def fake_live(sample, sink, log, sample_rate, **kw):
        seen.update(kw)
        return 1

    import tensixinet.cli as cli_module
    worker = _Worker()
    real = cli_module.run_live
    cli_module.run_live = fake_live
    try:
        code = main(args(tmp_path, "--mock-chips", "4", "--live", "--player", "none", "--kernels", "host"), make_worker=lambda mode, chips: worker)
    finally:
        cli_module.run_live = real
    assert code == 0 and seen["backend"].worker is worker and worker.closed


def test_the_run_log_carries_the_backends_own_notes(tmp_path):
    from tensixinet.media.chip import ChipError

    class Dies(_Worker):
        def run(self, op, inputs, n, device=0):
            raise ChipError("the chip worker is gone")

    assert main(args(tmp_path, "--mock-chips", "4", "--duration", "2", "--kernels", "host"), make_worker=lambda mode, chips: Dies()) == 0
    notes = [e for e in read_events(tmp_path / "logs") if e["event"] == "backend" and "stopped" in e["what"]]
    assert len(notes) == 1 and "CPU" in notes[0]["what"] and "chip worker is gone" in notes[0]["what"]


def test_kernels_chips_holds_one_chip_for_every_chip_the_run_plays_and_says_where_each_device_runs(tmp_path):
    calls = []

    def make(mode, chips):
        calls.append((mode, chips))
        return _Worker(devices=chips)

    worker = None

    def keep(mode, chips):
        nonlocal worker
        worker = make(mode, chips)
        return worker

    assert main(args(tmp_path, "--mock-chips", "4", "--duration", "4", "--kernels", "chips"), make_worker=keep) == 0
    assert calls == [("chips", 4)] and worker.devices == 4
    assert {device for _, device in worker.where} >= {0, 2, 3}          # the kernels ran on several of the chips
    ev = [e for e in read_events(tmp_path / "logs") if e["event"] == "backend"]
    head = ev[0]
    assert "4 leased chips" in head["what"] and "chips mode" in head["what"]
    per_chip = [e for e in ev if e["what"].startswith("chip ")]
    assert len(per_chip) == 4
    assert "ttavdog" in per_chip[0]["what"] and "ultrasonic_shimmer" in per_chip[2]["what"] and "ttdrone" in per_chip[3]["what"]
    assert "gong" in per_chip[1]["what"] and "the mixer" in per_chip[0]["what"]
    assert all("kernels" in e["what"] for e in per_chip)


def test_a_chip_run_says_it_is_releasing_the_chips_because_that_takes_about_forty_seconds(tmp_path):
    assert main(args(tmp_path, "--mock-chips", "2", "--kernels", "chip"), make_worker=lambda mode, chips: _Worker()) == 0
    ev = read_events(tmp_path / "logs")
    release = [e for e in ev if e["event"] == "backend" and "releasing" in e["what"]]
    assert len(release) == 1 and "reset" in release[0]["why"] and "40 s" in release[0]["why"]
    host = tmp_path / "h"
    host.mkdir()
    assert main(["--duration", "0.5", "--sample-rate", "8000", "--out", str(host / "o.wav"), "--log-dir", str(host / "logs"),
                 "--mock-chips", "2", "--kernels", "host"], make_worker=lambda mode, chips: _Worker()) == 0
    assert not [e for e in read_events(host / "logs") if "releasing" in e["what"]]   # a host worker holds no chip


# --- colour and the banner -------------------------------------------------------------------------------------------

def test_a_piped_run_is_plain_text_even_with_the_art(tmp_path, capsys):
    assert main(args(tmp_path, "--mock-chips", "2")) == 0
    out = capsys.readouterr().out
    assert "\x1b[" not in out and "TENSIXINET" not in out and "Ciat-Lonbarde" in out   # the art is blocks; the name is in the tagline text
    assert "█" in out or "▀" in out
    assert "● blue" in out and "● grey" in out                                         # the legend of the jack colours


def test_color_always_colours_the_screen_but_not_the_text_log_or_the_jsonl(tmp_path, capsys):
    assert main(args(tmp_path, "--mock-chips", "2", "--color", "always")) == 0
    out = capsys.readouterr().out
    assert "\x1b[38;" in out and "\x1b[0m" in out
    from tensixinet.narrate.color import strip_ansi
    (txt,) = (tmp_path / "logs").glob("*.txt")
    assert "\x1b" not in txt.read_text() and strip_ansi(out) == txt.read_text()         # the same words, with and without colour
    (jsonl,) = (tmp_path / "logs").glob("*.jsonl")
    assert "\x1b" not in jsonl.read_text()


def test_color_never_and_no_color_keep_a_terminal_plain(tmp_path, capsys, monkeypatch):
    assert main(args(tmp_path, "--mock-chips", "2", "--color", "never")) == 0
    assert "\x1b[" not in capsys.readouterr().out
    monkeypatch.setenv("NO_COLOR", "1")
    other = tmp_path / "n"
    other.mkdir()
    assert main(["--duration", "0.5", "--sample-rate", "8000", "--out", str(other / "o.wav"), "--log-dir", str(other / "logs"),
                 "--mock-chips", "2"]) == 0
    assert "\x1b[" not in capsys.readouterr().out


def test_no_banner_leaves_the_run_log_to_start_at_the_first_line(tmp_path, capsys):
    assert main(args(tmp_path, "--mock-chips", "2", "--no-banner")) == 0
    out = capsys.readouterr().out
    assert out.startswith("║") and "● blue" not in out and "█" not in out
    (txt,) = (tmp_path / "logs").glob("*.txt")
    assert txt.read_text().startswith("║")


def test_the_banner_adapts_to_the_pane_it_is_in(tmp_path, capsys, monkeypatch):
    import os
    monkeypatch.setattr("shutil.get_terminal_size", lambda *a, **k: os.terminal_size((40, 24)))
    assert main(args(tmp_path, "--mock-chips", "2")) == 0
    out = capsys.readouterr().out
    assert all(len(line) <= 40 for line in out.splitlines() if "o.wav" not in line)   # a file path cannot be wrapped
    assert "█" not in out and "TENSIXINET" in out


# --- --style ----------------------------------------------------------------------------------------------------------------------

def test_a_style_draws_a_patch_in_that_style_saves_it_and_says_so(tmp_path, capsys):
    from tensixinet.music.patch import load_patch
    assert main(args(tmp_path, "--mock-chips", "4", "--style", "blues", "--patch-seed", "5")) == 0
    (saved,) = (tmp_path / "logs").glob("patch_*.json")
    patch = load_patch(saved)
    from tensixinet.music.styles import STYLES
    assert patch.style == "blues" and patch.scale_name in STYLES["blues"].scales and patch.seed == 5
    ev = read_events(tmp_path / "logs")
    line = next(e for e in ev if e["event"] == "patch" and e["what"].startswith("style blues"))
    assert STYLES["blues"].blurb in line["why"] and "ttrax" in line["why"]
    assert any("blues style" in e["why"] for e in ev if e["what"].startswith("patch seed"))


def test_every_style_plays_a_run_through(tmp_path):
    from tensixinet.music.styles import STYLES
    for name in STYLES:
        out = tmp_path / name
        out.mkdir()
        assert main(["--duration", "1", "--sample-rate", "8000", "--out", str(out / "o.wav"), "--log-dir", str(out / "logs"),
                     "--mock-chips", "3", "--style", name, "--no-banner"]) == 0 and (out / "o.wav").exists()


def test_a_style_cannot_be_combined_with_a_saved_patch(tmp_path, capsys):
    with pytest.raises(SystemExit):
        main(args(tmp_path, "--mock-chips", "2", "--style", "enya", "--patch", str(tmp_path / "p.json")))
    err = capsys.readouterr().err
    assert "--style" in err and "--patch" in err


def test_an_unknown_style_is_refused_by_the_parser(tmp_path, capsys):
    with pytest.raises(SystemExit):
        main(args(tmp_path, "--mock-chips", "2", "--style", "polka"))
    assert "chiptune" in capsys.readouterr().err


# --- stereo is the default; --mono is the old sound -----------------------------------------------------------------------------------

def test_a_run_is_stereo_by_default_and_mono_on_request(tmp_path):
    import soundfile as sf
    assert main(args(tmp_path, "--mock-chips", "3", "--no-banner")) == 0
    assert sf.read(str(tmp_path / "o.wav"))[0].shape[1] == 2
    mono = tmp_path / "m"
    mono.mkdir()
    assert main(["--duration", "0.5", "--sample-rate", "8000", "--out", str(mono / "o.wav"), "--log-dir", str(mono / "logs"),
                 "--mock-chips", "3", "--mono", "--no-banner"]) == 0
    assert sf.read(str(mono / "o.wav"))[0].ndim == 1
    assert not [e for e in read_events(mono / "logs") if e["event"] == "space"]            # nothing to say about space in mono
    assert [e for e in read_events(tmp_path / "logs") if e["event"] == "space"]


def test_live_mode_asks_for_two_channels_by_default_and_one_with_mono(tmp_path):
    code, sink, asked = live(tmp_path / "a")
    assert code == 0 and asked.channels == [2] and all(c.ndim == 2 and c.shape[1] == 2 for c in sink.chunks)
    code, sink, asked = live(tmp_path / "b", "--mono")
    assert code == 0 and asked.channels == [1] and all(c.ndim == 1 for c in sink.chunks)


def test_the_options_say_what_stereo_is():
    from tensixinet.cli import build_parser
    help_text = build_parser().format_help()
    assert "--mono" in help_text and "stereo" in help_text


# --- eviction from the command line ------------------------------------------------------------------------------------------------------------

class _StepClock:
    """A clock that moves a fixed step each time it is read, so a run that does not wait for its music still sees time pass."""

    def __init__(self, step):
        self.now, self.step = 1000.0, step

    def __call__(self):
        self.now += self.step
        return self.now


def test_a_mock_load_cycle_in_live_mode_evicts_a_chip_and_says_so(tmp_path):
    code, sink, _ = live(tmp_path, "--mock-chips", "2", "--mock-load", "40", "--duration", "120", "--no-banner", clock=_StepClock(3.0))
    assert code == 0
    ev = read_events(tmp_path / "logs")
    kinds = [e["event"] for e in ev]
    assert "evict" in kinds and "fill" in kinds


def test_no_evict_leaves_a_load_cycle_without_consequences(tmp_path):
    code, _, _ = live(tmp_path, "--mock-chips", "2", "--mock-load", "40", "--duration", "120", "--no-evict", "--no-banner", clock=_StepClock(3.0))
    assert code == 0 and not [e for e in read_events(tmp_path / "logs") if e["event"] in ("evict", "fill")]


def test_mock_load_needs_mock_chips_and_live_mode(tmp_path, capsys):
    with pytest.raises(SystemExit):
        main(args(tmp_path, "--mock-load", "40"))
    assert "--mock-load" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(args(tmp_path, "--mock-chips", "2", "--mock-load", "40"))
    assert "--live" in capsys.readouterr().err


def test_kernels_sim_holds_a_pretend_chip_for_every_chip_the_run_plays_and_says_it_is_the_simulator(tmp_path):
    calls = []

    def make(mode, chips):
        calls.append((mode, chips))
        return _Worker(devices=chips)

    assert main(args(tmp_path, "--mock-chips", "4", "--duration", "4", "--kernels", "sim"), make_worker=make) == 0
    assert calls == [("sim", 4)]
    ev = [e for e in read_events(tmp_path / "logs") if e["event"] == "backend"]
    assert "TT-Lang simulator" in ev[0]["what"] and "no chip" in ev[0]["what"] and "4 pretend chips" in ev[0]["what"] and "leased" not in ev[0]["what"]
    per_chip = [e for e in ev if e["what"].startswith("chip ")]
    assert len(per_chip) == 4 and all("simulated chip" in e["why"] and "leased" not in e["why"] for e in per_chip)


def test_the_default_worker_for_sim_mode_is_the_simulator_pool_with_no_lease(monkeypatch):
    import tensixinet.media.chip as chip
    monkeypatch.setattr(chip, "find_simulator", lambda *a, **k: "/venv/bin/tt-lang-sim")
    from tensixinet.cli import default_make_worker
    w = default_make_worker("sim", 4)
    assert "--sim=/venv/bin/tt-lang-sim" in w.command and "--devices=4" in w.command and "gozer" not in w.command and w.devices == 4


def test_midi_file_writes_a_standard_midi_file_and_logs_it(tmp_path):
    mid = tmp_path / "m" / "run.mid"
    code = main(args(tmp_path, "--mock-chips", "2", "--duration", "3", "--midi-file", str(mid)))
    assert code == 0 and mid.read_bytes()[:4] == b"MThd"
    assert any(e["event"] == "midi file" and str(mid) in e["what"] for e in read_events(tmp_path / "logs"))


def test_midi_file_does_not_change_the_audio(tmp_path):
    import numpy as np
    import soundfile as sf
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    base = ["--mock-chips", "2", "--duration", "2", "--patch-seed", "3"]
    assert main(args(a, *base)) == 0
    assert main(args(b, *base, "--midi-file", str(b / "x.mid"))) == 0
    assert np.array_equal(sf.read(str(a / "o.wav"))[0], sf.read(str(b / "o.wav"))[0])


def test_midi_file_is_accepted_with_live_and_the_file_is_written(tmp_path):
    mid = tmp_path / "live.mid"
    code = main(args(tmp_path, "--mock-chips", "2", "--duration", "1", "--live", "--player", "none", "--midi-file", str(mid)))
    assert code == 0 and mid.read_bytes()[:4] == b"MThd"
    from tests.score.test_smf import read_smf
    assert [e for track in read_smf(mid)[3] for _, e in track if e[0] & 0xF0 == 0x90], "a live run wrote no note-on"


def test_midi_map_needs_a_midi_output_flag(tmp_path, capsys):
    routes = tmp_path / "r.json"
    routes.write_text("{}")
    with pytest.raises(SystemExit) as e:
        main(args(tmp_path, "--mock-chips", "2", "--midi-map", str(routes)))
    err = capsys.readouterr().err
    assert e.value.code == 2 and "--midi-map needs" in err and "--midi-file" in err


def test_a_bad_midi_map_exits_2_and_names_the_key(tmp_path, capsys):
    routes = tmp_path / "r.json"
    routes.write_text('{"pulse_s": -1}')
    with pytest.raises(SystemExit) as e:
        main(args(tmp_path, "--mock-chips", "2", "--midi-file", str(tmp_path / "x.mid"), "--midi-map", str(routes)))
    assert e.value.code == 2 and "pulse_s" in capsys.readouterr().err
    assert not (tmp_path / "x.mid").exists()


def test_a_midi_map_that_is_not_json_exits_2(tmp_path, capsys):
    routes = tmp_path / "r.json"
    routes.write_text("not json")
    with pytest.raises(SystemExit) as e:
        main(args(tmp_path, "--mock-chips", "2", "--midi-file", str(tmp_path / "x.mid"), "--midi-map", str(routes)))
    assert e.value.code == 2 and "cannot read" in capsys.readouterr().err
    with pytest.raises(SystemExit) as e:
        main(args(tmp_path, "--mock-chips", "2", "--midi-file", str(tmp_path / "x.mid"), "--midi-map", str(tmp_path / "missing.json")))
    assert e.value.code == 2 and "cannot read" in capsys.readouterr().err


def test_midi_map_changes_the_routes_in_the_file(tmp_path):
    from tests.score.test_smf import read_smf
    routes = tmp_path / "r.json"
    routes.write_text('{"note_channels": {"lead": 5, "ttquantus": 5}}')
    mid = tmp_path / "x.mid"
    code = main(args(tmp_path, "--mock-chips", "2", "--duration", "3", "--midi-file", str(mid), "--midi-map", str(routes)))
    assert code == 0
    ons = [e[0] & 0x0F for track in read_smf(mid)[3] for _, e in track if e[0] & 0xF0 == 0x90]
    assert 5 in ons and 0 not in ons and 1 not in ons


# --- the live MIDI port ----------------------------------------------------------------------------------------------
from tensixinet import cli as cli_module  # noqa: E402
from tensixinet.score.sinks import MidiPortError  # noqa: E402


class FakeMidiPort:
    def __init__(self):
        self.sent, self.closed = [], 0

    def send(self, raw):
        self.sent.append(bytes(raw))

    def close(self):
        self.closed += 1


def live_args(tmp_path, *extra):
    return args(tmp_path, "--mock-chips", "2", "--duration", "1", "--live", "--player", "none", *extra)


def test_midi_port_needs_live(tmp_path, capsys):
    with pytest.raises(SystemExit) as e:
        main(args(tmp_path, "--mock-chips", "2", "--midi-port", "virtual"), open_midi=lambda name: FakeMidiPort())
    assert e.value.code == 2 and "--midi-port needs --live" in capsys.readouterr().err


def test_midi_offset_needs_a_port(tmp_path, capsys):
    with pytest.raises(SystemExit) as e:
        main(live_args(tmp_path, "--midi-offset-ms", "50"))
    assert e.value.code == 2 and "--midi-offset-ms needs --midi-port" in capsys.readouterr().err


def test_midi_offset_is_a_number(tmp_path, capsys):
    with pytest.raises(SystemExit) as e:
        main(live_args(tmp_path, "--midi-port", "virtual", "--midi-offset-ms", "soon"), open_midi=lambda name: FakeMidiPort())
    assert e.value.code == 2


def test_a_port_that_cannot_open_is_one_message_and_exit_2_before_any_chip_is_read(tmp_path, capsys):
    read, sinks = [], []

    def sample():
        read.append(1)
        return mock_telemetry(2)

    def opener(name):
        raise MidiPortError(f"no MIDI output called {name!r}; the outputs are none")

    with pytest.raises(SystemExit) as e:
        main(live_args(tmp_path, "--midi-port", "Nope"), sample=sample, open_midi=opener,
             make_sink=lambda *a, **k: sinks.append(1))
    err = capsys.readouterr().err
    assert e.value.code == 2 and "no MIDI output called 'Nope'" in err
    assert read == [] and sinks == []


def test_a_missing_midi_library_names_the_extra_and_exits_2(tmp_path, capsys, monkeypatch):
    monkeypatch.setitem(sys.modules, "mido", None)
    with pytest.raises(SystemExit) as e:
        main(live_args(tmp_path, "--midi-port", "virtual"))
    assert e.value.code == 2 and "pip install -e .[midi]" in capsys.readouterr().err


def test_the_port_gets_the_live_runs_notes_and_is_closed(tmp_path):
    port = FakeMidiPort()
    code = main(live_args(tmp_path, "--midi-port", "Synth", "--duration", "2"), open_midi=lambda name: port)
    assert code == 0 and port.closed == 1
    # the run ends the moment the last segment is made; only what was due by then has gone out, plus the close's resets
    assert any(m[0] & 0xF0 == 0xB0 and m[1] == 123 for m in port.sent) or port.sent == []


def test_the_name_and_offset_reach_the_port_sink_and_the_sink_is_closed(tmp_path, monkeypatch):
    made = []

    class Spy:
        def __init__(self, name, routes=None, offset_ms=0.0, open_output=None, on_error=None, **kw):
            made.append((name, offset_ms, routes))
            self.closed = 0
            self.scores = []

        def add(self, score, start_s):
            self.scores.append((score, start_s))

        def close(self):
            self.closed += 1
            made.append("closed")

        def summary(self):
            return (1, 2, 3)

    monkeypatch.setattr(cli_module, "MidiPortSink", Spy)
    assert main(live_args(tmp_path, "--midi-port", "Synth", "--midi-offset-ms", "-35.5")) == 0
    assert made[0][:2] == ("Synth", -35.5) and made[-1] == "closed"
    assert main(live_args(tmp_path / "b", "--midi-port", "virtual")) == 0
    assert made[-2][:2] == ("virtual", 0.0)


def test_the_port_sink_gets_the_midi_map_routes(tmp_path, monkeypatch):
    seen = []

    class Spy:
        def __init__(self, name, routes=None, **kw):
            seen.append(routes)

        def add(self, score, start_s): pass
        def close(self): pass

    monkeypatch.setattr(cli_module, "MidiPortSink", Spy)
    routes = tmp_path / "r.json"
    routes.write_text('{"pulse_s": 0.5}')
    assert main(live_args(tmp_path, "--midi-port", "Synth", "--midi-map", str(routes))) == 0
    assert seen[0].pulse_s == 0.5


def test_midi_map_is_accepted_with_a_port_alone(tmp_path, monkeypatch):
    routes = tmp_path / "r.json"
    routes.write_text("{}")
    assert main(live_args(tmp_path, "--midi-port", "Synth", "--midi-map", str(routes)), open_midi=lambda n: FakeMidiPort()) == 0


def test_a_file_and_a_port_together_both_get_the_score(tmp_path):
    from tests.score.test_smf import read_smf
    mid, port = tmp_path / "both.mid", FakeMidiPort()
    code = main(live_args(tmp_path, "--midi-file", str(mid), "--midi-port", "Synth"), open_midi=lambda name: port)
    assert code == 0 and port.closed == 1
    assert [e for track in read_smf(mid)[3] for _, e in track if e[0] & 0xF0 == 0x90]
    done = [e for e in read_events(tmp_path / "logs") if e["event"] == "midi file"]
    assert done and str(mid) in done[0]["what"] and " 0 notes" not in done[0]["what"]


def test_a_port_alone_is_logged_with_its_counts_and_no_path(tmp_path):
    main(live_args(tmp_path, "--midi-port", "Synth"), open_midi=lambda name: FakeMidiPort())
    done = [e for e in read_events(tmp_path / "logs") if e["event"] in ("midi", "midi file")]
    assert done and "notes" in done[0]["what"] and " to " not in done[0]["what"]


def test_a_sink_that_fails_to_close_does_not_change_the_exit_code_and_is_logged(tmp_path, monkeypatch):
    class Bad:
        def __init__(self, *a, **k): pass
        def add(self, score, start_s): pass
        def summary(self): return (0, 0, 0)
        path = None

        def close(self):
            raise OSError("disk full")

    monkeypatch.setattr(cli_module, "MidiFileSink", Bad)
    assert main(live_args(tmp_path, "--midi-file", str(tmp_path / "x.mid"))) == 0
    assert any("disk full" in e["what"] for e in read_events(tmp_path / "logs"))


def test_a_close_failure_does_not_hide_a_player_failure_exit_code(tmp_path, monkeypatch):
    class Bad:
        def __init__(self, *a, **k): pass
        def add(self, score, start_s): pass
        def summary(self): return (0, 0, 0)
        path = None

        def close(self):
            raise OSError("disk full")

    class Dying:
        def write(self, audio):
            raise PlayerError("the player stopped")

        def close(self):
            pass

    monkeypatch.setattr(cli_module, "MidiFileSink", Bad)
    code = main(live_args(tmp_path, "--midi-file", str(tmp_path / "x.mid")), make_sink=lambda *a, **k: (Dying(), "dying"))
    assert code == 3


def test_a_port_is_closed_when_the_run_ends_before_playing(tmp_path):
    port = FakeMidiPort()
    code = main(live_args(tmp_path, "--midi-port", "Synth"), open_midi=lambda name: port,
                make_sink=lambda *a, **k: (_ for _ in ()).throw(PlayerError("no player")))
    assert code == 3 and port.closed == 1
