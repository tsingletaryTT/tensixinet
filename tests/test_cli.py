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
    asked.options = []
    tmp_path.mkdir(exist_ok=True)

    def make_sink(player, rate, out, channels=1, **options):
        asked.append((player, rate, out))
        asked.channels.append(channels)
        asked.options.append(options)
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
    code = main(live_args(tmp_path, "--midi-port", "Synth", "--duration", "3"), open_midi=lambda name: port)
    assert code == 0 and port.closed == 1
    assert any(m[0] & 0xF0 == 0x90 and m[2] > 0 for m in port.sent), "no note-on reached the port"
    assert any(m[0] & 0xF0 == 0xB0 and m[1] == 123 for m in port.sent)  # the close's all-notes-off


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


class DyingPort(FakeMidiPort):
    def send(self, raw):
        raise OSError("the synth was unplugged")


def test_a_port_that_dies_is_one_log_line_and_the_run_still_ends_well(tmp_path):
    port = DyingPort()
    code = main(live_args(tmp_path, "--midi-port", "Synth", "--duration", "7"), open_midi=lambda name: port)
    stopped = [e for e in read_events(tmp_path / "logs") if e["event"] == "midi" and "port stopped" in e["what"]]
    assert code == 0 and len(stopped) == 1 and "unplugged" in stopped[0]["what"]
    assert port.closed == 1


# --- fast mode: --interval -------------------------------------------------------------------------------------------------------------------

def fast(tmp_path, *extra, sample=None, sink=None, **kw):
    """Run the CLI with --interval and a memory sink. Returns (exit code, the sink, the keyword arguments make_sink was given)."""
    sink = sink or MemorySink()
    asked = []
    tmp_path.mkdir(exist_ok=True)

    def make_sink(player, rate, out, channels=1, **options):
        asked.append(dict(player=player, rate=rate, channels=channels, **options))
        return sink, "a memory sink"

    argv = ["--live", "--sample-rate", "8000", "--out", str(tmp_path / "o.wav"), "--log-dir", str(tmp_path / "logs"), "--no-banner", *extra]
    if "--duration" not in extra:
        argv += ["--duration", "1"]
    if sample is None and "--mock-chips" not in extra:
        argv += ["--mock-chips", "2"]
    code = main(argv, sample=sample, make_sink=kw.pop("make_sink", make_sink), **kw)
    return code, sink, asked


@pytest.mark.parametrize("value", ["9", "1001", "0", "-5"])
def test_an_interval_outside_10_to_1000_ms_is_a_parser_error_naming_the_range(tmp_path, capsys, value):
    with pytest.raises(SystemExit) as e:
        fast(tmp_path, "--interval", value)
    err = capsys.readouterr().err
    assert e.value.code == 2 and "--interval" in err and "10" in err and "1000" in err


def test_the_interval_ends_of_the_range_are_allowed():
    from tensixinet.cli import build_parser
    assert build_parser().parse_args(["--interval", "10"]).interval == 10
    assert build_parser().parse_args(["--interval", "1000"]).interval == 1000


def test_an_interval_needs_live(tmp_path, capsys):
    with pytest.raises(SystemExit) as e:
        main(args(tmp_path, "--mock-chips", "2", "--interval", "100"))
    assert e.value.code == 2 and "--interval needs --live" in capsys.readouterr().err


@pytest.mark.parametrize("flag", [["--midi-file", "x.mid"], ["--midi-port", "virtual"]])
def test_an_interval_is_refused_with_a_midi_output(tmp_path, capsys, flag):
    # fast mode does not build a score yet, so the MIDI flags would be accepted and write nothing
    with pytest.raises(SystemExit) as e:
        main(args(tmp_path, "--mock-chips", "2", "--live", "--interval", "100", *flag))
    err = capsys.readouterr().err
    assert e.value.code == 2 and "--interval cannot be used with --midi-file or --midi-port" in err


def test_an_interval_is_refused_with_the_kernels(tmp_path, capsys):
    with pytest.raises(SystemExit) as e:
        fast(tmp_path, "--interval", "100", "--kernels", "sim")
    err = capsys.readouterr().err
    assert e.value.code == 2 and "--kernels" in err and "hold the chips" in err and "render budget" in err


def test_an_interval_with_mock_chips_plays_and_logs_the_interval(tmp_path):
    code, sink, asked = fast(tmp_path, "--interval", "100", "--mock-chips", "4", "--player", "none")
    assert code == 0 and sink.closed == 1 and sum(len(c) for c in sink.chunks) >= 8000
    assert asked == [dict(player="none", rate=8000, channels=2, latency_ms=20)]  # the player is asked for a small buffer
    ev = read_events(tmp_path / "logs")
    assert any("100 ms" in e["what"] for e in ev)
    assert any("synthetic" in e["what"] for e in ev)  # the note says where the readings come from
    assert any(e["event"] == "discover" and "4 chips" in e["what"] for e in ev)


def test_an_interval_with_a_mock_load_cycle_runs(tmp_path):
    code, _, _ = fast(tmp_path, "--interval", "200", "--mock-chips", "2", "--mock-load", "40")
    assert code == 0


def test_an_interval_hands_the_flags_to_the_fast_loop(tmp_path, monkeypatch):
    import tensixinet.cli as cli
    seen = {}

    def fake(source, sink, log, sample_rate, interval_s, duration, patch, patch_source, **kw):
        seen.update(source=source, rate=sample_rate, interval_s=interval_s, duration=duration, **kw)
        return 0

    monkeypatch.setattr(cli, "run_fast", fake)
    monkeypatch.setattr(cli, "run_live", lambda *a, **k: pytest.fail("--interval must not call run_live"))
    code, _, _ = fast(tmp_path, "--interval", "50", "--no-eq", "--no-cables", "--mono", "--no-evict", "--only", "gong", "--player", "aplay",
                      "--duration", "0")
    assert code == 0 and seen["interval_s"] == 0.05 and seen["duration"] is None and seen["rate"] == 8000
    assert (seen["eq"], seen["cables"], seen["stereo"], seen["evict"], seen["only"], seen["normalise"]) == (False, False, False, False, ["gong"], True)
    assert seen["player_name"] == "aplay" and seen["latency_ms"] == 20


def test_without_an_interval_live_mode_is_run_live_as_before(tmp_path, monkeypatch):
    import tensixinet.cli as cli
    monkeypatch.setattr(cli, "run_fast", lambda *a, **k: pytest.fail("no --interval, so no fast loop"))
    code, _, asked = live(tmp_path)
    assert code == 0 and asked and len(asked.channels) == 1


def test_an_interval_on_real_chips_reads_through_the_fast_source_and_stops_it(tmp_path, monkeypatch):
    import tensixinet.cli as cli
    made = []

    class FakeFast:
        def __init__(self, fetch=None, **kw):
            self.fetch, self.started, self.stopped = fetch, 0, 0
            self.fast, self.note = True, "Readings come from hwmon (2 chips), with tt-smi for the rest."
            made.append(self)

        def start(self):
            self.started += 1

        def stop(self):
            self.stopped += 1

        def sample(self):
            return mock_telemetry(2)

    monkeypatch.setattr(cli, "FastSource", FakeFast)
    argv = ["--live", "--interval", "100", "--duration", "0.5", "--sample-rate", "8000", "--out", str(tmp_path / "o.wav"),
            "--log-dir", str(tmp_path / "logs"), "--no-banner"]
    code = main(argv, make_sink=lambda player, rate, out, channels=1, **o: (MemorySink(), "a memory sink"))
    assert code == 0 and len(made) == 1 and made[0].started >= 1 and made[0].stopped >= 1
    ev = read_events(tmp_path / "logs")
    assert any("hwmon (2 chips)" in e["what"] for e in ev)


def test_the_fast_source_is_stopped_when_the_player_cannot_start(tmp_path, monkeypatch):
    import tensixinet.cli as cli
    made = []

    class FakeFast:
        fast, note = True, "Readings come from hwmon (2 chips), with tt-smi for the rest."

        def __init__(self, fetch=None, **kw):
            self.stopped = 0
            made.append(self)

        def start(self):
            pass

        def stop(self):
            self.stopped += 1

        def sample(self):
            return mock_telemetry(2)

    def make_sink(player, rate, out, channels=1, **options):
        raise PlayerError("cannot start paplay: not installed")

    monkeypatch.setattr(cli, "FastSource", FakeFast)
    argv = ["--live", "--interval", "100", "--sample-rate", "8000", "--out", str(tmp_path / "o.wav"), "--log-dir", str(tmp_path / "logs"),
            "--no-banner"]
    assert main(argv, make_sink=make_sink) == 3 and made[0].stopped == 1  # its tt-smi thread does not outlive the run


def test_ctrl_c_in_fast_mode_closes_the_wav_and_the_log(tmp_path, monkeypatch):
    import time
    import soundfile as sf
    monkeypatch.setattr(time, "sleep", lambda s: None)  # the null sink would otherwise take as long as the music
    calls = []

    def sample():
        calls.append(1)
        if len(calls) == 8:
            raise KeyboardInterrupt
        return mock_telemetry(2)

    argv = ["--live", "--interval", "100", "--player", "none", "--duration", "0", "--sample-rate", "8000",
            "--out", str(tmp_path / "o.wav"), "--log-dir", str(tmp_path / "logs"), "--no-banner"]
    assert main(argv, sample=sample) == 0
    audio, rate = sf.read(str(tmp_path / "o.wav"))
    assert rate == 8000 and len(audio) == 5 * 800 + 400  # five blocks (one read drew the patch, one began the run) and the tail
    (txt,) = (tmp_path / "logs").glob("*.txt")
    assert "Ctrl-C" in txt.read_text() and read_events(tmp_path / "logs")[-1]["event"] == "done"


def test_a_player_gone_after_ctrl_c_in_fast_mode_still_exits_0_and_is_logged(tmp_path):
    calls = []

    def sample():
        calls.append(1)
        if len(calls) == 8:
            raise KeyboardInterrupt
        return mock_telemetry(2)

    class GoneSink(MemorySink):
        def write(self, audio):  # the player went with the Ctrl-C: the tail, written after it, finds the pipe broken
            if len(self.chunks) >= 5:
                raise PlayerError("paplay stopped before the audio was done")
            super().write(audio)

    sink = GoneSink()
    code, _, _ = fast(tmp_path, "--interval", "100", "--duration", "0", sample=sample, sink=sink)
    assert code == 0 and sink.closed == 1
    assert any(e["event"] == "player" and "after Ctrl-C" in e["what"] for e in read_events(tmp_path / "logs"))


def test_an_interval_under_100_ms_is_logged_at_the_100_ms_the_loop_runs(tmp_path):
    code, _, _ = fast(tmp_path, "--interval", "10", "--mock-chips", "2", "--player", "none")
    ev = read_events(tmp_path / "logs")
    assert code == 0 and not any("every 10 ms" in e["what"] + e["why"] for e in ev)
    # one line states the interval: fast mode's own, which says the interval was raised
    (stated,) = [e for e in ev if e["event"] == "live" and "read every" in e["what"] + e["why"]]
    assert stated["what"].startswith("fast mode") and "--interval 10 was raised to 100 ms, the shortest block" in stated["what"]


def test_an_interval_of_100_ms_states_the_interval_once_and_the_mock_note_names_the_source(tmp_path):
    fast(tmp_path, "--interval", "100", "--mock-chips", "2", "--player", "none")
    ev = read_events(tmp_path / "logs")
    stated = [e["what"] for e in ev if e["event"] == "live" and "every 100 ms" in e["what"] + e["why"]]  # the delay lines give it as a part
    assert stated == ["fast mode: the chips are read every 100 ms, in blocks of 0.10 s"]
    assert any(e["what"] == "The readings are synthetic (--mock-chips): no hardware is read." for e in ev)


def _rate_seen(tmp_path, monkeypatch, *argv):
    """The sample rate the CLI hands to whichever runner it picks (run_fast, run_live or run), and the event log."""
    import tensixinet.cli as cli
    seen = []
    monkeypatch.setattr(cli, "run_fast", lambda source, sink, log, rate, *a, **k: seen.append(("fast", rate)) or 0)
    monkeypatch.setattr(cli, "run_live", lambda sample, sink, log, rate, **k: seen.append(("live", rate)))
    monkeypatch.setattr(cli, "run", lambda chips, duration, out, log, rate, **k: seen.append(("run", rate)))
    rates = []

    def make_sink(player, rate, out, channels=1, **options):
        rates.append(rate)
        return MemorySink(), "a memory sink"

    tmp_path.mkdir(exist_ok=True)
    full = ["--out", str(tmp_path / "o.wav"), "--log-dir", str(tmp_path / "logs"), "--no-banner", "--mock-chips", "2", "--duration", "1", *argv]
    assert main(full, make_sink=make_sink) == 0
    ev = read_events(tmp_path / "logs")
    for f in (tmp_path / "logs").iterdir():
        f.unlink()
    return seen, rates, ev


def test_fast_mode_plays_at_22050_hz_unless_a_rate_is_given_and_says_which_and_why(tmp_path, monkeypatch):
    seen, rates, ev = _rate_seen(tmp_path, monkeypatch, "--live", "--interval", "100")
    assert seen == [("fast", 22050)] and rates == [22050]
    (line,) = [e for e in ev if e["event"] == "live" and "a memory sink" in e["what"]]
    assert "22050 Hz" in line["why"] and "--sample-rate" in line["why"]
    seen, rates, ev = _rate_seen(tmp_path, monkeypatch, "--live", "--interval", "100", "--sample-rate", "44100")
    assert seen == [("fast", 44100)] and rates == [44100]  # a rate given on the command line wins
    (line,) = [e for e in ev if e["event"] == "live" and "a memory sink" in e["what"]]
    assert "44100 Hz" in line["why"] and "as --sample-rate asked" in line["why"]


def test_without_an_interval_the_sample_rate_is_still_44100(tmp_path, monkeypatch):
    seen, rates, _ = _rate_seen(tmp_path, monkeypatch, "--live")
    assert seen == [("live", 44100)] and rates == [44100]
    seen, rates, _ = _rate_seen(tmp_path, monkeypatch)
    assert seen == [("run", 44100)] and rates == []
    seen, _, _ = _rate_seen(tmp_path, monkeypatch, "--live", "--sample-rate", "8000")
    assert seen == [("live", 8000)]


# --- --with-tt-toplike -----------------------------------------------------------------------------------------------------------------------

from tensixinet.media import toplike as toplike_module  # noqa: E402


def toplike(tmp_path, *extra, which=None, exists=False, monkeypatch=None):
    """Run the CLI with --with-tt-toplike and a fake tmux. Returns (exit code, the tmux commands that ran)."""
    ran = []

    own = list(toplike_module.own_tmux())

    def tmux_run(cmd, **kw):
        if cmd[:len(own)] == own:  # outside tmux every command carries the session's own socket and config; the checks below look at the verb
            cmd = ["tmux"] + list(cmd[len(own):])
        ran.append(cmd)
        return 0 if (cmd[1] == "has-session" and exists) else (1 if cmd[1] == "has-session" else 0)

    code = main(["--live", "--with-tt-toplike", "--log-dir", str(tmp_path / "logs"), *extra],
                which=which or (lambda name: f"/bin/{name}"), tmux_run=tmux_run)
    return code, ran


def test_with_tt_toplike_builds_the_session_and_returns_the_attach_code(tmp_path, monkeypatch):
    monkeypatch.delenv("TMUX", raising=False)
    code, ran = toplike(tmp_path, "--player", "none")
    verbs = [c[1] for c in ran]
    assert code == 0 and verbs == ["has-session", "new-session", "split-window", "set-option", "attach-session"]
    assert not (tmp_path / "logs").exists()  # the panes make the log; this process read no chips and wrote nothing
    assert "--interval 100" in ran[1][-1] and "--interval 100" in ran[2][-1]
    assert "--with-tt-toplike" not in ran[1][-1] and "--player none" in ran[1][-1]


def test_with_tt_toplike_uses_the_interval_and_toplike_args_it_was_given(tmp_path, monkeypatch):
    monkeypatch.delenv("TMUX", raising=False)
    _, ran = toplike(tmp_path, "--interval", "40", "--toplike-args", "--mode arcade")
    assert "--interval 40" in ran[1][-1] and "--interval 40" in ran[2][-1] and "--mode arcade" in ran[2][-1]
    assert "--toplike-args" not in ran[1][-1] and "arcade" not in ran[1][-1]


def test_a_new_session_takes_the_size_of_the_terminal_it_will_be_attached_to(tmp_path, monkeypatch):
    import os
    import shutil
    monkeypatch.delenv("TMUX", raising=False)
    monkeypatch.setattr(shutil, "get_terminal_size", lambda *a, **k: os.terminal_size((132, 41)))
    _, ran = toplike(tmp_path)
    new = ran[1]
    assert new[1] == "new-session" and new[new.index("-x") + 1] == "132" and new[new.index("-y") + 1] == "41"
    monkeypatch.setenv("TMUX", "/tmp/tmux-1/default,1,0")  # inside tmux the current window is split, and it has its own size
    _, ran = toplike(tmp_path)
    assert all("-x" not in c for c in ran)


@pytest.mark.parametrize("extra,words", [
    (["--only", "nosuch"], "--only: unknown device nosuch"),
    (["--style", "blues", "--patch", "p.json"], "--style draws a new patch"),
    (["--mock-load", "30"], "--mock-load needs --mock-chips"),
])
def test_the_ordinary_argument_checks_run_before_a_session_is_built(tmp_path, capsys, monkeypatch, extra, words):
    # otherwise the error would first show inside the synth pane, behind "Press Enter"
    monkeypatch.delenv("TMUX", raising=False)
    ran = []
    with pytest.raises(SystemExit) as e:
        main(["--live", "--with-tt-toplike", "--log-dir", str(tmp_path / "logs"), *extra], which=lambda n: f"/bin/{n}",
             tmux_run=lambda c, **k: ran.append(c) or 1)
    assert e.value.code == 2 and words in capsys.readouterr().err and ran == []


def test_inside_tmux_it_splits_without_checking_for_a_session(tmp_path, monkeypatch):
    monkeypatch.setenv("TMUX", "/tmp/tmux-1/default,1,0")
    _, ran = toplike(tmp_path, exists=True)  # a session of that name is no obstacle here
    assert [c[1] for c in ran] == ["split-window", "split-window", "set-option"]


@pytest.mark.parametrize("extra,which,exists,words", [
    ([], lambda n: "/bin/tt-toplike" if n == "tt-toplike" else None, False, "tmux"),
    ([], lambda n: "/bin/tmux" if n == "tmux" else None, False, "tt-toplike"),
    (["--mock-chips", "2"], None, False, "different chips"),
    (["--kernels", "host"], None, False, "--kernels"),
    ([], None, True, "tensixinet"),
])
def test_each_refusal_is_a_parser_error_with_a_sentence(tmp_path, capsys, monkeypatch, extra, which, exists, words):
    monkeypatch.delenv("TMUX", raising=False)
    with pytest.raises(SystemExit) as e:
        toplike(tmp_path, *extra, which=which, exists=exists)
    assert e.value.code == 2 and words in capsys.readouterr().err


def test_an_existing_session_message_says_how_to_end_it(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("TMUX", raising=False)
    with pytest.raises(SystemExit):
        toplike(tmp_path, exists=True)
    err = capsys.readouterr().err
    assert "tmux -L tensixinet kill-session -t tensixinet" in err and "tmux -L tensixinet attach -t tensixinet" in err  # the session is on its own socket


def raw_toplike_commands(tmp_path):
    """Every tmux command the CLI runs for --with-tt-toplike, exactly as given to tmux (the socket prefix kept)."""
    ran = []
    main(["--live", "--with-tt-toplike", "--log-dir", str(tmp_path / "logs")], which=lambda n: f"/bin/{n}",
         tmux_run=lambda c, **k: ran.append(list(c)) or (1 if c[-3:-2] == ["has-session"] or "has-session" in c else 0))
    return ran


def test_outside_tmux_the_session_uses_the_recording_style_tmux_config(tmp_path, monkeypatch):
    monkeypatch.delenv("TMUX", raising=False)
    ran = raw_toplike_commands(tmp_path)
    own = list(toplike_module.own_tmux())
    assert len(ran) == 5 and all(c[:len(own)] == own for c in ran)  # the has-session check too, or it would look at the wrong server


def test_inside_tmux_the_owners_own_server_is_left_alone(tmp_path, monkeypatch):
    monkeypatch.setenv("TMUX", "/tmp/tmux-1/default,1,0")
    ran = raw_toplike_commands(tmp_path)
    assert ran and all(c[0] == "tmux" and "-L" not in c and "-f" not in c[:4] for c in ran)


def test_with_tt_toplike_needs_live(tmp_path, capsys):
    with pytest.raises(SystemExit) as e:
        main(["--with-tt-toplike", "--log-dir", str(tmp_path)], which=lambda n: "/bin/x", tmux_run=lambda *a, **k: 0)
    assert e.value.code == 2 and "--with-tt-toplike needs --live" in capsys.readouterr().err


def test_toplike_args_without_the_flag_are_refused(tmp_path, capsys):
    with pytest.raises(SystemExit) as e:
        main(["--live", "--toplike-args", "--mode arcade", "--mock-chips", "2"])
    assert e.value.code == 2 and "--toplike-args needs --with-tt-toplike" in capsys.readouterr().err


def test_an_abbreviated_flag_is_an_unknown_argument(capsys):
    # --with-tt used to parse as --with-tt-toplike, and the synth pane (which strips only the full spelling) would start sessions without end
    from tensixinet.cli import build_parser
    for flag in ("--with-tt", "--toplike-a", "--inter"):
        with pytest.raises(SystemExit) as e:
            build_parser().parse_args(["--live", flag, "5"])
        assert e.value.code == 2
    assert "unrecognized" in capsys.readouterr().err


def test_with_tt_toplike_inside_its_own_session_is_refused(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("TENSIXINET_IN_SESSION", "1")
    monkeypatch.setenv("TMUX", "/tmp/tmux-1/default,1,0")
    ran = []
    with pytest.raises(SystemExit) as e:
        main(["--live", "--with-tt-toplike"], which=lambda n: "/bin/x", tmux_run=lambda c, **k: ran.append(c) or 0)
    assert e.value.code == 2 and ran == []
    assert "started inside its own session; it cannot start another" in capsys.readouterr().err


def test_the_session_commands_use_the_found_toplike_path_and_the_callers_path(tmp_path, monkeypatch):
    monkeypatch.delenv("TMUX", raising=False)
    monkeypatch.setenv("PATH", "/caller/bin")
    _, ran = toplike(tmp_path, which=lambda n: f"/found/{n}")
    assert "/found/tt-toplike --interval 100" in ran[2][-1]
    assert "PATH=/caller/bin" in ran[1] and "TENSIXINET_IN_SESSION=1" in ran[1]


# --- --tui: the full-screen view of what is heard ------------------------------------------------------------------------------

import re  # noqa: E402
import time  # noqa: E402

from tensixinet.tui.playhead import Playhead, TimedSink  # noqa: E402
from tests.tui.test_app import FakeTerminal  # noqa: E402
from tests.tui.test_frame import Screen  # noqa: E402


class SlowSink(MemorySink):
    """A memory sink that takes a moment over each write, so the TUI's thread draws frames while the run plays."""

    def __init__(self, pause=0.25, on_first=None):
        super().__init__()
        self.pause, self.on_first = pause, on_first

    def write(self, audio):
        if not self.chunks and self.on_first is not None:
            self.on_first()
        super().write(audio)
        time.sleep(self.pause)


def _split_at_leave(term):
    """The terminal's writes before its first `leave` and after it, each joined."""
    before, after, left = [], [], False
    for kind, text in term.snapshot():
        if kind == "leave":
            left = True
        elif kind == "write":
            (after if left else before).append(text)
    return "".join(before), "".join(after)


def screen(term):
    """What the alternate screen showed last, as plain text: the TUI's writes up to `leave`, fed through a small emulator."""
    w, h = term.size_
    s = Screen(w, h)
    s.feed(_split_at_leave(term)[0])
    return "\n".join("".join(row) for row in s.grid)


def after_leave(term):
    return _split_at_leave(term)[1]


def txt_log(tmp_path):
    (txt,) = (tmp_path / "logs").glob("*.txt")
    return txt.read_text()


def test_tui_needs_live(tmp_path, capsys):
    with pytest.raises(SystemExit) as e:
        main(args(tmp_path, "--mock-chips", "2", "--tui"))
    assert e.value.code == 2
    assert "--tui needs --live: it draws what is playing as it plays, and an offline run plays nothing" in capsys.readouterr().err


def test_a_tui_run_with_mock_chips_draws_and_writes_the_wav_and_logs(tmp_path):
    term = FakeTerminal()
    code, sink, _ = live(tmp_path, "--tui", sink=SlowSink(), terminal=term)
    assert code == 0 and term.entered == 1 and term.left == 1
    shown = screen(term)
    assert "segment 0" in shown and "has the floor" in shown and "●0" in shown and "●1" in shown  # the heading and both chips
    assert sink.closed == 1 and sink.chunks  # the memory sink stands in for the player, and got the audio
    assert list((tmp_path / "logs").glob("*.txt")) and list((tmp_path / "logs").glob("*.jsonl"))
    assert str(tmp_path / "logs") in after_leave(term)  # the exit lines are on the normal screen


def test_in_tui_mode_the_plain_log_goes_only_to_the_files(tmp_path, capsys):
    live(tmp_path, "--tui", "--color", "always", terminal=FakeTerminal())
    captured = capsys.readouterr()
    assert captured.out == "" and "║ [00:" not in captured.err  # neither the coloured log nor the banner reaches the screen
    txt = txt_log(tmp_path)
    assert "segment 0" in txt and "\x1b" not in txt
    assert not txt.startswith("║ [00:") and txt.index("\n\n") < txt.index("║ [00:")  # the banner went to the text log, ahead of the first line


def test_with_no_banner_a_tui_runs_text_log_starts_at_the_first_line(tmp_path):
    live(tmp_path, "--tui", "--no-banner", terminal=FakeTerminal())
    assert txt_log(tmp_path).startswith("║ [00:")


def test_the_tui_log_has_every_record_the_jsonl_has(tmp_path, monkeypatch):
    import tensixinet.cli as cli
    made = []

    class Spy(Playhead):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.noted = []
            made.append(self)

        def note(self, record):
            self.noted.append(record)
            super().note(record)

    monkeypatch.setattr(cli, "Playhead", Spy)
    code, _, _ = live(tmp_path, "--tui", terminal=FakeTerminal())
    assert code == 0 and len(made) == 1
    assert made[0].noted == read_events(tmp_path / "logs") and len(made[0].noted) > 3


def test_tui_hands_the_live_loop_the_playheads_hook_a_timed_sink_and_counts_segments(tmp_path, monkeypatch):
    import tensixinet.cli as cli
    seen, apps, heads = {}, [], []

    def fake_live(sample, sink, log, sample_rate, **kw):
        seen.update(sink=sink, **kw)
        return 1

    class SpyApp(cli.TuiApp):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            apps.append(k)

    class SpyHead(Playhead):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            heads.append(self)

    monkeypatch.setattr(cli, "run_live", fake_live)
    monkeypatch.setattr(cli, "TuiApp", SpyApp)
    monkeypatch.setattr(cli, "Playhead", SpyHead)
    code, sink, _ = live(tmp_path, "--tui", terminal=FakeTerminal())
    assert code == 0 and len(heads) == 1
    hook = seen["on_segment"]
    assert hook.__self__ is heads[0] and hook.__func__ is Playhead.hand_over
    assert isinstance(seen["sink"], TimedSink) and seen["sink"]._inner is sink and seen["sink"]._playhead is heads[0]
    assert apps[0]["unit"] == "segment" and apps[0]["log_dir"] == str(tmp_path / "logs")
    assert heads[0]._buffer_s == 0.0  # plain live mode does not know the player's buffer


def test_without_tui_the_live_loop_gets_no_hook_and_no_terminal_is_made(tmp_path, monkeypatch):
    import tensixinet.cli as cli
    seen = {}

    def fake_live(sample, sink, log, sample_rate, **kw):
        seen.update(sink=sink, **kw)
        return 1

    monkeypatch.setattr(cli, "run_live", fake_live)
    monkeypatch.setattr(cli, "TtyTerminal", lambda *a, **k: pytest.fail("no --tui, so no terminal is taken over"))
    code, sink, _ = live(tmp_path)
    assert code == 0 and seen["sink"] is sink and seen.get("on_segment") is None


def test_without_a_terminal_given_tui_takes_the_real_one(tmp_path, monkeypatch):
    import tensixinet.cli as cli
    made = []

    def factory():
        made.append(FakeTerminal())
        return made[-1]

    monkeypatch.setattr(cli, "TtyTerminal", factory)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)  # a real run's --tui needs its output to be a terminal
    code, _, _ = live(tmp_path, "--tui")
    assert code == 0 and len(made) == 1 and made[0].entered == 1 and made[0].left == 1


def _spy_fast(monkeypatch):
    """run_fast replaced by a spy, and the Playhead and TuiApp the CLI makes recorded. Returns (what run_fast got, the apps' keywords, the heads)."""
    import tensixinet.cli as cli
    seen, apps, heads = {}, [], []

    def fake_fast(source, sink, log, sample_rate, interval_s, duration, patch, patch_source, **kw):
        seen.update(sink=sink, **kw)
        return 1

    class SpyApp(cli.TuiApp):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            apps.append(k)

    class SpyHead(Playhead):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            heads.append(self)

    monkeypatch.setattr(cli, "run_fast", fake_fast)
    monkeypatch.setattr(cli, "run_live", lambda *a, **k: pytest.fail("--interval must not call run_live"))
    monkeypatch.setattr(cli, "TuiApp", SpyApp)
    monkeypatch.setattr(cli, "Playhead", SpyHead)
    return seen, apps, heads


def test_tui_with_interval_passes_the_hook_and_counts_phrases(tmp_path, monkeypatch):
    seen, apps, heads = _spy_fast(monkeypatch)
    sink = MemorySink()
    code, _, _ = fast(tmp_path, "--interval", "100", "--tui", sink=sink, terminal=FakeTerminal())
    assert code == 0 and len(heads) == 1
    hook = seen["on_segment"]
    assert hook.__self__ is heads[0] and hook.__func__ is Playhead.hand_over
    assert isinstance(seen["sink"], TimedSink) and seen["sink"]._inner is sink and seen["sink"]._playhead is heads[0]
    assert apps[0]["unit"] == "phrase"  # fast mode's view counts phrases; plain live mode's counts segments (the test above)


def test_the_playheads_buffer_is_the_delay_reports_in_fast_mode_and_zero_otherwise(tmp_path, monkeypatch):
    import tensixinet.cli as cli
    real_live = cli.run_live
    seen, _, heads = _spy_fast(monkeypatch)
    fast(tmp_path / "a", "--interval", "100", "--tui", "--player", "paplay", terminal=FakeTerminal())
    assert heads[-1]._buffer_s == pytest.approx(0.020)  # paplay is asked for PLAYER_LATENCY_MS, the buffer the delay report counts
    assert seen["player_name"] == "paplay" and seen["latency_ms"] == 20
    fast(tmp_path / "b", "--interval", "100", "--tui", "--player", "none", terminal=FakeTerminal())
    assert heads[-1]._buffer_s == 0.0  # no player, no buffer
    monkeypatch.setattr(cli, "run_live", real_live)
    code, _, _ = live(tmp_path / "c", "--tui", terminal=FakeTerminal())
    assert code == 0 and heads[-1]._buffer_s == 0.0  # plain live mode does not know the player's buffer


def test_without_tui_fast_mode_gets_no_hook_and_its_own_sink(tmp_path, monkeypatch):
    seen, apps, heads = _spy_fast(monkeypatch)
    sink = MemorySink()
    code, _, _ = fast(tmp_path, "--interval", "100", sink=sink)
    assert code == 0 and seen["sink"] is sink and seen.get("on_segment") is None and not apps and not heads


def test_a_tui_run_in_fast_mode_draws_phrases_and_writes_the_wav_and_logs(tmp_path):
    term = FakeTerminal()
    code, sink, _ = fast(tmp_path, "--interval", "100", "--tui", "--duration", "3", sink=SlowSink(pause=0.05), terminal=term)
    assert code == 0 and term.entered == 1 and term.left == 1 and sink.closed == 1 and sink.chunks
    shown = screen(term)
    assert "phrase " in shown and "has the floor" in shown and "●0" in shown and "●1" in shown
    assert list((tmp_path / "logs").glob("*.jsonl")) and str(tmp_path / "logs") in after_leave(term)


def test_with_tt_toplike_and_tui_the_synth_pane_runs_the_tui_and_the_session_has_two_panes(tmp_path, monkeypatch):
    monkeypatch.delenv("TMUX", raising=False)
    code, ran = toplike(tmp_path, "--tui", "--player", "none")
    verbs = [c[1] for c in ran]
    assert code == 0 and verbs == ["has-session", "new-session", "split-window", "set-option", "attach-session"]  # two panes
    synth, toplike_pane = ran[1][-1], ran[2][-1]
    assert "--tui" in synth and "--interval 100" in synth and "--with-tt-toplike" not in synth
    assert "--tui" not in toplike_pane
    assert not (tmp_path / "logs").exists()  # the launcher took no terminal and wrote no log; the synth pane does both


def test_tui_keeps_the_midi_rules_of_the_loop(tmp_path, capsys):
    with pytest.raises(SystemExit) as e:
        main(["--live", "--tui", "--interval", "100", "--midi-file", str(tmp_path / "x.mid"), "--mock-chips", "2"])
    assert e.value.code == 2 and "--interval cannot be used with --midi-file or --midi-port yet" in capsys.readouterr().err
    # plain live mode keeps its MIDI file with the TUI
    code, _, _ = live(tmp_path / "m", "--tui", "--midi-file", str(tmp_path / "y.mid"), terminal=FakeTerminal())
    assert code == 0 and (tmp_path / "y.mid").exists()


def test_q_in_the_tui_ends_a_live_run_like_ctrl_c(tmp_path):
    term = FakeTerminal()
    sink = SlowSink(pause=0.05, on_first=lambda: term.keys.put("q"))  # pressed once the music is playing
    code, _, _ = live(tmp_path, "--tui", "--duration", "0", sink=sink, terminal=term)
    assert code == 0 and term.left == 1 and sink.closed == 1
    assert any(e["what"] == "stopped by Ctrl-C" for e in read_events(tmp_path / "logs"))  # the run's own Ctrl-C path, which writes the logs
    assert "tensixinet: played" in after_leave(term) and str(tmp_path / "logs") in after_leave(term)  # the run's last line, on the normal screen


def test_a_tui_run_with_no_chips_leaves_the_alternate_screen_before_the_error(tmp_path):
    term = FakeTerminal()
    code, _, _ = live(tmp_path, "--tui", sample=lambda: [], terminal=term)
    assert code == 2 and term.left == 1
    out = after_leave(term)
    assert "no chips found" in out and str(tmp_path / "logs") in out
    assert "no chips found" not in _split_at_leave(term)[0]  # not drawn under the alternate screen only


def test_a_run_that_ends_before_the_tui_started_still_prints_the_exit_lines(tmp_path):
    def make_sink(player, rate, out, channels=1, **options):
        raise PlayerError("cannot start paplay: not installed")
    term = FakeTerminal()
    code, _, _ = live(tmp_path, "--tui", make_sink=make_sink, terminal=term)
    out = "".join(term.writes)
    assert code == 3 and term.entered == 0
    assert "the audio output could not be used" in out and str(tmp_path / "logs") in out


def test_an_exception_in_a_tui_run_restores_the_terminal_prints_it_and_is_raised(tmp_path, monkeypatch):
    import tensixinet.cli as cli

    def boom(*a, **k):
        raise RuntimeError("the loop broke")

    monkeypatch.setattr(cli, "run_live", boom)
    term = FakeTerminal()
    with pytest.raises(RuntimeError):
        live(tmp_path, "--tui", terminal=term)
    assert term.left == 1 and "RuntimeError: the loop broke" in after_leave(term)
    assert str(tmp_path / "logs") in after_leave(term)


def test_the_tui_closes_before_the_log(tmp_path, monkeypatch):
    import tensixinet.cli as cli
    from tensixinet.narrate.runlog import RunLog
    order = []
    real_close, real_log_close = cli.TuiApp.close, RunLog.close
    monkeypatch.setattr(cli.TuiApp, "close", lambda self, code: order.append(("tui", code)) or real_close(self, code))
    monkeypatch.setattr(RunLog, "close", lambda self: order.append(("log",)) or real_log_close(self))
    live(tmp_path, "--tui", sample=lambda: [], terminal=FakeTerminal())
    assert order == [("tui", 2), ("log",)]


def test_tui_respects_no_color(tmp_path, monkeypatch):
    sgr_colour = re.compile(r"\x1b\[[0-9;]*38;")
    monkeypatch.setenv("FORCE_COLOR", "1")  # the captured stdout is not a terminal; this makes "auto" colour it
    monkeypatch.delenv("NO_COLOR", raising=False)
    coloured = FakeTerminal()
    live(tmp_path / "a", "--tui", "--color", "auto", sink=SlowSink(), terminal=coloured)
    assert sgr_colour.search("".join(coloured.writes))
    monkeypatch.setenv("NO_COLOR", "1")
    plain = FakeTerminal()
    live(tmp_path / "b", "--tui", "--color", "auto", sink=SlowSink(), terminal=plain)
    assert not sgr_colour.search("".join(plain.writes)) and "has the floor" in screen(plain)


def test_tui_works_with_kernels_host(tmp_path):
    worker = _Worker(2)
    term = FakeTerminal(size=(160, 400))  # tall enough that the opening lines have not scrolled off
    code, _, _ = live(tmp_path, "--tui", "--kernels", "host", sink=SlowSink(), terminal=term,
                      make_worker=lambda mode, chips: worker)
    assert code == 0 and worker.closed and term.left == 1
    assert "TT-Lang kernels on" in screen(term)


# --- --tui: q and Ctrl-C end the same way; the terminal it needs ---------------------------------------------------------------

def test_with_tui_the_player_runs_in_its_own_session_and_without_it_as_before(tmp_path):
    # A player in Python's process group dies of a terminal Ctrl-C; in its own session only Python gets it, so Ctrl-C and q drain alike.
    _, _, asked = live(tmp_path / "a", "--tui", terminal=FakeTerminal())
    assert asked.options == [{"own_session": True}]
    _, _, asked = live(tmp_path / "b")
    assert asked.options == [{}]


class _DiesWithCtrlC(MemorySink):
    """A player that shares the terminal's Ctrl-C: once the run is interrupted, every write fails as a dead paplay's would."""

    def __init__(self, interrupted):
        super().__init__()
        self.interrupted = interrupted

    def write(self, audio):
        if self.interrupted:
            raise PlayerError("paplay stopped before the audio was done")
        super().write(audio)


def _ctrl_c_on_the_third_read(interrupted):
    reads = []

    def sample():
        reads.append(1)
        if len(reads) == 3:
            interrupted.append(1)
            raise KeyboardInterrupt
        return mock_telemetry(2)
    return sample


def test_a_player_that_dies_with_the_ctrl_c_ends_the_run_as_ctrl_c_does(tmp_path):
    interrupted = []
    code, _, _ = live(tmp_path, "--duration", "0", sample=_ctrl_c_on_the_third_read(interrupted), sink=_DiesWithCtrlC(interrupted))
    assert code == 130  # a normal end by Ctrl-C, not exit code 3
    events = read_events(tmp_path / "logs")
    assert any(e["what"] == "stopped by Ctrl-C" for e in events)
    assert any(e["event"] == "player" and "after Ctrl-C" in e["what"] for e in events)


def test_with_tui_a_player_that_dies_with_the_ctrl_c_prints_no_error(tmp_path):
    interrupted = []
    term = FakeTerminal()
    code, _, _ = live(tmp_path, "--tui", "--duration", "0", sample=_ctrl_c_on_the_third_read(interrupted),
                      sink=_DiesWithCtrlC(interrupted), terminal=term)
    out = after_leave(term)
    assert code == 130 and term.left == 1
    assert "why:" not in out and out.count("\n") == 1 and str(tmp_path / "logs") in out  # one exit line, no error dump


def test_a_player_error_with_no_ctrl_c_is_still_exit_code_3(tmp_path):
    code, _, _ = live(tmp_path, sink=_DiesWithCtrlC([1]))
    assert code == 3


def test_the_tui_is_disarmed_once_the_run_loop_returns_and_before_the_chips_are_released(tmp_path, monkeypatch):
    import tensixinet.cli as cli
    order = []

    class SpyApp(cli.TuiApp):
        def disarm(self):
            order.append("disarm")
            super().disarm()

    class Worker:
        def start(self):
            pass

        def close(self):
            order.append("release")

    def fake_live(*a, **k):
        order.append("run")
        return 1

    monkeypatch.setattr(cli, "TuiApp", SpyApp)
    monkeypatch.setattr(cli, "run_live", fake_live)
    code, _, _ = live(tmp_path, "--tui", "--kernels", "chip", terminal=FakeTerminal(), make_worker=lambda mode, chips: Worker())
    assert code == 0 and order[:3] == ["run", "disarm", "release"]


def test_fast_mode_sets_the_playheads_buffer_through_set_buffer(tmp_path, monkeypatch):
    seen, _, heads = _spy_fast(monkeypatch)
    calls = []
    real = Playhead.set_buffer
    monkeypatch.setattr(Playhead, "set_buffer", lambda self, s: calls.append(s) or real(self, s))
    fast(tmp_path, "--interval", "100", "--tui", "--player", "paplay", terminal=FakeTerminal())
    assert calls == [pytest.approx(0.020)] and heads[-1].heard_s() is None  # the public setter, not the private field


def test_tui_refuses_when_its_output_is_not_a_terminal(tmp_path, capsys):
    # The captured stdout here is not a terminal, as a pipe or a file is not; no terminal is handed in.
    with pytest.raises(SystemExit) as e:
        main(["--live", "--tui", "--mock-chips", "2", "--duration", "1", "--log-dir", str(tmp_path / "logs"), "--out", str(tmp_path / "o.wav")])
    err = capsys.readouterr().err
    assert e.value.code == 2 and "--tui needs a terminal" in err
    assert not (tmp_path / "logs").exists()  # refused before any log was made


def test_inside_tmux_with_tui_the_synth_and_tt_toplike_get_a_new_window_of_their_own(tmp_path, monkeypatch):
    monkeypatch.setenv("TMUX", "/tmp/tmux-1000/default,1,0")
    code, ran = toplike(tmp_path, "--tui", "--player", "none")
    assert code == 0 and [c[1] for c in ran] == ["new-window", "split-window", "set-option"]
    assert "--tui" in ran[0][-1]
    code, ran = toplike(tmp_path, "--player", "none")  # without --tui the current window is split as before
    assert [c[1] for c in ran] == ["split-window", "split-window", "set-option"]
