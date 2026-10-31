# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Where live audio goes: a player process, a WAV file, or nowhere at the right speed."""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from tensixinet.media.player import (
    NullSink,
    make_sink,
    PlayerError,
    PlayerSink,
    TeeSink,
    WavSink,
    find_player,
    pcm16,
    player_command,
    player_buffer_ms,
    FFPLAY_BUFFER_MS,
    LATENCY_FLAGS,
)

SR = 8000


# --- float audio to the bytes a player takes --------------------------------------------------------------------

def test_pcm16_is_little_endian_signed_sixteen_bit_with_full_scale_at_one():
    data = pcm16(np.array([0.0, 1.0, -1.0, 0.5], dtype=np.float32))
    assert np.frombuffer(data, dtype="<i2").tolist() == [0, 32767, -32767, 16384]


def test_pcm16_clips_instead_of_wrapping():
    assert np.frombuffer(pcm16(np.array([2.0, -3.0])), dtype="<i2").tolist() == [32767, -32767]


# --- choosing a player -------------------------------------------------------------------------------------------

def test_each_known_player_is_told_the_format_sample_rate_and_mono():
    assert player_command("paplay", 44100) == ["paplay", "--raw", "--format=s16le", "--rate=44100", "--channels=1"]
    assert player_command("aplay", 44100) == ["aplay", "-q", "-t", "raw", "-f", "S16_LE", "-r", "44100", "-c", "1"]
    assert player_command("ffplay", 22050) == [
        "ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", "-f", "s16le", "-ar", "22050", "-ac", "1", "-i", "pipe:0"]


def test_an_unknown_player_is_an_error():
    with pytest.raises(ValueError):
        player_command("winamp", 44100)


def test_find_player_takes_the_first_one_installed_and_reports_none_when_there_is_none():
    have = {"aplay", "ffplay"}
    which = lambda name: f"/usr/bin/{name}" if name in have else None
    assert find_player(SR, which=which)[0] == "aplay"
    assert find_player(SR, which=lambda name: None) is None
    assert find_player(SR, which=lambda name: "/x/paplay" if name == "paplay" else None)[0] == "paplay"
    assert find_player(SR, which=lambda name: f"/x/{name}")[0] == "paplay"  # all three installed: the first of the list


# --- the sinks ---------------------------------------------------------------------------------------------------

def test_a_player_sink_hands_the_bytes_to_a_real_process(tmp_path):
    out = tmp_path / "heard.raw"
    sink = PlayerSink(["sh", "-c", f"cat > {out}"])
    audio = np.array([0.0, 0.25, -0.5, 1.0], dtype=np.float32)
    sink.write(audio)
    sink.write(audio)
    sink.close()
    assert out.read_bytes() == pcm16(audio) * 2


def test_closing_a_player_sink_waits_for_the_player_to_finish_playing(tmp_path):
    marker = tmp_path / "done"
    sink = PlayerSink(["sh", "-c", f"cat > /dev/null; sleep 0.3; touch {marker}"])
    sink.write(np.zeros(100, dtype=np.float32))
    sink.close()
    assert marker.exists()


def test_a_player_runs_in_a_session_of_its_own_so_a_terminal_ctrl_c_reaches_only_python():
    sink = PlayerSink(["sh", "-c", "cat > /dev/null"], own_session=True)
    try:
        pid = sink._proc.pid
        assert os.getsid(pid) == pid != os.getsid(0)  # its own session, so its own process group: the terminal's SIGINT is not sent to it
    finally:
        sink.close()


def test_a_ctrl_c_to_the_whole_process_group_leaves_the_player_playing(tmp_path):
    # A child Python starts a player, then sends SIGINT to its own process group, as a terminal's Ctrl-C does (or tmux's send-keys C-c). The
    # player must still take the audio written after it and finish. The child runs in a session of its own, so the signal stays inside it.
    out = tmp_path / "heard.raw"
    script = (
        "import os, signal, time, numpy as np\n"
        "from tensixinet.media.player import PlayerSink\n"
        f"sink = PlayerSink(['sh', '-c', 'cat > {out}'], own_session=True)\n"
        "try:\n"
        "    os.killpg(os.getpgrp(), signal.SIGINT)\n"
        "    time.sleep(2)\n"
        "except KeyboardInterrupt:\n"
        "    pass\n"
        "time.sleep(0.2)\n"
        "sink.write(np.full(4, 0.5, dtype=np.float32))\n"
        "sink.close()\n"
    )
    done = subprocess.run([sys.executable, "-c", script], start_new_session=True, capture_output=True, text=True, timeout=30,
                          cwd=Path(__file__).resolve().parents[2])
    assert done.returncode == 0, done.stderr
    assert out.read_bytes() == pcm16(np.full(4, 0.5, dtype=np.float32))


def test_a_player_that_exits_early_is_reported_not_ignored():
    sink = PlayerSink(["true"])
    with pytest.raises(PlayerError):
        for _ in range(200):  # the pipe breaks as soon as the process is gone
            sink.write(np.zeros(SR, dtype=np.float32))
    sink.close()


def test_a_player_that_is_not_installed_is_reported_when_the_sink_is_made():
    with pytest.raises(PlayerError):
        PlayerSink(["definitely-not-a-player-binary"])


def test_a_null_sink_takes_as_long_as_the_audio_would_to_play():
    slept = []
    sink = NullSink(SR, sleep=slept.append)
    sink.write(np.zeros(2 * SR, dtype=np.float32))
    sink.write(np.zeros(SR // 2, dtype=np.float32))
    sink.close()
    assert slept == [2.0, 0.5]


def test_a_wav_sink_writes_what_it_was_given_as_one_file_of_sixteen_bit_audio(tmp_path):
    path = tmp_path / "live.wav"
    sink = WavSink(path, SR)
    a, b = np.linspace(-0.5, 0.5, 100, dtype=np.float32), np.linspace(0.5, -0.5, 60, dtype=np.float32)
    sink.write(a)
    sink.write(b)
    sink.close()
    data, rate = sf.read(str(path), dtype="float32")
    assert rate == SR and sf.info(str(path)).subtype == "PCM_16" and len(data) == 160
    assert np.allclose(data, np.concatenate([a, b]), atol=1e-4)


def test_a_tee_sink_feeds_every_sink_and_closes_every_sink_even_if_one_fails():
    class Spy:
        def __init__(self, fail=False):
            self.got, self.closed, self.fail = [], False, fail
        def write(self, audio):
            self.got.append(len(audio))
        def close(self):
            self.closed = True
            if self.fail:
                raise PlayerError("boom")
    a, b = Spy(fail=True), Spy()
    tee = TeeSink(a, b)
    tee.write(np.zeros(5, dtype=np.float32))
    assert a.got == b.got == [5]
    with pytest.raises(PlayerError):
        tee.close()
    assert a.closed and b.closed


# --- make_sink: the sink for a --player choice, always recording the WAV -------------------------------------------

class FakePlayer:
    def __init__(self, command):
        self.command, self.got, self.closed = list(command), [], False

    def write(self, audio):
        self.got.append(len(audio))

    def close(self):
        self.closed = True


def _installed(*names):
    return lambda name: f"/usr/bin/{name}" if name in names else None


def test_a_named_player_is_started_with_its_command_and_the_wav_records_what_it_plays(tmp_path):
    started = []
    sink, where = make_sink("aplay", SR, tmp_path / "heard.wav", which=_installed("aplay"),
                            start=lambda cmd: started.append(FakePlayer(cmd)) or started[-1])
    audio = np.linspace(-0.5, 0.5, 80, dtype=np.float32)
    sink.write(audio)
    sink.close()
    assert started[0].command == player_command("aplay", SR) and started[0].got == [80] and started[0].closed
    assert len(sf.read(str(tmp_path / "heard.wav"))[0]) == 80
    assert "aplay" in where and "heard.wav" in where


def test_auto_takes_the_first_installed_player(tmp_path):
    started = []
    make_sink("auto", SR, tmp_path / "h.wav", which=_installed("ffplay", "aplay"), start=lambda cmd: started.append(cmd) or FakePlayer(cmd))
    assert started == [player_command("aplay", SR)]


def test_auto_with_no_player_installed_keeps_time_in_silence_and_says_so(tmp_path):
    started = []
    sink, where = make_sink("auto", SR, tmp_path / "h.wav", which=_installed(), start=lambda cmd: started.append(cmd))
    assert not started and "no audio player" in where and "paplay" in where and "h.wav" in where
    assert any(isinstance(s, NullSink) for s in sink.sinks) and any(isinstance(s, WavSink) for s in sink.sinks)
    sink.close()


def test_none_means_no_sound_and_never_looks_for_a_player(tmp_path):
    sink, where = make_sink("none", SR, tmp_path / "h.wav", which=lambda name: pytest.fail("looked for a player"), start=lambda cmd: pytest.fail("started one"))
    assert "no sound" in where and any(isinstance(s, NullSink) for s in sink.sinks)
    sink.close()


def test_a_named_player_that_is_not_installed_is_an_error_that_names_it(tmp_path):
    with pytest.raises(PlayerError, match="ffplay"):
        make_sink("ffplay", SR, tmp_path / "h.wav", which=_installed("aplay"), start=lambda cmd: FakePlayer(cmd))
    assert not (tmp_path / "h.wav").exists()  # nothing is recorded for a run that never starts


def test_a_player_that_will_not_start_does_not_leave_the_wav_open(tmp_path, monkeypatch):
    closed = []
    real = WavSink.close
    monkeypatch.setattr(WavSink, "close", lambda self: closed.append(1) or real(self))

    def refuses(command):
        raise PlayerError("cannot start aplay: permission denied")

    with pytest.raises(PlayerError, match="permission denied"):
        make_sink("aplay", SR, tmp_path / "h.wav", which=_installed("aplay"), start=refuses)
    assert closed == [1]


def test_the_wav_gets_each_chunk_before_the_player_does_so_a_dying_player_loses_nothing_from_the_file(tmp_path):
    class Dies(FakePlayer):
        def write(self, audio):
            raise PlayerError("aplay stopped")

    sink, _ = make_sink("aplay", SR, tmp_path / "h.wav", which=_installed("aplay"), start=lambda cmd: Dies(cmd))
    with pytest.raises(PlayerError):
        sink.write(np.linspace(-0.5, 0.5, 80, dtype=np.float32))
    sink.close()
    assert len(sf.read(str(tmp_path / "h.wav"))[0]) == 80


# --- two channels --------------------------------------------------------------------------------------------------------------------

def test_stereo_audio_goes_to_a_player_as_interleaved_left_right_samples():
    data = pcm16(np.array([[1.0, -1.0], [0.5, 0.0]], dtype=np.float32))
    assert np.frombuffer(data, dtype="<i2").tolist() == [32767, -32767, 16384, 0]     # L, R, L, R


def test_the_players_are_asked_for_two_channels_when_the_audio_has_two():
    assert player_command("paplay", 44100, channels=2)[-1] == "--channels=2"
    assert player_command("aplay", 44100, channels=2)[-2:] == ["-c", "2"]
    assert player_command("ffplay", 44100, channels=2)[player_command("ffplay", 44100, channels=2).index("-ac") + 1] == "2"
    assert player_command("paplay", 44100) == player_command("paplay", 44100, channels=1)       # one channel is still the default
    assert find_player(44100, which=lambda n: "/usr/bin/" + n if n == "aplay" else None, channels=2)[-2:] == ["-c", "2"]


def test_a_stereo_wav_sink_records_both_channels(tmp_path):
    sink = WavSink(tmp_path / "s.wav", SR, channels=2)
    sink.write(np.stack([np.full(100, 0.25), np.full(100, -0.25)], axis=1).astype(np.float32))
    sink.close()
    data, rate = sf.read(str(tmp_path / "s.wav"))
    assert rate == SR and data.shape == (100, 2) and data[0, 0] == pytest.approx(0.25, abs=1e-3) and data[0, 1] == pytest.approx(-0.25, abs=1e-3)


def test_a_null_sink_paces_stereo_by_its_frames_not_its_samples():
    slept = []
    NullSink(SR, sleep=slept.append).write(np.zeros((SR, 2), dtype=np.float32))
    assert slept == [pytest.approx(1.0)]


def test_make_sink_asks_the_player_and_the_wav_for_the_channels_it_was_given(tmp_path):
    started = []
    sink, where = make_sink("aplay", SR, tmp_path / "h.wav", which=_installed("aplay"), start=lambda cmd: started.append(cmd) or FakePlayer(cmd), channels=2)
    assert started[0][-2:] == ["-c", "2"]
    sink.write(np.zeros((50, 2), dtype=np.float32))
    sink.close()
    assert sf.read(str(tmp_path / "h.wav"))[0].shape == (50, 2)
    quiet, _ = make_sink("none", SR, tmp_path / "n.wav", which=lambda name: None, start=lambda cmd: None, channels=2)
    quiet.sinks[1]._sleep = lambda seconds: None  # the null half would sleep for the audio's length
    quiet.write(np.zeros((50, 2), dtype=np.float32))
    quiet.close()
    assert sf.read(str(tmp_path / "n.wav"))[0].shape == (50, 2)


# --- a small player buffer for fast sampling ----------------------------------------------------------------------------------------
# Checked on this box on 2026-10-23 with one second of zero bytes (head -c 176400 /dev/zero | player ...), 44100 Hz stereo:
#   paplay ... --latency-msec=50                                  exit 0, no output
#   aplay -q ... --buffer-time=50000                              exit 0, no output (a sound device was usable)
#   ffplay ... -fflags nobuffer -flags low_delay -probesize 32 -analyzeduration 0   exit 0, no output
# All three accepted the flags, so nothing was swapped for a nearer one.

def test_each_player_gets_its_small_buffer_flag_when_a_latency_is_asked_for():
    assert player_command("paplay", 44100, 2, latency_ms=50) == [
        "paplay", "--raw", "--format=s16le", "--rate=44100", "--channels=2", "--latency-msec=50"]
    assert player_command("aplay", 44100, 2, latency_ms=50) == [
        "aplay", "-q", "-t", "raw", "-f", "S16_LE", "-r", "44100", "-c", "2", "--buffer-time=50000"]
    assert player_command("ffplay", 44100, 2, latency_ms=50) == [
        "ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet",
        "-fflags", "nobuffer", "-flags", "low_delay", "-probesize", "32", "-analyzeduration", "0",
        "-f", "s16le", "-ar", "44100", "-ac", "2", "-i", "pipe:0"]


def test_aplay_takes_its_buffer_in_microseconds_so_other_latencies_scale_by_a_thousand():
    assert player_command("aplay", 44100, latency_ms=120)[-1] == "--buffer-time=120000"
    assert player_command("paplay", 44100, latency_ms=120)[-1] == "--latency-msec=120"


def test_without_a_latency_every_command_line_is_the_old_one():
    for name in ("paplay", "aplay", "ffplay"):
        assert player_command(name, 44100, 2, latency_ms=None) == player_command(name, 44100, 2)
    assert "--latency-msec" not in " ".join(player_command("paplay", 44100))
    assert "nobuffer" not in player_command("ffplay", 44100)


def test_latency_flags_document_every_player():
    assert set(LATENCY_FLAGS) == {"paplay", "aplay", "ffplay"}


def test_player_buffer_ms_says_what_the_delay_report_adds():
    assert FFPLAY_BUFFER_MS == 100
    assert player_buffer_ms("paplay", 50) == 50
    assert player_buffer_ms("aplay", 75) == 75
    assert player_buffer_ms("ffplay", 50) == 100
    assert player_buffer_ms("ffplay", 300) == 100  # ffplay has no millisecond setting, so the stated constant is used


def test_find_player_passes_the_latency_through():
    cmd = find_player(44100, which=_installed("aplay"), channels=2, latency_ms=40)
    assert cmd[-1] == "--buffer-time=40000"
    assert find_player(44100, which=_installed("aplay"))[-1] == "1"  # no latency: the old line


def test_make_sink_passes_the_latency_to_the_player_it_starts(tmp_path):
    started = []
    make_sink("paplay", SR, tmp_path / "a.wav", which=_installed("paplay"), start=lambda c, **kw: started.append(c) or FakePlayer(c), latency_ms=30)
    make_sink("auto", SR, tmp_path / "b.wav", which=_installed("aplay"), start=lambda c, **kw: started.append(c) or FakePlayer(c), latency_ms=30)
    make_sink("aplay", SR, tmp_path / "c.wav", which=_installed("aplay"), start=lambda c: started.append(c) or FakePlayer(c))
    assert started[0][-1] == "--latency-msec=30"
    assert started[1][-1] == "--buffer-time=30000"
    assert started[2][-1] == "1"


def test_a_player_shares_the_terminals_process_group_unless_it_is_given_a_session_of_its_own():
    plain = PlayerSink(["sh", "-c", "cat > /dev/null"])
    apart = PlayerSink(["sh", "-c", "cat > /dev/null"], own_session=True)
    try:
        assert os.getpgid(plain._proc.pid) == os.getpgid(0)
        assert os.getpgid(apart._proc.pid) != os.getpgid(0)
    finally:
        plain.close()
        apart.close()


def test_make_sink_gives_the_player_its_own_session_only_when_a_small_buffer_is_asked_for(tmp_path):
    seen = []

    def start(command, **kwargs):
        seen.append(kwargs)
        return FakePlayer(command)

    make_sink("paplay", SR, tmp_path / "a.wav", which=_installed("paplay"), start=start)
    make_sink("paplay", SR, tmp_path / "b.wav", which=_installed("paplay"), start=start, latency_ms=20)
    assert seen == [{}, {"own_session": True}]


def test_make_sink_gives_the_player_its_own_session_when_asked_even_without_a_small_buffer(tmp_path):
    # --tui asks for it in plain live mode, so a terminal Ctrl-C reaches only Python and q and Ctrl-C drain the same way.
    seen = []

    def start(command, **kwargs):
        seen.append(kwargs)
        return FakePlayer(command)

    make_sink("paplay", SR, tmp_path / "a.wav", which=_installed("paplay"), start=start, own_session=True)
    make_sink("auto", SR, tmp_path / "b.wav", which=_installed("aplay"), start=start, own_session=True)
    make_sink("paplay", SR, tmp_path / "c.wav", which=_installed("paplay"), start=start, own_session=False)
    assert seen == [{"own_session": True}, {"own_session": True}, {}]
