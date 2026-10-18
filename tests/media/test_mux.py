"""Soundtrack: lay the chorus WAV under a recorded demo video."""
import pytest

from tensixinet.media.mux import mux, mux_command


def test_mux_command_copies_video_and_encodes_audio_trimmed_to_the_video():
    cmd = mux_command("demo.mp4", "chorus.wav", "out.mp4")
    assert cmd[0] == "ffmpeg" and "-y" in cmd
    assert cmd[cmd.index("-i") + 1] == "demo.mp4"
    assert cmd.count("-i") == 2 and "chorus.wav" in cmd
    assert cmd[cmd.index("-c:v") + 1] == "copy"  # no video re-encode
    assert cmd[cmd.index("-c:a") + 1] == "aac"
    assert "-shortest" in cmd  # audio longer than the video is cut, never extends it
    assert cmd[-1] == "out.mp4"
    # video from input 0, audio from input 1, whatever audio the video already had is dropped
    maps = [cmd[i + 1] for i, a in enumerate(cmd) if a == "-map"]
    assert maps == ["0:v:0", "1:a:0"]


class Result:
    def __init__(self, code, err=""): self.returncode, self.stderr = code, err


def test_mux_runs_the_command_and_returns_the_output_path():
    calls = []
    out = mux("v.mp4", "a.wav", "o.mp4", run=lambda c, **k: calls.append(c) or Result(0))
    assert out == "o.mp4" and calls[0][-1] == "o.mp4"


def test_mux_surfaces_ffmpeg_errors():
    with pytest.raises(RuntimeError, match="no such file"):
        mux("v.mp4", "a.wav", "o.mp4", run=lambda c, **k: Result(1, "no such file"))


def test_cli_muxes_and_reports_the_output(capsys):
    from tensixinet.media.mux import main
    calls = []
    code = main(["v.mp4", "a.wav", "o.mp4"], run=lambda c, **k: calls.append(c) or Result(0))
    assert code == 0 and calls[0][-1] == "o.mp4"
    assert "o.mp4" in capsys.readouterr().out


def test_cli_returns_nonzero_and_says_why_when_ffmpeg_fails(capsys):
    from tensixinet.media.mux import main
    assert main(["v.mp4", "a.wav", "o.mp4"], run=lambda c, **k: Result(1, "bad input")) == 1
    assert "bad input" in capsys.readouterr().err
