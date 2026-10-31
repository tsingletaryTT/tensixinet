# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Where live audio goes: a player process, a WAV file, or nowhere at the right speed.

Playing audio never needs `sounddevice` here (the hygiene test forbids it). The sound goes to the
standard input of an ordinary command line player as raw 16-bit audio (mono, or interleaved stereo), and the player's own buffer
paces the run: a write blocks while the player is full.

A sink has `write(audio)` for a float32 array in [-1, 1] and `close()`.
"""
from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path
from typing import Callable, List, Optional, Sequence

import numpy as np
import soundfile as sf

# In the order they are tried. Each reads raw signed 16-bit little-endian samples from standard input, mono or interleaved stereo.
PLAYERS = ("paplay", "aplay", "ffplay")

# What each player is given to keep its own buffer small, for fast sampling. `latency_ms` is the buffer asked for.
#   paplay  --latency-msec=N        N milliseconds.
#   aplay   --buffer-time=N*1000    its unit is microseconds.
#   ffplay  four low-delay flags    it has no millisecond setting, so its buffer is the constant below.
# Each was checked with one second of silence (see tests/media/test_player.py).
LATENCY_FLAGS = {
    "paplay": "--latency-msec=<ms>",
    "aplay": "--buffer-time=<ms * 1000>",
    "ffplay": "-fflags nobuffer -flags low_delay -probesize 32 -analyzeduration 0",
}

# The buffer the delay report counts for ffplay, which cannot be told a size. A stated estimate, not a measurement.
FFPLAY_BUFFER_MS = 100


class PlayerError(RuntimeError):
    """The player is missing, or it stopped while there was still audio to play."""


def pcm16(audio: np.ndarray) -> bytes:
    """Float audio as 16-bit little-endian samples, full scale at 1 and clipped past it."""
    return np.round(np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes()


def player_command(name: str, sample_rate: int, channels: int = 1, latency_ms: Optional[int] = None) -> List[str]:
    """The command line that plays raw 16-bit audio with `channels` channels at `sample_rate` from standard input.

    With `latency_ms` the player is also asked for a small buffer (see `LATENCY_FLAGS`). Without it the line is the plain one."""
    if name == "paplay":
        extra = [f"--latency-msec={latency_ms}"] if latency_ms is not None else []
        return ["paplay", "--raw", "--format=s16le", f"--rate={sample_rate}", f"--channels={channels}"] + extra
    if name == "aplay":
        extra = [f"--buffer-time={latency_ms * 1000}"] if latency_ms is not None else []
        return ["aplay", "-q", "-t", "raw", "-f", "S16_LE", "-r", str(sample_rate), "-c", str(channels)] + extra
    if name == "ffplay":
        extra = ["-fflags", "nobuffer", "-flags", "low_delay", "-probesize", "32", "-analyzeduration", "0"] if latency_ms is not None else []
        return ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"] + extra + ["-f", "s16le", "-ar", str(sample_rate), "-ac", str(channels), "-i", "pipe:0"]
    raise ValueError(f"unknown player {name!r}; known: {', '.join(PLAYERS)}")


def player_buffer_ms(name: str, latency_ms: int) -> int:
    """The buffer, in milliseconds, that the delay report adds for this player when it was started with `latency_ms`."""
    if name == "ffplay":
        return FFPLAY_BUFFER_MS
    if name in ("paplay", "aplay"):
        return latency_ms
    raise ValueError(f"unknown player {name!r}; known: {', '.join(PLAYERS)}")


def find_player(sample_rate: int, which: Callable[[str], Optional[str]] = shutil.which, channels: int = 1,
                latency_ms: Optional[int] = None) -> Optional[List[str]]:
    """The command for the first installed player, or None when there is none. `latency_ms` asks for a small buffer."""
    for name in PLAYERS:
        if which(name):
            return player_command(name, sample_rate, channels, latency_ms)
    return None


class PlayerSink:
    """Raw audio into a player's standard input.

    With `own_session` the player runs in a session of its own (`start_new_session=True`), so a Ctrl-C in the terminal (or tmux's
    `send-keys C-c`) reaches only Python. A player in Python's own process group would get the SIGINT too, exit at once, and the run's last
    writes (the fading tail) would fail with a broken pipe. Fast mode asks for this. Plain live mode leaves the player in the group, so a
    Ctrl-C stops it with Python and does not let queued segments play out, unless `--tui` is on: the TUI's `q` and a terminal Ctrl-C must end
    the run the same way, so it asks for this too. `close()` still lets the player finish what it was given."""

    def __init__(self, command: Sequence[str], own_session: bool = False):
        self.command = list(command)
        try:
            self._proc = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                          start_new_session=own_session)
        except OSError as e:
            raise PlayerError(f"cannot start {self.command[0]}: {e}") from e

    def write(self, audio: np.ndarray) -> None:
        try:
            self._proc.stdin.write(pcm16(audio))
            self._proc.stdin.flush()
        except OSError as e:  # includes a broken pipe
            raise PlayerError(f"{self.command[0]} stopped before the audio was done") from e

    def close(self) -> None:
        """Let the player finish what it has been given."""
        try:
            self._proc.stdin.close()
        except OSError:
            pass
        self._proc.wait()


class NullSink:
    """No sound. Each write takes as long as the audio would, so a run still goes at the speed of the music."""

    def __init__(self, sample_rate: int, sleep: Optional[Callable[[float], None]] = None):
        self.sample_rate = sample_rate
        self._sleep = sleep if sleep is not None else time.sleep  # looked up now, so a test can replace time.sleep

    def write(self, audio: np.ndarray) -> None:
        self._sleep(len(audio) / self.sample_rate)

    def close(self) -> None:
        pass


class WavSink:
    """A 16-bit WAV file that grows as the run plays, so what was heard is kept."""

    def __init__(self, path, sample_rate: int, channels: int = 1):
        self._file = sf.SoundFile(str(Path(path)), "w", samplerate=sample_rate, channels=channels, subtype="PCM_16")

    def write(self, audio: np.ndarray) -> None:
        self._file.write(audio)

    def close(self) -> None:
        self._file.close()


class TeeSink:
    """The same audio to several sinks. Closing closes all of them, then raises the first error."""

    def __init__(self, *sinks):
        self.sinks = sinks

    def write(self, audio: np.ndarray) -> None:
        for sink in self.sinks:
            sink.write(audio)

    def close(self) -> None:
        first: Optional[Exception] = None
        for sink in self.sinks:
            try:
                sink.close()
            except Exception as e:  # keep closing the rest
                first = first or e
        if first:
            raise first


def make_sink(
    player: str,
    sample_rate: int,
    wav_path,
    which: Callable[[str], Optional[str]] = shutil.which,
    start: Callable[[Sequence[str]], object] = PlayerSink,
    channels: int = 1,
    latency_ms: Optional[int] = None,
    own_session: bool = False,
):
    """(the sink for a `--player` choice, a sentence saying where the sound goes).

    `player` is `auto` (the first installed of `PLAYERS`, or silence at the speed of the music when none is
    installed), `none` (silence), or the name of one player, which must be installed. Every sink also records
    what is played to `wav_path`. `latency_ms` asks the player for a small buffer (None keeps its own). `channels` is 1 for mono audio and 2 for stereo. The player runs in a session of its own (see `PlayerSink`) when `latency_ms` is given
    or `own_session` is true. Raises `PlayerError` for a named player that is missing or will not start."""
    record = f"recording to {wav_path}"
    if player == "none":
        return TeeSink(WavSink(wav_path, sample_rate, channels), NullSink(sample_rate)), f"no sound, at the speed of the music; {record}"
    if player == "auto":
        command = find_player(sample_rate, which, channels, latency_ms)
        if command is None:
            return (
                TeeSink(WavSink(wav_path, sample_rate, channels), NullSink(sample_rate)),
                f"no audio player found ({', '.join(PLAYERS)}), so no sound, at the speed of the music; {record}",
            )
    else:
        if not which(player):
            raise PlayerError(f"{player} is not installed")
        command = player_command(player, sample_rate, channels, latency_ms)
    wav = WavSink(wav_path, sample_rate, channels)
    try:
        process = start(command, own_session=True) if latency_ms is not None or own_session else start(command)
    except PlayerError:
        wav.close()
        raise
    return TeeSink(wav, process), f"playing through {command[0]}; {record}"
