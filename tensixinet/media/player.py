# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Where live audio goes: a player process, a WAV file, or nowhere at the right speed.

Playing audio never needs `sounddevice` here (the hygiene test forbids it). The sound goes to the
standard input of an ordinary command line player as raw 16-bit mono, and the player's own buffer
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

# In the order they are tried. Each reads raw signed 16-bit little-endian mono from standard input.
PLAYERS = ("paplay", "aplay", "ffplay")


class PlayerError(RuntimeError):
    """The player is missing, or it stopped while there was still audio to play."""


def pcm16(audio: np.ndarray) -> bytes:
    """Float audio as 16-bit little-endian samples, full scale at 1 and clipped past it."""
    return np.round(np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes()


def player_command(name: str, sample_rate: int) -> List[str]:
    """The command line that plays raw 16-bit mono at `sample_rate` from standard input."""
    if name == "paplay":
        return ["paplay", "--raw", "--format=s16le", f"--rate={sample_rate}", "--channels=1"]
    if name == "aplay":
        return ["aplay", "-q", "-t", "raw", "-f", "S16_LE", "-r", str(sample_rate), "-c", "1"]
    if name == "ffplay":
        return ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", "-f", "s16le", "-ar", str(sample_rate), "-ac", "1", "-i", "pipe:0"]
    raise ValueError(f"unknown player {name!r}; known: {', '.join(PLAYERS)}")


def find_player(sample_rate: int, which: Callable[[str], Optional[str]] = shutil.which) -> Optional[List[str]]:
    """The command for the first installed player, or None when there is none."""
    for name in PLAYERS:
        if which(name):
            return player_command(name, sample_rate)
    return None


class PlayerSink:
    """Raw audio into a player's standard input."""

    def __init__(self, command: Sequence[str]):
        self.command = list(command)
        try:
            self._proc = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
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

    def __init__(self, path, sample_rate: int):
        self._file = sf.SoundFile(str(Path(path)), "w", samplerate=sample_rate, channels=1, subtype="PCM_16")

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
):
    """(the sink for a `--player` choice, a sentence saying where the sound goes).

    `player` is `auto` (the first installed of `PLAYERS`, or silence at the speed of the music when none is
    installed), `none` (silence), or the name of one player, which must be installed. Every sink also records
    what is played to `wav_path`. Raises `PlayerError` for a named player that is missing or will not start."""
    record = f"recording to {wav_path}"
    if player == "none":
        return TeeSink(WavSink(wav_path, sample_rate), NullSink(sample_rate)), f"no sound, at the speed of the music; {record}"
    if player == "auto":
        command = find_player(sample_rate, which)
        if command is None:
            return (
                TeeSink(WavSink(wav_path, sample_rate), NullSink(sample_rate)),
                f"no audio player found ({', '.join(PLAYERS)}), so no sound, at the speed of the music; {record}",
            )
    else:
        if not which(player):
            raise PlayerError(f"{player} is not installed")
        command = player_command(player, sample_rate)
    wav = WavSink(wav_path, sample_rate)
    try:
        process = start(command)
    except PlayerError:
        wav.close()
        raise
    return TeeSink(wav, process), f"playing through {command[0]}; {record}"
