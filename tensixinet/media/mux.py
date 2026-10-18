# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Lay the chorus under a recorded demo video with ffmpeg."""
from __future__ import annotations

import subprocess
from typing import Callable, List


def mux_command(video: str, audio: str, out: str) -> List[str]:
    """ffmpeg argv: video stream copied, audio encoded to AAC, trimmed to the video.

    Mapping is explicit (video from input 0, audio from input 1) so any audio
    track already in the recording is dropped rather than mixed in.
    """
    return [
        "ffmpeg", "-y", "-i", video, "-i", audio,
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", "-c:a", "aac", "-shortest", out,
    ]


def mux(video: str, audio: str, out: str, run: Callable = subprocess.run) -> str:
    result = run(mux_command(video, audio, out), capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.strip()[-500:]}")
    return out


def main(argv=None, run: Callable = subprocess.run) -> int:
    """`python3 -m tensixinet.mux VIDEO AUDIO OUT`"""
    import argparse
    import sys

    p = argparse.ArgumentParser(prog="tensixinet.mux", description="Lay a WAV under a video as its soundtrack.")
    p.add_argument("video")
    p.add_argument("audio")
    p.add_argument("out")
    a = p.parse_args(argv)
    try:
        print(f"wrote {mux(a.video, a.audio, a.out, run=run)}")
    except RuntimeError as e:
        print(e, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
