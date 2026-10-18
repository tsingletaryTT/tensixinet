# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""End-to-end: telemetry -> chorus -> WAV, logging what happens and why."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import soundfile as sf

from .music.chorus import chip_devices, render_chorus
from .music.eq import apply_eq
from .music.loudness import TARGET_RMS, normalise as normalise_level
from .music.patch import DEFAULT_PATCH, Patch
from .narrate.explain import (
    explain_chip,
    explain_events,
    explain_lead,
    explain_mix,
    explain_patch,
    explain_turn_weights,
    explain_turns,
)
from .narrate.runlog import RunLog
from .telemetry import Telemetry


def _dbfs(audio: np.ndarray) -> float:
    return 20 * math.log10(max(float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))), 1e-9))


def log_only(log: RunLog, only: Optional[Sequence[str]]) -> None:
    """One log line when the run is an audition of some devices, and nothing otherwise."""
    if only is not None:
        log.event("only", f"playing only {', '.join(only)}",
                  "the other devices are silent and there is no turn-taking, so each chosen device is heard at full level as it is")


def log_eq(log: RunLog, on: bool) -> None:
    """One log line saying what the final EQ does, or that it is off."""
    if on:
        log.event("eq", "final EQ: below 35 Hz cut, 70 to 250 Hz up to +4 dB, above 8 kHz falling to -14 dB at 14 kHz and -33 dB at 20 kHz",
                  "the lows (the gong and drone sit down to 65 Hz) are hard to hear on small speakers, so they are lifted, and the "
                  "shimmer at 12 to 20 kHz is harsh or inaudible, so it is taken down to a soft air; the curve is applied to the "
                  "whole mix, after the chips are mixed and before the loudness gain")
    else:
        log.event("eq", "final EQ off", "the mix is left as the devices made it: the lows are quiet and the shimmer is at full strength")


def describe_readings(
    telemetry: Sequence[Telemetry],
    duration: float,
    log: RunLog,
    source: str,
    patch: Patch,
    patch_source: str,
    turns: bool = True,
) -> None:
    """Log what the readings and the patch decide, one line each with its reason.

    `turns` adds the plan for every turn of a `duration` render, which a live run leaves out because it
    picks each turn's soloist as it goes."""
    n = len(telemetry)
    log.event("discover", f"found {n} chip{'' if n == 1 else 's'}",
              f"the chip count comes from {source}; nothing in the synth assumes a number")
    for d in explain_patch(patch, telemetry, patch_source):
        log.event("patch", d.what, d.why)
    for i, t in enumerate(telemetry):
        for d in explain_chip(i, t, n):
            log.event("chip", d.what, d.why, chip=i)
        if any(dev.__name__ == "lead" for dev in chip_devices(i, n)):
            d = explain_lead(t, duration)
            log.event("lead", d.what, d.why, chip=i)
    for i, t in enumerate(telemetry):
        d = explain_events(i, t)
        log.event("events", d.what, d.why, chip=i)
    for i, d in enumerate(explain_mix(telemetry)):
        log.event("mix", d.what, d.why, chip=i)
    for i, d in enumerate(explain_turn_weights(telemetry)):
        log.event("weight", d.what, d.why, chip=i)
    if turns:
        for k, d in enumerate(explain_turns(telemetry, duration, patch=patch)):
            log.event("turn", d.what, d.why, turn=k)


def run(
    telemetry: Sequence[Telemetry],
    duration: float,
    out_path: Path,
    log: RunLog,
    sample_rate: int = 44100,
    source: str = "the telemetry source",
    patch: Patch = DEFAULT_PATCH,
    patch_source: str = "the built-in default patch",
    normalise: bool = True,
    eq: bool = True,
    only: Optional[Sequence[str]] = None,
) -> np.ndarray:
    if not telemetry:
        log.event("discover", "no chips found, nothing to play",
                  f"{source} returned zero devices; refusing to invent any")
        raise ValueError("no chips in telemetry")
    describe_readings(telemetry, duration, log, source, patch, patch_source)
    log.event("render", f"rendering {duration:g}s at {sample_rate} Hz",
              "all chips are rendered, scaled by power, then mixed by current")
    log_only(log, only)
    audio = render_chorus(telemetry, duration, sample_rate, patch=patch, only=only)
    log_eq(log, eq)
    if eq:
        audio = apply_eq(audio, sample_rate)
    if normalise:
        raw_db = _dbfs(audio)
        audio, gain = normalise_level(audio)
        log.event("loudness", f"gain x{gain:.2f} ({20 * math.log10(gain):+.1f} dB): RMS {raw_db:.1f} to {_dbfs(audio):.1f} dBFS",
                  f"one gain for the whole run brings it toward {20 * math.log10(TARGET_RMS):.0f} dBFS RMS (the peak stays under -1 dBFS and "
                  "the gain is at most +20 dB), so runs land at the same level while the chips keep their level against each other")
    else:
        peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
        if peak > 1.0:  # the EQ lifts the lows, which can push a loud render past full scale
            audio = (audio / peak).astype(audio.dtype)
        log.event("loudness", f"level left as rendered (RMS {_dbfs(audio):.1f} dBFS)",
                  "normalising is off, so the level follows the chips' power and a quiet box stays quiet")
    out_path = Path(out_path)
    sf.write(str(out_path), audio, sample_rate, subtype="PCM_16")
    log.event("done", f"wrote {out_path}",
              f"peak {float(np.max(np.abs(audio))):.2f}; 16-bit WAV at {sample_rate} Hz so ffmpeg can mux it as a soundtrack",
              path=str(out_path))
    return audio
