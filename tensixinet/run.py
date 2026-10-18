# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""End-to-end: telemetry -> chorus -> WAV, logging what happens and why."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import soundfile as sf

from .music.chorus import chip_devices, render_chorus
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


def run(
    telemetry: Sequence[Telemetry],
    duration: float,
    out_path: Path,
    log: RunLog,
    sample_rate: int = 44100,
    source: str = "the telemetry source",
    patch: Patch = DEFAULT_PATCH,
    patch_source: str = "the built-in default patch",
) -> np.ndarray:
    if not telemetry:
        log.event("discover", "no chips found, nothing to play",
                  f"{source} returned zero devices; refusing to invent any")
        raise ValueError("no chips in telemetry")
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
    for k, d in enumerate(explain_turns(telemetry, duration, patch=patch)):
        log.event("turn", d.what, d.why, turn=k)
    log.event("render", f"rendering {duration:g}s at {sample_rate} Hz",
              "all chips are rendered, scaled by power, then mixed by current")
    audio = render_chorus(telemetry, duration, sample_rate, patch=patch)
    out_path = Path(out_path)
    sf.write(str(out_path), audio, sample_rate, subtype="PCM_16")
    log.event("done", f"wrote {out_path}",
              f"peak {float(np.max(np.abs(audio))):.2f}; 16-bit WAV at {sample_rate} Hz so ffmpeg can mux it as a soundtrack",
              path=str(out_path))
    return audio
