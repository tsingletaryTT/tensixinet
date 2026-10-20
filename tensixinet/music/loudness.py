# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""One gain for a finished render, so every run lands at the same level.

The chorus keeps power as volume, so how loud a render is depends on the chips: a quiet box peaks near
0.2 and a loaded one near 0.4. This scales the whole file by a single number. Because it is one number,
chips keep their level against each other and the music keeps its swells and gaps. Only the overall
level moves.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

TARGET_RMS = 10 ** (-18 / 20)  # -18 dBFS
PEAK_LIMIT = 10 ** (-1 / 20)  # -1 dBFS: nothing is pushed past this
MAX_GAIN = 10.0  # +20 dB: a near-silent render is not amplified into noise


def loudness_gain(audio: np.ndarray) -> float:
    """The gain that brings `audio` to the target RMS, held down by the peak limit and the maximum gain.
    Silence has no level, so it gets a gain of 1."""
    if len(audio) == 0:
        return 1.0
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    peak = float(np.max(np.abs(audio)))
    if rms == 0.0:
        return 1.0
    return min(TARGET_RMS / rms, PEAK_LIMIT / peak, MAX_GAIN)


def normalise(audio: np.ndarray) -> Tuple[np.ndarray, float]:
    """(`audio` scaled by `loudness_gain`, the gain), in the input's dtype."""
    gain = loudness_gain(audio)
    return (audio * gain).astype(audio.dtype), gain


SMOOTHING = 0.5  # each chunk moves the live gain this share of the way, in the log domain, to its own gain


class LiveGain:
    """Loudness while the music is still being made: a gain that follows the chunks as they arrive.

    A whole render gets one gain. A live run cannot wait for the end, so each chunk's own gain
    (`loudness_gain`) pulls the running gain part of the way, `SMOOTHING`, and the gain ramps in a straight line
    across the chunk so there is no step at the join. The chunk's peak limit applies at once. Silence holds the gain
    where it was, and a run that starts in silence takes its gain from its first sound."""

    def __init__(self) -> None:
        self.gain: Optional[float] = None

    def apply(self, chunk: np.ndarray) -> Tuple[np.ndarray, float]:
        """(`chunk` scaled by the ramp, the gain at the end of the chunk), in the input's dtype."""
        if len(chunk) == 0 or not np.any(chunk):
            held = 1.0 if self.gain is None else self.gain
            return chunk.copy(), held
        target = loudness_gain(chunk)
        start = target if self.gain is None else self.gain
        end = start * (target / start) ** SMOOTHING
        end = min(end, PEAK_LIMIT / float(np.max(np.abs(chunk))))
        ramp = np.linspace(start, end, len(chunk))
        if chunk.ndim == 2:
            ramp = ramp[:, None]  # the same ramp in both ears
        out = np.clip(chunk * ramp, -PEAK_LIMIT, PEAK_LIMIT)  # the start of the ramp can still be above this chunk's limit
        self.gain = end
        return out.astype(chunk.dtype), end
