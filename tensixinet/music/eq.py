# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""An EQ for the final mix, so what people hear is pleasant on ordinary speakers.

The devices span a wide range. The gong and the drone sit an octave below the chip's pitch (65 Hz at the
bottom), which small speakers barely reproduce and the ear is least sensitive to. The shimmer sits at 12 to
20 kHz, which many people cannot hear at all and which is harsh to those who can. One curve, applied to the
whole mix after the chips are mixed and before the loudness gain, deals with both:

* below 35 Hz is taken out, because it is rumble and costs headroom;
* 70 to 250 Hz is lifted by up to 4 dB, so the lows come through;
* the midrange is left alone;
* above 8 kHz it falls away, 14 dB down at 14 kHz and 33 dB down at 20 kHz, so the shimmer is a soft air
  and not a whistle.

The filter is a minimum-phase FIR made from the curve by the cepstral method. It is causal and has no
pre-ringing, it adds almost no delay (a live run needs that), and it can run on a stream in chunks
(`Equaliser`) and give the same samples as a single pass. It needs nothing beyond numpy.
"""
from __future__ import annotations

from functools import lru_cache
from typing import List

import numpy as np

# (frequency in Hz, gain in dB). Gains between points are interpolated on a log-frequency axis.
EQ_POINTS = (
    (10, -60.0),
    (20, -30.0),
    (35, -12.0),
    (50, -3.0),
    (70, 2.5),
    (100, 4.0),
    (160, 3.5),
    (250, 1.5),
    (500, 0.0),
    (1000, 0.0),
    (2500, -1.0),
    (4000, -2.0),
    (6000, -2.0),
    (8000, -3.0),
    (11000, -7.0),
    (14000, -14.0),
    (17000, -24.0),
    (22050, -40.0),
)
EQ_TAPS = 4096  # impulse response length: long enough for the 35 Hz corner to settle
_DESIGN_GRID = 1 << 16  # the frequency grid the curve is sampled on before the FIR is cut from it


def _curve_db(freqs: np.ndarray) -> np.ndarray:
    points_f = np.log10([f for f, _ in EQ_POINTS])
    points_g = [g for _, g in EQ_POINTS]
    return np.interp(np.log10(np.maximum(freqs, 1.0)), points_f, points_g)


@lru_cache(maxsize=8)
def design_fir(sample_rate: int, taps: int = EQ_TAPS) -> np.ndarray:
    """The minimum-phase FIR (float64, `taps` long) that has the EQ curve as its magnitude response."""
    freqs = np.fft.rfftfreq(_DESIGN_GRID, 1.0 / sample_rate)
    log_magnitude = np.log(10 ** (_curve_db(freqs) / 20))
    cepstrum = np.fft.irfft(log_magnitude, _DESIGN_GRID)  # real cepstrum of the magnitude
    fold = np.zeros(_DESIGN_GRID)
    fold[0], fold[_DESIGN_GRID // 2] = 1.0, 1.0
    fold[1:_DESIGN_GRID // 2] = 2.0  # double the causal half and drop the rest: that makes it minimum phase
    minimum_phase = np.exp(np.fft.rfft(cepstrum * fold))
    h = np.fft.irfft(minimum_phase, _DESIGN_GRID)[:taps]
    fade = int(taps * 0.25)
    h[taps - fade:] *= 0.5 * (1 + np.cos(np.pi * np.arange(fade) / fade))  # a smooth end, so cutting it does not ring
    h.setflags(write=False)  # shared by the cache
    return h


class Equaliser:
    """The EQ on a stream. `process` returns as many samples as it was given; `flush` returns the ring-out
    (`EQ_TAPS - 1` samples) that is left when the stream ends."""

    def __init__(self, sample_rate: int):
        self._h = design_fir(sample_rate)
        self._carry = np.zeros(len(self._h) - 1)

    def process(self, chunk: np.ndarray) -> np.ndarray:
        n = len(chunk)
        if n == 0:
            return np.zeros(0, dtype=np.float32)
        size = 1 << int(np.ceil(np.log2(n + len(self._h) - 1)))
        full = np.fft.irfft(np.fft.rfft(chunk, size) * np.fft.rfft(self._h, size), size)[: n + len(self._h) - 1]
        full[: len(self._carry)] += self._carry
        self._carry = full[n:].copy()
        return full[:n].astype(np.float32)

    def flush(self) -> np.ndarray:
        tail, self._carry = self._carry, np.zeros(0)
        return tail.astype(np.float32)


def apply_eq(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """The EQ over a whole render: the same length as `audio`, float32."""
    return Equaliser(sample_rate).process(audio)
