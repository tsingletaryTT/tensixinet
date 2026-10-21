# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The coco bus: the Cocoquantus as it is played, with other sounds patched into its delay.

The real Cocoquantus is not only a generator. It takes audio in, writes it into its 8-bit delay chips, and the
Quantussy's petals set how those chips are clocked, so the echoes bend and skip as the castles step. `ttquantus` and
`ttcoco` are the instrument's own voices (the petals playing notes, the delay taps echoing them). This module is the
other half: the chip's other voices go in as the audio input.

Each chip's send is the sum of its voices, each at its own `SEND_LEVELS` share, scaled by what is heard (the turn gain
and the style's balance) and by the patch's `coco_send`. Four taps, one for every second GDDR channel, echo it
(`ttcoco.tap_delay_ms`: a hotter channel echoes sooner; `tap_bits`: a longer delay holds fewer bits; the chip's pressure feeds the
repeats back). The petals' castles sway each tap's delay time, so a repeat lands early or late as a castle steps, the way a
changing delay clock bends a real echo. The echoes and the dry send then go through a room: a reverb made by convolving
with decaying, rolled-off noise (`reverb_ir`), a different room for each ear in stereo.

Alternate taps sit in alternate ears. Each segment starts with an empty buffer, so a tail that rings past the end of a segment
is cut there (the same limit the gongs have in live mode). Not modelled: looping the buffer, flip, skip, the Dolby switches.
"""
from __future__ import annotations

import numpy as np

from .devices import REFERENCE_AXICLK
from .mapping import aiclk_to_bpm, chip_pressure, personality
from .patch import Patch
from .ttcoco import ECHO_FLOOR, FEEDBACK_MAX, crunch_bits, tap_bits, tap_delay_ms
from .ttquantussy import N_PETALS, castle_offset, castle_output, petal_phases, petal_ratios
from .ttsteam import lowpass
from ..telemetry import Telemetry

# How much of each device goes into the Cocoquantus's input. The two coco devices send nothing: they are the instrument.
SEND_LEVELS = {
    "ttrollz": 0.5, "ttrax": 0.4, "ttquantus": 0.0, "ttdrone": 0.3, "lead": 0.7, "gong": 0.8, "ultrasonic_shimmer": 0.3,
    "ttsteam": 0.4, "ttavdog": 0.5, "ttchoir": 0.6, "ttcoco": 0.0,
}
BUS_TAPS = 4  # one per second GDDR channel; ttcoco's own eight are the instrument's voice
SWAY_DEPTH = 0.35  # a castle at its extremes moves a tap's delay by this share, either way
SWAY_SMOOTH_S = 0.01  # the castle's steps are glided over this long, so a step bends the echo and does not click
MAX_REPEATS = 12
MIN_FEEDBACK = 0.25  # pressure raises the feedback from here to ttcoco's ceiling, so a cool chip still echoes
TAP_LEVEL = 0.5
REVERB_PRE_DELAY_S = 0.02
REVERB_CUTOFF_HZ = 4000.0  # a room swallows the highs; the tail is darker than the sound that made it
REVERB_STREAM = 0x4E5B


def send_level(device_name: str) -> float:
    return SEND_LEVELS.get(device_name, 0.0)


def _glide(x: np.ndarray, width: int) -> np.ndarray:
    """A moving average of `width` samples, the same length as `x` (the edges use what is there)."""
    if width <= 1:
        return x
    c = np.concatenate([[0.0], np.cumsum(x)])
    hi = np.minimum(np.arange(len(x)) + width // 2 + 1, len(x))
    lo = np.maximum(np.arange(len(x)) - width // 2, 0)
    return (c[hi] - c[lo]) / (hi - lo)


def petal_sway(t: Telemetry, n: int, sample_rate: int, seed: int, tap: int) -> np.ndarray:
    """Tap `tap`'s delay control, 0 to 1: the castle of one of the five petals, held between its petal's peaks and valleys.
    The petals run at the chip's note rate, as the notes of `ttquantus` do, and the chip's identity picks which petal each castle samples."""
    rate = aiclk_to_bpm(t.aiclk) / 60.0 * 2 * t.axiclk / REFERENCE_AXICLK
    freqs = [rate / 2 * r for r in petal_ratios(seed)]
    return castle_output(tap % N_PETALS, freqs, petal_phases(seed), np.arange(n) / sample_rate, offset=castle_offset(personality(t)))


def swayed_tap(x: np.ndarray, base_samples: float, sway: np.ndarray, depth: float, feedback: float, bits: float) -> np.ndarray:
    """One delay tap's echoes (not the dry signal): repeat k is `x` delayed by k times a delay that the sway moves, `feedback` ** k as loud,
    and crunched to `bits`. At a sway of 0.5 the delay is `base_samples`; at 1 it is `depth` longer and at 0 `depth` shorter, as a share of it."""
    n = len(x)
    delay = max(1.0, base_samples) * (1.0 + depth * (2.0 * np.asarray(sway) - 1.0))
    index = np.arange(n)
    out = np.zeros(n)
    gain = feedback
    for k in range(1, MAX_REPEATS + 1):
        if gain <= ECHO_FLOOR or k * delay.min() >= n:
            break
        out += gain * crunch_bits(np.interp(index - k * delay, index, x, left=0.0), bits)
        gain *= feedback
    return out


def reverb_ir(seconds: float, sample_rate: int, seed: int, ear: int) -> np.ndarray:
    """A room's impulse response: noise that falls by 60 dB over `seconds`, rolled off, after a short pre-delay, with unit energy."""
    pre = int(REVERB_PRE_DELAY_S * sample_rate)
    length = int(seconds * sample_rate)
    noise = np.random.default_rng([seed, REVERB_STREAM, ear]).normal(size=length)
    tail = noise * np.exp(-np.log(1000.0) * np.arange(length) / (seconds * sample_rate))
    tail = lowpass(tail, min(REVERB_CUTOFF_HZ, 0.15 * sample_rate), sample_rate)
    ir = np.concatenate([np.zeros(pre), tail])
    return ir / np.sqrt(np.sum(ir ** 2))


def fft_convolve(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """`x` convolved with `h`, cut to the length of `x`."""
    size = 1 << int(np.ceil(np.log2(len(x) + len(h))))
    return np.fft.irfft(np.fft.rfft(x, size) * np.fft.rfft(h, size), size)[: len(x)]


def coco_bus(send: np.ndarray, t: Telemetry, sample_rate: int, seed: int, patch: Patch, stereo: bool) -> np.ndarray:
    """What the Cocoquantus makes of the chip's send: its echoes and the room, shape (n,) or, with `stereo`, (n, 2). Silence in is silence out."""
    n = len(send)
    if not np.any(send):
        return np.zeros((n, 2) if stereo else n)
    dry = send * patch.coco_send
    feedback = MIN_FEEDBACK + (FEEDBACK_MAX - MIN_FEEDBACK) * chip_pressure(t)
    ears = [np.zeros(n), np.zeros(n)]
    for tap in range(BUS_TAPS):
        delay_ms = tap_delay_ms(t.gddr_temps[(2 * tap) % len(t.gddr_temps)])
        sway = _glide(petal_sway(t, n, sample_rate, seed, tap), int(SWAY_SMOOTH_S * sample_rate))
        ears[tap % 2] += swayed_tap(dry, delay_ms / 1000.0 * sample_rate, sway, SWAY_DEPTH, feedback, tap_bits(delay_ms))
    wet = []
    for ear, echoes in enumerate(ears):
        echoes = TAP_LEVEL * echoes / np.sqrt(BUS_TAPS / 2)
        room = fft_convolve(dry + echoes, reverb_ir(patch.reverb_s, sample_rate, seed, ear)) if patch.coco_reverb > 0 else 0.0
        wet.append(echoes + patch.coco_reverb * room)
    return np.stack(wet, axis=1) if stereo else wet[0] + wet[1]
