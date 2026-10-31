# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Consonants for the choir, as pure signal functions.

There are three families, each ordered from soft to sharp:

* **Fricatives** (h, f, sh, s) are band noise. Each has a centre and a width, and the centre slides by
  `FRICATIVE_SWEEP` of itself over the sound (up for an onset, down for a coda). The sound fades in and out
  over `EDGE_FADE_S` so its ends do not click.
* **Plosives** (b, d, g, p, t, k) are a closure of near silence (`CLOSURE_S`, made by the caller), then a burst:
  `BURST_S` of band noise at the consonant's place, rising over `BURST_ATTACK_S`, decaying, and faded to zero
  over its last `EDGE_FADE_S`. Then the vowel glides in over `ONSET_GLIDE_S`. The voiced ones (b, d, g) hum a
  low "voice bar" through the closure (`VOICE_BAR_LEVEL`, under `VOICE_BAR_CUTOFF_HZ`).
* **Nasals** (m, n) are the vowel itself with a spectral notch (`nasal_gain`), as when the mouth closes and the
  sound leaves through the nose. They last `NASAL_S`.

Band noise is white noise low-passed to half the band's width with an rfft mask, then moved up to the band's
centre by multiplying by a cosine whose frequency follows the centre. A band's centre is held under
`CENTRE_LIMIT` of the sample rate and its width shrinks so its top edge stays under `TOP_LIMIT`, so a low
sample rate gives a duller "s" and nothing folds back past the Nyquist limit.

A burst is short, and its envelope spreads its spectrum. At 8000 Hz a "t" or "d" burst is 120 samples long. With
the usual 0.45 top edge, it put 1.5% of its power past 0.46 of the rate on average, and up to 8% on some draws.
So below `LOW_RATE_HZ` (16 kHz) a burst's band tops out at `LOW_RATE_BURST_TOP` (0.40) of the rate, with its
centre held 0.03 of the rate under that. From 16 kHz up a burst uses the usual limits.

`channel_sharpness` and `pick_consonant` choose a consonant: a hotter GDDR channel picks a sharper member of
the family (a second draw nudges it a little either way, so equal channels still vary), and a strong trigger
leans to plosives while a weak one leans to fricatives and nasals. Every family in the mix keeps a floor, so a
strong trigger still sings some fricatives and nasals.

This module imports nothing from the package, so `patch.py` can import from it.
"""
import math
from typing import Dict, NamedTuple, Sequence, Tuple

import numpy as np


class Consonant(NamedTuple):
    family: str  # "plosive", "fricative" or "nasal"
    voiced: bool
    centre_hz: float  # the band's centre (a nasal's notch)
    width_hz: float  # the band's full width (a nasal's notch width at half depth)
    level: float  # the noise's RMS level (a nasal has none of its own)


PLOSIVES = ("b", "d", "g", "p", "t", "k")
FRICATIVES = ("h", "f", "sh", "s")
NASALS = ("m", "n")
FAMILIES = ("plosive", "fricative", "nasal")  # the order of `choir_mix`
_MEMBERS = {"plosive": PLOSIVES, "fricative": FRICATIVES, "nasal": NASALS}

CONSONANTS: Dict[str, Consonant] = {
    "b": Consonant("plosive", True, 900.0, 1200.0, 0.5),
    "d": Consonant("plosive", True, 4500.0, 3000.0, 0.5),
    "g": Consonant("plosive", True, 2200.0, 1200.0, 0.5),
    "p": Consonant("plosive", False, 900.0, 1200.0, 0.5),
    "t": Consonant("plosive", False, 4500.0, 3000.0, 0.5),
    "k": Consonant("plosive", False, 2200.0, 1200.0, 0.5),
    "h": Consonant("fricative", False, 1500.0, 2500.0, 0.2),
    "f": Consonant("fricative", False, 5000.0, 5000.0, 0.15),
    "sh": Consonant("fricative", False, 3200.0, 1600.0, 0.4),
    "s": Consonant("fricative", False, 6500.0, 2500.0, 0.35),
    "m": Consonant("nasal", True, 1000.0, 300.0, 0.0),
    "n": Consonant("nasal", True, 1800.0, 400.0, 0.0),
}

FRICATIVE_S = 0.09  # an onset fricative's length
CODA_FRICATIVE_S = 0.12  # a fricative that ends a syllable
FRICATIVE_SWEEP = 0.15  # how far a fricative's centre moves over the sound, as a share of the centre
CLOSURE_S = 0.05  # a plosive's near silence before its burst
BURST_S = 0.015  # a plosive's burst
BURST_ATTACK_S = 0.001  # the burst's rise, so it does not click
ONSET_GLIDE_S = 0.03  # the vowel gliding in after a burst
NASAL_S = 0.07  # a nasal's length
NASAL_DEPTH = 0.95  # how far a nasal's notch cuts at its centre
VOICE_BAR_LEVEL = 0.12  # a voiced plosive's hum through its closure
VOICE_BAR_CUTOFF_HZ = 500.0  # the voice bar is low-passed here
EDGE_FADE_S = 0.005  # the fade at the ends of a noise

CENTRE_LIMIT = 0.42  # a band's centre stays under this share of the sample rate
TOP_LIMIT = 0.45  # and its top edge under this one
LOW_RATE_HZ = 16000  # below this rate a burst's band has a lower top edge
LOW_RATE_BURST_TOP = 0.40  # a burst's top edge below LOW_RATE_HZ, as a share of the rate
_EDGE_SHARE = 0.2  # the raised-cosine edge of the low-pass mask, as a share of its cutoff
_BURST_DECAY_SHARE = 1 / 3  # the burst decays with a time constant of this share of its length
_HALF_DEPTH_SIGMAS = 2 * math.sqrt(2 * math.log(2))  # a Gaussian's full width at half depth, in sigmas
_QUIET_HALF_SPREAD_C = 0.25  # under this half spread every channel is the middle one (as in `ttchoir.vowel_sequence`)
FAMILY_FLOOR = 0.3  # each family keeps at least this share of its mix weight whatever the trigger strength (`pick_consonant`)
MEMBER_NUDGE = 0.5  # the second draw moves the sharpness by up to half of this either way (`pick_consonant`)


def _consonant(name: str, family: str) -> Consonant:
    c = CONSONANTS.get(name)
    if c is None or c.family != family:
        raise ValueError(f"{name!r} is not one of the {family}s {_MEMBERS[family]}")
    return c


def _fade(n: int, fade: int) -> np.ndarray:
    """A raised-cosine rise from 0 over `fade` samples, then 1."""
    out = np.ones(n)
    fade = min(fade, n)
    if fade > 0:
        out[:fade] = 0.5 - 0.5 * np.cos(np.pi * np.arange(fade) / fade)
    return out


def band_noise(n: int, centre_hz: float, width_hz: float, sweep: float, sample_rate: int,
               rng: np.random.Generator, top_limit: float = TOP_LIMIT) -> np.ndarray:
    """`n` samples of unit-RMS noise in a band `width_hz` wide. The centre moves linearly from `centre_hz` to
    `centre_hz * (1 + sweep)` over the length. The width shrinks so the top edge stays under `top_limit` of the
    rate, and the centre is held `TOP_LIMIT - CENTRE_LIMIT` of the rate under that (`CENTRE_LIMIT` by default)."""
    if n <= 0:
        return np.zeros(0)
    centre_limit = top_limit - (TOP_LIMIT - CENTRE_LIMIT)
    centre = np.minimum(np.linspace(centre_hz, centre_hz * (1 + sweep), n), centre_limit * sample_rate)
    width = min(width_hz, 2 * (top_limit * sample_rate - float(centre.max())))
    cutoff = width / 2
    # Low-pass white noise to the half width, with a raised-cosine edge that ends at the cutoff.
    spectrum = np.fft.rfft(rng.standard_normal(n))
    freqs = np.fft.rfftfreq(n, 1 / sample_rate)
    edge = _EDGE_SHARE * cutoff
    mask = np.clip((cutoff - freqs) / edge, 0.0, 1.0)
    mask = 0.5 - 0.5 * np.cos(np.pi * mask)
    low = np.fft.irfft(spectrum * mask, n=n)
    # Move it up to the centre: a cosine whose frequency follows the centre's track.
    phase = 2 * np.pi * np.cumsum(centre) / sample_rate
    x = low * np.cos(phase)
    rms = float(np.sqrt(np.mean(x ** 2)))
    return x / rms if rms > 0 else x


def fricative_noise(name: str, n: int, sample_rate: int, rng: np.random.Generator, rising: bool = True) -> np.ndarray:
    """A fricative `n` samples long at its level. Its centre rises over the sound (`rising`) or falls, and its
    ends fade over `EDGE_FADE_S`."""
    c = _consonant(name, "fricative")
    sweep = FRICATIVE_SWEEP if rising else -FRICATIVE_SWEEP
    x = c.level * band_noise(n, c.centre_hz, c.width_hz, sweep, sample_rate, rng)
    fade = _fade(n, int(EDGE_FADE_S * sample_rate))
    return x * fade * fade[::-1]


def burst(name: str, sample_rate: int, rng: np.random.Generator) -> np.ndarray:
    """A plosive's burst, `BURST_S` long: band noise at the consonant's place that rises over `BURST_ATTACK_S`,
    decays, and fades to zero over its last `EDGE_FADE_S`. Below `LOW_RATE_HZ` the band's top edge is held under
    `LOW_RATE_BURST_TOP` of the rate, so the envelope's spread stays under the Nyquist limit."""
    c = _consonant(name, "plosive")
    n = int(BURST_S * sample_rate)
    top = LOW_RATE_BURST_TOP if sample_rate < LOW_RATE_HZ else TOP_LIMIT
    x = band_noise(n, c.centre_hz, c.width_hz, 0.0, sample_rate, rng, top_limit=top)
    t = np.arange(n) / sample_rate
    attack = np.clip(t / BURST_ATTACK_S, 0.0, 1.0) if BURST_ATTACK_S > 0 else np.ones(n)
    decay = np.exp(-np.maximum(t - BURST_ATTACK_S, 0.0) / (BURST_S * _BURST_DECAY_SHARE))
    tail = _fade(n, int(EDGE_FADE_S * sample_rate))[::-1]
    return c.level * x * attack * decay * tail


def nasal_gain(freqs: np.ndarray, name: str) -> np.ndarray:
    """The gain a nasal puts on each frequency: a Gaussian notch at its centre, `NASAL_DEPTH` deep, as wide at
    half depth as its width, and 1 far from it."""
    c = _consonant(name, "nasal")
    sigma = c.width_hz / _HALF_DEPTH_SIGMAS
    f = np.asarray(freqs, dtype=float)
    return 1.0 - NASAL_DEPTH * np.exp(-0.5 * ((f - c.centre_hz) / sigma) ** 2)


def channel_sharpness(gddr_temps: Sequence[float], channel: int) -> float:
    """How sharp a consonant the channel asks for: 0 for the coolest channel, 1 for the hottest, in a straight line
    between. Channels wrap round. When the channels are within 0.25 C of their middle (the rule in
    `ttchoir.vowel_sequence`) every channel gives 0.5."""
    temps = [float(x) for x in gddr_temps]
    low, high = min(temps), max(temps)
    if (high - low) / 2 < _QUIET_HALF_SPREAD_C:
        return 0.5
    return (temps[channel % len(temps)] - low) / (high - low)


def pick_consonant(sharpness: float, strength: float, mix: Tuple[float, float, float], u: float, u_member: float) -> str:
    """The consonant for one syllable.

    The family: with F = `FAMILY_FLOOR`, the weights are `mix[0] * (F + (1 - F) * strength)` for plosives and
    `mix[1] * (1 - (1 - F) * strength)` and `mix[2] * (1 - (1 - F) * strength)` for fricatives and nasals. A stronger trigger leans to plosives and a
    weaker one to fricatives and nasals, and every family in the mix keeps at least `FAMILY_FLOOR` of its weight, so a trigger of
    strength 1 still sings some fricatives and nasals. The mix is the base lean. The family is the first whose cumulative share
    passes `u` (in [0, 1)). When all three weights are 0 the mix alone weights them.

    The member: the step of the family nearest the sharpness (0 the softest, 1 the sharpest), nudged by the second draw
    `u_member` (in [0, 1)) by up to `MEMBER_NUDGE / 2` either way and held inside 0 to 1. A `u_member` of 0.5 does not move it."""
    lean = (1.0 - FAMILY_FLOOR) * strength
    weights = [mix[0] * (FAMILY_FLOOR + lean), mix[1] * (1.0 - lean), mix[2] * (1.0 - lean)]
    if sum(weights) <= 0:
        weights = list(mix)
    total = sum(weights)
    cumulative = 0.0
    family = [name for name, w in zip(FAMILIES, weights) if w > 0][-1]  # if rounding leaves the sum just under u
    for name, w in zip(FAMILIES, weights):
        cumulative += w / total
        if cumulative > u:
            family = name
            break
    members = _MEMBERS[family]
    nudged = float(np.clip(sharpness + MEMBER_NUDGE * (u_member - 0.5), 0.0, 1.0))
    return members[round(nudged * (len(members) - 1))]
