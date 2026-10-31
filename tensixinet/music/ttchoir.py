# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""ttchoir: a chip as a singer, for a little humanity in the mix.

A voice is a source and a filter. The source is a buzz at the singer's pitch with its harmonics falling away
(a glottal pulse). The filter is the vocal tract, a few resonances called formants. Moving the jaw, tongue and lips
moves the formants, and that is what makes one vowel sound like another: "oo" has a low first formant and a low second,
"ah" a high first and a middle second, "ee" a low first and a very high second.

This is the additive form of formant synthesis. At each moment the sound is a sum of the source's harmonics, each as loud as
the tract lets it be at its own frequency. That form suits the chips: every harmonic is a `sin` of a multiple of one phase.

What a chip sings:

* the pitch is the chip's pitch, snapped to the patch's scale, so a choir of chips sings in tune with itself;
* the vowel changes once a bar, and each bar reads the next GDDR channel: a channel hotter than the others sings toward
  "ee", a cooler one toward "oo", and equal temperatures sing "ah";
* the voice type (a vocal tract 0%, 8%, 17% or 25% shorter, from bass toward soprano) comes from the chip's identity;
* pressure flattens the glottal slope, so a chip near a limit sings with more effort and a brighter tone;
* each syllable swells in and dips between, as a singer breathes, and the pitch has a gentle vibrato.

That is the old choir, kept byte for byte as `_vowel_choir`. A patch with no onset source (`choir_onset == ""`, the default and
every patch saved before syllables existed) sings it, and every other choir field is ignored.

A patch with an onset source sings syllables instead (`_syllable_choir`). It owns no clock. Each syllable starts on a trigger from
the source the patch names (`devices.choir_triggers`, recomputed from this chip's own readings), and the voice is silent between
syllables apart from the last vowel dying away over `RELEASE_S`:

* triggers closer than `MIN_TRIGGER_GAP_S` merge, keeping the stronger;
* a syllable is an onset consonant, a vowel and sometimes a closing consonant (a coda), which sounds only when a trigger of the
  closing source lands inside the syllable. With the same source for both, every other trigger closes (`split_alternate`);
* a syllable is cut at the next trigger, its vowel crossfading into the next onset over `XFADE_S`, and it is never longer than
  a limit that drifts over bars (`syllable_limit`); dense triggers give short syllables and sparse ones long syllables;
* syllable j reads GDDR channel `personality + j`: the channel picks the vowel (as the old choir does) and how sharp the
  consonant is, and the trigger's strength leans to plosives (strong) or to fricatives and nasals (weak), with the patch's `choir_mix`
  as the base lean and a floor under every family; a second draw nudges the member, so equal channels still vary. `consonants.py`
  holds the consonants themselves;
* with `choir_drift` above 0 the syllable limit, the tract's length (the voice type) and the register (whole scale degrees,
  held for a syllable) wander on random walks with a knot every `DRIFT_BARS` bars, different on every chip.

A patch with a howl source (`choir_howl`) may also self-oscillate. A share (`choir_howl_odds`) of that source's triggers
fires a howl (`howl_times`), one at a time. A howl is the tract feeding back on itself with nothing driving it: three sines at
the first three formants of the syllable it lands on, growing from quiet, cut back whenever they would pass -1 dBFS, and
falling silent at its `HOWL_MAX_S` cap (`howl_voice`). The syllables are muted while it sounds, crossfading over `XFADE_S`.

Not modelled: words, breath noise between syllables, and the tract's zeros, which appear only as the nasals' notch.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from . import consonants as cons
from .devices import choir_triggers
from .mapping import _unit, aiclk_to_bpm, chip_pressure, personality, temp_to_pitch_hz
from .patch import DEFAULT_PATCH, Patch
from .rests import rest_flags
from .ttrolzer import SEGMENT_SEED_STRIDE
from ..telemetry import Telemetry

# Formants F1, F2, F3 in Hz for a male voice (Peterson and Barney), per vowel.
VOWELS = {
    "a": (730, 1090, 2440),
    "e": (530, 1840, 2480),
    "i": (270, 2290, 3010),
    "o": (570, 840, 2410),
    "u": (300, 870, 2240),
}
UPPER_FORMANTS = (3400.0, 4400.0)  # F4 and F5 barely move between vowels
BANDWIDTHS = (80.0, 100.0, 120.0, 160.0, 200.0)  # Hz, widening up the tract
VOWEL_ORDER = "uoaei"  # from the darkest (closed, back) to the brightest (closed, front), with "ah" in the middle
VOICE_SCALES = (1.0, 1.08, 1.17, 1.25)  # the tract is shorter, so every formant is higher

SYLLABLE_BEATS = 4  # a syllable is a bar
GLIDE_S = 0.12  # the formants move between vowels over this long
FRAME_HZ = 200  # the tract is updated this often and the harmonic levels are interpolated between frames
MAX_HARMONICS = 40
TILT_CALM, TILT_STRAINED = 1.2, 0.6  # the glottal source falls as 1 / k ** tilt; pressure flattens it
VIBRATO_HZ = 5.5
VIBRATO_SEMITONES = (0.25, 0.6)  # calm and strained
DRIFT_SEMITONES = 0.04  # the slow random wander of the pitch
BREATH_DIP = 0.7  # how far the level falls between syllables
BREATH_S = 0.08
FADE_IN_S = 0.1
CHOIR_LEVEL = 0.5
C3_HZ = 130.8128

# The syllable choir (a patch with an onset source).
XFADE_S = 0.02  # a syllable cut by the next trigger crossfades into the next onset over this long
RELEASE_S = 0.15  # a syllable that ends before the next trigger dies away over this long
MIN_TRIGGER_GAP_S = 0.06  # onset triggers closer than this to the last kept one merge into it
DRIFT_BARS = 3  # the drift walks have a knot every this many bars (a bar is SYLLABLE_BEATS beats)
SYLLABLE_SHRINK_OCTAVES = 3  # at full drift the syllable limit wanders from a bar down to 2 ** -3 of it (half a beat)
VOICE_DRIFT_OCTAVES = 0.16  # at full drift the tract scale wanders this far either side of the chip's voice type
VOICE_SCALE_RANGE = (0.92, 1.35)  # and stays inside this range
REGISTER_DEGREES = 3  # at full drift the pitch moves up to this many whole scale degrees, held for a syllable
CONSONANT_TAG = 0xC05  # the random stream that picks consonants
DRIFT_TAG = 0xD71F  # the three drift walks: + 0 the syllable limit, + 1 the tract, + 2 the register
NOISE_TAG = 0xB425  # the random stream of the consonants' noise
NASAL_LEVEL = 0.6  # the voice's level through a nasal
NOISE_KNEE = 0.4  # the consonants' summed noise passes unchanged up to this (before CHOIR_LEVEL)
NOISE_CEILING = 0.7  # and is bent smoothly toward this above it, so voice (at most 1) plus noise stays under 0.89 / CHOIR_LEVEL

# The howl: self-oscillation of the tract, fired by a share of the howl source's triggers.
HOWL_TAG = 0x4041  # the random stream (`rest_flags`) that picks which triggers fire a howl
HOWL_MAX_S = 4.0  # a howl lasts this long from its fire, and a trigger inside a running howl is ignored
HOWL_CEILING = 10 ** (-1 / 20)  # -1 dBFS: the howl's level never passes this
HOWL_START = 0.01  # the level a howl starts from
HOWL_GROWTH_PER_S = 30.0  # the feedback multiplies the level by this much a second while it is under the ceiling
HOWL_CUT = 0.5  # a frame whose growth would pass the ceiling has its gain multiplied by this
HOWL_RELEASE_S = 0.25  # over its last this long before HOWL_MAX_S the howl falls to silence
HOWL_WEIGHTS = (1.0, 0.6, 0.35)  # the levels of the sines at F1, F2 and F3 before they are normalised

# The speech-chip crush: a strained chip sings through a cheap speech chip. It holds each sample for a few samples (a lower
# effective sample rate) and rounds it to fewer bits. Below the onset pressure it does nothing.
CRUSH_ONSET_PRESSURE = 0.3
CRUSH_BITS = (12, 4)  # bit depth at depth 0 and at depth 1
CRUSH_HOLD = (1, 8)  # samples held at depth 0 and at depth 1


def formant_response(freqs: np.ndarray, formants: Sequence[float]) -> np.ndarray:
    """The vocal tract's gain at each frequency: a product of two-pole resonances, each 1 well below its formant and
    `formant / bandwidth` at it."""
    f = np.asarray(freqs, dtype=np.float64)
    gain = np.ones_like(f)
    for centre, bandwidth in zip(formants, BANDWIDTHS):
        r = f / centre
        q = centre / bandwidth
        gain = gain / np.sqrt((1 - r * r) ** 2 + (r / q) ** 2)
    return gain


def harmonic_amplitudes(f0: float, formants: Sequence[float], pressure: float, sample_rate: int) -> np.ndarray:
    """The level of each harmonic of a `f0` buzz after the tract, from harmonic 1 up, summing to 1. Harmonics stay below
    0.44 of the sample rate, with room for the vibrato."""
    count = max(1, min(MAX_HARMONICS, int(0.44 * sample_rate / (f0 * 1.05))))
    k = np.arange(1, count + 1)
    tilt = TILT_CALM + (TILT_STRAINED - TILT_CALM) * float(np.clip(pressure, 0.0, 1.0))
    amps = k ** (-tilt) * formant_response(k * f0, formants)
    return amps / amps.sum()


def choir_pitch(temp_c: float, scale: Sequence[int]) -> float:
    """The chip's pitch snapped to the nearest note of `scale` (semitones above C3, over three octaves)."""
    semis = 12 * math.log2(temp_to_pitch_hz(temp_c) / C3_HZ)
    notes = [12 * octave + degree for octave in range(3) for degree in scale]
    return C3_HZ * 2 ** (min(notes, key=lambda note: abs(note - semis)) / 12)


def voice_scale(identity: int) -> float:
    return VOICE_SCALES[identity % len(VOICE_SCALES)]


def vowel_sequence(gddr_temps: Sequence[float], count: int, offset: int) -> List[str]:
    """The vowel of each syllable, from `offset`. Each syllable reads the next GDDR channel: hotter than the average
    channel sings toward "ee", cooler toward "oo". Equal temperatures sing "ah"."""
    temps = list(gddr_temps)
    mean = sum(temps) / len(temps)
    half_spread = (max(temps) - min(temps)) / 2
    out = []
    for j in range(count):
        c = temps[(offset + j) % len(temps)]
        position = 2.0 if half_spread < 0.25 else 2 + 2 * (c - mean) / half_spread
        out.append(VOWEL_ORDER[int(np.clip(round(position), 0, len(VOWEL_ORDER) - 1))])
    return out


def pitch_modulation(sample_t: np.ndarray, pressure: float, seed: int) -> np.ndarray:
    """How far the pitch is from the note, in semitones at each time: a vibrato (deeper under pressure) and a slow random
    drift, so the voice is not a machine's."""
    rng = np.random.default_rng(seed)
    if len(sample_t) == 0:
        return np.zeros(0)
    knots = np.arange(0.0, float(sample_t[-1]) + 0.2, 0.2)
    wander = np.convolve(np.pad(rng.standard_normal(len(knots)), 1, mode="edge"), np.ones(3) / 3, mode="valid")  # three-point average
    drift = np.interp(sample_t, knots, wander) * DRIFT_SEMITONES
    depth = VIBRATO_SEMITONES[0] + (VIBRATO_SEMITONES[1] - VIBRATO_SEMITONES[0]) * float(np.clip(pressure, 0.0, 1.0))
    return depth * np.sin(2 * math.pi * VIBRATO_HZ * sample_t) + drift


def _vowel_choir(t: Telemetry, duration: float, sample_rate: int, seed: int, patch: Patch = DEFAULT_PATCH) -> np.ndarray:
    n = int(duration * sample_rate)
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    syllable_s = SYLLABLE_BEATS * 60.0 / aiclk_to_bpm(t.aiclk)
    identity = personality(t)
    pressure = chip_pressure(t)
    f0 = choir_pitch(t.asic_temperature, patch.scale)
    voice = voice_scale(identity)

    # The tract, frame by frame: each syllable's vowel sets the target formants and the formants glide between them.
    frames = max(2, int(duration * FRAME_HZ) + 2)
    frame_t = np.arange(frames) / FRAME_HZ
    vowels = vowel_sequence(t.gddr_temps, int(duration / syllable_s) + 2, identity % len(t.gddr_temps))
    syllable_of_frame = (frame_t / syllable_s).astype(int)
    targets = np.array([list(VOWELS[vowels[s]]) + list(UPPER_FORMANTS) for s in syllable_of_frame], dtype=np.float64) * voice
    width = max(1, int(GLIDE_S * FRAME_HZ))
    padded = np.pad(targets, ((width // 2, width - width // 2 - 1), (0, 0)), mode="edge")
    smooth = np.stack([np.convolve(padded[:, i], np.ones(width) / width, mode="valid") for i in range(targets.shape[1])], axis=1)

    harmonics = np.array([harmonic_amplitudes(f0, smooth[f], pressure, sample_rate) for f in range(frames)])
    sample_t = np.arange(n) / sample_rate

    semitones = pitch_modulation(sample_t, pressure, seed)
    phase = 2 * math.pi * np.cumsum(f0 * 2 ** (semitones / 12)) / sample_rate

    out = np.zeros(n)
    for k in range(harmonics.shape[1]):
        out += np.interp(sample_t, frame_t, harmonics[:, k]) * np.sin((k + 1) * phase)

    # A syllable swells in and dips where it meets the next, and the whole voice fades in.
    distance = np.abs(sample_t - np.round(sample_t / syllable_s) * syllable_s)
    envelope = (1 - BREATH_DIP * np.exp(-((distance / BREATH_S) ** 2))) * np.minimum(1.0, sample_t / FADE_IN_S)
    return (CHOIR_LEVEL * envelope * out).astype(np.float32)


def ttchoir(t: Telemetry, duration: float, sample_rate: int, seed: int, patch: Patch = DEFAULT_PATCH) -> np.ndarray:
    """One chip's choir voice. With no onset source in the patch it is the old vowel choir on its bar clock, byte for byte;
    otherwise it sings syllables on the source's triggers."""
    if patch.choir_onset == "":
        return _vowel_choir(t, duration, sample_rate, seed, patch)
    return _syllable_choir(t, duration, sample_rate, seed, patch)


# --- the syllable choir ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Syllable:
    start: float  # seconds from the render's start, on an onset trigger
    length: float  # seconds until the syllable ends (its release, if any, follows)
    onset: str  # the onset consonant (a name in consonants.CONSONANTS)
    vowel: str  # a letter of VOWEL_ORDER
    coda: str  # the closing consonant, or "" for none
    coda_at: float  # seconds from the render's start of the closing trigger; start + length when there is no coda
    strength: float  # the onset trigger's strength
    channel: int  # the GDDR channel the syllable reads


def _stream(seed: int, tag: int, t: Telemetry) -> np.random.Generator:
    return np.random.default_rng([abs(int(seed)), tag, personality(t)])


def merge_triggers(times: np.ndarray, strengths: np.ndarray, min_gap: float = MIN_TRIGGER_GAP_S) -> Tuple[np.ndarray, np.ndarray]:
    """Triggers with any closer than `min_gap` after the last kept one dropped. The kept one takes the larger strength."""
    times = np.asarray(times, dtype=float)
    strengths = np.asarray(strengths, dtype=float)
    kept_t: List[float] = []
    kept_s: List[float] = []
    for when, strength in zip(times, strengths):
        if kept_t and when - kept_t[-1] < min_gap:
            kept_s[-1] = max(kept_s[-1], float(strength))
            continue
        kept_t.append(float(when))
        kept_s.append(float(strength))
    return np.array(kept_t, dtype=float), np.array(kept_s, dtype=float)


def split_alternate(times: np.ndarray, strengths: np.ndarray) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """((onset times, strengths), (coda times, strengths)): the even triggers start syllables and the odd ones close them. With
    one source for both, a closing trigger would otherwise always be the next onset and never land inside a syllable."""
    times = np.asarray(times, dtype=float)
    strengths = np.asarray(strengths, dtype=float)
    return (times[0::2], strengths[0::2]), (times[1::2], strengths[1::2])


def _bar_s(t: Telemetry) -> float:
    return SYLLABLE_BEATS * 60.0 / aiclk_to_bpm(t.aiclk)


def _walk(t: Telemetry, seed: int, which: int, at: np.ndarray) -> np.ndarray:
    """A drift walk in [0, 1] at each time: a straight line between random knots `DRIFT_BARS` bars apart, from the stream
    `[seed, DRIFT_TAG + which, personality(t)]`. The knots do not depend on how far the times reach."""
    at = np.asarray(at, dtype=float)
    if at.size == 0:
        return np.zeros(0)
    spacing = DRIFT_BARS * _bar_s(t)
    count = int(max(0.0, float(at.max())) // spacing) + 2
    knots = _stream(seed, DRIFT_TAG + which, t).random(count)
    return np.interp(at, np.arange(count) * spacing, knots)


def syllable_limit(t: Telemetry, seed: int, patch: Patch, at: np.ndarray) -> np.ndarray:
    """The longest a syllable starting at each time may be, in seconds: a bar (`SYLLABLE_BEATS` beats) with no drift, and with
    drift a bar shrunk by up to `SYLLABLE_SHRINK_OCTAVES * choir_drift` octaves along the first walk."""
    shrink = SYLLABLE_SHRINK_OCTAVES * patch.choir_drift * _walk(t, seed, 0, at)
    return _bar_s(t) * 2.0 ** -shrink


def tract_scales(t: Telemetry, seed: int, patch: Patch, at: np.ndarray) -> np.ndarray:
    """The tract scale at each time: the chip's voice type moved by up to `VOICE_DRIFT_OCTAVES * choir_drift` octaves either way
    along the second walk, inside `VOICE_SCALE_RANGE`."""
    swing = VOICE_DRIFT_OCTAVES * patch.choir_drift * (2.0 * _walk(t, seed, 1, at) - 1.0)
    return np.clip(voice_scale(personality(t)) * 2.0 ** swing, *VOICE_SCALE_RANGE)


def syllable_pitches(t: Telemetry, seed: int, patch: Patch, at: np.ndarray) -> np.ndarray:
    """The pitch (Hz) of a syllable starting at each time: the chip's note moved by `round(REGISTER_DEGREES * choir_drift * w)`
    whole degrees of the patch's scale, with w in [-1, 1] along the third walk."""
    base = 12 * math.log2(choir_pitch(t.asic_temperature, patch.scale) / C3_HZ)
    notes = sorted({12 * octave + degree for octave in range(-2, 5) for degree in patch.scale})
    home = min(range(len(notes)), key=lambda i: abs(notes[i] - base))
    shift = np.round(REGISTER_DEGREES * patch.choir_drift * (2.0 * _walk(t, seed, 2, at) - 1.0)).astype(int)
    index = np.clip(home + shift, 0, len(notes) - 1)
    return C3_HZ * 2.0 ** (np.array(notes, dtype=float)[index] / 12.0)


def pitch_track(starts: np.ndarray, pitches: np.ndarray, at: np.ndarray) -> np.ndarray:
    """The voice's pitch (Hz, before the vibrato) at each time: each syllable's pitch from its start, gliding from the last one's
    over `GLIDE_S` (or over the gap to the next syllable, if that is shorter), in a straight line in octaves. Before the first
    syllable it is the first pitch."""
    log_f0 = np.log2(np.asarray(pitches, dtype=float))
    knot_t, knot_v = [0.0], [log_f0[0]]
    for j in range(1, len(starts)):
        glide = min(GLIDE_S, (starts[j + 1] - starts[j]) if j + 1 < len(starts) else GLIDE_S)
        knot_t += [starts[j], starts[j] + glide]
        knot_v += [log_f0[j - 1], log_f0[j]]
    return 2.0 ** np.interp(at, knot_t, knot_v)


def _coda_length(name: str) -> float:
    family = cons.CONSONANTS[name].family
    if family == "fricative":
        return cons.CODA_FRICATIVE_S
    if family == "nasal":
        return cons.NASAL_S
    return cons.CLOSURE_S + cons.BURST_S


def plan_syllables(t: Telemetry, duration: float, seed: int, patch: Patch, onsets: Tuple[np.ndarray, np.ndarray],
                   codas: Optional[Tuple[np.ndarray, np.ndarray]]) -> List[Syllable]:
    """The syllables the triggers make. The onset triggers are merged and each starts a syllable, as long as
    `min(limit, gap to the next onset, duration - start + RELEASE_S)` and at least `MIN_TRIGGER_GAP_S`. Syllable j reads GDDR
    channel `personality + j` for its vowel and its onset. A coda is the first closing trigger strictly inside the syllable; it
    reads the next channel, and it ends the syllable its own length after the trigger, or at the next onset if that is first.
    The patch's `choir_pick` (the Play tab's) replaces each syllable's onset and vowel where it is set."""
    times, strengths = merge_triggers(*onsets)
    if len(times) == 0:
        return []
    close_t, close_s = merge_triggers(*codas) if codas is not None else (np.zeros(0), np.zeros(0))
    limits = syllable_limit(t, seed, patch, times)
    stream = _stream(seed, CONSONANT_TAG, t)
    picks = stream.random((len(times), 2))  # an onset's and a coda's family draw for each syllable
    nudges = stream.random((len(times), 2))  # and their member nudges, drawn after them so the family draws keep their place
    temps = t.gddr_temps
    identity = personality(t)
    pick_onset, pick_vowel = patch.choir_pick
    out = []
    for j, (start, strength) in enumerate(zip(times, strengths)):
        following = float(times[j + 1]) if j + 1 < len(times) else math.inf
        length = max(MIN_TRIGGER_GAP_S, min(float(limits[j]), following - start, duration - start + RELEASE_S))
        channel = (identity + j) % len(temps)
        vowel = vowel_sequence(temps, 1, channel)[0]
        onset = cons.pick_consonant(cons.channel_sharpness(temps, channel), float(strength), patch.choir_mix, float(picks[j, 0]),
                                   float(nudges[j, 0]))
        # The Play tab's pick overrides the readings' choice of onset and vowel for every syllable; the draws above still happen, so the
        # coda's draw and the timing are the same with or without a pick.
        onset, vowel = pick_onset or onset, pick_vowel or vowel
        coda, coda_at = "", start + length
        inside = np.flatnonzero((close_t > start) & (close_t < start + length))
        if len(inside):
            k = int(inside[0])
            coda_at = float(close_t[k])
            coda = cons.pick_consonant(cons.channel_sharpness(temps, channel + 1), float(close_s[k]), patch.choir_mix,
                                       float(picks[j, 1]), float(nudges[j, 1]))
            length = min(coda_at + _coda_length(coda), following) - start
        out.append(Syllable(float(start), float(length), onset, vowel, coda, float(coda_at), float(strength), int(channel)))
    return out


def _ramp(x: np.ndarray, start: float, span: float, a: float, b: float) -> np.ndarray:
    """a before `start`, b after `start + span`, a straight line between (a step at `start` when span is 0)."""
    if span <= 0:
        return np.where(x < start, a, b)
    return a + (b - a) * np.clip((x - start) / span, 0.0, 1.0)


def _voiced_shape(s: Syllable, following: float, tau: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For times `tau` (seconds from the syllable's start): the voice's level, a mask of the times that hum only the voice bar,
    and the nasal (or "") sounding at each time."""
    level = np.ones_like(tau)
    bar = np.zeros(len(tau), dtype=bool)
    nasal = np.full(len(tau), "", dtype=object)
    onset = cons.CONSONANTS[s.onset]
    if onset.family == "plosive":
        closed = cons.CLOSURE_S + cons.BURST_S
        hum = cons.VOICE_BAR_LEVEL * _ramp(tau, 0.0, XFADE_S, 0.0, 1.0) if onset.voiced else np.zeros_like(tau)
        level = np.where(tau < closed, hum, _ramp(tau, closed, cons.ONSET_GLIDE_S, hum, 1.0))
        bar |= onset.voiced & (tau < closed)
    elif onset.family == "fricative":
        level = _ramp(tau, cons.FRICATIVE_S - cons.ONSET_GLIDE_S, cons.ONSET_GLIDE_S, 0.0, 1.0)
    else:  # nasal
        level = np.where(tau < cons.NASAL_S, NASAL_LEVEL * _ramp(tau, 0.0, XFADE_S, 0.0, 1.0),
                         _ramp(tau, cons.NASAL_S, cons.ONSET_GLIDE_S, NASAL_LEVEL, 1.0))
        nasal[tau < cons.NASAL_S] = s.onset
    end = s.length
    if s.coda:
        c = s.coda_at - s.start
        coda = cons.CONSONANTS[s.coda]
        after = tau >= c
        if coda.family == "nasal":
            target = NASAL_LEVEL
            nasal[after] = s.coda
        elif coda.family == "plosive" and coda.voiced:
            target = cons.VOICE_BAR_LEVEL
            bar |= after
        else:
            target = 0.0
        share = _ramp(tau, c, XFADE_S, 0.0, 1.0)
        level = level * (1 - share) + target * share
        if coda.family == "plosive":
            level = np.where(tau >= c + cons.CLOSURE_S, 0.0, level)  # the voice bar stops at the burst
    if s.start + end < following - 1e-9:  # it ends before the next trigger: it dies away
        level = level * _ramp(tau, end, RELEASE_S, 1.0, 0.0)
    if math.isfinite(following):  # the next trigger cuts it, crossfading into the next onset
        level = level * _ramp(tau, following - s.start, XFADE_S, 1.0, 0.0)
    return level, bar, nasal


def _fit(x: np.ndarray, room: int, sample_rate: int) -> np.ndarray:
    """`x` cut to `room` samples, with a fade over its new last `EDGE_FADE_S` when it was cut."""
    if room >= len(x):
        return x
    if room <= 0:
        return np.zeros(0)
    y = x[:room].copy()
    fade = min(room, int(cons.EDGE_FADE_S * sample_rate))
    if fade > 0:
        y[-fade:] *= 0.5 + 0.5 * np.cos(np.pi * np.arange(1, fade + 1) / fade)
    return y


def _add(out: np.ndarray, x: np.ndarray, at: int) -> None:
    """Add `x` into `out` from sample `at`, dropping whatever falls outside."""
    lo, hi = max(0, at), min(len(out), at + len(x))
    if hi > lo:
        out[lo:hi] += x[lo - at: hi - at]


def noise_headroom(x: np.ndarray) -> np.ndarray:
    """The consonants' summed noise with its peaks bent under `NOISE_CEILING`: unchanged up to `NOISE_KNEE`, and above it a tanh
    curve that leaves the knee with slope 1 and never reaches the ceiling."""
    x = np.asarray(x, dtype=np.float64)
    span = NOISE_CEILING - NOISE_KNEE
    over = np.abs(x) - NOISE_KNEE
    bent = np.sign(x) * (NOISE_KNEE + span * np.tanh(np.maximum(over, 0.0) / span))
    return np.where(over > 0, bent, x)


def howl_times(t: Telemetry, duration: float, seed: int, patch: Patch, sample_rate: int = 44100) -> np.ndarray:
    """The times (seconds) a howl fires. Trigger k of `patch.choir_howl` fires when `rest_flags(seed, HOWL_TAG, n,
    choir_howl_odds)` marks step k. A fire less than `HOWL_MAX_S` after the last kept fire lands while that howl is sounding,
    and is ignored. Empty when the patch has no onset source or no howl source. `sample_rate` is passed to the source (the lead
    and the quantus place their notes on samples)."""
    if not patch.choir_onset or not patch.choir_howl or patch.choir_howl_odds <= 0.0 or duration <= 0.0:
        return np.zeros(0)
    times = choir_triggers(patch.choir_howl, t, duration, sample_rate, seed // SEGMENT_SEED_STRIDE, patch)[0]
    kept: List[float] = []
    for when in np.asarray(times, dtype=float)[rest_flags(seed, HOWL_TAG, len(times), patch.choir_howl_odds)]:
        if not kept or when - kept[-1] >= HOWL_MAX_S:
            kept.append(float(when))
    return np.array(kept, dtype=float)


def howl_levels(frames: int) -> np.ndarray:
    """The howl's level at each frame (`FRAME_HZ` a second) from its fire. It starts at `HOWL_START`. Each frame the level is
    multiplied by the feedback gain, `HOWL_GROWTH_PER_S ** (1 / FRAME_HZ)`. A frame whose growth would pass `HOWL_CEILING` has
    its gain multiplied by `HOWL_CUT`, so the level drops back and grows again. Over the last `HOWL_RELEASE_S` before
    `HOWL_MAX_S` the level falls in a straight line to 0."""
    step = HOWL_GROWTH_PER_S ** (1.0 / FRAME_HZ)
    level = np.empty(frames)
    current = HOWL_START
    for f in range(frames):
        level[f] = current
        grown = current * step
        current = grown if grown <= HOWL_CEILING else grown * HOWL_CUT
    frame_t = np.arange(frames) / FRAME_HZ
    return level * np.clip((HOWL_MAX_S - frame_t) / HOWL_RELEASE_S, 0.0, 1.0)


def howl_voice(formants: Sequence[float], n: int, sample_rate: int) -> np.ndarray:
    """One howl of `n` samples (at most `HOWL_MAX_S` of them) from its fire: sines at the first three `formants` (each held under
    0.45 of the sample rate), weighted by `HOWL_WEIGHTS` and normalised so their summed peak is 1, times `howl_levels`
    interpolated to samples. Nothing buzzes under it, so it is a whistle or growl at the formants, never above `HOWL_CEILING`."""
    n = min(int(n), int(round(HOWL_MAX_S * sample_rate)))
    if n <= 0:
        return np.zeros(0)
    sample_t = np.arange(n) / sample_rate
    weights = np.array(HOWL_WEIGHTS) / sum(HOWL_WEIGHTS)
    tone = np.zeros(n)
    for weight, centre in zip(weights, list(formants)[:3]):
        tone += weight * np.sin(2 * math.pi * min(float(centre), 0.45 * sample_rate) * sample_t)
    frames = int(n / sample_rate * FRAME_HZ) + 2
    return np.interp(sample_t, np.arange(frames) / FRAME_HZ, howl_levels(frames)) * tone


def _howl_formants(t: Telemetry, seed: int, patch: Patch, syllables: List[Syllable], fire: float) -> List[float]:
    """The formants a howl fired at `fire` sings: those of the syllable sounding then (the last to start at or before it, or the
    first if none has), scaled by the tract at that time, as the frame loop scales them."""
    if syllables:
        before = [s for s in syllables if s.start <= fire]
        vowel = (before[-1] if before else syllables[0]).vowel
    else:
        vowel = vowel_sequence(t.gddr_temps, 1, personality(t) % len(t.gddr_temps))[0]
    return (np.array(VOWELS[vowel], dtype=float) * float(tract_scales(t, seed, patch, np.array([fire]))[0])).tolist()


def _with_howls(sung: np.ndarray, t: Telemetry, duration: float, sample_rate: int, seed: int, patch: Patch,
                syllables: List[Syllable]) -> np.ndarray:
    """`sung` with each howl in place of the syllables while it sounds. The syllables crossfade out over `XFADE_S` at the fire
    and back in over the `XFADE_S` before the howl's cap. With no howl, `sung` comes back unchanged."""
    fires = howl_times(t, duration, seed, patch, sample_rate)
    if len(fires) == 0:
        return sung
    n = len(sung)
    sample_t = np.arange(n) / sample_rate
    howl = np.zeros(n)
    share = np.zeros(n)  # how much of the output is the howl
    for fire in fires:
        at = int(round(fire * sample_rate))
        voice = howl_voice(_howl_formants(t, seed, patch, syllables, float(fire)), int(round(HOWL_MAX_S * sample_rate)), sample_rate)
        _add(howl, _fit(voice, n - at, sample_rate), at)
        inside = np.minimum(sample_t - fire, fire + HOWL_MAX_S - sample_t) / XFADE_S
        share = np.maximum(share, np.clip(inside, 0.0, 1.0))
    return sung * (1.0 - share) + howl * share


def crush_depth(pressure: float, crush: float) -> float:
    """How hard the crush bites: the patch's `crush` at full pressure, nothing at or below `CRUSH_ONSET_PRESSURE`."""
    return crush * _unit(pressure, CRUSH_ONSET_PRESSURE, 1.0)


def speech_crush(x: np.ndarray, depth: float) -> np.ndarray:
    """`x` with each sample held for `round(1 + 7 * depth)` samples and rounded toward zero to `round(12 - 8 * depth)` bits.
    Depth 0 returns `x` itself. Rounding toward zero keeps every sample's size at or under the size of the sample it holds,
    so the crush cannot raise a peak (nearest rounding could lift 0.8913 above the howl's ceiling)."""
    if depth <= 0.0:
        return x
    hold = round(CRUSH_HOLD[0] + (CRUSH_HOLD[1] - CRUSH_HOLD[0]) * depth)
    bits = round(CRUSH_BITS[0] + (CRUSH_BITS[1] - CRUSH_BITS[0]) * depth)
    held = np.repeat(x[::hold], hold)[: len(x)]
    step = 2.0 ** (1 - bits)  # the voice spans -1 to 1, so `bits` bits make steps this wide
    return np.trunc(held / step) * step


def _syllable_choir(t: Telemetry, duration: float, sample_rate: int, seed: int, patch: Patch) -> np.ndarray:
    """The syllable choir, with a hard clip at full scale as a safety net only (`syllable_mix` keeps the voice and noise under
    0.89 and a howl under `HOWL_CEILING`). The speech-chip crush comes after the howl and before the clip, so cables see it."""
    mix = speech_crush(syllable_mix(t, duration, sample_rate, seed, patch), crush_depth(chip_pressure(t), patch.choir_crush))
    return np.clip(mix, -1.0, 1.0).astype(np.float32)


def syllable_mix(t: Telemetry, duration: float, sample_rate: int, seed: int, patch: Patch) -> np.ndarray:
    """The syllable choir before its safety clip, in float64: `CHOIR_LEVEL * (voice + noise_headroom(noise))`, with any howls
    in place of it while they sound."""
    n = int(duration * sample_rate)
    if n == 0:
        return np.zeros(0)
    segment = seed // SEGMENT_SEED_STRIDE
    onsets = choir_triggers(patch.choir_onset, t, duration, sample_rate, segment, patch)
    codas: Optional[Tuple[np.ndarray, np.ndarray]] = None
    if patch.choir_coda == patch.choir_onset:
        onsets, codas = split_alternate(*merge_triggers(*onsets))
    elif patch.choir_coda:
        codas = choir_triggers(patch.choir_coda, t, duration, sample_rate, segment, patch)
    syllables = plan_syllables(t, duration, seed, patch, onsets, codas)
    if not syllables:
        return _with_howls(np.zeros(n), t, duration, sample_rate, seed, patch, syllables)

    pressure = chip_pressure(t)
    starts = np.array([s.start for s in syllables])
    log_f0 = np.log2(syllable_pitches(t, seed, patch, starts))

    frames = max(2, int(duration * FRAME_HZ) + 2)
    frame_t = np.arange(frames) / FRAME_HZ
    frame_f0 = pitch_track(starts, 2.0 ** log_f0, frame_t)
    scales = tract_scales(t, seed, patch, frame_t)

    # The voiced part, frame by frame: each sounding syllable adds its spectrum at its level.
    harmonics = np.zeros((frames, MAX_HARMONICS))
    for j, s in enumerate(syllables):
        following = float(starts[j + 1]) if j + 1 < len(syllables) else math.inf
        last = min(s.start + s.length + RELEASE_S, following + XFADE_S) + 1.0 / FRAME_HZ
        idx = np.flatnonzero((frame_t >= s.start) & (frame_t <= last))
        if len(idx) == 0:
            continue
        level, bar, nasal = _voiced_shape(s, following, frame_t[idx] - s.start)
        for f, lv, b, nz in zip(idx, level, bar, nasal):
            if lv <= 0:
                continue
            f0 = frame_f0[f]
            formants = (np.array(VOWELS[s.vowel] + UPPER_FORMANTS, dtype=float) * scales[f]).tolist()
            amps = harmonic_amplitudes(f0, formants, pressure, sample_rate)
            freqs = f0 * np.arange(1, len(amps) + 1)
            if b:  # the voice bar: only the harmonics under the cutoff, and always the fundamental
                amps = np.where((freqs < cons.VOICE_BAR_CUTOFF_HZ) | (np.arange(len(amps)) == 0), amps, 0.0)
                amps = amps / amps.sum()
            if nz:
                amps = amps * cons.nasal_gain(freqs, nz)
            harmonics[f, : len(amps)] += lv * amps

    sample_t = np.arange(n) / sample_rate
    semitones = pitch_modulation(sample_t, pressure, seed)
    phase = 2 * math.pi * np.cumsum(pitch_track(starts, 2.0 ** log_f0, sample_t) * 2 ** (semitones / 12)) / sample_rate
    out = np.zeros(n)
    for k in np.flatnonzero(harmonics.any(axis=0)):
        out += np.interp(sample_t, frame_t, harmonics[:, k]) * np.sin((k + 1) * phase)

    # The noise part: fricatives and bursts in the sample domain, each faded at its edges and kept inside its syllable.
    rng = _stream(seed, NOISE_TAG, t)
    noise_sum = np.zeros(n)
    for j, s in enumerate(syllables):
        following = float(starts[j + 1]) if j + 1 < len(syllables) else math.inf
        stop = min(following, s.start + s.length)
        sounds = [(s.onset, s.start, True)] + ([(s.coda, s.coda_at, False)] if s.coda else [])
        for name, at, is_onset in sounds:
            family = cons.CONSONANTS[name].family
            if family == "fricative":
                room = min(cons.FRICATIVE_S if is_onset else cons.CODA_FRICATIVE_S, stop - at)
                noise, begin = cons.fricative_noise(name, int(room * sample_rate), sample_rate, rng, rising=is_onset), at
            elif family == "plosive":
                begin = at + cons.CLOSURE_S
                noise = _fit(cons.burst(name, sample_rate, rng), int((stop - begin) * sample_rate), sample_rate)
            else:
                continue
            _add(noise_sum, noise, int(round(begin * sample_rate)))
    return _with_howls(CHOIR_LEVEL * (out + noise_headroom(noise_sum)), t, duration, sample_rate, seed, patch, syllables)
