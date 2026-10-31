# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The devices that have TT-Lang kernels, run through a chip worker instead of numpy.

Seven kernels exist (`kernels/`): the gong, the shimmer, the drone's ripple, the AVDog, the Cocoquantus's eight-tap mix, the
choir's harmonics and the mixer. Each does the per-sample arithmetic on the chip; the host does the rest, with the same helpers
the numpy device uses, so a chip device and its numpy device differ only where the chip's arithmetic does (about 1e-3 at the
worst, in the kernel checks). The choir's chip does its voiced part, the sum of up to forty harmonics, eight per kernel run; the
syllable plan, the consonants' noise, the howl and the crush stay on the host (`ttchoir.choir_voice` with a `sines` that calls the
chip). The devices with no kernel (ttrollz, ttrax, ttquantus, the lead and ttsteam) stay on the CPU, so a soundtrack on the
backend is the chips for those seven stages and the CPU for the rest.

A `KernelBackend` goes to `render_chorus(backend=...)`. If the worker fails it says so once and hands everything back to the
CPU, so a chip that goes away does not stop the sound. Only a `ChipError` is caught: a bug on the host side still shows.
"""
from __future__ import annotations

import numpy as np
from typing import Callable, Dict, Optional, Sequence

from . import packing as pk
from ..media.chip import ChipError
from ..music.devices import (
    current_mix,
    drone_chord,
    drone_ripple_inputs,
    drone_with_ripple,
    gong,
    gong_parameters,
    seconds_since,
    shimmer_parameters,
    strike_times,
    ttdrone,
    ultrasonic_shimmer,
)
from ..music.patch import DEFAULT_PATCH, Patch
from ..music.ttavdog import avdog_parameters, avdog_phase, avdog_pings, ttavdog
from ..music.ttchoir import choir_voice, ttchoir
from ..music.ttcoco import coco_taps, ttcoco
from ..music.ttrolzer import SEGMENT_SEED_STRIDE
from ..telemetry import Telemetry

KERNEL_DEVICES = ("gong", "ultrasonic_shimmer", "ttdrone", "ttavdog", "ttcoco", "ttchoir")  # by device name; the mixer is `KernelBackend.mix`


class KernelBackend:
    def __init__(self, worker, log: Optional[Callable[[str], None]] = None):
        self.worker = worker
        self.log = log or (lambda message: None)
        self.failed = False
        self._by_device: Dict[int, Dict[str, Callable]] = {}
        self.devices = self._for_device(0)  # the kernel devices as they run on the worker's first chip

    def _for_device(self, device: int) -> Dict[str, Callable]:
        """The kernel devices bound to the worker's chip number `device`."""
        if device not in self._by_device:
            self._by_device[device] = {
                "gong": self._guarded(gong, self._gong, device),
                "ultrasonic_shimmer": self._guarded(ultrasonic_shimmer, self._shimmer, device),
                "ttdrone": self._guarded(ttdrone, self._drone, device),
                "ttavdog": self._guarded(ttavdog, self._avdog, device),
                "ttcoco": self._guarded(ttcoco, self._coco, device),
                "ttchoir": self._guarded(ttchoir, self._choir, device),
            }
        return self._by_device[device]

    def device_for(self, name: str, chip: int = 0) -> Optional[Callable]:
        """The chip version of device `name` for telemetry chip number `chip`, or None when the device has no kernel. The
        telemetry chip's own kernels run it: chip `c` runs on the worker's chip `c`, or `c` modulo the chips it holds."""
        if name not in KERNEL_DEVICES:
            return None
        return self._for_device(chip % max(1, getattr(self.worker, "devices", 1)))[name]

    # -- failure: one note, then the CPU for everything --------------------------------------------------------

    def _fail(self, error: ChipError) -> None:
        if not self.failed:
            self.failed = True
            self.log(f"the chip backend stopped ({error}); the sound continues on the CPU")

    def _guarded(self, cpu_device: Callable, chip_device: Callable, device: int = 0) -> Callable:
        def on_chip(t: Telemetry, duration: float, sample_rate: int, seed: int, patch: Patch = DEFAULT_PATCH) -> np.ndarray:
            if not self.failed:
                try:
                    return chip_device(t, duration, sample_rate, seed, patch, device)
                except ChipError as e:
                    self._fail(e)
            return cpu_device(t, duration, sample_rate, seed, patch)
        on_chip.__name__ = cpu_device.__name__
        return on_chip

    # -- the devices -------------------------------------------------------------------------------------------

    def _gong(self, t, duration, sample_rate, seed, patch, device=0):
        n = int(duration * sample_rate)
        strike = gong_parameters(t, sample_rate, patch)
        strikes = strike_times(t, duration, patch, seed // SEGMENT_SEED_STRIDE)
        runs = []
        for back in range(pk.GONG_OVERLAP):  # one run for each strike that can still be ringing
            if len(strikes) <= back:
                break
            tiles = pk.gong_inputs(pk.strike_ages(strikes, n, sample_rate, strike.length, back), strike, sample_rate)
            runs.append(self.worker.run("ttgong", tiles, n, device=device))
        return pk.gong_combine(runs).astype(np.float32) if runs else np.zeros(n, dtype=np.float32)

    def _shimmer(self, t, duration, sample_rate, seed, patch, device=0):
        n = int(duration * sample_rate)
        f1, f2, level = shimmer_parameters(t, sample_rate, patch)
        since = seconds_since(strike_times(t, duration, patch, seed // SEGMENT_SEED_STRIDE), n, sample_rate)
        return self.worker.run("ttshimmer", pk.shimmer_inputs(since, f1, f2, level, n, sample_rate), n, device=device).astype(np.float32)

    def _drone(self, t, duration, sample_rate, seed, patch, device=0):
        n = int(duration * sample_rate)
        since, rate_hz, decay_s = drone_ripple_inputs(t, duration, sample_rate, patch, seed // SEGMENT_SEED_STRIDE)
        ripple = self.worker.run("ttshock", pk.shock_inputs(since, rate_hz, decay_s), n, device=device)
        return drone_with_ripple(drone_chord(t, duration, sample_rate), ripple)

    def _avdog(self, t, duration, sample_rate, seed, patch, device=0):
        n = int(duration * sample_rate)
        pitch, shape, rate_hz, decay_s = avdog_parameters(t)
        since = seconds_since(avdog_pings(t, duration, patch, seed // SEGMENT_SEED_STRIDE), n, sample_rate)
        phase = avdog_phase(t, n, sample_rate, patch) if "pitch" in patch.steam_routes else None
        tiles = pk.avdog_inputs(since, rate_hz, decay_s, pitch, shape, n, sample_rate, phase)
        return self.worker.run("ttavdog", tiles, n, device=device).astype(np.float32)

    def _coco(self, t, duration, sample_rate, seed, patch, device=0):
        n = int(duration * sample_rate)
        return self.worker.run("ttcoco", pk.coco_inputs(coco_taps(t, duration, sample_rate, seed, patch), n), n, device=device).astype(np.float32)

    def _choir(self, t, duration, sample_rate, seed, patch, device=0):
        """The choir with its voiced part (the sum of sines) on the chip, eight harmonics a run. The worker packs the groups and adds the runs; this sends it the phase once and the level tables, where sixteen packed tiles a group went before. The
        choir's own code does the rest on the host, so the plan, the consonants, the howl and the crush are the numpy device's."""
        def on_chip(sample_t, frame_t, harmonics, phase, ks):
            return self.worker.run("ttchoir_voice", pk.choir_compact(sample_rate, frame_t, harmonics, phase, ks), len(sample_t), device=device)
        return choir_voice(t, duration, sample_rate, seed, patch, sines=on_chip)

    # -- the mixer ---------------------------------------------------------------------------------------------

    def mix(self, signals: Sequence[np.ndarray], currents: Sequence[float]) -> np.ndarray:
        """`devices.current_mix` on the chip for up to four chips (the mixer kernel's inputs), on the CPU for more."""
        if self.failed or len(signals) > pk.MIX_INPUTS:
            return current_mix(signals, currents)
        n = len(signals[0])
        try:
            return self.worker.run("ttmix", pk.mix_inputs(signals, currents, n), n)
        except ChipError as e:
            self._fail(e)
            return current_mix(signals, currents)
