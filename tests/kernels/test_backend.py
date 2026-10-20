# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The kernel backend: the devices that have TT-Lang kernels, run through a chip worker, in place of their numpy versions.

The worker here is in host mode (each kernel's formula in numpy), so these tests check the packing, the wiring and the
fallback. That the kernels themselves match the devices on a real chip is checked in `kernels/cycle.py`."""
import dataclasses

import numpy as np
import pytest

from tensixinet.kernels.backend import KERNEL_DEVICES, KernelBackend
from tensixinet.media.chip import ChipError, ChipWorker, host_command
from tensixinet.music.chorus import DEVICES, render_chorus
from tensixinet.music.devices import current_mix, gong, ttdrone, ultrasonic_shimmer
from tensixinet.music.patch import DEFAULT_PATCH, STEAM_ROUTES
from tensixinet.music.ttavdog import ttavdog
from tensixinet.music.ttcoco import ttcoco
from tensixinet.telemetry import FALLBACK
from tensixinet.telemetry.mock import mock_telemetry

SR = 8000
CPU = {"gong": gong, "ultrasonic_shimmer": ultrasonic_shimmer, "ttdrone": ttdrone, "ttavdog": ttavdog, "ttcoco": ttcoco}
OPS = {"gong": {"ttgong"}, "ultrasonic_shimmer": {"ttshimmer"}, "ttdrone": {"ttshock"}, "ttavdog": {"ttavdog"}, "ttcoco": {"ttcoco"}}


class Spy:
    """A host-mode worker that remembers which kernels it was asked for."""

    def __init__(self, devices=1):
        self.devices = devices
        self.inner = ChipWorker(host_command(chunk_tiles=16, devices=devices), devices=devices)
        self.calls = []
        self.where = []  # (op, device) for every call

    def run(self, op, inputs, n, device=0):
        self.calls.append(op)
        self.where.append((op, device))
        return self.inner.run(op, inputs, n, device=device)

    def close(self):
        self.inner.close()


class Broken:
    devices = 1

    def __init__(self):
        self.calls = 0

    def run(self, op, inputs, n, device=0):
        self.calls += 1
        raise ChipError("the chip worker is gone")


@pytest.fixture(scope="module")
def spy():
    s = Spy()
    yield s
    s.close()


def _chip(i=1, **kw):
    return dataclasses.replace(mock_telemetry(4, 7)[i], **kw)


def _patches():
    all_routes = dataclasses.replace(DEFAULT_PATCH, seed=12, steam_routes=STEAM_ROUTES)
    return [DEFAULT_PATCH, all_routes, dataclasses.replace(DEFAULT_PATCH, seed=5, steam_routes=("pitch",)),
            dataclasses.replace(DEFAULT_PATCH, seed=9, steam_routes=("strikes", "avdog"))]


def test_the_backend_replaces_exactly_the_devices_that_have_kernels(spy):
    assert set(KERNEL_DEVICES) == set(CPU) and set(KernelBackend(spy).devices) == set(CPU)
    assert set(CPU) <= {d.__name__ for d in DEVICES}


@pytest.mark.parametrize("name", sorted(CPU))
@pytest.mark.parametrize("patch_index", range(4))
def test_each_chip_device_matches_its_numpy_device_and_asks_for_its_own_kernel(spy, name, patch_index):
    patch, backend = _patches()[patch_index], KernelBackend(spy)
    for chip in range(4):
        t = _chip(chip)
        before = len(spy.calls)
        got = backend.devices[name](t, 12.0, SR, 3, patch)
        want = CPU[name](t, 12.0, SR, 3, patch)
        assert got.dtype == np.float32 and got.shape == want.shape
        assert np.allclose(got, want, atol=2e-3), (name, chip, float(np.abs(got - want).max()))
        assert set(spy.calls[before:]) == OPS[name] or (name == "gong" and not spy.calls[before:])  # a gong with no strike needs no chip


def test_the_chip_is_really_asked_the_checks_above_are_not_on_silence(spy):
    backend = KernelBackend(spy)
    for name in CPU:
        assert np.abs(backend.devices[name](_chip(1), 12.0, SR, 3, DEFAULT_PATCH)).max() > 0.01, name


def test_the_gong_runs_one_kernel_per_strike_that_can_still_be_ringing():
    s = Spy()
    try:
        t = _chip(1)
        KernelBackend(s).devices["gong"](t, 40.0, SR, 3, DEFAULT_PATCH)
        from tensixinet.music.devices import strike_times
        from tensixinet.kernels.packing import GONG_OVERLAP
        assert s.calls == ["ttgong"] * min(GONG_OVERLAP, len(strike_times(t, 40.0)))
    finally:
        s.close()


def test_the_pitch_route_reaches_the_chip_as_the_dogtones_phase(spy):
    backend, t = KernelBackend(spy), _chip(2)
    steady = backend.devices["ttavdog"](t, 12.0, SR, 3, DEFAULT_PATCH)
    stepped = backend.devices["ttavdog"](t, 12.0, SR, 3, dataclasses.replace(DEFAULT_PATCH, seed=5, steam_routes=("pitch",)))
    assert not np.allclose(steady, stepped)


def test_a_segment_moves_the_chip_devices_with_the_numpy_ones(spy):
    from tensixinet.music.ttrolzer import SEGMENT_SEED_STRIDE
    backend, t = KernelBackend(spy), _chip(1)
    for name in ("gong", "ultrasonic_shimmer", "ttdrone", "ttavdog"):
        a = backend.devices[name](t, 12.0, SR, 2, DEFAULT_PATCH)
        b = backend.devices[name](t, 12.0, SR, 2 + 3 * SEGMENT_SEED_STRIDE, DEFAULT_PATCH)
        want = CPU[name](t, 12.0, SR, 2 + 3 * SEGMENT_SEED_STRIDE, DEFAULT_PATCH)
        assert np.allclose(b, want, atol=2e-3) and not np.allclose(a, b, atol=2e-3), name


def test_the_mixer_runs_on_the_chip_for_up_to_four_chips_and_on_the_cpu_for_more(spy):
    backend, rng = KernelBackend(spy), np.random.default_rng(1)
    four = [rng.uniform(-1, 1, 5000) for _ in range(4)]
    before = len(spy.calls)
    got = backend.mix(four, [30.0, 40.0, 50.0, 60.0])
    assert spy.calls[before:] == ["ttmix"] and np.allclose(got, current_mix(four, [30.0, 40.0, 50.0, 60.0]), atol=1e-3)
    five = four + [rng.uniform(-1, 1, 5000)]
    before = len(spy.calls)
    assert np.allclose(backend.mix(five, [1.0] * 5), current_mix(five, [1.0] * 5), atol=1e-12)
    assert spy.calls[before:] == []


@pytest.mark.parametrize("patch_index", range(4))
def test_a_whole_chorus_on_the_backend_matches_the_cpu_chorus_and_uses_every_kernel(patch_index):
    patch, s = _patches()[patch_index], Spy()
    try:
        chips = mock_telemetry(4, 3)
        got = render_chorus(chips, 4.0, SR, patch, backend=KernelBackend(s))
        want = render_chorus(chips, 4.0, SR, patch)
        assert got.dtype == np.float32 and got.shape == want.shape
        assert np.allclose(got, want, atol=3e-3), float(np.abs(got - want).max())
        assert {"ttshock", "ttshimmer", "ttavdog", "ttcoco", "ttmix"} <= set(s.calls)
    finally:
        s.close()


def test_without_a_backend_the_chorus_is_what_it_was(spy):
    chips = mock_telemetry(4, 3)
    assert np.array_equal(render_chorus(chips, 2.0, SR), render_chorus(chips, 2.0, SR, DEFAULT_PATCH, backend=None))


def test_a_worker_that_fails_hands_the_sound_back_to_the_cpu_once_and_says_so():
    notes, broken = [], Broken()
    backend = KernelBackend(broken, log=notes.append)
    chips = mock_telemetry(4, 3)
    got = render_chorus(chips, 3.0, SR, DEFAULT_PATCH, backend=backend)
    assert np.array_equal(got, render_chorus(chips, 3.0, SR))      # exactly the CPU sound
    assert backend.failed and broken.calls == 1                     # asked once, then not again
    assert len(notes) == 1 and "CPU" in notes[0] and "chip worker is gone" in notes[0]
    render_chorus(chips, 2.0, SR, DEFAULT_PATCH, backend=backend)
    assert broken.calls == 1 and len(notes) == 1


def test_only_a_chip_error_is_caught(spy):
    class Bug:
        devices = 1

        def run(self, op, inputs, n, device=0):
            raise ZeroDivisionError("a bug in the host side")
    with pytest.raises(ZeroDivisionError):
        KernelBackend(Bug()).devices["ultrasonic_shimmer"](_chip(1), 4.0, SR, 0, DEFAULT_PATCH)


# --- one chip's devices on that chip -------------------------------------------------------------------------------------

def test_each_chips_devices_run_on_that_chips_own_kernels_and_the_mixer_on_the_first():
    s = Spy(devices=4)
    try:
        chips = mock_telemetry(4, 3)
        render_chorus(chips, 6.0, SR, DEFAULT_PATCH, backend=KernelBackend(s))
        from tensixinet.music.chorus import assign_device, slot_chip
        home = {assign_device(slot).__name__: slot_chip(slot, 4) for slot in range(len(DEVICES))}
        assert (home["ultrasonic_shimmer"], home["ttdrone"], home["ttavdog"], home["ttcoco"]) == (2, 3, 0, 2)  # slot % chips
        assert {("ttshimmer", home["ultrasonic_shimmer"]), ("ttshock", home["ttdrone"]), ("ttavdog", home["ttavdog"]),
                ("ttcoco", home["ttcoco"]), ("ttmix", 0)} <= set(s.where)
        assert {device for _, device in s.where} >= {0, 2, 3}
        if ("ttgong", home["gong"]) not in s.where:
            assert not any(op == "ttgong" for op, _ in s.where)  # a gong with no strike in a short render asks for nothing
    finally:
        s.close()


def test_with_fewer_chips_than_the_run_plays_a_device_goes_to_chip_modulo_the_chips_held():
    s = Spy(devices=2)
    try:
        render_chorus(mock_telemetry(4, 3), 4.0, SR, DEFAULT_PATCH, backend=KernelBackend(s))
        assert {device for _, device in s.where} <= {0, 1}
        assert ("ttshimmer", 0) in s.where and ("ttshock", 1) in s.where   # chip 2 -> device 0, chip 3 -> device 1
    finally:
        s.close()


def test_a_chorus_on_four_chips_matches_the_cpu_chorus_whichever_chip_each_device_ran_on():
    s = Spy(devices=4)
    try:
        chips = mock_telemetry(4, 3)
        got = render_chorus(chips, 4.0, SR, DEFAULT_PATCH, backend=KernelBackend(s))
        assert np.allclose(got, render_chorus(chips, 4.0, SR, DEFAULT_PATCH), atol=3e-3)
    finally:
        s.close()


def test_device_for_names_the_device_and_the_chip_and_the_default_is_the_first_chip(spy):
    backend = KernelBackend(spy)
    assert backend.device_for("ttavdog", 0) is backend.devices["ttavdog"]
    assert backend.device_for("ttavdog", 0).__name__ == "ttavdog"
    assert backend.device_for("ttrollz", 3) is None          # no kernel for it
