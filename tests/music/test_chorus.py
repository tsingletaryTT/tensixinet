# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""A chorus of devices, one per chip, for any chip count."""
import dataclasses

import numpy as np
import pytest

from tensixinet.music.chorus import DEVICES, assign_device, chip_devices, plan_turns, render_chorus
from tensixinet.music.mapping import aiclk_to_bpm
from tensixinet.telemetry import FALLBACK

SR = 8000


def chips(n):
    return [FALLBACK] * n


@pytest.mark.parametrize("n", [1, 2, 3, 4, 9, 32])
def test_chorus_renders_for_any_chip_count(n):
    audio = render_chorus(chips(n), 0.5, SR)
    assert audio.shape == (SR // 2,) and audio.dtype == np.float32
    assert np.isfinite(audio).all() and np.abs(audio).max() < 1.0


def test_devices_cycle_by_chip_index_so_every_character_appears_once_n_is_large_enough():
    names = [assign_device(i).__name__ for i in range(len(DEVICES) * 2)]
    assert names[: len(DEVICES)] == names[len(DEVICES):]
    assert len(set(names)) == len(DEVICES) >= 4


def test_chorus_is_deterministic():
    assert np.array_equal(render_chorus(chips(4), 0.5, SR), render_chorus(chips(4), 0.5, SR))


def test_empty_system_is_an_error():
    with pytest.raises(ValueError):
        render_chorus([], 0.5, SR)


# Wiring test: the README claims every signal shapes the music. For each field
# we change ONE chip's reading and require the audio to change. A field that
# is parsed but never reaches the sound would fail here.
CHANGES = {
    "asic_temperature": 90.0,
    "aiclk": 1350.0,
    "power": 190.0,
    "current": 5.0,
    "gddr_temps": (26.0, 99.0) * 4,
    "ddr_status": (1, 1, 1, 1),
    "voltage": 0.90,
    "board_power": 300.0,
    "axiclk": 1920.0,
    "arcclk": 1600.0,
    "heartbeat": 8,
    "tensix_mask": 0x1235,  # picked so the lead offset and gong detune both differ from the default chip
    "eth_mask": 0x0F0C,
    "pcie_usage": 5,
    "corr_errs": 5,
    "uncorr_errs": 1,
    "thermal_trips": 1,
    "thm_limit_throttle": 45.0,
    "tdp_limit": 40.0,
    "busy": 3,
}


@pytest.mark.parametrize("field", sorted(CHANGES))
def test_every_telemetry_field_reaches_the_audio(field):
    base = [dataclasses.replace(FALLBACK, ddr_status=(0, 0, 0, 0)) for _ in range(4)]
    baseline = render_chorus(base, 1.0, SR)
    for chip in range(4):  # whichever chip's device consumes the field
        changed = list(base)
        changed[chip] = dataclasses.replace(base[chip], **{field: CHANGES[field]})
        if not np.allclose(render_chorus(changed, 1.0, SR), baseline):
            return
    pytest.fail(f"changing {field} on any chip left the audio identical")


def test_voltage_detunes_the_drone_so_it_reaches_the_chorus():
    base = [FALLBACK] * 4
    hi = [dataclasses.replace(FALLBACK, voltage=0.99)] * 4
    assert not np.array_equal(render_chorus(base, 0.5, SR), render_chorus(hi, 0.5, SR))


# --- several devices per chip ----------------------------------------------------

@pytest.mark.parametrize("n", [1, 2, 3, 4, 6, 7, 8, 13, 32])
def test_every_device_plays_exactly_once_per_round_whatever_the_chip_count(n):
    played = [d for i in range(n) for d in chip_devices(i, n)]
    assert set(played) == set(DEVICES)
    assert len(played) == max(n, len(DEVICES))


def test_a_one_chip_box_plays_the_whole_ensemble():
    assert chip_devices(0, 1) == list(DEVICES)


def test_with_enough_chips_each_chip_plays_one_device_and_the_order_cycles():
    n = len(DEVICES) + 2
    assert chip_devices(0, n) == [DEVICES[0]]
    assert chip_devices(n - 1, n) == [DEVICES[(n - 1) % len(DEVICES)]]


def test_a_small_system_still_plays_more_than_one_device_per_chip():
    assert all(len(chip_devices(i, 3)) >= 2 for i in range(3))


# --- turn-taking follows the system ------------------------------------------------

def test_turns_cover_the_whole_duration_and_last_two_bars():
    soloist_slots, turn_s = plan_turns(chips(4), 50.0)
    assert turn_s == pytest.approx(2 * 4 * 60.0 / aiclk_to_bpm(FALLBACK.aiclk))
    assert len(soloist_slots) * turn_s >= 50.0 > (len(soloist_slots) - 1) * turn_s


def test_a_faster_clock_means_shorter_turns():
    slow = [dataclasses.replace(FALLBACK, aiclk=400.0)] * 2
    fast = [dataclasses.replace(FALLBACK, aiclk=1350.0)] * 2
    assert plan_turns(slow, 30.0)[1] > plan_turns(fast, 30.0)[1]


def test_the_chip_working_hardest_solos_most():
    chips3 = [dataclasses.replace(FALLBACK, power=p) for p in (190.0, 10.0, 10.0)]
    slots, _ = plan_turns(chips3, 400.0)
    n_slots = max(3, len(DEVICES))
    busy = [s for s in range(n_slots) if s % 3 == 0]
    share = sum(1 for s in slots if s in busy) / len(slots)
    assert share > 0.6  # chip 0 owns 3 of the 8 slots (38%) but draws 90% of the power


def test_the_soloist_changes_with_system_load():
    quiet = [dataclasses.replace(FALLBACK, power=10.0) for _ in range(3)]
    loaded = [dataclasses.replace(FALLBACK, power=p) for p in (190.0, 10.0, 10.0)]
    assert plan_turns(quiet, 60.0)[0] != plan_turns(loaded, 60.0)[0]


def test_the_chorus_is_the_soloist_weighted_sum_of_its_devices_mixed_by_current():
    # Composition check for one chip, built from the parts that are tested on their own:
    # every device is scaled by its turn gain, the chip's devices are summed and divided by
    # sqrt(device count), then scaled by power and mixed by current.
    import math
    from tensixinet.music.devices import current_mix
    from tensixinet.music.mapping import power_to_volume
    from tensixinet.music.turns import turn_gains
    t = dataclasses.replace(FALLBACK, ddr_status=(1, 1, 1, 1), gddr_temps=(30.0, 38.0, 34.0, 29.0) * 2)
    dur = 20.0
    solo, turn_s = plan_turns([t], dur)
    gains = turn_gains(solo, len(DEVICES), int(dur * SR), SR, turn_s)
    total = sum(gains[s] * assign_device(s)(t, dur, SR, seed=s) for s in range(len(DEVICES)))
    expected = current_mix([total / math.sqrt(len(DEVICES)) * power_to_volume(t.power)], [t.current])
    assert np.allclose(render_chorus([t], dur, SR), expected.astype(np.float32), atol=1e-6)


# --- patches ---------------------------------------------------------------------

from tensixinet.music.patch import DEFAULT_PATCH, draw_patch  # noqa: E402


def test_the_default_patch_leaves_the_chorus_as_it_was():
    t = chips(3)
    assert np.array_equal(render_chorus(t, 4.0, SR), render_chorus(t, 4.0, SR, patch=DEFAULT_PATCH))


def test_a_different_patch_changes_the_chorus():
    t = [dataclasses.replace(FALLBACK, gddr_temps=(30.0, 40.0) * 4)] * 3
    assert not np.allclose(render_chorus(t, 6.0, SR), render_chorus(t, 6.0, SR, patch=draw_patch(9, t)))


def test_the_patch_sets_the_turn_length():
    two = plan_turns(chips(2), 30.0, patch=dataclasses.replace(DEFAULT_PATCH, bars_per_turn=2))[1]
    four = plan_turns(chips(2), 30.0, patch=dataclasses.replace(DEFAULT_PATCH, bars_per_turn=4))[1]
    assert four == pytest.approx(2 * two)


def test_the_patch_sets_how_far_non_soloists_drop():
    quiet = dataclasses.replace(DEFAULT_PATCH, floor=0.05)
    loud = dataclasses.replace(DEFAULT_PATCH, floor=0.9)
    t = chips(3)
    a, b = render_chorus(t, 6.0, SR, patch=quiet), render_chorus(t, 6.0, SR, patch=loud)
    assert float(np.sqrt(np.mean(b ** 2))) > 1.1 * float(np.sqrt(np.mean(a ** 2)))


@pytest.mark.parametrize("change", [
    {"motif": (0, 1, 0, 1, 0, 1, 0, 1)},
    {"scale_name": "whole_tone", "scale": (0, 2, 4, 6, 8, 10)},
    {"gong_beat_hz": 1.2},
    {"gong_partials": (1.0, 3.3), "gong_amps": (1.0, 1.0)},
    {"shimmer_band": (6500.0, 7500.0)},
], ids=lambda c: next(iter(c)))
def test_each_device_field_of_a_patch_reaches_the_chorus(change):
    # Only one field changes, and the turn length and floor stay the same, so a difference
    # can only come from that field being passed down to its device.
    t = chips(1)
    base = render_chorus(t, 30.0, 22050, patch=DEFAULT_PATCH)
    changed = render_chorus(t, 30.0, 22050, patch=dataclasses.replace(DEFAULT_PATCH, **change))
    assert not np.allclose(base, changed)


# --- the wider readings decide who solos and add event sounds ------------------------------

from tensixinet.music.chorus import slot_chip  # noqa: E402


def _share(telemetry, chip, duration=3000.0):
    slots, _ = plan_turns(telemetry, duration)
    return sum(slot_chip(s, len(telemetry)) == chip for s in slots) / len(slots)


@pytest.mark.parametrize("change", [
    {"board_power": 300.0},  # a busier board than the others
    {"busy": 3},  # processes are using it
    {"asic_temperature": 88.0},  # close to the throttle temperature, same power
])
def test_a_chip_under_load_gets_more_turns_than_an_ordinary_one(change):
    base = [dataclasses.replace(FALLBACK, busy=0 if "busy" in change else None) for _ in range(4)]
    loaded = list(base)
    loaded[1] = dataclasses.replace(base[1], **change)
    assert _share(loaded, 1) > _share(base, 1) + 0.05


def test_a_chip_that_reports_no_process_list_is_not_treated_as_idle_or_busy():
    plain = [FALLBACK] * 3
    assert plan_turns(plain, 200.0) == plan_turns([dataclasses.replace(FALLBACK, busy=0)] * 3, 200.0)


def test_errors_are_added_to_the_chips_signal_whoever_has_the_floor():
    base = [FALLBACK] * 3
    sick = [FALLBACK, dataclasses.replace(FALLBACK, corr_errs=30, thermal_trips=1), FALLBACK]
    diff = render_chorus(sick, 4.0, SR) - render_chorus(base, 4.0, SR)
    assert np.abs(diff).max() > 0.02 and np.abs(render_chorus(sick, 4.0, SR)).max() < 1.0


def test_each_chip_gets_its_own_event_seed_so_two_sick_chips_do_not_tick_in_unison(monkeypatch):
    from tensixinet.music import chorus as chorus_module
    seen = []
    real = chorus_module.error_events
    monkeypatch.setattr(chorus_module, "error_events", lambda t, d, sr, seed: seen.append(seed) or real(t, d, sr, seed))
    render_chorus([FALLBACK] * 3, 0.3, SR)
    assert len(seen) == 3 and len(set(seen)) == 3
