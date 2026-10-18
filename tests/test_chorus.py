"""A chorus of devices, one per chip, for any chip count."""
import dataclasses

import numpy as np
import pytest

from tensixinet.chorus import DEVICES, assign_device, chip_devices, plan_turns, render_chorus
from tensixinet.mapping import aiclk_to_bpm
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


def test_voltage_is_documented_as_unused():
    # voltage has no musical mapping in the README; this pins that decision so
    # adding one later is a deliberate, tested change.
    base = [FALLBACK] * 4
    hi = [dataclasses.replace(FALLBACK, voltage=0.99)] * 4
    assert np.array_equal(render_chorus(base, 0.5, SR), render_chorus(hi, 0.5, SR))


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
    assert share > 0.6  # chip 0 owns 3 of the 7 slots (43%) but draws 90% of the power


def test_the_soloist_changes_with_system_load():
    quiet = [dataclasses.replace(FALLBACK, power=10.0) for _ in range(3)]
    loaded = [dataclasses.replace(FALLBACK, power=p) for p in (190.0, 10.0, 10.0)]
    assert plan_turns(quiet, 60.0)[0] != plan_turns(loaded, 60.0)[0]


def test_the_chorus_is_the_soloist_weighted_sum_of_its_devices_mixed_by_current():
    # Composition check for one chip, built from the parts that are tested on their own:
    # every device is scaled by its turn gain, the chip's devices are summed and divided by
    # sqrt(device count), then scaled by power and mixed by current.
    import math
    from tensixinet.devices import current_mix
    from tensixinet.mapping import power_to_volume
    from tensixinet.turns import turn_gains
    t = dataclasses.replace(FALLBACK, ddr_status=(1, 1, 1, 1), gddr_temps=(30.0, 38.0, 34.0, 29.0) * 2)
    dur = 20.0
    solo, turn_s = plan_turns([t], dur)
    gains = turn_gains(solo, len(DEVICES), int(dur * SR), SR, turn_s)
    total = sum(gains[s] * assign_device(s)(t, dur, SR, seed=s) for s in range(len(DEVICES)))
    expected = current_mix([total / math.sqrt(len(DEVICES)) * power_to_volume(t.power)], [t.current])
    assert np.allclose(render_chorus([t], dur, SR), expected.astype(np.float32), atol=1e-6)
