"""Self-logging: the project says what it is doing and why, and the claims are checked."""
import dataclasses

import numpy as np
import pytest

from tensixinet.music.chorus import DEVICES, assign_device
from tensixinet.music.devices import DEVICE_INPUTS
from tensixinet.narrate.explain import explain_chip, explain_mix
from tensixinet.telemetry import FALLBACK

SR = 8000
CHANGES = {
    "asic_temperature": 90.0,
    "aiclk": 1350.0,
    "gddr_temps": (26.0, 99.0) * 4,
    "ddr_status": (1, 1, 1, 1),
}


# --- the claim "device X uses fields Y" is true --------------------------------

@pytest.mark.parametrize("device", DEVICES, ids=lambda d: d.__name__)
@pytest.mark.parametrize("field", sorted(CHANGES))
def test_device_inputs_match_what_the_device_actually_reads(device, field):
    base = dataclasses.replace(FALLBACK, ddr_status=(0, 0, 0, 0))
    changed = dataclasses.replace(base, **{field: CHANGES[field]})
    differs = not np.allclose(device(base, 1.0, SR, 0), device(changed, 1.0, SR, 0))
    assert differs == (field in DEVICE_INPUTS[device.__name__]), (
        f"{device.__name__}: DEVICE_INPUTS says {sorted(DEVICE_INPUTS[device.__name__])}, "
        f"but changing {field} {'did' if differs else 'did not'} change the audio"
    )


def test_every_device_is_described():
    assert {d.__name__ for d in DEVICES} == set(DEVICE_INPUTS)


# --- explanations --------------------------------------------------------------

def subjects(decisions):
    return " | ".join(d.what for d in decisions)


def test_each_decision_has_what_and_why():
    for d in explain_chip(0, FALLBACK):
        assert d.what.strip() and d.why.strip()


def test_explanation_reports_the_numbers_the_mapping_actually_produces():
    t = dataclasses.replace(FALLBACK, aiclk=1350.0, asic_temperature=100.0, power=200.0)
    text = subjects(explain_chip(0, t))
    assert "180.0 BPM" in text  # aiclk 1350 -> 180 BPM
    assert "523.3 Hz" in text  # 100 C -> C5
    assert "volume 1.00" in text  # 200 W


def test_explanation_only_mentions_inputs_the_chips_device_consumes():
    rollz_chip = subjects(explain_chip(0, FALLBACK))  # chip 0 plays rollz
    sidrax_chip = subjects(explain_chip(1, FALLBACK))  # chip 1 plays sidrax
    assert "DDR" in rollz_chip and "GDDR" not in rollz_chip
    assert "GDDR" in sidrax_chip and "DDR status" not in sidrax_chip


def test_explanation_names_the_device_and_why_it_was_chosen():
    first = explain_chip(2, FALLBACK)[0]
    assert assign_device(2).__name__ in first.what
    assert "2" in first.why and "cycle" in first.why


def test_explain_mix_gives_each_chip_its_current_share_summing_to_one():
    chips = [dataclasses.replace(FALLBACK, current=c) for c in (10.0, 30.0)]
    decisions = explain_mix(chips)
    assert len(decisions) == 2
    assert "25%" in decisions[0].what and "75%" in decisions[1].what


def test_explain_mix_handles_zero_current():
    chips = [dataclasses.replace(FALLBACK, current=0.0)] * 2
    assert "50%" in explain_mix(chips)[0].what


# --- several devices per chip, the lead, and turn-taking --------------------------

from tensixinet.narrate.explain import explain_lead, explain_turns  # noqa: E402


def test_a_one_chip_box_says_it_plays_every_device_and_explains_the_inputs_of_all_of_them():
    decisions = explain_chip(0, FALLBACK, n_chips=1)
    first = decisions[0]
    assert all(d.__name__ in first.what for d in DEVICES)
    text = subjects(decisions)
    assert "DDR status" in text and "GDDR" in text  # rollz and sidrax both read their fields


def test_the_dealing_rule_is_explained_with_the_real_slot_numbers():
    first = explain_chip(1, FALLBACK, n_chips=3)[0]
    assert "slot" in first.why and "3 chips" in first.why


def test_the_lead_log_says_which_phrases_shift_or_invert_and_why():
    t = dataclasses.replace(FALLBACK, gddr_temps=(30.0, 40.0, 30.0, 30.0))
    d = explain_lead(t, 8.0)
    assert "invert" in d.what and "shift" in d.what
    assert "GDDR" in d.why and "average" in d.why


def test_the_lead_log_says_flat_readings_loop():
    d = explain_lead(dataclasses.replace(FALLBACK, gddr_temps=(33.0,) * 8), 8.0)
    assert "loop" in d.what


def test_turns_are_logged_with_the_soloist_and_the_power_reason():
    chips = [dataclasses.replace(FALLBACK, power=p) for p in (190.0, 10.0)]
    decisions = explain_turns(chips, 30.0)
    assert decisions and all(d.what.strip() and d.why.strip() for d in decisions)
    assert "solo" in decisions[0].what and "chip" in decisions[0].what
    assert "power" in decisions[0].why


def test_the_dealing_text_uses_correct_plurals():
    one = explain_chip(0, FALLBACK, n_chips=1)[0].why
    assert "1 chips" not in one and "1 chip" in one and "slots 0-6" in one
    many = explain_chip(2, FALLBACK, n_chips=3)[0].why
    assert "3 chips" in many and "slots 2, 5" in many
    single = explain_chip(2, FALLBACK, n_chips=9)[0].why
    assert "slot 2 " in single
