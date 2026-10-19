# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Self-logging: the project says what it is doing and why, and the claims are checked."""
import dataclasses

import numpy as np
import pytest

from tensixinet.music.chorus import DEVICES, assign_device
from tensixinet.music.devices import DEVICE_INPUTS
from tensixinet.narrate.explain import explain_chip, explain_mix
from tensixinet.telemetry import FALLBACK

SR = 8000
LONG = 15.0
# The snare plays the holes in the Tensix mask, so its base chip needs one for the other fields to matter.
BASE_OVERRIDES = {"ttsteam": {"tensix_mask": 0x37F7}}
CHANGES = {
    "asic_temperature": 90.0,
    "aiclk": 1350.0,
    "gddr_temps": (26.0, 99.0) * 4,
    "ddr_status": (1, 1, 1, 1),
    "voltage": 0.90,
    "power": 125.0,
    "axiclk": 1920.0,
    "arcclk": 1600.0,
    "heartbeat": 8,
    "tensix_mask": 0x1235,  # picked so the lead offset and gong detune both differ from the default chip
    "eth_mask": 0x0F0C,
    "pcie_usage": 5,
    "thm_limit_throttle": 45.0,
    "tdp_limit": 40.0,
    "temp_rate": 1.0,
    "power_rate": 25.0,
    # read by the chorus or by events.py, never by a device: every device must stay the same
    "board_power": 300.0,
    "busy": 3,
    "corr_errs": 5,
    "uncorr_errs": 1,
    "thermal_trips": 1,
}


# --- the claim "device X uses fields Y" is true --------------------------------

@pytest.mark.parametrize("device", DEVICES, ids=lambda d: d.__name__)
@pytest.mark.parametrize("field", sorted(CHANGES))
def test_device_inputs_match_what_the_device_actually_reads(device, field):
    base = dataclasses.replace(FALLBACK, ddr_status=(0, 0, 0, 0), **BASE_OVERRIDES.get(device.__name__, {}))
    changed = dataclasses.replace(base, **{field: CHANGES[field]})
    # long enough for a second gong strike, because the strike times depend on the clock, the heartbeat and the identity
    differs = not np.allclose(device(base, LONG, SR, 0), device(changed, LONG, SR, 0))
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
    ttrollz_chip = subjects(explain_chip(0, FALLBACK))  # chip 0 plays ttrollz
    ttrax_chip = subjects(explain_chip(1, FALLBACK))  # chip 1 plays ttrax
    assert "DDR" in ttrollz_chip and "GDDR" not in ttrollz_chip
    assert "GDDR" in ttrax_chip and "DDR status" not in ttrax_chip


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
    assert "DDR status" in text and "GDDR" in text  # ttrollz and ttrax both read their fields


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
    assert "1 chips" not in one and "1 chip" in one and f"slots 0-{len(DEVICES) - 1}" in one
    many = explain_chip(2, FALLBACK, n_chips=3)[0].why
    assert "3 chips" in many and "slots 2, 5" in many
    single = explain_chip(2, FALLBACK, n_chips=9)[0].why
    assert "slot 2 " in single


# --- explaining a patch -------------------------------------------------------------

from tensixinet.music.chorus import plan_turns  # noqa: E402
from tensixinet.music.patch import DEFAULT_PATCH, draw_patch, memory_to_asic_ratio  # noqa: E402
from tensixinet.narrate.explain import explain_patch  # noqa: E402

UNEVEN = [dataclasses.replace(FALLBACK, gddr_temps=(30.0, 54.0) * 4)] * 2


def joined(decisions):
    return " | ".join(d.what + " || " + d.why for d in decisions)


def test_every_patch_decision_has_what_and_why():
    ds = explain_patch(draw_patch(11, UNEVEN), UNEVEN, "test seed")
    assert len(ds) >= 6 and all(d.what.strip() and d.why.strip() for d in ds)


def test_the_patch_log_gives_the_seed_and_where_it_came_from():
    p = draw_patch(11, UNEVEN)
    first = explain_patch(p, UNEVEN, "chip readings plus entropy")[0]
    assert "11" in first.what and "chip readings plus entropy" in first.why


def test_the_scale_is_explained_from_the_gddr_spread_that_chose_it():
    p = draw_patch(11, UNEVEN)
    text = joined(explain_patch(p, UNEVEN, "s"))
    assert p.scale_name in text and "24" in text  # the spread is 54 - 30 = 24 C
    assert "GDDR" in text and "3 C" in text


def test_the_gong_tuning_is_explained_from_the_temperature_ratio():
    p = draw_patch(11, UNEVEN)
    text = joined(explain_patch(p, UNEVEN, "s"))
    assert p.gong_tuning in text and f"{memory_to_asic_ratio(UNEVEN):.2f}" in text


def test_the_drawn_choices_are_reported_with_their_real_values():
    p = draw_patch(11, UNEVEN)
    text = joined(explain_patch(p, UNEVEN, "s"))
    assert str(p.motif) in text
    assert f"{p.shimmer_band[0]:.0f}" in text and f"{p.shimmer_band[1]:.0f}" in text
    assert f"{p.bars_per_turn} bar" in text and f"{p.floor:.0%}" in text
    assert f"{p.gong_beat_hz:g} Hz" in text


def test_turn_times_follow_the_patch_turn_length():
    chips2 = [FALLBACK] * 2
    two = explain_turns(chips2, 60.0, patch=dataclasses.replace(DEFAULT_PATCH, bars_per_turn=2))
    four = explain_turns(chips2, 60.0, patch=dataclasses.replace(DEFAULT_PATCH, bars_per_turn=4))
    secs = lambda d: int(d.what.split("(")[1].split("s)")[0])
    assert abs(secs(four[1]) - 2 * secs(two[1])) <= 1  # the log rounds to whole seconds


def test_a_long_turn_list_says_how_many_more_turns_follow():
    chips2 = [FALLBACK] * 2
    one_bar = dataclasses.replace(DEFAULT_PATCH, bars_per_turn=1)
    ds = explain_turns(chips2, 60.0, limit=5, patch=one_bar)
    total = len(plan_turns(chips2, 60.0, one_bar)[0])
    assert len(ds) == 6 and total > 5
    assert f"{total - 5} more turns" in ds[-1].what and "limit" not in ds[-1].what.lower()
    assert len(explain_turns(chips2, 60.0, limit=total, patch=one_bar)) == total  # nothing left out, no extra line


# --- the wider readings are explained, each on the chip whose device uses it ---------------

from tensixinet.music.devices import gong_detune_cents  # noqa: E402
from tensixinet.music.lead import identity_offset  # noqa: E402
from tensixinet.music.mapping import chip_identity, chip_pressure, voltage_to_cents  # noqa: E402
from tensixinet.narrate.explain import explain_events, explain_turn_weights  # noqa: E402

LOADED = dataclasses.replace(
    FALLBACK, voltage=0.82, pcie_usage=4, arcclk=1600.0, axiclk=1920.0, eth_mask=0x3EDF,
    tensix_mask=0x37F7, heartbeat=8, asic_temperature=70.0, power=100.0,
)


def _chip_text(index):
    return subjects(explain_chip(index, LOADED, n_chips=len(DEVICES)))  # with one chip per device, chip i plays DEVICES[i] alone


def test_ttrollz_says_how_many_ethernet_links_are_up_and_what_lanes_that_makes():
    text = _chip_text(0)
    assert "12 of 16 ethernet links" in text and "lanes" in text and "GDDR" not in text


def test_ttrax_and_ttquantus_say_what_the_other_two_clocks_do():
    assert "ARC clock 1600 MHz" in _chip_text(1) and "x2.00" in _chip_text(1)
    assert "AXI clock 1920 MHz" in _chip_text(2) and "x2.00" in _chip_text(2)


def test_the_drone_says_its_detune_and_voice_count():
    text = _chip_text(3)
    assert f"voltage 0.82 V -> {voltage_to_cents(0.82):+.0f} cents" in text
    assert "PCIe usage 4 -> 4 drone voices" in text


def test_lead_and_shimmer_say_their_pressure_and_the_limit_it_came_from():
    p = chip_pressure(LOADED)
    for index in (4, 6):
        text = subjects(explain_chip(index, LOADED, n_chips=7))
        assert f"pressure {p:.2f}" in text
    why = " ".join(d.why for d in explain_chip(4, LOADED, n_chips=7))
    assert "90" in why and "125" in why  # the throttle temperature and the power limit


def test_lead_and_gong_say_what_the_chips_identity_does():
    ident = chip_identity(LOADED.tensix_mask, LOADED.eth_mask)
    assert f"+{identity_offset(ident)} degrees" in _chip_text(4)
    assert f"{gong_detune_cents(ident):+d} cents" in _chip_text(5)
    assert "heartbeat 8" in _chip_text(5)


def test_a_device_that_does_not_read_a_field_does_not_mention_it():
    assert "PCIe" not in _chip_text(0) and "voltage" not in _chip_text(0)
    assert "ethernet" not in _chip_text(3)


def test_turn_weights_show_each_factor_and_the_chip_they_belong_to():
    chips = [FALLBACK, dataclasses.replace(FALLBACK, board_power=240.0, busy=2, asic_temperature=80.0, power=90.0)]
    ds = explain_turn_weights(chips)
    assert len(ds) == 2 and "chip 1" in ds[1].what
    assert "2 processes" in ds[1].what and "board" in ds[1].what and "pressure" in ds[1].what
    assert "processes" not in ds[0].what  # busy is unknown for the first chip


def test_events_say_nothing_is_wrong_when_nothing_is_wrong():
    assert "no memory errors" in explain_events(0, FALLBACK).what


def test_events_name_each_kind_and_its_count_and_sound():
    d = explain_events(2, dataclasses.replace(FALLBACK, corr_errs=7, uncorr_errs=1, thermal_trips=2))
    assert "7 correctable" in d.what and "1 uncorrectable" in d.what and "2 thermal trips" in d.what
    assert "tick" in d.why and "breath" in d.why and "sweep" in d.why


def test_turn_reason_mentions_load_not_just_power():
    why = explain_turns([FALLBACK, FALLBACK], 30.0)[0].why
    assert "pressure" in why and "process" in why and "board" in why


def test_each_wider_decision_appears_only_on_chips_that_play_its_device():
    texts = {i: _chip_text(i) for i in range(len(DEVICES))}  # chip i plays DEVICES[i] alone
    others = lambda keep: [i for i in texts if i not in keep]
    assert "heartbeat" in texts[5] and not any("heartbeat" in texts[i] for i in others({5}))
    assert "ethernet" in texts[0] and not any("ethernet" in texts[i] for i in others({0}))
    assert "ARC clock" in texts[1] and not any("ARC clock" in texts[i] for i in others({1}))
    assert "AXI clock" in texts[2] and not any("AXI clock" in texts[i] for i in others({2}))
    assert all("pressure" in texts[i] for i in (1, 4, 6, 7, 8, 9, 10)) and not any("pressure" in texts[i] for i in others({1, 4, 6, 7, 8, 9, 10}))  # ttrax's chaos, the lead, the shimmer, the snare, the ttavdog's wave shape, the ttchoir's glottal slope, ttcoco's feedback
    assert "vowels" in texts[9] and "voice" in texts[9] and not any("vowels" in texts[i] for i in others({9}))
    assert "wave shape" in texts[8] and not any("wave shape" in texts[i] for i in others({8}))
    assert "snare" in texts[7] and not any("snare" in texts[i] for i in others({7}))
    assert "chaos" in texts[1] and "barre walls" in texts[1] and "barre walls" not in texts[4]
    assert "gong tuned" in texts[5] and "lead starts" in texts[4] and "lead starts" not in texts[5]


def test_the_snare_says_which_steps_fire_how_tight_it_is_and_how_much_dust_there_is():
    text = subjects(explain_chip(7, dataclasses.replace(LOADED, tensix_mask=0x37F7, pcie_usage=4), n_chips=len(DEVICES)))
    assert "steps 3, 11" in text and "harvested" in " ".join(d.why for d in explain_chip(7, LOADED, n_chips=len(DEVICES)))
    assert "dust" in text and "60" in text  # four PCIe steps, fifteen pops each


def test_a_chip_with_every_column_enabled_says_the_snare_stays_silent():
    text = subjects(explain_chip(7, dataclasses.replace(LOADED, tensix_mask=0x3FFF), n_chips=len(DEVICES)))
    assert "no harvested" in text


def test_the_ttquantus_says_which_petal_its_castle_samples_and_why():
    from tensixinet.music.mapping import personality
    from tensixinet.music.ttquantussy import castle_offset
    decisions = explain_chip(2, LOADED, n_chips=len(DEVICES))
    text = subjects(decisions)
    assert f"{castle_offset(personality(LOADED))} places" in text and "castle" in text
    assert "castle" not in _chip_text(0) and "castle" not in _chip_text(7)


def test_the_gong_says_when_its_first_strikes_land_and_how_the_trigger_response_made_them():
    from tensixinet.music.devices import strike_times
    text = _chip_text(5)
    first = ", ".join(f"{s:.1f}" for s in strike_times(LOADED, 60.0)[:4])
    assert f"strikes at {first} s" in text
    why = " ".join(d.why for d in explain_chip(5, LOADED, n_chips=len(DEVICES)))
    assert "blocking window" in why and "clock" in why


def test_the_drone_and_the_shimmer_say_they_follow_the_gong_strikes():
    drone = _chip_text(3)
    assert "ripples" in drone and "gong strike" in drone
    why = " ".join(d.why for d in explain_chip(3, LOADED, n_chips=len(DEVICES)))
    assert "AVDog" in why
    assert "gong strike" in " ".join(d.why for d in explain_chip(6, LOADED, n_chips=len(DEVICES)))
    assert "gong strike" not in _chip_text(0) and "ripples" not in _chip_text(5)


def test_a_still_chip_says_chaos_is_its_pressure_and_a_moving_one_says_how_fast_it_moves():
    still = " ".join(d.what + " " + d.why for d in explain_chip(1, dataclasses.replace(FALLBACK, asic_temperature=65.0), n_chips=len(DEVICES)))
    assert "moving" not in still and "chaos 50%" in still
    moving = dataclasses.replace(FALLBACK, asic_temperature=65.0, temp_rate=0.5, power_rate=-25.0)
    decisions = explain_chip(1, moving, n_chips=len(DEVICES))
    line = next(d for d in decisions if "chaos" in d.what)
    assert "chaos 100%" in line.what and "+0.5 C/s" in line.why and "-25 W/s" in line.why and "moving" in line.why


# --- patch cables -----------------------------------------------------------------------------------------------

def test_each_cable_is_explained_with_its_kind_depth_chip_and_push():
    from tensixinet.music.cables import Cable, strength
    from tensixinet.music.chorus import slot_chip
    from tensixinet.music.mapping import chip_chaos
    from tensixinet.narrate.explain import explain_cables
    hot = dataclasses.replace(FALLBACK, asic_temperature=85.0, power=150.0)
    cold = dataclasses.replace(FALLBACK, asic_temperature=30.0, power=20.0)
    chips = [hot, cold]
    cables = [Cable("ttrollz", "ttchoir", "fm", 0.8), Cable("ttrax", "ttrollz", "crunch", 0.5)]
    out = explain_cables(cables, chips)
    assert len(out) == 2
    chaos = chip_chaos(chips[slot_chip(0, 2)])  # ttrollz is slot 0
    assert "ttrollz" in out[0].what and "ttchoir" in out[0].what and "fm" in out[0].what and "0.80" in out[0].what
    assert "chip 0" in out[0].what and f"{strength(cables[0], chaos):.2f}" in out[0].what
    assert "chip 1" in out[1].what  # ttrax is slot 1
    assert "chaos" in out[0].why and "mutates" in out[0].why
    assert "degrades" in out[1].why


def test_a_cable_kind_is_described_in_plain_words():
    from tensixinet.music.cables import KINDS, Cable
    from tensixinet.narrate.explain import explain_cables
    words = {"fm": "mutates", "crunch": "degrades", "inflate": "inflates", "fold": "folds"}
    assert set(words) == set(KINDS)
    for kind, word in words.items():
        assert word in explain_cables([Cable("ttrollz", "ttchoir", kind, 0.5)], [FALLBACK])[0].why


# --- ttrax's bar count in the log, for Tetrax and Quatrax too -----------------------------------------------------

def test_ttrax_reports_its_own_bar_count_not_a_hardcoded_seven():
    import dataclasses
    from tensixinet.music.patch import DEFAULT_PATCH
    why = lambda decisions: " | ".join(d.why for d in decisions)
    sidrax_why = why(explain_chip(1, LOADED, n_chips=len(DEVICES)))
    assert "7 ttrax barres" in sidrax_why
    tetrax = dataclasses.replace(DEFAULT_PATCH, rax_bars=4)
    tetrax_why = why(explain_chip(1, LOADED, n_chips=len(DEVICES), patch=tetrax))
    assert "4 ttrax barres" in tetrax_why and "7 ttrax barres" not in tetrax_why


def test_ttcoco_says_its_tap_spread_and_its_feedback():
    text = subjects(explain_chip(10, LOADED, n_chips=len(DEVICES)))
    why = " ".join(d.why for d in explain_chip(10, LOADED, n_chips=len(DEVICES)))
    assert "delay taps" in text and "bits" in text
    assert "two such chips" in why and "tap feeds back" in why
