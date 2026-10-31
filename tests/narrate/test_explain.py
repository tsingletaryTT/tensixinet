# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Self-logging: the project says what it is doing and why, and the claims are checked."""
import dataclasses

import numpy as np
import pytest

from tensixinet.music.chorus import DEVICES, assign_device
from tensixinet.music.devices import DEVICE_INPUTS
from tensixinet.music.patch import DEFAULT_PATCH
from tensixinet.narrate.explain import explain_chip, explain_mix
from tensixinet.telemetry import FALLBACK

SR = 8000
LONG = 15.0
# The snare plays the holes in the Tensix mask, so its base chip needs one for the other fields to matter.
# The gong follows one of the chip's rolls or its clock, picked by the chip's identity; the devices that follow the gong's strikes
# (and ttrollz, whose rolls they follow) read the DDR status, pressure and drift only when it follows a roll, so their base chip
# has an identity whose gong does. 0x3002 is one where every listed field below moves the strikes.
ROLL_LINKED = {"tensix_mask": 0x3002}
BASE_OVERRIDES = {
    "ttsteam": {"tensix_mask": 0x37F7},
    "gong": ROLL_LINKED, "ttdrone": ROLL_LINKED, "ultrasonic_shimmer": ROLL_LINKED, "ttavdog": ROLL_LINKED, "ttrollz": ROLL_LINKED,
}
# The looping line's vibrato reads the chip's pressure only when the patch moves its pitch, so those two devices are judged with a patch that does.
PATCH_OVERRIDES = {"ttquantus": dataclasses.replace(DEFAULT_PATCH, quantus_mod=0.8), "ttcoco": dataclasses.replace(DEFAULT_PATCH, quantus_mod=0.8)}
from tests.telemetry_changes import CHANGES  # noqa: E402  (shared with tests/music/test_choir_triggers.py)


# --- the claim "device X uses fields Y" is true --------------------------------

@pytest.mark.parametrize("device", DEVICES, ids=lambda d: d.__name__)
@pytest.mark.parametrize("field", sorted(CHANGES))
def test_device_inputs_match_what_the_device_actually_reads(device, field):
    if device.__name__ == "ttchoir":
        pytest.skip("one patch cannot exercise every trigger source; tests/music/test_choir_triggers.py runs the choir once per source")
    base = dataclasses.replace(FALLBACK, ddr_status=(0, 0, 0, 0), **BASE_OVERRIDES.get(device.__name__, {}))
    changed = dataclasses.replace(base, **{field: CHANGES[field]})
    patch = PATCH_OVERRIDES.get(device.__name__, DEFAULT_PATCH)
    # long enough for a second gong strike, because the strike times depend on the clock, the heartbeat and the identity
    differs = not np.allclose(device(base, LONG, SR, 0, patch), device(changed, LONG, SR, 0, patch))
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
from tensixinet.music.patch import DEFAULT_PATCH as DEFAULT_PATCH_FOR_ROUTES  # noqa: E402

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


def test_the_patch_log_says_which_steam_routes_are_on_and_what_each_does():
    off = joined(explain_patch(dataclasses.replace(DEFAULT_PATCH_FOR_ROUTES, steam_routes=()), UNEVEN, "s"))
    assert "steam routes: none" in off
    on = explain_patch(dataclasses.replace(DEFAULT_PATCH_FOR_ROUTES, steam_routes=("strikes", "pitch")), UNEVEN, "s")
    route_lines = [d for d in on if d.what.startswith("steam routes")]
    assert len(route_lines) == 1
    line = route_lines[0]
    assert "strikes, pitch" in line.what
    assert "gong" in line.why and "dogtone" in line.why          # what the two routes do
    assert "ttrollz clock" not in line.why and "pings the AVDog's filter" not in line.why  # and nothing about the routes that are off


def test_the_log_says_how_the_rolls_are_wired_how_hard_they_pull_and_how_far_they_drift():
    from tensixinet.music.devices import ttrollz_network
    from tensixinet.music.mapping import chip_pressure, personality
    from tensixinet.music.ttrolzer import draw_links, roll_drift
    t = dataclasses.replace(FALLBACK, ddr_status=(1, 1, 1, 1), eth_mask=0x00FF, asic_temperature=70.0)
    patch = dataclasses.replace(DEFAULT_PATCH_FOR_ROUTES, seed=21)
    net = ttrollz_network(t)
    links = draw_links(len(net.rolls), 21, personality(t), chip_pressure(t), [lane.index for lane in net.lanes])
    ds = explain_chip(0, t, patch=patch)  # chip 0 plays ttrollz
    wiring = [d for d in ds if "Browns patched" in d.what]
    assert len(wiring) == 1
    text = wiring[0].what + " || " + wiring[0].why
    assert f"{len(links)} Browns patched between its {len(net.rolls)} rolls" in text and f"{len(net.lanes)} of them heard" in text
    for l in links[:6]:
        src, dst = net.rolls[l.source], net.rolls[l.target]
        assert f"{src.sides}-roll ({src.bpm:.0f} BPM) to {dst.sides}-roll ({dst.bpm:.0f} BPM)" in text  # who is patched into whom
    if len(links) > 6:
        assert f"and {len(links) - 6} more" in text
    assert f"{roll_drift(70.0):.1%}" in text and f"{max(l.pull for l in links):.0%}" in text
    assert "Sandrode" in text and "cycle" in text and "accent" in text


def test_the_log_says_what_the_gong_is_patched_to():
    from tensixinet.music.devices import gong_trigger_roll, ttrollz_network
    from tensixinet.music.mapping import personality
    for mask in range(0x3000, 0x3100):
        t = dataclasses.replace(FALLBACK, ddr_status=(1, 1, 1, 1), eth_mask=0xFFFF, tensix_mask=mask)
        which = gong_trigger_roll(t, DEFAULT_PATCH_FOR_ROUTES)
        text = " | ".join(d.what + " || " + d.why for d in explain_chip(5, t))  # slot 5 is the gong
        if which is None:
            assert "gong is patched to the chip's own ttrollz clock" in text
        else:
            roll = ttrollz_network(t).rolls[which]
            assert f"gong is patched to roll {which} of 40, a {roll.sides}-roll ({roll.bpm:.0f} BPM)" in text
            assert "the roll's own pulses" in text
    chipless = dataclasses.replace(FALLBACK, ddr_status=(0, 0, 0, 0), eth_mask=0x0000)
    assert "gong is patched to the chip's own ttrollz clock" in " | ".join(d.what for d in explain_chip(5, chipless))


def test_the_log_says_where_each_chip_sits_how_far_it_orbits_and_what_the_field_is():
    from tensixinet.music.mapping import chip_chaos
    from tensixinet.music.space import ORBIT_CALM, ORBIT_CHAOS, chip_azimuth
    from tensixinet.narrate.explain import explain_space
    chips = mock_telemetry_for_space()
    ds = explain_space(chips, DEFAULT_PATCH_FOR_ROUTES)
    assert len(ds) == 1 + len(chips) and ds[0].what.startswith("stereo field")
    assert "azimuth" in ds[0].why and "width" in ds[0].why and "depth" in ds[0].why and "orbit" in ds[0].why
    for i, (t, d) in enumerate(zip(chips, ds[1:])):
        assert d.what.startswith(f"chip {i} sits ") and f"{chip_azimuth(i, len(chips), t):+.2f}" in d.what
        reach = ORBIT_CALM + ORBIT_CHAOS * min(1.0, chip_chaos(t))
        assert f"orbits {reach:.2f}" in d.what or f"orbits {reach:.2f}" in d.why
    assert "left" in ds[1].what and "right" in ds[-1].what
    assert "centre" in explain_space(chips[:1], DEFAULT_PATCH_FOR_ROUTES)[1].what


def mock_telemetry_for_space():
    from tensixinet.telemetry.mock import mock_telemetry
    return mock_telemetry(4, 3)


def test_the_patch_log_says_how_much_of_the_snare_plays_and_why():
    quiet = [d for d in explain_patch(dataclasses.replace(DEFAULT_PATCH_FOR_ROUTES, snare=0.0), UNEVEN, "s") if d.what.startswith("snare")]
    some = [d for d in explain_patch(dataclasses.replace(DEFAULT_PATCH_FOR_ROUTES, snare=0.2), UNEVEN, "s") if d.what.startswith("snare")]
    full = [d for d in explain_patch(DEFAULT_PATCH_FOR_ROUTES, UNEVEN, "s") if d.what.startswith("snare")]
    assert len(quiet) == len(some) == len(full) == 1
    assert "silent" in quiet[0].what and "20%" in some[0].what and "every hit" in full[0].what
    assert "40%" in some[0].why                       # the odds a drawn patch has no snare


def test_the_patch_log_says_how_much_goes_through_the_coco_and_how_big_the_room_is():
    p = dataclasses.replace(DEFAULT_PATCH_FOR_ROUTES, coco_send=0.6, coco_reverb=0.3, reverb_s=2.5)
    lines = [d for d in explain_patch(p, UNEVEN, "s") if d.what.startswith("coco bus")]
    assert len(lines) == 1
    assert "60%" in lines[0].what and "2.5 s" in lines[0].what and "30%" in lines[0].what
    assert "petals" in lines[0].why and "8-bit" in lines[0].why
    off = [d for d in explain_patch(DEFAULT_PATCH_FOR_ROUTES, UNEVEN, "s") if d.what.startswith("coco bus")]
    assert len(off) == 1 and "off" in off[0].what


def test_the_steam_route_reason_gives_the_odds_the_seed_used():
    why = [d.why for d in explain_patch(dataclasses.replace(DEFAULT_PATCH_FOR_ROUTES, steam_routes=("strikes",)), UNEVEN, "s")
           if d.what.startswith("steam routes")][0]
    assert "60%" not in why and "none 10%" in why and "three 20%" in why


def test_the_patch_log_says_how_many_of_the_cocos_notes_rest():
    on = [d for d in explain_patch(dataclasses.replace(DEFAULT_PATCH_FOR_ROUTES, rests=0.3), UNEVEN, "s") if d.what.startswith("rests")]
    off = [d for d in explain_patch(DEFAULT_PATCH_FOR_ROUTES, UNEVEN, "s") if d.what.startswith("rests")]
    assert len(on) == len(off) == 1 and "30%" in on[0].what and "none" in off[0].what
    assert "castle" in on[0].why and "15 to 45%" in on[0].why


def test_the_patch_log_says_how_much_of_the_lead_and_the_rolls_rests():
    on = explain_patch(dataclasses.replace(DEFAULT_PATCH_FOR_ROUTES, lead_rests=0.2, roll_rests=0.15), UNEVEN, "s")
    off = explain_patch(DEFAULT_PATCH_FOR_ROUTES, UNEVEN, "s")
    for name, share in (("lead rests", "20%"), ("roll rests", "15%")):
        a, b = [[d for d in ds if d.what.startswith(name)] for ds in (on, off)]
        assert len(a) == len(b) == 1 and share in a[0].what and "none" in b[0].what and "10 to 30%" in a[0].why
    assert "downbeat" in [d for d in on if d.what.startswith("lead rests")][0].why


def test_the_patch_log_says_how_far_the_looping_lines_pitch_is_moved():
    on = [d for d in explain_patch(dataclasses.replace(DEFAULT_PATCH_FOR_ROUTES, quantus_mod=0.8), UNEVEN, "s") if d.what.startswith("pitch movement")]
    off = [d for d in explain_patch(DEFAULT_PATCH_FOR_ROUTES, UNEVEN, "s") if d.what.startswith("pitch movement")]
    assert len(on) == len(off) == 1 and "80%" in on[0].what and "none" in off[0].what
    for text in ("3 scale degrees", "glide", "vibrato", "0.35 to 0.9"):
        assert text in on[0].why, text


def test_the_patch_log_names_the_choirs_sources_and_numbers():
    p = dataclasses.replace(DEFAULT_PATCH, choir_onset="gong", choir_coda="rolls", choir_howl="register",
                            choir_howl_odds=0.05, choir_mix=(0.5, 0.3, 0.2), choir_crush=0.6, choir_drift=0.4)
    ds = explain_patch(p, UNEVEN, "test")
    text = " | ".join(d.what for d in ds)
    assert "choir syllables on the gong's strikes" in text and "codas on the rolls" in text
    assert "5% of the register's hits" in text and "4 s" in " ".join(d.why for d in ds)
    assert "plosive 50%" in text and "fricative 30%" in text and "nasal 20%" in text
    assert "speech-chip crush 60% at full pressure" in text
    assert "drift 40%" in text
    off = " | ".join(d.what for d in explain_patch(DEFAULT_PATCH, UNEVEN, "test"))
    assert "choir: vowels on the bar" in off and "choir syllables" not in off


def test_the_choir_log_says_where_each_value_came_from_and_handles_no_coda_or_howl():
    p = dataclasses.replace(DEFAULT_PATCH, choir_onset="rax", choir_mix=(0.1, 0.3, 0.6), choir_crush=0.0, style="enya")
    ds = explain_patch(p, UNEVEN, "test")
    what = " | ".join(d.what for d in ds)
    why = " | ".join(d.why for d in ds)
    assert "choir syllables on ttrax's gestures" in what
    assert "no codas" in what and "no howl" in what and "crush: none" in what
    crushed = explain_patch(dataclasses.replace(p, choir_crush=0.5), UNEVEN, "test")
    assert "speech-chip crush 50% at full pressure, none at rest" in " | ".join(d.what for d in crushed)
    assert "set by the enya style" in why
    drawn = explain_patch(dataclasses.replace(DEFAULT_PATCH, choir_onset="lead"), UNEVEN, "test")
    assert "drawn from the seed" in " ".join(d.why for d in drawn)


def _choir_lines(**kw):
    """The patch log's choir lines as 'what | why' strings, keyed by the start of their subject."""
    ds = explain_patch(dataclasses.replace(DEFAULT_PATCH, **kw), UNEVEN, "test")
    return {d.what.split(" ")[0] + " " + d.what.split(" ")[1]: d.what + " | " + d.why for d in ds}


def test_the_consonant_mix_line_says_the_trigger_strength_scales_the_mix():
    line = _choir_lines(choir_onset="gong", choir_mix=(0.5, 0.3, 0.2))["consonant mix:"]
    assert "plosive 50%" in line and "base lean" in line
    assert "a stronger trigger leans to plosives, a weaker one to fricatives and nasals" in line
    assert "at least 30% of its weight" in line
    assert "shares of the consonants that are sung" not in line


def test_the_crush_line_names_the_bits_from_the_crush_constants():
    line = _choir_lines(choir_onset="gong", choir_crush=0.5)["speech-chip crush"]
    assert "8-bit" not in line and "12 bits" in line and "8 bits at full pressure" in line  # round(12 - 8 * 0.5) = 8
    assert "4 bits at the strongest crush" in line
    assert "7 bits at full pressure" in _choir_lines(choir_onset="gong", choir_crush=0.6)["speech-chip crush"]  # round(7.2)


def test_the_choir_lines_build_their_numbers_from_the_constants(monkeypatch):
    from tensixinet.narrate import explain as explain_module
    monkeypatch.setattr(explain_module, "XFADE_S", 0.035)
    monkeypatch.setattr(explain_module, "CHOIR_CODA_ODDS", (0.5, 0.2, 0.3))
    monkeypatch.setattr(explain_module, "CRUSH_BITS", (16, 2))
    monkeypatch.setattr(explain_module, "FAMILY_FLOOR", 0.25)
    lines = _choir_lines(choir_onset="gong", choir_coda="rolls", choir_crush=0.5)
    assert "35 ms" in lines["choir syllables"] and "20 ms" not in lines["choir syllables"]
    assert "none 50% of the time, the onset's own source 20%, another source 30%" in lines["codas on"]
    assert "16 bits" in lines["speech-chip crush"] and "9 bits at full pressure" in lines["speech-chip crush"]  # 16 - 14 * 0.5
    assert "2 bits at the strongest crush" in lines["speech-chip crush"]
    assert "at least 25% of its weight" in lines["consonant mix:"]
    assert "no coda source 50% of the time" in _choir_lines(choir_onset="gong")["no codas"]


def test_the_choir_lines_say_what_a_style_sets_and_what_the_seed_draws():
    # A style sets the consonant mix and the crush; the sources (onset, coda, howl) and the drift come from the seed in every style.
    styled = _choir_lines(choir_onset="gong", choir_crush=0.5, style="enya")
    assert "set by the enya style" in styled["consonant mix:"] and "set by the enya style" in styled["speech-chip crush"]
    assert "drawn from the seed" in styled["choir syllables"] and "enya" in styled["choir syllables"]
    plain = _choir_lines(choir_onset="gong", choir_crush=0.5)
    assert "drawn from the seed" in plain["choir syllables"] and "style" not in plain["choir syllables"]
    assert "drawn from the seed" in plain["consonant mix:"]


# --- the choir's chip lines and its howls -------------------------------------------------------------------------

from tensixinet.music.chorus import SEGMENT_SEED_STRIDE  # noqa: E402
from tensixinet.music.ttchoir import howl_times  # noqa: E402
from tensixinet.narrate.explain import explain_howls  # noqa: E402

HARVESTED_CHIP = dataclasses.replace(FALLBACK, tensix_mask=0x37F7, ddr_status=(1, 1, 1, 1))  # its register fires


def test_the_chip_log_says_what_triggers_the_choir():
    p = dataclasses.replace(DEFAULT_PATCH, choir_onset="lead", choir_coda="rax")
    text = subjects(explain_chip(9, FALLBACK, 11, p))  # slot 9 plays ttchoir on 11 chips
    assert "syllables on the lead's notes" in text and "codas on ttrax's gestures" in text
    whys = " | ".join(d.why for d in explain_chip(9, FALLBACK, 11, p))
    assert "each bar" not in whys and "on each trigger" in whys
    old = explain_chip(9, FALLBACK, 11)  # the old choir's line is unchanged: its reason (not its subject) says "each bar"
    assert "each bar" in " | ".join(d.why for d in old) and "syllables" not in subjects(old)


def test_a_choir_with_no_coda_says_so_on_the_chip_line():
    p = dataclasses.replace(DEFAULT_PATCH, choir_onset="lead")
    text = subjects(explain_chip(9, FALLBACK, 11, p))
    assert "syllables on the lead's notes" in text and "codas on" not in text


def test_a_source_that_cannot_fire_on_this_chip_is_said():
    p = dataclasses.replace(DEFAULT_PATCH, choir_onset="register")
    waiting = [d for d in explain_chip(9, FALLBACK, 11, p, duration=10.0) if "waiting" in d.what]  # every Tensix column is on: no hits
    assert len(waiting) == 1 and "register" in waiting[0].what and "silent" in waiting[0].what
    firing = explain_chip(9, HARVESTED_CHIP, 11, p, duration=10.0)
    assert not any("waiting" in d.what for d in firing)
    assert not any("waiting" in d.what for d in explain_chip(9, FALLBACK, 11, p))  # no duration, nothing to check
    assert not any("waiting" in d.what for d in explain_chip(9, FALLBACK, 11, duration=10.0))  # the old choir has no source
    assert not any("waiting" in d.what for d in explain_chip(0, FALLBACK, 11, p, duration=10.0))  # a chip that plays no choir


def test_explain_howls_gives_one_line_per_fire_with_its_chip_and_time():
    p = dataclasses.replace(DEFAULT_PATCH, choir_onset="gong", choir_howl="register", choir_howl_odds=1.0)
    lines = explain_howls([HARVESTED_CHIP] * 4, 20.0, p, sample_rate=8000)
    expected = howl_times(HARVESTED_CHIP, 20.0, 0 * SEGMENT_SEED_STRIDE + 9, p, 8000)
    assert len(expected) > 0 and len(lines) == len(expected)
    assert all(chip == 9 % 4 for chip, _ in lines)
    assert "howls" in lines[0][1].what and "-1 dBFS" in lines[0][1].why
    assert f"{expected[0]:.2f}" in lines[0][1].what


def test_explain_howls_uses_the_seed_the_chorus_gives_the_slot_and_its_segment():
    p = dataclasses.replace(DEFAULT_PATCH, choir_onset="gong", choir_howl="register", choir_howl_odds=0.5)
    chips = [HARVESTED_CHIP] * 4
    for segment in (0, 3):
        got = [round(float(w), 3) for w in howl_times(HARVESTED_CHIP, 40.0, segment * SEGMENT_SEED_STRIDE + 9, p, 8000)]
        said = [d.what for _, d in explain_howls(chips, 40.0, p, segment=segment, sample_rate=8000)]
        assert len(said) == len(got) and all(f"{w:.2f}" in line for w, line in zip(got, said))


def test_explain_howls_is_empty_without_a_howl_source_or_without_a_choir_slot():
    assert explain_howls([HARVESTED_CHIP] * 4, 20.0, dataclasses.replace(DEFAULT_PATCH, choir_onset="gong"), sample_rate=8000) == []
    assert explain_howls([HARVESTED_CHIP] * 4, 20.0, DEFAULT_PATCH) == []


def test_the_howl_line_builds_its_numbers_from_the_choirs_constants(monkeypatch):
    from tensixinet.narrate import explain as explain_module
    p = dataclasses.replace(DEFAULT_PATCH, choir_onset="gong", choir_howl="register", choir_howl_odds=1.0)
    monkeypatch.setattr(explain_module, "HOWL_MAX_S", 2.5)
    monkeypatch.setattr(explain_module, "HOWL_CEILING", 10 ** (-3 / 20))
    why = explain_howls([HARVESTED_CHIP] * 4, 20.0, p, sample_rate=8000)[0][1].why
    assert "2.5 s" in why and "-3 dBFS" in why and "4 s" not in why and "-1 dBFS" not in why
    patch_why = " ".join(d.why for d in explain_module.explain_patch(p, [HARVESTED_CHIP] * 4, "test"))
    assert "2.5 s" in patch_why and "-3 dBFS" in patch_why


def test_howls_are_not_claimed_for_a_choir_that_is_not_rendered():
    p = dataclasses.replace(DEFAULT_PATCH, choir_onset="gong", choir_howl="register", choir_howl_odds=1.0)
    chips = [HARVESTED_CHIP] * 4
    assert explain_howls(chips, 20.0, p, sample_rate=8000)
    assert explain_howls(chips, 20.0, p, sample_rate=8000, only=["gong"]) == []
    assert explain_howls(chips, 20.0, p, sample_rate=8000, only=["ttchoir", "gong"])
    assert explain_howls(chips, 20.0, p, sample_rate=8000, silent={9 % 4}) == []  # an idle (evicted) chip is silent
    assert explain_howls(chips, 20.0, p, sample_rate=8000, silent={0})


def test_explain_howls_can_leave_out_the_crossfade_tail():
    p = dataclasses.replace(DEFAULT_PATCH, choir_onset="gong", choir_howl="register", choir_howl_odds=1.0)
    every = explain_howls([HARVESTED_CHIP] * 4, 20.0, p, sample_rate=8000)
    times = howl_times(HARVESTED_CHIP, 20.0, 9, p, 8000)
    assert len(times) >= 2
    cut = (times[0] + times[1]) / 2
    kept = explain_howls([HARVESTED_CHIP] * 4, 20.0, p, sample_rate=8000, before=cut)
    assert len(every) == len(times) and len(kept) == 1 and f"{times[0]:.2f}" in kept[0][1].what


def test_the_waiting_check_uses_the_segment_it_is_given(monkeypatch):
    from tensixinet.narrate import explain as explain_module
    seen = []
    real = explain_module.choir_triggers
    monkeypatch.setattr(explain_module, "choir_triggers", lambda *a, **k: seen.append(a[4]) or real(*a, **k))
    p = dataclasses.replace(DEFAULT_PATCH, choir_onset="register")
    explain_chip(9, FALLBACK, 11, p, duration=10.0, segment=7)
    assert seen == [7]


def test_the_patch_log_says_when_a_saved_patch_picks_the_choirs_consonant_and_vowel():
    def texts(**kw):
        ds = explain_patch(dataclasses.replace(DEFAULT_PATCH, **kw), UNEVEN, "test")
        return [d.what + " | " + d.why for d in ds if "pick" in d.what]
    assert texts(choir_onset="gong") == [] and texts(choir_onset="gong", choir_pick=("", "")) == []
    both = texts(choir_onset="gong", choir_pick=("sh", "i"))
    assert len(both) == 1 and "sh" in both[0] and "ee" in both[0]
    assert "consonant" in texts(choir_onset="gong", choir_pick=("t", ""))[0] and "vowel" not in texts(choir_onset="gong", choir_pick=("t", ""))[0].split("|")[0]
    assert "vowel" in texts(choir_onset="gong", choir_pick=("", "o"))[0].split("|")[0]
    assert "no onset source" in texts(choir_pick=("t", "a"))[0]
