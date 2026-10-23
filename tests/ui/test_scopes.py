# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The Play tab's scopes: one device, a module, or the whole system."""
import pytest

from tensixinet.music.chorus import device_names
from tensixinet.ui.scopes import MODULES, SYSTEM, heard_on, resolve_scope, scope_choices

DEVICES = ["ttrollz", "ttrax", "ttquantus", "ttdrone", "lead", "gong", "ultrasonic_shimmer", "ttsteam", "ttavdog", "ttchoir", "ttcoco"]


def test_the_modules_are_the_ones_the_plan_chose():
    assert MODULES == {
        "rhythm section": ("ttrollz", "gong", "ttavdog", "ttsteam"),
        "drones and air": ("ttdrone", "ultrasonic_shimmer", "ttchoir"),
        "melody": ("lead", "ttquantus", "ttrax"),
        "delay and echo": ("ttquantus", "ttcoco"),
    }
    assert SYSTEM == "the whole system"


def test_every_module_member_is_a_real_device():
    assert device_names() == DEVICES  # the device list this file was written against
    for members in MODULES.values():
        assert set(members) <= set(DEVICES)


def test_scope_choices_start_with_the_system_then_modules_then_every_device_once():
    choices = scope_choices()
    values = [v for _, v in choices]
    assert choices[0] == (SYSTEM, "system")
    assert len(set(values)) == len(values)
    assert [v for v in values if v.startswith("module:")] == [f"module:{m}" for m in MODULES]
    assert [v for v in values if v.startswith("device:")] == [f"device:{d}" for d in DEVICES]
    kinds = [v.split(":")[0] for v in values[1:]]
    assert kinds == sorted(kinds, key=["module", "device"].index)  # modules before devices


def test_resolve_scope_round_trips_every_choice():
    assert resolve_scope("system") is None
    for module, members in MODULES.items():
        assert resolve_scope(f"module:{module}") == list(members)
    for device in DEVICES:
        assert resolve_scope(f"device:{device}") == [device]


@pytest.mark.parametrize("bad", ["", "module:nope", "device:nope", "devices:gong", "gong"])
def test_resolve_scope_rejects_an_unknown_value_and_names_the_choices(bad):
    with pytest.raises(ValueError) as e:
        resolve_scope(bad)
    assert "system" in str(e.value)


@pytest.mark.parametrize("chips", [1, 2, 3, 4])
def test_every_device_is_dealt_to_some_chip_whatever_the_chip_count(chips):
    # n_slots is at least the number of devices, so even one chip plays all eleven (they were not all dealt in the first prototype).
    assert heard_on(None, chips) == DEVICES
    assert heard_on(["gong", "ttcoco"], chips) == ["gong", "ttcoco"]


def test_heard_on_keeps_the_order_of_the_scope_and_drops_a_name_that_is_not_a_device():
    assert heard_on(["ttcoco", "gong", "nope"], 2) == ["ttcoco", "gong"]


def test_heard_on_needs_at_least_one_chip():
    with pytest.raises(ValueError):
        heard_on(None, 0)
