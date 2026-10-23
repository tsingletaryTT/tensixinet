# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
import json

import pytest

from tensixinet.telemetry import mock_telemetry
from tensixinet.ui.instrument import InstrumentSession
from tensixinet.ui.knobs import KNOB_RANGES, Knobs
from tensixinet.ui.scopes import scope_choices

from tensixinet.music.patch import DEFAULT_PATCH, save_patch
from tensixinet.music.styles import STYLES
from tensixinet.ui.forms import DRAW, PLAIN, SOURCES, new_instrument, new_session, patch_choices, read_chip_knobs, style_choices

SLEEP = lambda s: None


def test_the_style_choices_are_a_plain_draw_then_every_style():
    assert style_choices() == ["plain draw"] + list(STYLES)


def test_the_patch_choices_are_the_seeded_draw_then_each_saved_patch_by_name(tmp_path):
    (tmp_path / "b.patch.json").write_text("{}")
    (tmp_path / "a.patch.json").write_text("{}")
    (tmp_path / "notes.txt").write_text("")
    assert patch_choices([tmp_path]) == [DRAW, "a.patch.json", "b.patch.json"]
    assert patch_choices([tmp_path / "nowhere"]) == [DRAW]


def test_a_new_session_from_the_form_draws_the_patch_in_the_chosen_style_from_the_seed():
    s, patch = new_session(source=SOURCES[0], style="enya", seed=7, chips=4, load_s=0, patch_name=DRAW, patch_dirs=[], sleep=SLEEP)
    assert patch.style == "enya" and patch.seed != 0
    s2, patch2 = new_session(source=SOURCES[0], style="enya", seed=7, chips=4, load_s=0, patch_name=DRAW, patch_dirs=[], sleep=SLEEP)
    assert patch2 == patch  # the same form gives the same patch


def test_a_plain_draw_has_no_style():
    _, patch = new_session(source=SOURCES[0], style="plain draw", seed=7, chips=4, load_s=0, patch_name=DRAW, patch_dirs=[], sleep=SLEEP)
    assert patch.style == ""


def test_a_saved_patch_is_used_as_it_is_whatever_the_style_and_seed_say(tmp_path):
    save_patch(DEFAULT_PATCH, tmp_path / "mine.patch.json")
    _, patch = new_session(source=SOURCES[0], style="aphex", seed=99, chips=4, load_s=0, patch_name="mine.patch.json", patch_dirs=[tmp_path], sleep=SLEEP)
    assert patch == DEFAULT_PATCH


def test_the_synthetic_source_has_the_chosen_number_of_chips_and_a_load_cycle_when_asked():
    s, _ = new_session(source=SOURCES[0], style="plain draw", seed=1, chips=3, load_s=0, patch_name=DRAW, patch_dirs=[], sleep=SLEEP)
    assert len(s._sample()) == 3
    s2, _ = new_session(source=SOURCES[0], style="plain draw", seed=1, chips=4, load_s=45, patch_name=DRAW, patch_dirs=[], sleep=SLEEP)
    assert len(s2._sample()) == 4


def test_an_unknown_source_or_a_missing_saved_patch_is_refused_with_the_reason(tmp_path):
    with pytest.raises(ValueError, match="source"):
        new_session(source="the moon", style="plain draw", seed=1, chips=4, load_s=0, patch_name=DRAW, patch_dirs=[], sleep=SLEEP)
    with pytest.raises(ValueError, match="patch"):
        new_session(source=SOURCES[0], style="plain draw", seed=1, chips=4, load_s=0, patch_name="gone.patch.json", patch_dirs=[tmp_path], sleep=SLEEP)


def test_the_real_source_draws_its_patch_from_the_first_reading_and_a_box_with_no_chips_is_refused():
    from tensixinet.telemetry import mock_telemetry
    from tensixinet.ui.session import make_patch
    chips = mock_telemetry(4, 5)
    s, patch = new_session(source=SOURCES[1], style="blues", seed=3, chips=4, load_s=0, patch_name=DRAW, patch_dirs=[], sleep=SLEEP,
                           real=lambda: (lambda: chips))
    assert patch == make_patch("blues", 3, chips) and s._sample() == chips
    with pytest.raises(ValueError, match="no chips"):
        new_session(source=SOURCES[1], style="blues", seed=3, chips=4, load_s=0, patch_name=DRAW, patch_dirs=[], sleep=SLEEP, real=lambda: (lambda: []))


def test_the_load_cycle_moves_the_synthetic_chips_powers_and_zero_leaves_them_still():
    now = [0.0]
    cycling, _ = new_session(SOURCES[0], PLAIN, 1, 4, 60, DRAW, [], SLEEP, clock=lambda: now[0])
    still, _ = new_session(SOURCES[0], PLAIN, 1, 4, 0, DRAW, [], SLEEP, clock=lambda: now[0])
    a_c, a_s = [c.power for c in cycling._sample()], [c.power for c in still._sample()]
    now[0] = 20.0
    assert [c.power for c in cycling._sample()] != a_c and [c.power for c in still._sample()] == a_s


# --- the kernels in the simulator, and choices that depend on what the machine has ------------------------------------------

class _HostWorker:
    """A host-mode worker standing in for the simulator pool: it records what was asked of it and whether it was started and closed."""

    def __init__(self, devices):
        from tensixinet.media.chip import ChipWorker, host_command
        self.devices, self.inner, self.calls, self.started, self.closed = devices, ChipWorker(host_command(chunk_tiles=16, devices=devices), devices=devices), [], False, False

    def start(self):
        self.started = True
        self.inner.start()

    def run(self, op, inputs, n, device=0):
        self.calls.append((op, device))
        return self.inner.run(op, inputs, n, device=device)

    def close(self):
        self.closed = True
        self.inner.close()


def test_the_kernel_choices_offer_the_simulator_only_when_it_is_there():
    from tensixinet.ui.forms import CPU, SIMULATED, kernel_choices
    assert kernel_choices(lambda: "/x/tt-lang-sim") == [CPU, SIMULATED]
    assert kernel_choices(lambda: None) == [CPU]


def test_the_source_choices_offer_the_real_chips_only_when_tt_smi_is_there():
    from tensixinet.ui.forms import source_choices
    assert source_choices(lambda name: "/usr/bin/tt-smi") == list(SOURCES)
    assert source_choices(lambda name: None) == [SOURCES[0]]


def test_a_session_with_the_simulated_kernels_runs_the_kernel_devices_through_the_worker_and_closes_it():
    from tensixinet.ui.forms import SIMULATED
    from tensixinet.ui.session import Session
    made = []
    s, _ = new_session(SOURCES[0], PLAIN, 4, 4, 0, DRAW, [], SLEEP, kernels=SIMULATED,
                       worker_factory=lambda n: made.append(_HostWorker(n)) or made[-1])
    s.start()
    import time
    end = time.monotonic() + 40
    while time.monotonic() < end and not s.snapshot().state:
        time.sleep(0.05)
    try:
        assert s.snapshot().state is not None and made[0].devices == 4 and made[0].started
        assert {op for op, _ in made[0].calls} >= {"ttmix", "ttshimmer"}
        assert any("simulator" in r["what"] for r in s.snapshot().records if r["event"] == "backend")
    finally:
        s.stop()
    assert made[0].closed


def test_cpu_only_never_starts_a_worker():
    made = []
    s, _ = new_session(SOURCES[0], PLAIN, 4, 4, 0, DRAW, [], SLEEP, worker_factory=lambda n: made.append(n))
    assert s._worker is None and made == []


def test_a_simulator_that_cannot_start_ends_the_session_with_the_reason_and_still_closes_the_worker():
    from tensixinet.media.chip import ChipError
    from tensixinet.ui.forms import SIMULATED

    class Broken:
        closed = False

        def start(self):
            raise ChipError("no simulator today")

        def close(self):
            Broken.closed = True

    s, _ = new_session(SOURCES[0], PLAIN, 4, 4, 0, DRAW, [], SLEEP, kernels=SIMULATED, worker_factory=lambda n: Broken())
    s.start()
    import time
    end = time.monotonic() + 10
    while time.monotonic() < end and s.snapshot().running:
        time.sleep(0.05)
    assert "no simulator today" in (s.snapshot().error or "") and Broken.closed


def test_the_lead_goes_to_the_session_and_defaults_to_none():
    from tensixinet.ui.forms import SOURCES, new_session
    args = (SOURCES[0], "plain draw", 1, 2, 0, "draw from the seed and the style", [], lambda s: None)
    assert new_session(*args)[0]._lead_s == 0.0
    assert new_session(*args, lead_s=7.5)[0]._lead_s == 7.5


# --- the Play tab ------------------------------------------------------------------------------------------------------------------

def _instrument(scope="system", style="plain draw", seed=7, chips=4, knobs=None, **kw):
    knobs = knobs or Knobs(4)
    return new_instrument(scope, style, seed, chips, True, True, True, knobs, sleep=SLEEP, **kw), knobs


def test_new_instrument_builds_a_session_and_a_patch_for_every_scope_choice():
    for _, value in scope_choices():
        (session, patch), _ = _instrument(scope=value)
        assert isinstance(session, InstrumentSession) and session.config.scope == value and session.patch is patch


def test_new_instrument_draws_the_patch_in_the_style_from_the_seed_and_ignores_the_knobs():
    (_, patch), knobs = _instrument(style="enya", seed=7)
    knobs.set(0, "power", 140.0)
    (_, again), _ = _instrument(style="enya", seed=7, knobs=knobs)
    (_, other), _ = _instrument(style="enya", seed=8)
    assert patch.style == "enya" and again == patch and other != patch
    (_, plain), _ = _instrument(style=PLAIN)
    assert plain.style == ""


def test_new_instrument_sizes_the_knobs_to_the_chip_count_and_keeps_pending_events():
    knobs = Knobs(4)
    knobs.trigger(1, "memory_error")
    _instrument(chips=2, knobs=knobs)
    assert len(knobs.values()) == 2 and knobs.take_events() == [(1, "memory_error")]  # drawing the patch must not eat the performer's events


def test_new_instrument_records_the_style_the_seed_and_the_switches_it_was_given():
    session, patch = new_instrument("system", PLAIN, 11, 2, False, False, False, Knobs(2), sleep=SLEEP)
    assert (session.config.style, session.config.seed) == ("", 11)
    assert (session.config.stereo, session.config.cables, session.config.evict) == (False, False, False)
    assert session._sleep is SLEEP
    session, _ = new_instrument("system", "blues", 12, 2, True, True, True, Knobs(2), sleep=SLEEP)
    assert (session.config.style, session.config.seed) == ("blues", 12)


def test_new_instrument_passes_the_choices_on():
    (session, _), knobs = _instrument(chips=3)
    assert session.config.chips == 3 and session.config.stereo and session.config.cables and session.config.evict and session.knobs is knobs
    (session, _), _ = _instrument(lead_s=0.25)
    assert session._lead_s == 0.25


@pytest.mark.parametrize("kwargs", [dict(scope="device:nope"), dict(scope="gong"), dict(style="no such style"), dict(chips=0), dict(chips=5)])
def test_new_instrument_rejects_what_it_cannot_play(kwargs):
    with pytest.raises(ValueError):
        _instrument(**kwargs)


def test_read_chip_knobs_copies_a_reading_into_the_knobs():
    knobs = Knobs(3)
    readings = mock_telemetry(4, seed=2)
    status = read_chip_knobs(knobs, sample=lambda: readings)
    for chip in range(4):
        t = readings[chip]
        low, high, _ = KNOB_RANGES["temperature"]
        assert knobs.get(chip, "temperature") == min(high, max(low, t.asic_temperature))
        assert knobs.get(chip, "aiclk") == min(KNOB_RANGES["aiclk"][1], max(KNOB_RANGES["aiclk"][0], t.aiclk))
        assert knobs.get(chip, "power") == min(KNOB_RANGES["power"][1], max(KNOB_RANGES["power"][0], t.power))
        assert knobs.get(chip, "voltage") == min(KNOB_RANGES["voltage"][1], max(KNOB_RANGES["voltage"][0], t.voltage))
        assert knobs.get(chip, "gddr_spread") == pytest.approx(max(t.gddr_temps) - min(t.gddr_temps))
    assert status == "loaded 4 chips from tt-smi" and knobs.get(3, "power") != 30.0  # a hidden chip's knobs are filled too


def test_read_chip_knobs_with_fewer_chips_than_the_knobs_leaves_the_rest_alone():
    knobs = Knobs(4)
    knobs.set(3, "power", 99.0)
    read_chip_knobs(knobs, sample=lambda: mock_telemetry(2, seed=2))
    assert knobs.get(3, "power") == 99.0 and knobs.get(1, "power") != 30.0


def test_read_chip_knobs_reports_a_failing_tt_smi_and_leaves_the_knobs_alone():
    knobs = Knobs(2)

    def broken():
        raise FileNotFoundError("tt-smi")

    status = read_chip_knobs(knobs, sample=broken)
    assert "tt-smi" in status and knobs.get(0, "power") == 30.0
    assert "no chips" in read_chip_knobs(knobs, sample=lambda: [])
