# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
import json

import pytest

from tensixinet.music.patch import DEFAULT_PATCH, save_patch
from tensixinet.music.styles import STYLES
from tensixinet.ui.forms import DRAW, PLAIN, SOURCES, new_session, patch_choices, style_choices

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
