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
