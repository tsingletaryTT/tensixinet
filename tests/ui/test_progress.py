# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Progress: what the page shows before the first segment plays (the simulator starting, the first segment rendering, the buffer filling)."""
from tensixinet.ui.progress import ACTIVE, DONE, TODO, Progress, Step, progress_html


def _p(*states, fraction=None, elapsed=0.0):
    labels = ["start the TT-Lang simulator", "render the first segment", "fill the buffer"]
    return Progress(tuple(Step(l, s, secs) for (l, (s, secs)) in zip(labels, states)), fraction, elapsed)


def test_every_step_is_listed_with_its_state_and_the_active_one_says_how_long_it_has_taken():
    html = progress_html(_p((DONE, 14.2), (ACTIVE, 3.1), (TODO, 0.0), elapsed=17.3))
    for label in ("start the TT-Lang simulator", "render the first segment", "fill the buffer"):
        assert label in html
    assert "14 s" in html and "3 s" in html and "17 s" in html          # a finished step's time, the active step's time and the total
    assert html.index("start the TT-Lang simulator") < html.index("render the first segment") < html.index("fill the buffer")
    assert html.count('data-state="done"') == 1 and html.count('data-state="active"') == 1 and html.count('data-state="todo"') == 1


def test_a_step_with_no_known_length_is_an_indeterminate_bar_and_one_with_a_fraction_is_filled_to_it():
    assert "indeterminate" in progress_html(_p((ACTIVE, 2.0), (TODO, 0), (TODO, 0)))
    html = progress_html(_p((DONE, 1.0), (DONE, 1.0), (ACTIVE, 4.0), fraction=0.4))
    assert "indeterminate" not in html and "width:40%" in html


def test_the_fraction_is_kept_between_zero_and_all():
    assert "width:100%" in progress_html(_p((DONE, 1), (DONE, 1), (ACTIVE, 1), fraction=7.0))
    assert "width:0%" in progress_html(_p((DONE, 1), (DONE, 1), (ACTIVE, 1), fraction=-1.0))


def test_the_page_says_what_it_is_waiting_for_in_plain_words():
    assert "press Play" not in progress_html(_p((ACTIVE, 0), (TODO, 0), (TODO, 0)))
    assert "starting" in progress_html(_p((ACTIVE, 0), (TODO, 0), (TODO, 0))).lower()
