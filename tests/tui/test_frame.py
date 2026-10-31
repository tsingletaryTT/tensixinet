# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
import random
import re

from tensixinet.music.evict import PRESENCE
from tensixinet.music.patch import DEFAULT_PATCH
from tensixinet.narrate.color import _cube
from tensixinet.telemetry import mock_telemetry
from tensixinet.tui.frame import DIM_SCALE, Frame, Span, Style, diff, dim_level, fit, sgr
from tensixinet.tui.layout import render
from tensixinet.ui.flow import build_flow


def test_dim_levels_follow_the_presence_table():
    assert [dim_level(PRESENCE[p]) for p in ("running", "filling", "evicting", "idle")] == [0, 1, 2, 3]


def test_dim_level_boundaries_are_the_pinned_thresholds():
    assert [dim_level(x) for x in (0.9, 0.89, 0.5, 0.49, 0.2, 0.19)] == [0, 1, 1, 2, 2, 3]


def test_the_dim_scale_is_pinned():
    assert DIM_SCALE == (1.0, 0.7, 0.45, 0.25)


def test_sgr_is_empty_without_colour_and_bold_only_when_asked():
    assert sgr(Style("gong", bold=True), colour=False, truecolor=True) == ""
    assert "1;" in sgr(Style("gong", bold=True), True, True) and "1;" not in sgr(Style("gong"), True, True)


def test_sgr_starts_with_a_reset_and_a_style_without_a_jack_is_the_reset_alone():
    assert sgr(Style(), True, True) == "\x1b[0m"
    assert sgr(Style("orange"), True, True) == "\x1b[0m\x1b[38;2;255;154;61m"


def test_a_device_name_takes_its_own_jack():
    # gong plays on the orange jack (narrate/color.DEVICE_JACKS)
    assert sgr(Style("gong"), True, True) == sgr(Style("orange"), True, True)


def test_a_dimmer_level_scales_the_jacks_rgb_by_the_pinned_factors():
    # JACK_HEX["orange"] = FF9A3D -> level 2 is (255, 154, 61) * 0.45, rounded
    assert "38;2;115;69;27" in sgr(Style("orange", dim=2), True, True)
    # Python rounds halves to even: 255 * 0.7 = 178.5 -> 178, 154 * 0.25 = 38.5 -> 38
    assert "38;2;178;108;43" in sgr(Style("orange", dim=1), True, True)
    assert "38;2;64;38;15" in sgr(Style("orange", dim=3), True, True)


def test_without_truecolor_the_scaled_colour_is_the_nearest_cube_colour():
    assert sgr(Style("orange", dim=2), True, False) == f"\x1b[0m\x1b[38;5;{_cube((115, 69, 27))}m"


def test_cells_pad_every_row_to_the_width_and_the_frame_to_the_height():
    f = Frame(((Span("ab"),),), width=5, height=3)
    assert [len(r) for r in f.cells()] == [5, 5, 5] and f.text()[0] == "ab"
    assert [ch for ch, _ in f.cells()[0]] == ["a", "b", " ", " ", " "]


def test_find_gives_the_row_column_and_style_of_each_span():
    bold = Style("gong", bold=True)
    f = Frame(((Span("x "), Span("gong", bold)), (Span("gong"),)), width=10, height=2)
    assert f.find("gong") == [(0, 2, bold), (1, 0, Style())]
    assert f.cells()[0][2] == ("g", bold)


def test_fit_cuts_with_an_ellipsis_and_leaves_short_text_alone():
    assert fit("ttquantus", 5) == "ttqu…" and fit("gong", 5) == "gong" and fit("gong", 4) == "gong"
    assert fit("gong", 0) == ""


# --- diff: the escapes that turn the screen showing one frame into the screen showing the next -----------------------------

_ESCAPE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


class Screen:
    """A small terminal emulator for the tests: it understands the escapes `diff` and the TUI write (clear, cursor move, SGR,
    the private modes) and keeps the characters on a grid, so a test can read what a real terminal would show."""

    def __init__(self, width=200, height=80):
        self.width, self.height = width, height
        self.clear()

    def clear(self):
        self.grid = [[" "] * self.width for _ in range(self.height)]
        self.row = self.col = 0

    def feed(self, text):
        i = 0
        while i < len(text):
            m = _ESCAPE.match(text, i)
            if m:
                code = m.group(0)
                if code == "\x1b[2J":
                    self.clear()
                elif code.endswith("H"):
                    parts = code[2:-1].split(";") if code[2:-1] else ["1", "1"]
                    self.row, self.col = int(parts[0]) - 1, int(parts[1]) - 1
                i = m.end()
                continue
            ch = text[i]
            if ch == "\n":
                self.row, self.col = self.row + 1, 0
            elif 0 <= self.row < self.height and 0 <= self.col < self.width:
                self.grid[self.row][self.col] = ch
                self.col += 1
            else:
                self.col += 1
            i += 1
        return self

    def lines(self, width, height):
        return ["".join(row[:width]) for row in self.grid[:height]]


def strip(text):
    return _ESCAPE.sub("", text)


def _cells(segment=0, width=80, height=24, colour=True):
    s = build_flow(mock_telemetry(4), DEFAULT_PATCH, segment=segment, soloist_slot=2, seconds=4.0)
    return render(s, [], width, height, colour).cells()


def _text(cells):
    return ["".join(ch for ch, _ in row) for row in cells]


def test_an_unchanged_frame_writes_nothing():
    c = _cells()
    assert diff(c, [list(row) for row in c], True, True) == ""


def test_one_changed_cell_moves_the_cursor_there_and_writes_only_it():
    c = _cells()
    new = [list(row) for row in c]
    new[5][10] = ("X", Style())
    out = diff(c, new, False, True)
    assert out.startswith("\x1b[6;11H") and strip(out) == "X"
    assert out.endswith("\x1b[0m")


def test_a_run_of_changed_cells_is_one_cursor_move_with_the_runs_colour():
    c = [[(" ", Style())] * 10 for _ in range(3)]
    new = [list(row) for row in c]
    new[1][2:5] = [("a", Style("gong")), ("b", Style("gong")), ("c", Style("gong"))]
    new[1][8] = ("z", Style())
    out = diff(c, new, True, True)
    assert out.count("H") == 2 and "\x1b[2;3H" in out and "\x1b[2;9H" in out
    assert sgr(Style("gong"), True, True) + "abc" in out


def test_a_new_size_clears_and_draws_every_row():
    c80, c100 = _cells(width=80), _cells(width=100, height=30)
    out = diff(c80, c100, False, True)
    assert out.startswith("\x1b[2J")
    assert Screen().feed(out).lines(100, 30) == _text(c100)


def test_with_no_previous_frame_the_screen_is_cleared_and_every_row_drawn():
    c = _cells()
    out = diff(None, c, True, True)
    assert out.startswith("\x1b[2J") and Screen().feed(out).lines(80, 24) == _text(c)


def test_diffs_applied_one_after_another_leave_exactly_the_newest_frame_on_the_screen():
    rng = random.Random(7)
    screen = Screen()
    old = None
    cur = _cells()
    for _ in range(30):
        screen.feed(diff(old, cur, True, True))
        assert screen.lines(80, 24) == _text(cur)
        old = cur
        cur = [list(row) for row in cur]
        for _ in range(rng.randint(1, 12)):
            r, c = rng.randrange(24), rng.randrange(80)
            cur[r][c] = (rng.choice("xyz#. "), Style(rng.choice([None, "gong", "orange"])))


def test_a_new_width_alone_clears_and_draws_every_row():
    c80, c100 = _cells(width=80), _cells(width=100)
    out = diff(c80, c100, False, True)
    assert out.startswith("\x1b[2J") and Screen().feed(out).lines(100, 24) == _text(c100)


def test_a_style_change_inside_a_run_writes_the_new_style_before_its_cell():
    c = [[(" ", Style())] * 4]
    new = [[("a", Style("gong")), ("b", Style("orange")), (" ", Style()), (" ", Style())]]
    out = diff(c, new, True, True)
    assert out == "\x1b[1;1H" + sgr(Style("gong"), True, True) + "a" + sgr(Style("orange"), True, True) + "b\x1b[0m"
