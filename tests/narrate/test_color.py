# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The colour layer of the run log: the Ciat-Lonbarde jack colours, applied to what is on screen and nowhere else."""
import io
import json
import re
from pathlib import Path

import pytest

from tensixinet.music.chorus import device_names
from tensixinet.narrate.color import (
    CHIP_JACKS, DEVICE_JACKS, EVENT_JACKS, JACK_HEX, JACK_MEANING, Palette, banner, bar, decorate, detect, legend, strip_ansi,
)
from tensixinet.narrate.runlog import RunLog

ESC = "\x1b["


class Tty(io.StringIO):
    def isatty(self):
        return True


# --- when colour is on -------------------------------------------------------------------------------------------

def test_auto_colour_needs_a_terminal_and_respects_no_color_and_a_dumb_term():
    assert not detect(io.StringIO(), "auto", {}).enabled                       # a pipe or a file
    assert detect(Tty(), "auto", {"TERM": "xterm-256color"}).enabled
    assert not detect(Tty(), "auto", {"NO_COLOR": "1"}).enabled                 # https://no-color.org
    assert not detect(Tty(), "auto", {"TERM": "dumb"}).enabled
    assert detect(io.StringIO(), "auto", {"FORCE_COLOR": "1"}).enabled          # a recording that is not a tty
    assert not detect(Tty(), "auto", {"FORCE_COLOR": "1", "NO_COLOR": "1"}).enabled


def test_always_and_never_decide_regardless_of_the_terminal():
    assert detect(io.StringIO(), "always", {"NO_COLOR": "1"}).enabled
    assert not detect(Tty(), "never", {"FORCE_COLOR": "1"}).enabled


def test_true_colour_is_used_when_the_terminal_says_it_can_and_256_colours_otherwise():
    assert detect(Tty(), "always", {"COLORTERM": "truecolor"}).truecolor
    assert detect(Tty(), "always", {"COLORTERM": "24bit"}).truecolor
    assert not detect(Tty(), "always", {"TERM": "xterm-256color"}).truecolor


# --- painting ---------------------------------------------------------------------------------------------------

def test_a_disabled_palette_returns_the_text_untouched():
    p = Palette(enabled=False)
    assert p.paint("hello", "blue", bold=True) == "hello" and p.dim("x") == "x"


def test_a_true_colour_palette_writes_the_exact_rgb_of_the_jack_and_resets_after():
    p = Palette(enabled=True, truecolor=True)
    r, g, b = (int(JACK_HEX["orange"][i:i + 2], 16) for i in (0, 2, 4))
    assert p.paint("x", "orange") == f"{ESC}38;2;{r};{g};{b}mx{ESC}0m"
    assert p.paint("x", "orange", bold=True).startswith(f"{ESC}1;38;2;{r};{g};{b}m")


def test_a_256_colour_palette_picks_the_nearest_cube_colour():
    p = Palette(enabled=True, truecolor=False)
    code = re.match(rf"{re.escape(ESC)}38;5;(\d+)mx", p.paint("x", "red"))
    assert code and 16 <= int(code.group(1)) <= 255
    assert p.paint("x", "red") != p.paint("x", "blue")


def test_stripping_the_escapes_gives_back_the_text():
    p = Palette(enabled=True)
    assert strip_ansi(p.paint("a", "red") + p.dim("b") + p.paint("c", "teal", bold=True)) == "abc"


# --- the jack colours ---------------------------------------------------------------------------------------------

def test_the_nine_jack_colours_of_the_plumbutter_guide_are_all_there_with_the_guides_meanings():
    guide = {"blue", "green", "purple", "white", "orange", "red", "yellow", "brown", "grey"}
    assert guide <= set(JACK_HEX) and guide <= set(JACK_MEANING)
    assert "controlling" in JACK_MEANING["blue"] and "trigger" in JACK_MEANING["green"] and "innards" in JACK_MEANING["purple"]
    assert "audio" in JACK_MEANING["white"] and "control voltages" in JACK_MEANING["orange"] and "pulses" in JACK_MEANING["red"]
    assert "pulses" in JACK_MEANING["brown"] and "dust" in JACK_MEANING["grey"] and "modular" in JACK_MEANING["yellow"]
    assert all(re.fullmatch(r"[0-9A-Fa-f]{6}", h) for h in JACK_HEX.values()) and len(set(JACK_HEX.values())) == len(JACK_HEX)


def test_every_kind_of_log_line_the_code_writes_has_a_colour():
    source = "".join(p.read_text() for p in Path("tensixinet").rglob("*.py"))
    kinds = set(re.findall(r'log\.event\(\s*"([a-z_]+)"', source)) | set(re.findall(r'log\.event\(\s*\n\s*"([a-z_]+)"', source))
    assert kinds and kinds <= set(EVENT_JACKS), sorted(kinds - set(EVENT_JACKS))
    assert set(EVENT_JACKS.values()) <= set(JACK_HEX)


def test_every_device_and_chip_has_a_jack_colour():
    assert set(device_names()) <= set(DEVICE_JACKS) and set(DEVICE_JACKS.values()) <= set(JACK_HEX)
    assert len(CHIP_JACKS) >= 4 and len(set(CHIP_JACKS[:4])) == 4 and set(CHIP_JACKS) <= set(JACK_HEX)


# --- decorating a line --------------------------------------------------------------------------------------------

def test_devices_and_chips_are_coloured_by_their_own_jack_and_nothing_else_changes():
    p = Palette(enabled=True)
    line = "ttrollz and gong on chip 2 play 3 steps"
    out = decorate(line, p)
    assert strip_ansi(out) == line
    assert p.paint("ttrollz", DEVICE_JACKS["ttrollz"], bold=True) in out and p.paint("gong", DEVICE_JACKS["gong"], bold=True) in out
    assert p.paint("chip 2", CHIP_JACKS[2], bold=True) in out
    assert decorate(line, Palette(enabled=False)) == line


def test_a_device_name_inside_a_longer_word_is_left_alone():
    p = Palette(enabled=True)
    assert decorate("leadership is not the lead device", p).count(ESC) == decorate("the lead", p).count(ESC)
    assert strip_ansi(decorate("a gong strikes; gongs ring", p)) == "a gong strikes; gongs ring"


def test_numbers_with_units_are_lifted_out_of_the_line():
    p = Palette(enabled=True)
    out = decorate("beat 1.98 Hz at 55 C and 4.3 s", p)
    assert strip_ansi(out) == "beat 1.98 Hz at 55 C and 4.3 s"
    for token in ("1.98 Hz", "55 C", "4.3 s"):
        assert p.paint(token, "yellow", bold=True) in out


# --- the run log on screen and in the file ---------------------------------------------------------------------------

def _log(tmp_path, palette=None, width=None):
    screen, plain = io.StringIO(), io.StringIO()
    log = RunLog(jsonl=open(tmp_path / "r.jsonl", "w"), stream=screen, plain_stream=plain, palette=palette, width=width, clock=iter([0.0, 1.0, 2.0, 3.0]).__next__)
    return log, screen, plain


def test_the_screen_is_coloured_and_the_file_and_the_jsonl_are_plain(tmp_path):
    log, screen, plain = _log(tmp_path, Palette(enabled=True))
    log.event("chip", "chip 1 plays gong at 55 C", "the ASIC temperature sets the pitch", chip=1)
    shown, filed = screen.getvalue(), plain.getvalue()
    log.close()
    assert ESC in shown and ESC not in filed and strip_ansi(shown) == filed
    assert ESC not in (tmp_path / "r.jsonl").read_text() and json.loads((tmp_path / "r.jsonl").read_text())["what"] == "chip 1 plays gong at 55 C"


def test_a_wrapped_line_is_the_same_in_colour_and_plain(tmp_path):
    log, screen, plain = _log(tmp_path, Palette(enabled=True), width=40)
    log.event("patch", "scale hirajoshi, mode 1 (GDDR spread 6 C) with a gong on ttavdog and chip 3", "each 3 C moves one scale along the list")
    shown, filed = screen.getvalue(), plain.getvalue()
    log.close()
    assert strip_ansi(shown) == filed and filed.count("\n") > 3
    assert all(line.startswith("║") for line in filed.splitlines())    # the left bar survives wrapping


def test_the_bar_takes_the_colour_of_the_kind_and_a_segment_takes_its_soloists(tmp_path):
    p = Palette(enabled=True)
    log, screen, _ = _log(tmp_path, p)
    log.event("cable", "ttrollz bends ttavdog", "a purple jack")
    log.event("live", "segment 3: 4.3 s, gong on chip 1 has the floor, gain x2.01", "the soloist is drawn", segment=3, chip=1, device="gong", seconds=4.3)
    shown = screen.getvalue()
    log.close()
    first, second = shown.splitlines()[0], [l for l in shown.splitlines() if "segment 3" in l][0]
    assert p.paint("║", EVENT_JACKS["cable"], bold=True) in first
    assert p.paint("║", DEVICE_JACKS["gong"], bold=True) in second
    assert "━" in strip_ansi(second)                                                 # a bar as long as the segment


def test_without_a_palette_the_log_is_exactly_what_it_was(tmp_path):
    screen = io.StringIO()
    log = RunLog(jsonl=open(tmp_path / "a.jsonl", "w"), stream=screen, clock=iter([0.0, 1.5]).__next__)
    log.event("chip", "chip 1 plays gong", "because")
    shown = screen.getvalue()
    log.close()
    assert shown == "║ [00:01.5] chip 1 plays gong\n║          why: because\n"


# --- ascii art, with a left bar and no right one -----------------------------------------------------------------------

def test_a_bar_is_a_row_of_blocks_in_proportion_and_clamped():
    assert bar(0.5, 10) == "█████░░░░░" and bar(0, 4) == "░░░░" and bar(1, 4) == "████" and bar(7, 4) == "████" and bar(-1, 4) == "░░░░"


def test_the_banner_is_gradient_art_that_fits_the_pane_and_has_no_right_border():
    p = Palette(enabled=True)
    lines = banner(80, p)
    plain = [strip_ansi(l) for l in lines]
    assert len(lines) >= 4 and ESC in "".join(lines)
    assert all(len(l) <= 80 for l in plain)
    assert not any(l.rstrip().endswith(tuple("│┃║┐┘╗╝┤┓┛╮╯")) for l in plain)       # the global rule: left and bottom edges only
    assert any("█" in l or "▀" in l for l in plain)                                  # the art
    assert any("Ciat-Lonbarde" in l for l in plain)
    narrow = [strip_ansi(l) for l in banner(40, p)]
    assert all(len(l) <= 40 for l in narrow) and not any("█" in l for l in narrow)   # too narrow for the art: a word instead
    assert [strip_ansi(l) for l in banner(80, Palette(enabled=False))] == plain      # plain text is the same art


def test_the_legend_names_every_jack_colour_and_what_it_means_and_fits():
    p = Palette(enabled=True)
    plain = [strip_ansi(l) for l in legend(78, p)]
    text = "\n".join(plain)
    for name in ("blue", "green", "purple", "white", "orange", "red", "yellow", "brown", "grey"):
        assert name in text
    assert all(len(l) <= 78 for l in plain) and text.count("●") >= 9
    assert not any(l.rstrip().endswith(tuple("│┃║┐┘╗╝┤┓┛╮╯")) for l in plain)
