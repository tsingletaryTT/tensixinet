# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The Play tab's readout: pitch, tempo and pressure for each chip, and what is playing."""
import dataclasses
import re

import pytest

from tensixinet.music.live import Tracker
from tensixinet.narrate.color import CHIP_JACKS, JACK_HEX
from tensixinet.ui.instrument import InstrumentSnapshot
from tensixinet.ui.knobs import Knobs, to_telemetry
from tensixinet.ui.readout import READOUT_WAITING, note_name, readout_html


def snap(knobs=None, chips=2, playing=("gong",), notes=()):
    knobs = knobs or Knobs(chips)
    readings = tuple(to_telemetry(knobs, Tracker(), 0.0))
    return InstrumentSnapshot(True, None, 3, readings, tuple(playing), tuple(notes))



@pytest.mark.parametrize("hz,name", [(440.0, "A4"), (261.63, "C4"), (130.81, "C3"), (523.25, "C5"), (466.16, "A#4"), (432.0, "A4")])
def test_note_names(hz, name):
    assert note_name(hz) == name


def test_an_empty_snapshot_gives_the_waiting_text():
    empty = InstrumentSnapshot(False, None, -1, (), (), ())
    assert readout_html(empty) == READOUT_WAITING and "press Play" in READOUT_WAITING


def test_one_row_per_chip_each_in_its_own_jack_colour():
    html = readout_html(snap(chips=3))
    assert html.count('class="chip"') == 3
    for i in range(3):
        assert JACK_HEX[CHIP_JACKS[i]] in html
    assert [f"chip {i}" for i in range(3)] == re.findall(r"chip \d", html)[:3]


def test_a_row_shows_the_pitch_hz_and_note_the_tempo_and_the_pressure():
    knobs = Knobs(1)
    knobs.set(0, "temperature", 25.0)  # mapping: 25 C is the bottom of the range, C3
    knobs.set(0, "aiclk", 1350.0)  # the top of the tempo range, 180 BPM
    html = readout_html(snap(knobs))
    assert "C3" in html and "130.8 Hz" in html and "180 BPM" in html


def test_a_hotter_chip_has_a_higher_note_and_a_wider_pressure_bar():
    cool, hot = Knobs(1), Knobs(1)
    cool.set(0, "temperature", 30.0)
    cool.set(0, "power", 10.0)
    hot.set(0, "temperature", 95.0)
    hot.set(0, "power", 140.0)
    width = lambda h: int(re.search(r"width:(\d+)%", re.search(r'class="bar".*?</div>', h, re.S).group(0)).group(1))
    assert width(readout_html(snap(hot))) > width(readout_html(snap(cool)))
    assert width(readout_html(snap(cool))) < 40 and width(readout_html(snap(hot))) > 90


def test_the_pressure_bar_is_clamped_to_the_box():
    knobs = Knobs(1)
    knobs.set(0, "power", 150.0)
    knobs.set(0, "temperature", 105.0)
    assert "width:100%" in readout_html(snap(knobs))


def test_the_devices_playing_and_the_notes_are_listed_and_the_notes_are_escaped():
    html = readout_html(snap(playing=("ttrollz", "gong"), notes=("ttchoir <b>is</b> silent",)))
    assert "ttrollz" in html and "gong" in html
    assert "&lt;b&gt;" in html and "<b>" not in html
    assert "&lt;i&gt;" in readout_html(snap(playing=("<i>x",))) and "<i>" not in readout_html(snap(playing=("<i>x",)))


def test_a_snapshot_with_nothing_playing_says_so():
    assert "nothing" in readout_html(snap(playing=())).lower()


def test_the_readout_sits_on_its_own_dark_panel_so_it_reads_on_any_page_theme():
    for html in (readout_html(snap()), READOUT_WAITING):
        assert html.startswith('<div class="readout" style="background:#092221') and html.endswith("</div>")
