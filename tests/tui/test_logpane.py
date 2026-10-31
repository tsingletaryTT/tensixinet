# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The TUI log pane: edge colours, token colours, stamps, reasons, wrapping."""
from tensixinet.narrate.color import CHIP_JACKS, EVENT_JACKS, JACK_HEX
from tensixinet.tui.frame import Style
from tensixinet.tui.logpane import edge_jack, log_rows, spans


def rec(t=0.0, what="a line", why="a reason", event="live", **kw):
    return {"t": t, "event": event, "what": what, "why": why, **kw}


def text(row):
    return "".join(s.text for s in row)


def test_a_segment_line_takes_its_soloists_jack_and_other_kinds_their_own():
    assert edge_jack({"event": "live", "device": "gong"}) == "orange"
    assert edge_jack({"event": "live"}) == "green"
    assert edge_jack({"event": "evict"}) == "red" and edge_jack({"event": "delay"}) == "yellow" and edge_jack({"event": "midi"}) == "yellow"
    assert edge_jack({"event": "backend"}) == "teal" and edge_jack({"event": "unknown"}) == "grey"
    assert edge_jack({"event": "chip", "device": "gong"}) == "blue"  # only live lines follow the device


def test_devices_chips_and_numbers_with_units_are_coloured_as_in_the_terminal_log():
    sp = spans("gong on chip 2 at 41 W", Style())
    assert [(s.text, s.style.jack, s.style.bold) for s in sp if s.style.bold] == [("gong", "orange", True), ("chip 2", CHIP_JACKS[2], True), ("41 W", "yellow", True)]
    assert "".join(s.text for s in sp) == "gong on chip 2 at 41 W"
    assert [s.style for s in sp if not s.style.bold] == [Style(), Style()]


def test_a_chip_number_past_eight_wraps_round_the_chip_colours():
    sp = spans("chip 9", Style())
    assert sp[0].style.jack == CHIP_JACKS[1]


def test_plain_text_keeps_the_base_style():
    base = Style("grey", dim=1)
    assert [sp.style for sp in spans("nothing special", base)] == [base]


def test_each_line_has_the_stamp_the_what_and_the_why_in_grey():
    rows = log_rows([rec(t=65.25, what="segment 4: ...", why="because", device="gong")], 80)
    assert text(rows[0]).startswith("║ [01:05.2] segment 4")
    assert rows[0][0].style == Style("orange", bold=True)  # a live segment's bar takes its soloist's jack: the gong is orange
    assert rows[0][2].text == "[01:05.2]" and rows[0][2].style.jack == "grey"
    assert text(rows[1]) == "║          why: because"
    assert rows[1][-1].style == Style("grey", dim=1) and rows[1][2].text == "why:"


def test_the_left_edge_takes_the_records_jack_and_is_bold():
    rows = log_rows([rec(device="gong", what="x")], 80)
    assert rows[0][0].text == "║" and rows[0][0].style == Style("orange", bold=True)
    assert rows[1][0].style == Style("orange", bold=True)


def test_a_reason_already_shown_is_left_out():
    rows = log_rows([rec(why="same"), rec(why="same"), rec(why="other")], 80)
    assert [text(r) for r in rows if "why:" in text(r)] == ["║          why: same", "║          why: other"]


def test_long_lines_wrap_inside_the_width_and_keep_the_left_edge():
    rows = log_rows([rec(what="word " * 40, why="reason " * 20)], 60)
    assert len(rows) > 4
    assert all(len(text(r)) <= 60 and text(r).startswith("║") for r in rows)
    assert text(rows[1]).startswith("║" + " " * 11)  # hangs under the text, as RunLog does


def test_a_line_exactly_the_width_does_not_wrap():
    what = "x" * (60 - len("║ [00:00.0] "))
    assert len(log_rows([rec(what=what, why="")], 60)) == 1


def test_a_missing_reason_adds_no_why_row():
    assert len(log_rows([rec(why="")], 80)) == 1


def test_records_come_out_oldest_first():
    rows = log_rows([rec(what="first", why=""), rec(what="second", why="")], 80)
    assert [("first" in text(r), "second" in text(r)) for r in rows] == [(True, False), (False, True)]


def test_every_event_kind_the_loops_log_has_a_colour():
    for kind in ("delay", "midi", "backend", "evict", "fill", "events", "live"):
        assert kind in EVENT_JACKS
    for kind, jack in EVENT_JACKS.items():
        assert jack in JACK_HEX and edge_jack({"event": kind}) == jack
