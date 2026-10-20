# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
import xml.etree.ElementTree as ET

from tensixinet.narrate.color import EVENT_JACKS, JACK_HEX
from tensixinet.ui.logview import MAX_LINES, log_html


def rec(t, event, what, why="because", **kw):
    return {"t": t, "event": event, "what": what, "why": why, **kw}


def _rows(html):
    root = ET.fromstring(html)
    return [e for e in root.iter() if "ln" in (e.get("class") or "").split()]


def test_each_event_is_a_row_with_its_time_what_and_why_and_the_jack_colour_of_its_kind():
    html = log_html([rec(4.4, "patch", "scale major_pentatonic"), rec(65.0, "evict", "chip 1 evicted", "power fell back")])
    rows = _rows(html)
    assert len(rows) == 2
    assert f"#{JACK_HEX[EVENT_JACKS['patch']]}" in rows[0].get("style")
    assert f"#{JACK_HEX[EVENT_JACKS['evict']]}" in rows[1].get("style")
    text = "".join(ET.fromstring(html).itertext())  # a coloured token splits the markup but not the words
    for s in ("00:04.4", "scale major_pentatonic", "01:05.0", "chip 1 evicted", "power fell back"):
        assert s in text, s


def test_a_live_segment_takes_its_soloists_colour():
    from tensixinet.narrate.color import DEVICE_JACKS
    row = _rows(log_html([rec(1.0, "live", "segment 0", device="gong", chip=1)]))[0]
    assert f"#{JACK_HEX[DEVICE_JACKS['gong']]}" in row.get("style")


def test_the_newest_row_is_last_on_the_page_and_only_the_last_lines_are_kept():
    records = [rec(float(i), "live", f"segment {i}", device="gong", chip=0) for i in range(MAX_LINES + 20)]
    rows = _rows(log_html(records))
    assert len(rows) == MAX_LINES
    assert f"segment {MAX_LINES + 19}" in ET.tostring(rows[-1], encoding="unicode") and "segment 19<" not in "".join(ET.tostring(r, encoding="unicode") for r in rows)


def test_text_is_escaped():
    html = log_html([rec(0.0, "patch", "a<b>&c", "x<y")])
    assert "a<b>" not in html and "a&lt;b&gt;&amp;c" in html and len(_rows(html)) == 1


def test_a_repeated_reason_is_shown_once():
    rows = _rows(log_html([rec(0.0, "chip", "one", "same reason"), rec(1.0, "chip", "two", "same reason")]))
    assert "same reason" in ET.tostring(rows[0], encoding="unicode") and "same reason" not in ET.tostring(rows[1], encoding="unicode")


def test_no_events_gives_a_waiting_line():
    assert "waiting" in log_html([]).lower()


def test_every_row_sets_its_own_text_colour_because_a_host_page_theme_can_make_inherited_text_dark():
    for row in _rows(log_html([rec(0.0, "patch", "x"), rec(1.0, "live", "y", device="gong", chip=0)])):
        assert "color:#cfe4e2" in row.get("style")


def _spans(html):
    root = ET.fromstring(html)
    return {(e.text, ("color:#" + e.get("style").split("color:#")[1][:6]).upper()) for e in root.iter("span") if e.get("data-token")}


def test_devices_chips_and_numbers_with_units_are_coloured_inline_like_the_terminal_log():
    from tensixinet.narrate.color import CHIP_JACKS, DEVICE_JACKS
    html = log_html([rec(0.0, "patch", "chip 2 plays ttsteam and gong at 62 W, 4.5 kHz and 12 s")])
    spans = _spans(html)
    hexc = lambda jack: "COLOR:#" + JACK_HEX[jack].upper()
    assert ("ttsteam", hexc(DEVICE_JACKS["ttsteam"])) in spans and ("gong", hexc(DEVICE_JACKS["gong"])) in spans
    assert ("chip 2", hexc(CHIP_JACKS[2])) in spans
    for number in ("62 W", "4.5 kHz", "12 s"):
        assert (number, hexc("yellow")) in spans


def test_plain_words_stay_plain_and_text_around_a_token_is_still_escaped():
    html = log_html([rec(0.0, "patch", "a<b> gong & more")])
    assert "a&lt;b&gt; " in html and "&amp; more" in html and ("gong", "COLOR:#" + JACK_HEX["orange"].upper()) in _spans(html)
