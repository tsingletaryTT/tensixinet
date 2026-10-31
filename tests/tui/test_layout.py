# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The TUI's instrument-centric layout, checked on the text and the styles it draws (which glyphs, which cells are bold, which
dim level): the readings strip (one line per chip), the modules grouped by role, the cables, the stereo line and the log."""
import dataclasses

from tensixinet.music.chorus import device_names
from tensixinet.music.evict import PRESENCE
from tensixinet.music.patch import DEFAULT_PATCH
from tensixinet.narrate.color import CHIP_JACKS, DEVICE_JACKS, bar
from tensixinet.telemetry import mock_telemetry
from tensixinet.tui.frame import Span, Style, dim_level
from tensixinet.tui.layout import (
    BAR_W, CABLE_ROWS, CABLE_STEPS, GROUPS, LABELS, LOG_MIN_ROWS, LOG_SHORT_MIN_ROWS, MIN_COLS, MIN_ROWS, PHASE_MARK, QUIT_HINT,
    READINGS_KEEP, SHORT_CABLE_ROWS, STEREO_CODES, TARGET, WIDE_COLS, heading, label, render,
)
from tensixinet.ui.chart import CABLE_COLOURS
from tensixinet.ui.flow import build_flow

EDGES = ("│", "┃", "║", "╎", "┐", "┘", "┓", "┛", "╗", "╝")


def _state(n=4, soloist=2, **kw):
    chips = mock_telemetry(n)
    return chips, build_flow(chips, DEFAULT_PATCH, segment=3, soloist_slot=soloist, seconds=4.0, **kw)


def log_rows_of(k):
    return [(Span(f"log line {i}", Style("grey")),) for i in range(k)]


def _first(f, predicate, start=0):
    return next(i for i, line in enumerate(f.text()) if i >= start and predicate(line))


def stereo_row(f):
    return _first(f, lambda line: line.startswith("L ") and line.endswith(" R"))


def stereo_col(az, width):
    """Where a device's two-letter code starts on the stereo line: the field runs from column 2 to width - 3."""
    return 2 + round((az + 1) / 2 * (width - 4 - 2))


def log_floor(h):
    """The log rows the optional rows leave, at any height (so a taller screen never has a shorter log for them)."""
    return LOG_SHORT_MIN_ROWS


EXTRAS = ("CABLES", "STEREO", "legend", "READINGS")  # the rows drawn only when there is room, kept in this order


def extras(lines):
    """Which of the optional rows a frame drew."""
    found = []
    if "CABLES" in lines:
        found.append("CABLES")
    if "STEREO" in lines:
        found.append("STEREO")
    if any(line.startswith(f"{STEREO_CODES['gong']} gong") for line in lines):
        found.append("legend")
    if any(line.startswith("READINGS") for line in lines):
        found.append("READINGS")
    return found


def modules_row(f):
    """The row of the group headings RHYTHM, MELODY and AIR."""
    return _first(f, lambda line: line.startswith("RHYTHM"))


def cable_rows(f):
    return [i for i, line in enumerate(f.text()) if "═▶" in line]


def readings_row(f, chip):
    """The readings line of the chip with this index (its mark and index follow the left bar)."""
    return _first(f, lambda line: line[1:2] in "●◌…" and line[2:].split(" ")[0] == str(chip), start=1)


def col_w(f):
    return f.width // (4 if f.width >= WIDE_COLS else 3)


def module_hits(f, d):
    """(row, column, style) of the module row for device `d`: its name, in the modules section, followed by its chip's index."""
    lines = f.text()
    top, bottom = modules_row(f), (cable_rows(f) or [stereo_row(f)])[0]
    hits = []
    for r, c, st in f.find(label(d.name)):
        if top < r < bottom:
            rest = lines[r][c:(c // col_w(f) + 1) * col_w(f)]  # to the end of its own column
            if any(f"{m}{d.chip} " in rest + " " for m in "●◌…"):
                hits.append((r, c, st))
    return hits


def bold_cells(f):
    return {(r, c) for r, row in enumerate(f.cells()) for c, (ch, st) in enumerate(row) if st.bold and ch.strip()}


# --- constants ------------------------------------------------------------------------------------------------------------

def test_pinned_constants():
    assert (MIN_COLS, MIN_ROWS, TARGET, CABLE_ROWS, BAR_W, LOG_MIN_ROWS, READINGS_KEEP, WIDE_COLS) == (
        60, 16, (80, 24), 5, 4, 3, 4, 100)
    assert (SHORT_CABLE_ROWS, LOG_SHORT_MIN_ROWS, QUIT_HINT) == (3, 6, "q quit")
    assert STEREO_CODES == {
        "gong": "go", "ttrollz": "rz", "ttsteam": "st", "lead": "ld", "ttquantus": "qu", "ttrax": "tx",
        "ttchoir": "ch", "ttdrone": "dr", "ultrasonic_shimmer": "sh", "ttavdog": "av", "ttcoco": "co",
    }
    assert PHASE_MARK == {"running": "●", "filling": "…", "evicting": "◌", "idle": "◌"}
    assert CABLE_STEPS == ((0.5, 0), (0.25, 1), (0.1, 2))
    assert LABELS == {"ultrasonic_shimmer": "shimmer"} and label("ultrasonic_shimmer") == "shimmer" and label("gong") == "gong"
    assert GROUPS == (
        ("RHYTHM", ("gong", "ttrollz", "ttsteam")),
        ("MELODY", ("lead", "ttquantus", "ttrax")),
        ("AIR", ("ttchoir", "ttdrone", "ultrasonic_shimmer")),
        ("MOD/DELAY", ("ttavdog", "ttcoco")),
    )


def test_every_device_has_a_group():
    grouped = [name for _, names in GROUPS for name in names]
    assert sorted(grouped) == sorted(device_names()) and len(set(grouped)) == len(grouped)


# --- the modules ----------------------------------------------------------------------------------------------------------

def test_group_headings_are_in_order_and_mod_delay_takes_a_second_row_below_100_columns():
    _, s = _state(4)
    f = render(s, [], 80, 24, True)
    top = f.text()[modules_row(f)]
    assert top.index("RHYTHM") == 0 and top.index("MELODY") == col_w(f) and top.index("AIR") == 2 * col_w(f)
    assert "MOD/DELAY" not in top
    second = _first(f, lambda line: line.startswith("MOD/DELAY"))
    assert second > modules_row(f) + 3
    f = render(s, [], 100, 30, True)
    top = f.text()[modules_row(f)]
    assert top.index("MOD/DELAY") == 3 * col_w(f) == 75


def test_each_device_sits_under_its_role():
    for w in (80, 100):
        _, s = _state(4)
        f = render(s, [], w, 30, True)
        lines = f.text()
        for gi, (group, names) in enumerate(GROUPS):
            for d in (d for d in s.devices if d.name in names):
                (r, c, _), = module_hits(f, d)
                heading_row = _first(f, lambda line, g=group: g in line, start=modules_row(f))
                gx = lines[heading_row].index(group)
                assert c - 2 >= gx and c - 2 < gx + col_w(f), (w, d.name)  # the name follows the left bar (and the "▶ " of the soloist)
                assert r > heading_row
                # nothing but blanks or other module rows sit between the heading and this row in that column
                assert all(lines[k][gx:gx + 1] in ("╎", "┃") for k in range(heading_row + 1, r)), (w, d.name)


def test_the_roles_are_the_owners_table():
    """Written out here, so a change to GROUPS in the module cannot move the expectation with it."""
    _, s = _state(4)
    f = render(s, [], 100, 30, True)
    lines = f.text()
    top = modules_row(f)
    cw = col_w(f)
    under = {}
    for r in range(top + 1, (cable_rows(f) or [stereo_row(f)])[0]):
        if lines[r] == "CABLES":
            break
        for k, title in enumerate(("RHYTHM", "MELODY", "AIR", "MOD/DELAY")):
            cell = lines[r][k * cw:(k + 1) * cw].split()
            if cell:
                under.setdefault(title, []).append([w for w in cell if w.strip("╎┃▶")][0].strip("╎┃▶[]"))
    assert under == {"RHYTHM": ["gong", "ttrollz", "ttsteam"], "MELODY": ["lead", "ttquantus", "ttrax"],
                     "AIR": ["ttchoir", "ttdrone", "shimmer"], "MOD/DELAY": ["ttavdog", "ttcoco"]}


def test_a_device_not_in_the_table_goes_in_other():
    _, s = _state(4)
    devices = tuple(dataclasses.replace(d, name="ttnew") if d.name == "ttcoco" else d for d in s.devices)
    f = render(dataclasses.replace(s, devices=devices), [], 132, 50, True)
    text = "\n".join(f.text())
    assert "OTHER" in text and "ttnew" in text
    other = _first(f, lambda line: "OTHER" in line)
    (r, c, _), = [h for h in f.find("ttnew") if other < h[0] < stereo_row(f)]
    assert f.text()[other].index("OTHER") <= c - 2


def _hint(s, d):
    """The strongest cable that touches `d`, as its module row names it, or None."""
    touching = sorted((k for k in s.cables if d.name in (k.source, k.target)), key=lambda k: -k.push)
    if not touching:
        return None
    k = touching[0]
    return f"◂ {k.kind} {label(k.source)}" if k.target == d.name else f"▸ {k.kind} {label(k.target)}"


def test_a_module_row_carries_its_chips_mark_index_and_share_bar():
    _, s = _state(4, soloist=0, phases={1: "evicting"})
    f = render(s, [], 100, 30, True)  # 100 columns: no hint fits, so every row has its share bar
    lines = f.text()
    for d in s.devices:
        (r, c, st), = module_hits(f, d)
        mark = PHASE_MARK[s.chips[d.chip].phase]
        segment = lines[r][c:c + col_w(f)]
        assert f"{mark}{d.chip}" in segment
        assert bar(min(1.0, d.share * 4), BAR_W) in segment, d.name
        dot = [h for h in f.find(f"{mark}{d.chip}") if h[0] == r and c < h[1] < c + col_w(f)]
        assert dot and dot[0][2].jack == CHIP_JACKS[d.chip % len(CHIP_JACKS)]


def test_wide_rows_say_faint_and_name_the_strongest_cable_into_a_module():
    _, s = _state(4, soloist=0, phases={1: "evicting"})
    f = render(s, [], 200, 60, True)
    lines = f.text()
    for d in s.devices:
        (r, c, _), = module_hits(f, d)
        segment = lines[r][c:c + col_w(f)]
        assert ("faint" in segment) == (d.chip == 1), d.name
        touching = sorted((k for k in s.cables if d.name in (k.source, k.target)), key=lambda k: -k.push)
        if touching:
            k = touching[0]
            hint = f"◂ {k.kind} {label(k.source)}" if k.target == d.name else f"▸ {k.kind} {label(k.target)}"
            assert hint in segment, (d.name, segment)
            (hr, hc, hst), = [h for h in f.find(hint) if h[0] == r and c < h[1] < c + col_w(f)]
            assert hst.jack == CABLE_COLOURS[k.kind] and not hst.bold


def test_no_module_row_reaches_past_its_column():
    """The last column of every module column is a blank gap."""
    # 69 to 71, 145 and 160 columns are widths where a row one cell wider would reach the gap
    for n, w in ((1, 60), (1, 69), (1, 70), (1, 71), (1, 145), (1, 160), (4, 69), (4, 80), (4, 100), (12, 132), (4, 81)):
        _, s = _state(n, soloist=0)
        f = render(s, [], w, 50, True)
        cells = f.cells()
        top = modules_row(f)
        bottom = (cable_rows(f) or [stereo_row(f)])[0]
        cw = col_w(f)
        for r in range(top, bottom):
            for k in range(f.width // cw):
                assert cells[r][(k + 1) * cw - 1][0] == " ", (n, w, r, k)


# --- the readings strip ---------------------------------------------------------------------------------------------------

def test_a_readings_line_names_the_chip_its_phase_its_readings_and_what_it_drives():
    _, s = _state(4, soloist=0, phases={1: "evicting", 2: "idle", 3: "filling"})
    f = render(s, [], 132, 30, True)
    lines = f.text()
    for c in s.chips:
        line = lines[readings_row(f, c.index)]
        assert line.startswith(("┃" if c.index == 0 else "╎") + PHASE_MARK[c.phase] + str(c.index))
        assert c.phase in line and f"{c.temp_c:.0f}C" in line and f"{c.power_w:.0f}W" in line and f"{c.pressure * 100:.0f}%" in line
        drives = [label(d.name) for d in s.devices if d.chip == c.index]
        assert line.rstrip().endswith(" ".join(drives)), line
    # the readings come straight under the heading (and the READINGS row, when there is room), one per chip in order
    first = 2 if lines[1].startswith("READINGS") else 1
    assert [readings_row(f, i) for i in range(4)] == [first, first + 1, first + 2, first + 3]


def test_readings_lines_are_dimmed_by_their_phase():
    _, s = _state(4, soloist=0, phases={1: "evicting", 2: "idle", 3: "filling"})
    f = render(s, [], 80, 24, True)
    for c in s.chips[1:]:
        r = readings_row(f, c.index)
        st = f.rows[r][0].style
        assert st.dim == dim_level(PRESENCE[c.phase]) and not st.bold and st.jack == CHIP_JACKS[c.index]


def test_chips_that_are_not_numbered_from_zero_keep_their_own_index():
    _, s = _state(4, soloist=0)
    new = {0: 2, 1: 5, 2: 7, 3: 9}
    chips = tuple(dataclasses.replace(c, index=new[c.index], phase="evicting" if c.index == 1 else "running") for c in s.chips)
    devices = tuple(dataclasses.replace(d, chip=new[d.chip]) for d in s.devices)
    s = dataclasses.replace(s, chips=chips, devices=devices, soloist_chip=new[s.soloist_chip])
    f = render(s, [], 132, 40, True)
    lines = f.text()
    line5 = lines[readings_row(f, 5)]
    assert line5.startswith("╎◌5") and "evicting" in line5
    assert line5.rstrip().endswith(" ".join(label(d.name) for d in devices if d.chip == 5))
    assert lines[readings_row(f, 2)].startswith("┃●2")
    for d in devices:
        (r, c, st), = module_hits(f, d)
        assert st.dim == (0 if d.soloist else 2 if d.chip == 5 else 1), d.name
        assert (PHASE_MARK["evicting"] if d.chip == 5 else PHASE_MARK["running"]) + str(d.chip) in lines[r][c:c + col_w(f)]


def test_more_chips_than_fit_collapse_to_four_lines_and_a_count_and_the_soloists_chip_stays():
    _, s = _state(12, soloist=0)
    f = render(s, [], 80, 20, True)
    lines = f.text()
    shown = [i for i in range(12) if any(line[1:].startswith(f"●{i} ") for line in lines[1:modules_row(f)])]
    assert shown == [0, 1, 2, 3]
    assert lines[5].strip() == "+8 more chips"
    _, s = _state(12, soloist=11)  # ttrollz on chip 11
    f = render(s, [], 80, 20, True)
    lines = f.text()
    assert [readings_row(f, i) for i in (0, 1, 2, 11)] == [1, 2, 3, 4]
    assert lines[4].startswith("┃●11") and lines[5].strip() == "+8 more chips"


# --- focus follows the signal ---------------------------------------------------------------------------------------------

def test_the_soloists_chip_line_is_bold_with_the_heavy_bar_and_the_others_light():
    _, s = _state(4, soloist=2)  # ttquantus on chip 2
    f = render(s, [], 80, 24, True)
    for c in s.chips:
        r = readings_row(f, c.index)
        first = f.rows[r][0]
        if c.index == s.soloist_chip:
            assert first.text.startswith("┃") and first.style.bold and first.style.dim == 0
        else:
            assert first.text.startswith("╎") and not first.style.bold


def test_the_soloists_module_row_is_bold_with_the_heavy_bar_and_the_marker():
    _, s = _state(4, soloist=2)
    f = render(s, [], 80, 24, True)
    lines = f.text()
    for d in s.devices:
        (r, c, st), = module_hits(f, d)
        if d.soloist:
            assert st.bold and st.dim == 0 and lines[r][c - 2:c] == "┃▶"
            # the whole row is bold: the bar, the name, the chip's mark and index, and the share bar (a cable hint is in the
            # cable's own colour and is not)
            x = (c // col_w(f)) * col_w(f)
            spans, col = [], 0
            for sp in f.rows[r]:
                if x <= col < x + col_w(f) and sp.text.strip() and not sp.text.startswith(("◂", "▸")):
                    spans.append(sp)
                col += len(sp.text)
            assert len(spans) >= 3 and all(sp.style.bold for sp in spans), spans
        else:
            assert not st.bold and st.dim >= 1 and lines[r][c - 2:c] == "╎ ", d.name


def test_cables_that_touch_the_soloist_are_full_brightness_the_rest_one_step_dimmer_and_none_bold():
    _, s = _state(4, soloist=2)  # ttquantus: two of the strongest cables run into it
    f = render(s, [], 80, 30, True)
    rows = cable_rows(f)
    expected = sorted(s.cables, key=lambda c: -c.push)[:5]
    assert len(rows) == 5
    touching = 0
    for r, c in zip(rows, expected):
        line = f.text()[r]
        assert line.startswith(f"╎ {label(c.source)} ═{c.kind} {c.depth:.2f}".replace(" 0.", " .") + "═▶ " + label(c.target))
        styles = [sp.style for sp in f.rows[r] if sp.text.strip()]
        assert all(not st.bold for st in styles)
        cable_style = [sp.style for sp in f.rows[r] if "═" in sp.text][0]
        assert cable_style.jack == CABLE_COLOURS[c.kind]
        if s.soloist_device in (c.source, c.target):
            touching += 1
            assert cable_style.dim == 0
        else:
            assert cable_style.dim >= 1
    assert touching == 2


def test_with_nobody_on_the_floor_a_hot_cable_brightens_by_colour_step_and_is_never_bold():
    _, s = _state(4, soloist=None)
    pushes = (0.6, 0.27, 0.15, 0.05, 0.5)
    pushes += (0.01,) * (len(s.cables) - len(pushes))  # the rest are weaker than all five
    cables = tuple(dataclasses.replace(c, push=p) for c, p in zip(s.cables, pushes))
    f = render(dataclasses.replace(s, cables=cables), [], 80, 30, True)
    styles = [[sp.style for sp in f.rows[r] if "═" in sp.text][0] for r in cable_rows(f)]
    assert [st.dim for st in styles] == [0, 0, 1, 2, 3]
    assert not any(st.bold for st in styles)


def test_bold_is_only_on_the_focused_readings_line_the_soloists_module_row_and_its_stereo_code():
    for n, soloist, size in ((4, 2, (80, 24)), (4, 5, (132, 50)), (12, 11, (200, 60)), (1, 0, (80, 24))):
        _, s = _state(n, soloist=soloist)
        f = render(s, log_rows_of(5), *size, True)
        cells = f.cells()
        focus_line = readings_row(f, s.soloist_chip)
        solo = next(d for d in s.devices if d.soloist)
        (mr, mc, _), = module_hits(f, solo)
        sr = stereo_row(f)
        bold = bold_cells(f)
        assert any(r == focus_line for r, _ in bold) and any(r == mr for r, _ in bold)
        for r, c in bold:
            if r == sr:
                assert cells[r][c][0] in STEREO_CODES[solo.name]
            elif r == mr:
                assert (mc // col_w(f)) * col_w(f) <= c < (mc // col_w(f) + 1) * col_w(f), (n, r, c)
            else:
                assert r == focus_line, (n, r, c, f.text()[r])


def test_only_the_soloists_slot_is_bright_when_its_name_is_dealt_twice():
    chips, s = _state(12, soloist=11)  # slot 11 is ttrollz on chip 11; slot 0 is ttrollz on chip 0
    assert [d.chip for d in s.devices if d.name == "ttrollz"] == [0, 11]
    f = render(s, [], 200, 60, True)
    bright = [r for r, c, st in f.find("ttrollz") if st.bold]
    assert len(bright) == 1
    lines = f.text()
    assert "●11" in lines[bright[0]]


def test_the_stereo_line_puts_each_device_at_its_azimuth_and_the_soloist_on_top():
    _, s = _state(4, soloist=2)
    solo = next(d for d in s.devices if d.soloist)
    other = next(d for d in s.devices if not d.soloist)
    # move one other device onto the soloist's place, so the two collide
    devices = tuple(dataclasses.replace(d, azimuth=solo.azimuth) if d is other else d for d in s.devices)
    s = dataclasses.replace(s, devices=devices)
    w = 80
    f = render(s, [], w, 24, True)
    sr = stereo_row(f)
    line = f.text()[sr]
    col = lambda az: stereo_col(az, w)  # noqa: E731
    taken = {}
    for d in s.devices:
        for k in (0, 1):
            taken.setdefault(col(d.azimuth) + k, []).append(d)
    for d in s.devices:
        c = col(d.azimuth)
        if len(taken[c]) == 1 and len(taken[c + 1]) == 1:  # nothing else drawn over either of its two cells
            assert line[c:c + 2] == STEREO_CODES[d.name], (c, d.name)
    assert line[col(solo.azimuth):col(solo.azimuth) + 2] == STEREO_CODES[solo.name]
    cell = f.cells()[sr][col(solo.azimuth)]
    assert cell[1].bold and cell[1].jack == DEVICE_JACKS[solo.name]
    assert line.startswith("L ") and line.endswith(" R") and len(line) == w


def test_stereo_codes_are_unique_and_cover_every_device():
    assert sorted(STEREO_CODES) == sorted(device_names()) and len(set(STEREO_CODES.values())) == len(STEREO_CODES)
    # two lower-case letters each, so none can be read as the L and R that label the ends of the line
    assert all(len(code) == 2 and code.isalpha() and code.islower() for code in STEREO_CODES.values())


def test_a_code_never_covers_the_l_and_r_at_the_ends_of_the_stereo_line():
    _, s = _state(4, soloist=2)
    for az in (-1.0, -0.999, 1.0, 0.999):
        devices = tuple(dataclasses.replace(d, azimuth=az) for d in s.devices)
        for w in (60, 80, 81, 133):
            line = render(dataclasses.replace(s, devices=devices), [], w, 24, True).text()[stereo_row(render(dataclasses.replace(s, devices=devices), [], w, 24, True))]
            assert line[:2] == "L " and line[-2:] == " R" and len(line) == w, (az, w, line)


# --- without colour -------------------------------------------------------------------------------------------------------

def test_without_colour_focus_and_phase_still_show():
    _, s = _state(4, soloist=2, phases={1: "evicting", 3: "idle"})
    f = render(s, [], 80, 24, colour=False)
    text = "\n".join(f.text())
    solo = next(d for d in s.devices if d.soloist)
    (r, c, _), = module_hits(f, solo)
    assert f.text()[r][c - 2:c] == "▶[" and c - 2 == (c // col_w(f)) * col_w(f)  # the module's row starts with the arrow
    assert f"▶[{label(s.soloist_device)}]" in text and "┃" in text  # and the chip's readings line keeps the heavy bar
    assert "evicting" in f.text()[readings_row(f, 1)] and "idle" in f.text()[readings_row(f, 3)]
    # the brackets mark the soloist only
    assert text.count("[") == 1


def test_colour_and_no_colour_frames_have_the_same_text_but_for_the_brackets():
    for soloist in range(11):
        for w in (60, 70, 80, 100, 132, 200):
            _, s = _state(4, soloist=soloist, phases={1: "evicting", 3: "idle"})
            plain = render(s, log_rows_of(4), w, 30, colour=False).text()
            coloured = render(s, log_rows_of(4), w, 30, colour=True).text()
            assert "[" not in "\n".join(coloured)
            assert len(plain) == len(coloured)
            differ = [i for i, (a, b) in enumerate(zip(plain, coloured)) if a != b]
            assert len(differ) == 1, (soloist, w)
            a, b = plain[differ[0]], coloured[differ[0]]
            assert a.replace("▶[", "┃▶").replace("]", " ") == b, (soloist, w, a, b)  # the arrow and brackets take cells: nothing moves


# --- sizes ----------------------------------------------------------------------------------------------------------------

SIZES = ((60, 16), (80, 24), (80, 30), (81, 25), (100, 30), (132, 50), (200, 60))
COUNTS = (1, 2, 3, 4, 5, 8, 12, 16)


def test_there_is_no_right_hand_border_at_any_size_and_chip_count():
    for n in COUNTS:
        for w, h in SIZES:
            for colour in (True, False):
                f = render(_state(n, soloist=n - 1)[1], log_rows_of(30), w, h, colour)
                for line in f.text():
                    assert line.rstrip()[-1:] not in EDGES, (n, w, h, line)
                    # and no closing corner anywhere in a line
                    assert not set("┐┘┓┛╗╝┤┫╣") & set(line), (n, w, h, line)


def test_no_row_is_wider_than_the_terminal_at_any_size_and_chip_count():
    for n in COUNTS:
        for w, h in SIZES:
            f = render(_state(n, soloist=n - 1)[1], log_rows_of(30), w, h, True)
            assert len(f.rows) <= h and all(len(line) <= w for line in f.text()), (n, w, h)
            assert all(len(r) == w for r in f.cells()) and len(f.cells()) == h


def test_the_log_keeps_its_minimum_rows_at_the_target_size_and_above_and_shows_the_newest():
    for n in COUNTS:
        for w, h in SIZES:
            if w < 80 or h < 24:
                continue
            lines = render(_state(n, soloist=n - 1)[1], log_rows_of(40), w, h, True).text()
            logs = [line for line in lines if line.startswith("log line")]
            assert len(logs) >= 3, (n, w, h)
            if n <= 4 and h <= TARGET[1]:
                assert len(logs) >= LOG_SHORT_MIN_ROWS, (n, w, h)  # on a 24-row screen the log has room for a few whys
            assert logs[-1] == "log line 39" and len(lines) <= h


def _shape(n, h, w=80):
    f = render(_state(n, soloist=0)[1], log_rows_of(40), w, h, True)
    lines = f.text()
    folded = not any(line.startswith("MOD/DELAY") or "MOD/DELAY" in line for line in lines)
    collapsed = any("more chips" in line for line in lines)
    logs = sum(line.startswith("log line") for line in lines)
    return len(cable_rows(f)), folded, collapsed, logs


def test_four_chips_at_the_target_size_need_no_degrading():
    cables, folded, collapsed, logs = _shape(4, 24)
    assert (cables, folded, collapsed) == (SHORT_CABLE_ROWS, False, False) and logs >= LOG_SHORT_MIN_ROWS
    # one row taller, the cables are five again
    assert _shape(4, 25)[:3] == (5, False, False)


def test_at_80_by_24_four_chips_leave_the_log_at_least_six_rows():
    for soloist in range(11):
        f = render(_state(4, soloist=soloist)[1], log_rows_of(40), 80, 24, True)
        assert sum(line.startswith("log line") for line in f.text()) >= 6, soloist
        assert len(cable_rows(f)) == 3


def test_short_screens_cut_cables_first_then_fold_mod_delay_then_collapse_the_readings():
    for n in (4, 6, 12):
        shapes = {h: _shape(n, h) for h in range(16, 45)}
        # the tallest screen shows everything
        assert shapes[44][:3] == (5, False, n > 44)
        for h in range(16, 44):
            cables, folded, collapsed, logs = shapes[h]
            up = shapes[h + 1]
            # once folded or collapsed at some height, every shorter screen is too
            assert not (up[1] and not folded) and not (up[2] and not collapsed), (n, h)
            # the fold comes only after the cables are down to one: the next taller screen was not folded and had one cable,
            # or was folded too
            if folded and not up[1]:
                assert up[0] == 1, (n, h, up)
            if collapsed and not up[2]:
                assert up[1], (n, h, up)  # the readings collapse only once mod/delay has folded
            if not folded:
                assert cables >= 1
        # what really happens at the shortest screen: four chips only fold (READINGS_KEEP is 4, so there is nothing to
        # collapse); six and twelve chips fold and collapse
        assert shapes[16][1] and shapes[16][2] == (n > READINGS_KEEP), (n, shapes[16])
        assert not any(shapes[h][2] for h in shapes) if n <= READINGS_KEEP else True


def test_below_sixty_by_sixteen_one_line_says_so():
    _, s = _state(4)
    f = render(s, [], 59, 24, True)
    assert len(f.rows) == 1 and "too small" in f.text()[0] and "59x24" in f.text()[0]
    assert len(render(s, [], 80, 15, True).rows) == 1
    assert render(s, [], 80, 15, True).text()[0] == "terminal too small for --tui (80x15, needs 60x16); the run goes on"
    assert len(render(s, [], 60, 16, True).rows) > 1


# --- the heading ----------------------------------------------------------------------------------------------------------

def test_before_the_first_segment_the_heading_says_it_is_waiting():
    f = render(None, [], 80, 24, True)
    assert f.text()[0].startswith("waiting for the first segment")
    assert render(None, [], 80, 24, True, unit="phrase").text()[0].startswith("waiting for the first phrase")
    assert heading(None, "segment") == "waiting for the first segment to play"


def test_the_same_state_renders_the_same_frame():
    _, s = _state(4)
    assert render(s, log_rows_of(3), 80, 24, True) == render(s, log_rows_of(3), 80, 24, True)


def test_the_heading_names_the_segment_the_soloist_and_the_patch():
    _, s = _state(4, soloist=5)
    assert heading(s, "segment").startswith("segment 3 · 4.0 s · gong on chip 1 has the floor · ")
    p = s.patch
    assert heading(s, "segment").endswith(f" · {p['style'] or 'plain patch'} · scale {p['scale']} · gong {p['gong']}")
    _, nobody = _state(4, soloist=None)
    assert heading(nobody, "segment").startswith("segment 3 · 4.0 s · nobody has the floor")


def test_the_heading_drops_the_patch_parts_that_do_not_fit():
    _, s = _state(4, soloist=5)
    line = render(s, [], 60, 24, True).text()[0]
    assert line.startswith("segment 3 · 4.0 s · gong on chip 1 has the floor") and len(line) <= 60 and "…" not in line


def test_fast_mode_heading_counts_phrases():
    _, s = _state(4, soloist=5)
    assert heading(s, "phrase").startswith("phrase 3 · gong on chip 1 has the floor")


# --- Task 1c: the soloist's cable, section labels, the hint at 80 columns ----------------------------------------------------

def _weak_solo_cables(soloist=2):
    """A state whose soloist's cables are all weaker than every other cable."""
    _, s = _state(4, soloist=soloist)
    solo = s.soloist_device
    cables = tuple(dataclasses.replace(c, push=(0.02 + 0.01 * k) if solo in (c.source, c.target) else 0.5 + 0.05 * k)
                   for k, c in enumerate(s.cables))
    return dataclasses.replace(s, cables=cables)


def test_the_soloists_strongest_cable_is_always_in_the_cut():
    s = _weak_solo_cables()
    solo = s.soloist_device
    touching = sorted((c for c in s.cables if solo in (c.source, c.target)), key=lambda c: -c.push)
    others = sorted((c for c in s.cables if solo not in (c.source, c.target)), key=lambda c: -c.push)
    assert touching and len(others) >= 5
    for h, n_rows in ((30, 5), (17, 1)):
        f = render(s, [], 80, h, True)
        rows = cable_rows(f)
        assert len(rows) == n_rows, h
        lines = [f.text()[r] for r in rows]
        top = touching[0]
        want = f"{label(top.source)} ═{top.kind}"
        # it takes the last place, after the strongest others, and the list stays strongest first
        assert lines[-1].startswith("╎ " + want), (h, lines)
        for line, c in zip(lines[:-1], others):
            assert line.startswith(f"╎ {label(c.source)} ═{c.kind}"), (h, lines)
        style = [sp.style for sp in f.rows[rows[-1]] if "═" in sp.text][0]
        assert style.dim == 0 and not style.bold


def test_the_soloists_cable_is_not_added_twice_when_it_is_already_strong():
    _, s = _state(4, soloist=2)  # ttquantus: two of the five strongest touch it
    f = render(s, [], 80, 30, True)
    expected = sorted(s.cables, key=lambda c: -c.push)[:5]
    assert [f.text()[r] for r in cable_rows(f)] == [
        f"╎ {label(c.source)} ═{c.kind} {c.depth:.2f}".replace(" 0.", " .") + "═▶ " + label(c.target) for c in expected]


def test_spare_rows_carry_the_cables_and_stereo_labels():
    _, s = _state(4, soloist=5)
    f = render(s, log_rows_of(40), 80, 24, True)
    lines = f.text()
    first_cable = cable_rows(f)[0]
    assert lines[first_cable - 1] == "CABLES" and lines[stereo_row(f) - 1] == "STEREO"
    for r in (first_cable - 1, stereo_row(f) - 1):
        (span,) = f.rows[r]
        assert span.style.jack == "grey" and span.style.dim >= 1 and not span.style.bold
    assert sum(line.startswith("log line") for line in lines) >= 3


def test_labels_take_only_spare_rows_and_go_before_anything_else():
    """A label never causes a cut: one row shorter, a screen with a label loses a label and keeps the same cables, fold and
    readings. And a label never shows when the log is below its minimum."""
    def shape(n, w, h):
        f = render(_state(n, soloist=0)[1], log_rows_of(60), w, h, True)
        lines = f.text()
        structure = (len(cable_rows(f)), "MOD/DELAY" in "\n".join(lines), any("more chips" in line for line in lines))
        return structure, len(extras(lines)), sum(line.startswith("log line") for line in lines)

    seen = 0
    for n in (1, 4, 6, 12):
        for w in (60, 80, 100):
            for h in range(MIN_ROWS + 1, 45):
                structure, labels, logs = shape(n, w, h)
                floor = log_floor(h)
                if logs < floor:
                    assert labels == 0, (n, w, h)
                if labels and h - 1 != TARGET[1]:  # from 25 to 24 rows the cables go from five to three: compared apart
                    seen += 1
                    assert logs >= floor, (n, w, h)
                    shorter, fewer, shorter_logs = shape(n, w, h - 1)
                    assert shorter == structure, (n, w, h)
                    # the row comes from the log while it has rows over its minimum, and from a label after that
                    assert fewer == labels - 1 if logs == floor else (fewer == labels and shorter_logs == logs - 1), (n, w, h)
    assert seen


def test_at_80_columns_a_module_names_the_cable_that_feeds_it_in_place_of_its_share_bar():
    _, s = _state(4, soloist=5)
    f = render(s, [], 80, 24, True)
    lines = f.text()
    steam = next(d for d in s.devices if d.name == "ttsteam")
    hint = _hint(s, steam)
    assert hint == "◂ fold ttrax"
    (r, c, _), = module_hits(f, steam)
    segment = lines[r][c:(c // col_w(f) + 1) * col_w(f)]
    assert hint in segment and "█" not in segment and "░" not in segment
    # every row has either its hint or its share bar
    for d in s.devices:
        (r, c, _), = module_hits(f, d)
        segment = lines[r][c:(c // col_w(f) + 1) * col_w(f)]
        assert (_hint(s, d) or "\0") in segment or bar(min(1.0, d.share * 4), BAR_W) in segment, (d.name, segment)


def test_with_one_spare_row_the_cables_label_comes_first():
    # Four chips at 80 columns take 16 rows with three cables; up to 24 rows the log keeps six.
    _, s = _state(4, soloist=5)
    for h, want in ((22, []), (23, ["CABLES"]), (24, ["CABLES", "STEREO"])):
        assert extras(render(s, log_rows_of(40), 80, h, True).text()) == want, h
    # Eight chips take 22 rows with five cables; the optional rows still leave six rows of log, and the READINGS heading goes first.
    _, s = _state(8, soloist=5)
    for h, want in ((28, []), (29, ["CABLES"]), (30, ["CABLES", "STEREO"]), (31, ["CABLES", "STEREO", "legend"]), (32, list(EXTRAS))):
        assert extras(render(s, log_rows_of(40), 80, h, True).text()) == want, h


# --- the fix wave: legend, READINGS heading, the q hint, the soloist's row in line ----------------------------------------------

def test_a_spare_row_carries_a_legend_of_the_stereo_codes_under_the_stereo_line():
    _, s = _state(4, soloist=2)
    f = render(s, log_rows_of(40), 200, 60, True)
    lines = f.text()
    legend = lines[stereo_row(f) + 1]
    for d in {d.name for d in s.devices}:
        assert f"{STEREO_CODES[d]} {label(d)}" in legend, d
    assert all(not sp.style.bold and sp.style.dim >= 1 for sp in f.rows[stereo_row(f) + 1] if sp.text.strip())
    # at 80 columns it shows what fits, whole pairs only, and never past the edge
    narrow = render(s, log_rows_of(40), 80, 40, True)
    row = narrow.text()[stereo_row(narrow) + 1]
    assert row.startswith(f"{STEREO_CODES['gong']} gong") and len(row) <= 80
    pairs = {f"{STEREO_CODES[name]} {label(name)}" for name in STEREO_CODES}
    assert row.split("  ")[-1] in pairs and all(p in pairs for p in row.split("  ")), row  # whole pairs, two blanks apart
    assert "legend" not in extras(render(s, log_rows_of(40), 80, 24, True).text())


def test_a_readings_heading_names_the_columns_over_the_readings_lines():
    _, s = _state(4, soloist=0)
    f = render(s, log_rows_of(40), 100, 40, True)
    lines = f.text()
    head = _first(f, lambda line: line.startswith("READINGS"))
    line = lines[readings_row(f, 1)]
    assert head + 1 == readings_row(f, 0)
    assert lines[head][14:18] == "temp" and line[17] == "C"
    assert lines[head][19:24] == "power" and line[23] == "W"
    assert lines[head][25:33] == "pressure" and line[28] == "%"
    first_drive = label(next(d.name for d in sorted(s.devices, key=lambda d: d.slot) if d.chip == 1))
    assert lines[head][36:42] == "drives" and line[36:].startswith(first_drive)
    assert all(not sp.style.bold and sp.style.dim >= 1 for sp in f.rows[head] if sp.text.strip())


def test_the_optional_rows_go_in_order_readings_heading_first_and_never_below_the_logs_floor():
    for n in (1, 4, 8):
        for w in (80, 100):
            for h in range(MIN_ROWS, 50):
                lines = render(_state(n, soloist=0)[1], log_rows_of(60), w, h, True).text()
                got = extras(lines)
                assert got == list(EXTRAS[:len(got)]), (n, w, h, got)
                if got:
                    assert sum(line.startswith("log line") for line in lines) >= log_floor(h), (n, w, h)


def test_the_heading_ends_with_a_dim_q_quit_hint_when_it_fits():
    _, s = _state(4, soloist=5)
    for w in (80, 100, 200):
        f = render(s, [], w, 24, True)
        top = f.text()[0]
        assert top.endswith(QUIT_HINT) and len(top) == w and "has the floor" in top, w
        assert top[-len(QUIT_HINT) - 2:-len(QUIT_HINT)] == "  ", w  # never run into the heading
        (hint,) = [sp for sp in f.rows[0] if sp.text == QUIT_HINT]
        assert hint.style.dim >= 1 and not hint.style.bold
    # where the heading itself needs the room, the hint is left out
    long = dataclasses.replace(s, soloist_device="ultrasonic_shimmer_with_a_very_long_name")
    assert QUIT_HINT not in render(long, [], 60, 24, True).text()[0]


def test_a_notice_takes_the_headings_place():
    _, s = _state(4, soloist=5)
    top = render(s, [], 80, 24, True, notice="stopping: playing out what is already made, Ctrl-C to stop now").text()[0]
    assert top.startswith("stopping: playing out what is already made") and "has the floor" not in top


def test_the_chip_marks_of_a_module_column_line_up_with_the_soloist_among_them():
    for colour in (True, False):
        for w in (80, 100, 200):
            for soloist in range(11):
                _, s = _state(4, soloist=soloist)
                f = render(s, [], w, 40, colour)
                cw = col_w(f)
                columns = {}
                for d in s.devices:
                    (r, c, _), = module_hits(f, d)
                    line = f.text()[r]
                    k = c // cw
                    mark = next(i for i in range(c, (k + 1) * cw) if line[i] in "●◌…")
                    columns.setdefault((r > modules_row(f) + 3, k), set()).add(mark)
                assert all(len(marks) == 1 for marks in columns.values()), (colour, w, soloist, columns)


def test_the_q_hint_never_touches_the_heading_at_any_width():
    for soloist in (None, 0, 5, 10):
        _, s = _state(4, soloist=soloist)
        for w in range(MIN_COLS, 160):
            top = render(s, [], w, 24, True).text()[0]
            if top.endswith(QUIT_HINT):
                assert top[-len(QUIT_HINT) - 2:-len(QUIT_HINT)] == "  ", (soloist, w, top)


def test_a_taller_screen_never_gives_the_log_fewer_rows_but_where_the_cables_go_from_three_to_five():
    # The optional rows always leave LOG_SHORT_MIN_ROWS of log, so they never take a taller screen's log below a shorter one's.
    # From 24 to 25 rows the cables go from three to five: two rows more for one, so the log may lose one row there.
    for n in (1, 4, 8):
        for w in (80, 100):
            heights = range(MIN_ROWS, 50)
            logs = {h: sum(line.startswith("log line") for line in render(_state(n, soloist=0)[1], log_rows_of(60), w, h, True).text())
                    for h in heights}
            for h in heights[1:]:
                assert logs[h] >= logs[h - 1] - (1 if h == TARGET[1] + 1 else 0), (n, w, h, logs)
