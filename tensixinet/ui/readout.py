# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The Play tab's readout: for each chip, the pitch, tempo and pressure its readings give, and which devices are playing.

The numbers come from `music/mapping.py` (the same functions the devices use), applied to the readings of the block the listener is hearing, so the
page and the sound agree. Inline styles only, so the string can be the HTML of any page.
"""
from __future__ import annotations

import math
from html import escape

from ..music.mapping import C3_HZ, aiclk_to_bpm, chip_pressure, temp_to_pitch_hz
from ..narrate.color import CHIP_JACKS, JACK_HEX
from .chart import flow_svg
from .instrument import InstrumentSnapshot

PANEL = '<div class="readout" style="background:#092221;border-radius:8px;padding:12px 14px">'  # the brand's dark teal: the readout's colours are written for it, whatever the page's theme
READOUT_WAITING = PANEL + '<div style="color:#7d9a98;font:14px ui-monospace,monospace">press Play: each chip\'s pitch, tempo and pressure show here as you move the knobs</div></div>'
NOTES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def note_name(hz: float) -> str:
    """The nearest equal-tempered note to `hz`, with its octave (A4 is 440 Hz)."""
    midi = round(69 + 12 * math.log2(max(hz, 1e-3) / 440.0))
    return f"{NOTES[midi % 12]}{midi // 12 - 1}"


def readout_html(snapshot: InstrumentSnapshot) -> str:
    """One row per chip (colour, pitch and note, tempo, a pressure bar) and a line for what is playing, then any notes."""
    if not snapshot.telemetry:
        return READOUT_WAITING
    rows = []
    for i, t in enumerate(snapshot.telemetry):
        colour = "#" + JACK_HEX[CHIP_JACKS[i % len(CHIP_JACKS)]]
        hz, bpm, pressure = temp_to_pitch_hz(t.asic_temperature), aiclk_to_bpm(t.aiclk), chip_pressure(t)  # 0 to 1
        rows.append(
            f'<div class="chip" style="margin:6px 0;font:13px ui-monospace,monospace;color:#cfe4e2">'
            f'<div style="color:#cfe4e2"><span style="color:{colour};font-weight:600">chip {i}</span> &nbsp; '  # the page's theme colours a div's text itself, so each sets its own
            f'{note_name(hz)} ({hz:.1f} Hz) &nbsp; {bpm:.0f} BPM &nbsp; pressure {pressure:.2f}</div>'
            f'<div class="bar" style="height:6px;background:#12302e;border-radius:3px;margin-top:3px">'
            f'<div style="width:{round(pressure * 100)}%;height:100%;background:{colour};border-radius:3px"></div></div></div>')
    playing = ", ".join(escape(n) for n in snapshot.playing) or "nothing (see below)"
    lines = [f'<div style="margin-top:8px;font:12px ui-monospace,monospace;color:#7d9a98">playing: {playing}</div>']
    lines += [f'<div style="font:12px ui-monospace,monospace;color:#f6bc42">{escape(n)}</div>' for n in snapshot.notes]
    return PANEL + "".join(rows + lines) + "</div>"


def play_html(snapshot: InstrumentSnapshot) -> str:
    """What the Play tab shows: the flow chart of the block being heard (the same drawing as the first tab, so a knob's effect on the chips, the cables,
    the floor and the stereo field is seen), and the readout under it. Without a flow it is the readout alone. The heading names who has the floor,
    and carries no segment or time, so the page changes only when something in it does."""
    if not snapshot.telemetry or snapshot.state is None:
        return readout_html(snapshot)
    s = snapshot.state
    heading = (f"{s.soloist_device} on chip {s.soloist_chip} has the floor" if s.soloist_device
               else "the chosen devices play at full level, no turn-taking")
    return flow_svg(s, heading) + readout_html(snapshot)
