# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Draw a `FlowState` as one animated SVG: the chips, the devices dealt to them, the cables between the devices and the stereo field.

Left to right: each chip (with its temperature, power and pressure) is wired to the devices it plays; the devices carry a bar for their
share of the floor, and the patch cables arc between them; on the right is the stereo field as seen from above, a dot for each voice
at its azimuth and depth. The chips' wires flow faster the more power a chip draws, a cable is thicker the harder it pushes, the voice
that has the floor glows, and a chip that is evicting or idle fades its voices by the same table the mix uses (music/evict.py).
No script, so it can be set as the HTML of any page.
"""
from __future__ import annotations

from html import escape
from typing import Optional, Tuple

from ..music.evict import PRESENCE
from ..narrate.color import CHIP_JACKS, DEVICE_JACKS, JACK_HEX
from .flow import FlowState

WIDTH, HEIGHT = 1100, 640
FIELD_CX, FIELD_W = 840.0, 230.0  # the field's centre and half width, in the page's units
FIELD_TOP, FIELD_BOTTOM = 150.0, 450.0  # a far voice is drawn at the top, a near one at the bottom
CABLE_COLOURS = {"fm": "blue", "crunch": "red", "inflate": "yellow", "fold": "purple"}  # the guide's jacks for each way a cable acts

STYLE = """
.flow text{font-family:ui-monospace,Menlo,Consolas,monospace;fill:#cfe4e2;font-size:12px}
.flow .dim{fill:#7d9a98}.flow .big{font-size:15px;font-weight:600;fill:#e9f1f0}
.flow .wire,.flow .cable{fill:none;stroke-dasharray:6 8;animation:flow 2s linear infinite}
.flow .cable{opacity:.85}
@keyframes flow{to{stroke-dashoffset:-28}}
@keyframes glow{50%{opacity:.55}}
.flow .solo rect,.flow .voice.solo{animation:glow 1.2s ease-in-out infinite}
.flow .chip.evicting rect.box,.flow .chip.idle rect.box{stroke-dasharray:4 4}
.flow .chip.idle{opacity:.45}.flow .chip.evicting{opacity:.7}
.flow .device.silent{opacity:.3}
"""


def hexc(name: str) -> str:
    return "#" + JACK_HEX[name]


def chip_colour(index: int) -> str:
    return hexc(CHIP_JACKS[index % len(CHIP_JACKS)])


def device_colour(name: str) -> str:
    return hexc(DEVICE_JACKS.get(name, "grey"))


def flow_seconds(power_w: float) -> float:
    """How long a wire takes to run its dashes once: 3 s for an idle chip, down to 0.7 s at 100 W and more."""
    return 3.0 - 2.3 * min(max(power_w, 0.0) / 100.0, 1.0)


def field_xy(azimuth: float, depth: float) -> Tuple[float, float]:
    """Where a voice is drawn in the field: azimuth -1 to +1 across, depth 0 (near, low) to 1 (far, high)."""
    return FIELD_CX + max(-1.0, min(1.0, azimuth)) * FIELD_W, FIELD_BOTTOM - max(0.0, min(1.0, depth)) * (FIELD_BOTTOM - FIELD_TOP)


def _bar(x: float, y: float, w: float, value: float, colour: str, label: str) -> str:
    value = max(0.0, min(1.0, value))
    return (f'<text class="dim" x="{x}" y="{y + 9}">{label}</text>'
            f'<rect x="{x + 44}" y="{y}" width="{w}" height="10" rx="3" fill="#16302f"/>'
            f'<rect x="{x + 44}" y="{y}" width="{w * value:.1f}" height="10" rx="3" fill="{colour}"/>')


SILENT_PRESENCE = 0.25  # a voice that is not played is drawn at this fraction of its presence


def flow_svg(state: FlowState, heading: Optional[str] = None) -> str:
    """The chart. `heading` replaces the first line (the segment, its time and who has the floor), for a page that has no segments to count."""
    n = len(state.chips)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" class="flow" viewBox="0 0 {WIDTH} {HEIGHT}" role="img" '
           f'aria-label="What is playing: {n} chips, {len(state.devices)} devices, {len(state.cables)} cables, and where each voice sits">',
           f"<style>{STYLE}</style>", f'<rect width="{WIDTH}" height="{HEIGHT}" rx="10" fill="#0b1f1e"/>']
    p = state.patch
    head = escape(heading) if heading is not None else (
        f"segment {state.segment} · {state.seconds:.1f} s · {escape(state.soloist_device)} on chip {state.soloist_chip} has the floor")
    patch = (f"{escape(p['style'] or 'plain patch')} · scale {escape(p['scale'])} · gong {escape(p['gong'])} · ttrax {p['rax_bars']} bars"
             f" · routes {escape(', '.join(p['routes']) or 'none')} · seed {p['seed']}")
    out.append(f'<text class="big" x="20" y="28">{head}</text><text class="dim" x="20" y="46">{patch}</text>')
    for x, label in ((20, "chips"), (300, "devices (the share of the floor in the bar)"), (840 - 60, "stereo field, from above")):
        out.append(f'<text class="dim" x="{x}" y="72">{label}</text>')

    # the chips, down the left
    top, room = 84.0, HEIGHT - 84.0 - 16.0
    chip_h = min(112.0, (room - 10.0 * (n - 1)) / n)
    chip_y = {c.index: top + c.index * (chip_h + 10.0) for c in state.chips}
    for c in state.chips:
        y, colour = chip_y[c.index], chip_colour(c.index)
        out.append(
            f'<g class="chip {c.phase}" data-chip="{c.index}">'
            f'<rect class="box" x="20" y="{y:.1f}" width="190" height="{chip_h:.1f}" rx="8" fill="#10292a" stroke="{colour}" stroke-width="2"/>'
            f'<text class="big" x="32" y="{y + 20:.1f}" style="fill:{colour}">chip {c.index}</text>'
            f'<text class="dim" x="108" y="{y + 20:.1f}">{escape(c.phase)}</text>'
            + _bar(32, y + 32, 110, c.temp_c / 100.0, colour, f"{c.temp_c:.0f}C")
            + _bar(32, y + 48, 110, c.power_w / 150.0, colour, f"{c.power_w:.0f}W")
            + _bar(32, y + 64, 110, c.pressure, colour, "press")
            + _bar(32, y + 80, 110, c.chaos, colour, "chaos")
            + "</g>")

    # the devices, in the middle, grouped by chip
    order = sorted(state.devices, key=lambda d: (d.chip, d.slot))
    pill_h = 36.0
    gap = (room - pill_h * len(order)) / max(1, len(order) - 1) if len(order) > 1 else 0.0
    pill_y = {d.name: top + i * (pill_h + min(gap, 12.0)) for i, d in enumerate(order)}
    powers = {c.index: c.power_w for c in state.chips}
    for d in order:  # wires first, so the pills sit on top
        y = pill_y[d.name] + pill_h / 2
        cy = chip_y[d.chip] + chip_h / 2
        out.append(f'<path class="wire" data-chip="{d.chip}" d="M210 {cy:.1f} C 255 {cy:.1f} 255 {y:.1f} 300 {y:.1f}" '
                   f'stroke="{chip_colour(d.chip)}" stroke-width="2" style="animation-duration:{flow_seconds(powers[d.chip]):.2f}s"/>')
    for c in state.cables:
        ya, yb = pill_y[c.source] + pill_h / 2, pill_y[c.target] + pill_h / 2
        bulge = 30.0 + abs(ya - yb) * 0.18
        colour = hexc(CABLE_COLOURS.get(c.kind, "grey"))
        out.append(f'<path class="cable {escape(c.kind)}" data-source="{escape(c.source)}" data-target="{escape(c.target)}" '
                   f'd="M470 {ya:.1f} C {470 + bulge:.1f} {ya:.1f} {470 + bulge:.1f} {yb:.1f} 470 {yb:.1f}" stroke="{colour}" '
                   f'stroke-width="{1.0 + 4.0 * c.push:.2f}" style="animation-duration:{flow_seconds(60.0 * (0.3 + c.push)):.2f}s"/>')
    for d in order:
        y, colour = pill_y[d.name], device_colour(d.name)
        out.append(
            f'<g class="device{" solo" if d.soloist else ""}{"" if d.heard else " silent"}" data-device="{escape(d.name)}" data-chip="{d.chip}">'
            f'<rect x="300" y="{y:.1f}" width="170" height="{pill_h}" rx="8" fill="#10292a" stroke="{colour}" stroke-width="{3 if d.soloist else 1.5}"/>'
            f'<text x="312" y="{y + 15:.1f}" style="fill:{colour}">{escape(d.name)}</text>'
            f'<text class="dim" x="440" y="{y + 15:.1f}">c{d.chip}</text>'
            f'<rect x="312" y="{y + 22:.1f}" width="146" height="6" rx="3" fill="#16302f"/>'
            f'<rect x="312" y="{y + 22:.1f}" width="{146 * min(1.0, d.share * 4):.1f}" height="6" rx="3" fill="{colour}"/>'
            "</g>")
    kinds = "".join(f'<tspan style="fill:{hexc(col)}"> ● {k}</tspan>' for k, col in CABLE_COLOURS.items())
    out.append(f'<text class="dim" x="488" y="{HEIGHT - 8}">cables:{kinds}</text>')

    # the stereo field
    out.append(f'<rect x="{FIELD_CX - FIELD_W - 20}" y="{FIELD_TOP - 30}" width="{2 * FIELD_W + 40}" height="{FIELD_BOTTOM - FIELD_TOP + 70}" rx="10" '
               f'fill="#0f2625" stroke="#1d3d3b"/>')
    out.append(f'<text class="dim" x="{FIELD_CX - FIELD_W}" y="{FIELD_BOTTOM + 42}">left</text>'
               f'<text class="dim" x="{FIELD_CX + FIELD_W - 28}" y="{FIELD_BOTTOM + 42}">right</text>'
               f'<text class="dim" x="{FIELD_CX - 24}" y="{FIELD_TOP - 14}">far</text><text class="dim" x="{FIELD_CX - 24}" y="{FIELD_BOTTOM + 28}">near</text>')
    for d in state.devices:
        x, y = field_xy(d.azimuth, d.depth)
        presence = PRESENCE[state.chips[d.chip].phase] * (1.0 if d.heard else SILENT_PRESENCE)
        r = 11 if d.soloist else 7
        label = f'<text class="{"" if d.soloist else "dim"}" x="{x + r + 3:.1f}" y="{y + 4:.1f}" opacity="{presence:.2f}">{escape(d.name)}</text>'
        out.append(f'<g><circle class="voice{" solo" if d.soloist else ""}{"" if d.heard else " silent"}" data-device="{escape(d.name)}" cx="{x:.1f}" cy="{y:.1f}" r="{r}" '
                   f'fill="{device_colour(d.name)}" stroke="{chip_colour(d.chip)}" stroke-width="2" opacity="{presence:.2f}"/>{label}</g>')
    out.append("</svg>")
    return "".join(out)
