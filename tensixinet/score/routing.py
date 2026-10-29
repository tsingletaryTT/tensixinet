# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Where each kind of event goes on the MIDI wire: channels, drum notes and controller numbers.

Channels are stored zero-based (0 to 15). The docs and the CLI talk in 1 to 16.
"""
import json
import math
from dataclasses import dataclass, fields, replace
from typing import Mapping, Tuple

CHANNELS = range(16)
DATA_BYTES = range(128)


@dataclass(frozen=True)
class Routes:
    """The route table. `note_channels` maps a pitched source to its channel;
    `trigger_notes` maps an unpitched source to a drum note (`ttrollz_clock` is the roll
    clock, lane -1; the other roll lanes use `roll_lane_notes`, cycling); `control_ccs`
    maps a control name to a CC number; `pulse_s` is how long a trigger's note stays on;
    `bend_range` is the pitch-bend span in semitones that the full wheel covers."""
    note_channels: Mapping[str, int]
    trigger_notes: Mapping[str, int]
    roll_lane_notes: Tuple[int, ...]
    control_ccs: Mapping[str, int]
    control_channel: int
    pulse_s: float
    bend_range: float
    trigger_channel: int = 9


DEFAULT_ROUTES = Routes(
    note_channels={"lead": 0, "ttquantus": 1},
    trigger_notes={"gong": 49, "ttsteam": 38, "ttrollz_clock": 36},
    roll_lane_notes=(42, 44, 46, 54, 56, 60, 61, 62, 63, 64),
    control_ccs={"temperature": 20, "power": 21, "pressure": 22, "chaos": 23},
    control_channel=15,
    pulse_s=0.05,
    bend_range=2.0,
    trigger_channel=9,
)


def _check_range(key: str, value, allowed: range) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value not in allowed:
        raise ValueError(f"{key}: {value!r} is outside {allowed.start}-{allowed.stop - 1}")


def _check_positive(key: str, value) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not value > 0 or not math.isfinite(value):
        raise ValueError(f"{key}: {value!r} must be a finite number above 0")


def load_routes(path) -> Routes:
    """Read a JSON file with any subset of the `Routes` fields and merge it over the defaults.
    Mappings merge key by key. Unknown keys and out-of-range numbers raise `ValueError`."""
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("the file must be a JSON object")
    known = {f.name for f in fields(Routes)}
    for key in data:
        if key not in known:
            raise ValueError(f"{key}: unknown routes key")
    merged = {}
    for key, value in data.items():
        if key in ("note_channels", "trigger_notes", "control_ccs"):
            if not isinstance(value, dict):
                raise ValueError(f"{key}: must be a JSON object")
            merged[key] = {**getattr(DEFAULT_ROUTES, key), **value}
        elif key == "roll_lane_notes":
            if not isinstance(value, list):
                raise ValueError(f"{key}: must be a list")
            merged[key] = tuple(value)
        else:
            merged[key] = value
    routes = replace(DEFAULT_ROUTES, **merged)
    for key in ("control_channel", "trigger_channel"):
        _check_range(key, getattr(routes, key), CHANNELS)
    for name, ch in routes.note_channels.items():
        _check_range(f"note_channels[{name}]", ch, CHANNELS)
    for name, n in routes.trigger_notes.items():
        _check_range(f"trigger_notes[{name}]", n, DATA_BYTES)
    for n in routes.roll_lane_notes:
        _check_range("roll_lane_notes", n, DATA_BYTES)
    for name, cc in routes.control_ccs.items():
        _check_range(f"control_ccs[{name}]", cc, DATA_BYTES)
    _check_positive("pulse_s", routes.pulse_s)
    _check_positive("bend_range", routes.bend_range)
    if not routes.roll_lane_notes:
        raise ValueError("roll_lane_notes: needs at least one note")
    return routes
