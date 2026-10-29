# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Write timed MIDI messages as a format 1 Standard MIDI File.

Track 0 holds the tempo and a name. Chip `k` gets its own track, named "tensixinet chip k",
and a chip with no messages gets none (the tracks stay in chip order).
"""
import struct
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

END_OF_TRACK = b"\xff\x2f\x00"


def vlq(n: int) -> bytes:
    """A variable-length quantity: 7 bits per byte, high bit set on every byte but the last."""
    if n < 0:
        raise ValueError("a variable-length quantity cannot be negative")
    out = [n & 0x7F]
    n >>= 7
    while n:
        out.append((n & 0x7F) | 0x80)
        n >>= 7
    return bytes(reversed(out))


def _meta(kind: int, payload: bytes) -> bytes:
    return bytes([0xFF, kind]) + vlq(len(payload)) + payload


def _chunk(tag: bytes, body: bytes) -> bytes:
    return tag + struct.pack(">I", len(body)) + body


def write_midi_file(path, messages: Sequence[Tuple[float, bytes, int]], bpm: float, ppq: int = 480) -> None:
    """Write `messages` (seconds, raw bytes, chip) to `path`. Each event's tick is
    `round(seconds * bpm / 60 * ppq)`, and the delta stored is the difference between ticks, so rounding never builds up."""
    micros = round(60_000_000 / bpm)
    tempo = [b"\x00" + _meta(0x03, b"tensixinet"), b"\x00" + _meta(0x51, micros.to_bytes(3, "big")), b"\x00" + END_OF_TRACK]
    by_chip: Dict[int, List[Tuple[int, int, bytes]]] = {}
    for order, (seconds, data, chip) in enumerate(messages):
        by_chip.setdefault(chip, []).append((round(seconds * bpm / 60 * ppq), order, data))
    chunks = [_chunk(b"MTrk", b"".join(tempo))]
    for chip in sorted(by_chip):
        parts = [b"\x00" + _meta(0x03, f"tensixinet chip {chip}".encode())]
        last = 0
        for tick, _, data in sorted(by_chip[chip], key=lambda e: (e[0], e[1])):
            parts.append(vlq(tick - last) + data)
            last = tick
        parts.append(b"\x00" + END_OF_TRACK)
        chunks.append(_chunk(b"MTrk", b"".join(parts)))
    header = _chunk(b"MThd", struct.pack(">HHH", 1, len(chunks), ppq))
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(header + b"".join(chunks))
