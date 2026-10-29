# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Tests for the Standard MIDI File writer, read back with a small reader written here."""
import struct

import pytest

from tensixinet.score.smf import vlq, write_midi_file


def read_vlq(data, i):
    value = 0
    while True:
        b = data[i]
        i += 1
        value = (value << 7) | (b & 0x7F)
        if not b & 0x80:
            return value, i


def read_smf(path):
    """(format, ntracks, ppq, tracks); a track is a list of (absolute_tick, bytes) with meta events as FF..."""
    data = open(path, "rb").read()
    assert data[:4] == b"MThd" and struct.unpack(">I", data[4:8])[0] == 6
    fmt, ntracks, ppq = struct.unpack(">HHH", data[8:14])
    i, tracks = 14, []
    while i < len(data):
        assert data[i : i + 4] == b"MTrk"
        (n,) = struct.unpack(">I", data[i + 4 : i + 8])
        body, i = data[i + 8 : i + 8 + n], i + 8 + n
        events, j, tick, status = [], 0, 0, 0
        while j < len(body):
            delta, j = read_vlq(body, j)
            tick += delta
            if body[j] == 0xFF:
                length, k = read_vlq(body, j + 2)
                events.append((tick, body[j : k + length]))
                j = k + length
            else:
                status = body[j]
                size = 2 if status & 0xF0 in (0x80, 0x90, 0xA0, 0xB0, 0xE0) else 1
                events.append((tick, body[j : j + 1 + size]))
                j += 1 + size
        tracks.append(events)
    return fmt, ntracks, ppq, tracks


MSGS = [
    (0.0, bytes([0x90, 60, 100]), 0),
    (0.5, bytes([0x80, 60, 0]), 0),
    (1.0, bytes([0x99, 38, 90]), 2),
    (2.37, bytes([0xB0, 20, 64]), 2),
]


@pytest.mark.parametrize("n,encoded", [(0, b"\x00"), (127, b"\x7f"), (128, b"\x81\x00"), (16383, b"\xff\x7f"), (16384, b"\x81\x80\x00")])
def test_variable_length_quantities_round_trip(n, encoded):
    assert vlq(n) == encoded
    assert read_vlq(encoded, 0) == (n, len(encoded))


def test_header_tempo_and_one_track_per_chip_that_has_messages(tmp_path):
    p = tmp_path / "a.mid"
    write_midi_file(p, MSGS, bpm=100.0)
    fmt, ntracks, ppq, tracks = read_smf(p)
    assert (fmt, ppq) == (1, 480)
    assert ntracks == 3 and len(tracks) == 3  # tempo track, chip 0, chip 2
    tempo = [e for _, e in tracks[0] if e[:2] == b"\xff\x51"]
    assert len(tempo) == 1 and int.from_bytes(tempo[0][3:6], "big") == 600000  # 60_000_000 / 100
    name = [e for _, e in tracks[0] if e[:2] == b"\xff\x03"]
    assert name and b"tensixinet" in name[0]
    assert any(e[:2] == b"\xff\x03" and e.endswith(b"chip 0") for _, e in tracks[1])
    assert any(e[:2] == b"\xff\x03" and e.endswith(b"chip 2") for _, e in tracks[2])


def test_ticks_are_the_rounded_time_in_beats(tmp_path):
    p = tmp_path / "a.mid"
    write_midi_file(p, MSGS, bpm=100.0)
    _, _, _, tracks = read_smf(p)
    chip0 = [(t, e) for t, e in tracks[1] if e[0] != 0xFF]
    chip2 = [(t, e) for t, e in tracks[2] if e[0] != 0xFF]
    assert [t for t, _ in chip0] == [round(s * 100 / 60 * 480) for s in (0.0, 0.5)]
    assert [t for t, _ in chip2] == [round(s * 100 / 60 * 480) for s in (1.0, 2.37)]
    assert [e for _, e in chip2] == [MSGS[2][1], MSGS[3][1]]


def test_every_track_ends_with_end_of_track(tmp_path):
    p = tmp_path / "a.mid"
    write_midi_file(p, MSGS, bpm=100.0)
    _, _, _, tracks = read_smf(p)
    assert all(t[-1][1] == b"\xff\x2f\x00" for t in tracks)


def test_an_empty_score_is_a_valid_file_with_only_the_tempo_track(tmp_path):
    p = tmp_path / "e.mid"
    write_midi_file(p, [], bpm=120.0)
    fmt, ntracks, _, tracks = read_smf(p)
    assert (fmt, ntracks) == (1, 1) and tracks[0][-1][1] == b"\xff\x2f\x00"


def test_a_long_gap_uses_a_multi_byte_delta(tmp_path):
    p = tmp_path / "g.mid"
    write_midi_file(p, [(0.0, bytes([0x90, 60, 100]), 0), (60.0, bytes([0x80, 60, 0]), 0)], bpm=120.0)
    _, _, _, tracks = read_smf(p)
    assert [t for t, e in tracks[1] if e[0] != 0xFF] == [0, 60 * 2 * 480]


def test_mido_reads_the_same_notes(tmp_path):
    mido = pytest.importorskip("mido")
    p = tmp_path / "m.mid"
    write_midi_file(p, MSGS, bpm=100.0)
    mid = mido.MidiFile(str(p))
    assert len(mid.tracks) == 3
    notes = [(m.type, m.note) for m in mid.tracks[1] if m.type in ("note_on", "note_off")]
    assert notes == [("note_on", 60), ("note_off", 60)]


def test_a_tick_that_is_not_whole_rounds_to_the_nearest(tmp_path):
    p = tmp_path / "r.mid"
    write_midi_file(p, [(0.0021, bytes([0x90, 60, 100]), 0)], bpm=100.0)  # 1.68 ticks: nearest is 2, truncation is 1
    _, _, _, tracks = read_smf(p)
    assert [t for t, e in tracks[1] if e[0] != 0xFF] == [2]
