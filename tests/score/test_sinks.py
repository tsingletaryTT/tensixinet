# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Tests for the score sinks."""
from tensixinet.score.events import Control, Note, Score, Trigger
from tensixinet.score.sinks import MidiFileSink, TeeScoreSink
from tests.score.test_smf import read_smf


def score(bpm=120.0, *events):
    return Score(events=tuple(events), duration=2.0, bpm=bpm)


def note_on_ticks(path):
    _, _, _, tracks = read_smf(path)
    return [t for track in tracks[1:] for t, e in track if e[0] == 0x90]


def test_a_second_add_lands_start_s_later(tmp_path):
    p = tmp_path / "x.mid"
    sink = MidiFileSink(p)
    s = score(120.0, Note(0.5, 0.2, 440.0, 0.5, "lead", 0))
    sink.add(s, 0.0)
    sink.add(s, 2.0)
    sink.close()
    assert note_on_ticks(p) == [round(0.5 * 2 * 480), round(2.5 * 2 * 480)]


def test_close_makes_the_folder_and_the_file(tmp_path):
    p = tmp_path / "deep" / "er" / "x.mid"
    sink = MidiFileSink(p)
    sink.add(score(120.0, Trigger(0.1, 0.8, "gong", 1)), 0.0)
    sink.close()
    assert p.exists()


def test_the_files_tempo_is_the_first_scores(tmp_path):
    p = tmp_path / "x.mid"
    sink = MidiFileSink(p)
    sink.add(score(100.0, Note(0.0, 0.2, 440.0, 0.5, "lead", 0)), 0.0)
    sink.add(score(150.0, Note(0.0, 0.2, 440.0, 0.5, "lead", 0)), 1.0)
    sink.close()
    _, _, _, tracks = read_smf(p)
    (tempo,) = [e for _, e in tracks[0] if e[:2] == b"\xff\x51"]
    assert int.from_bytes(tempo[3:6], "big") == 600000


def test_summary_counts_notes_triggers_and_controls(tmp_path):
    sink = MidiFileSink(tmp_path / "x.mid")
    sink.add(score(120.0, Note(0.0, 0.2, 440.0, 0.5, "lead", 0), Trigger(0.1, 0.5, "gong", 0), Trigger(0.2, 0.5, "gong", 0),
                   Control(0.0, "power", 0.5, 0)), 0.0)
    assert sink.summary() == (1, 2, 1)


def test_an_empty_run_still_writes_a_file(tmp_path):
    p = tmp_path / "x.mid"
    sink = MidiFileSink(p)
    sink.close()
    assert p.exists() and read_smf(p)[1] == 1


def test_tee_forwards_add_and_close_to_every_sink():
    calls = []

    class Spy:
        def __init__(self, name):
            self.name = name

        def add(self, s, start_s):
            calls.append((self.name, "add", start_s))

        def close(self):
            calls.append((self.name, "close"))

    tee = TeeScoreSink(Spy("a"), Spy("b"))
    tee.add(score(), 1.5)
    tee.close()
    assert calls == [("a", "add", 1.5), ("b", "add", 1.5), ("a", "close"), ("b", "close")]
