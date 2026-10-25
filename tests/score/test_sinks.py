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


# --- the live port sink ---------------------------------------------------------------------------------------------
import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import types  # noqa: E402

import pytest  # noqa: E402

from tensixinet.score.routing import DEFAULT_ROUTES  # noqa: E402
from tensixinet.score.sinks import MidiPortError, MidiPortSink  # noqa: E402


class FakePort:
    """Records raw messages; can be told to fail on the nth send."""

    def __init__(self, fail_on=None):
        self.sent, self.closed, self.fail_on = [], 0, fail_on

    def send(self, raw):
        if self.fail_on is not None and len(self.sent) + 1 >= self.fail_on:
            raise OSError("port went away")
        self.sent.append(bytes(raw))

    def close(self):
        self.closed += 1


class FakeClock:
    def __init__(self, now=100.0):
        self.now = now

    def __call__(self):
        return self.now


def port_sink(port=None, offset_ms=0.0, clock=None, errors=None, **kw):
    port = port or FakePort()
    clock = clock or FakeClock()
    sink = MidiPortSink("fake", offset_ms=offset_ms, open_output=lambda name: port, clock=clock, sleep=lambda s: None,
                        on_error=(errors.append if errors is not None else None), threaded=False, **kw)
    return sink, port, clock


def ons(port):
    return [m for m in port.sent if m[0] & 0xF0 == 0x90 and m[2] > 0]


def test_a_message_goes_out_only_when_its_time_has_come():
    sink, port, clock = port_sink()
    sink.add(score(120.0, Note(0.5, 0.2, 440.0, 0.5, "lead", 0)), 0.0)
    sink._pump(clock.now + 0.4)
    assert ons(port) == []
    sink._pump(clock.now + 0.5)
    assert len(ons(port)) == 1


def test_messages_leave_in_time_order_from_the_base_set_at_the_first_add():
    sink, port, clock = port_sink()
    sink.add(score(120.0, Note(0.9, 0.05, 880.0, 0.5, "lead", 0), Note(0.2, 0.05, 440.0, 0.5, "lead", 0)), 0.0)
    clock.now += 50  # a later add does not move the base
    sink._pump(clock.now + 5)
    assert [m[1] for m in ons(port)] == [69, 81]


def test_the_offset_moves_every_message_later_by_that_many_milliseconds():
    sink, port, clock = port_sink(offset_ms=250.0)
    base = clock.now
    sink.add(score(120.0, Note(0.5, 0.2, 440.0, 0.5, "lead", 0)), 0.0)
    sink._pump(base + 0.74)
    assert ons(port) == []
    sink._pump(base + 0.76)
    assert len(ons(port)) == 1


def test_a_negative_offset_moves_messages_earlier():
    sink, port, clock = port_sink(offset_ms=-200.0)
    base = clock.now
    sink.add(score(120.0, Note(0.5, 0.2, 440.0, 0.5, "lead", 0)), 0.0)
    sink._pump(base + 0.31)
    assert len(ons(port)) == 1


def test_a_later_add_lands_start_s_later():
    sink, port, clock = port_sink()
    base = clock.now
    s = score(120.0, Note(0.5, 0.2, 440.0, 0.5, "lead", 0))
    sink.add(s, 0.0)
    sink.add(s, 2.0)
    sink._pump(base + 0.6)
    assert len(ons(port)) == 1
    sink._pump(base + 2.4)
    assert len(ons(port)) == 1
    sink._pump(base + 2.6)
    assert len(ons(port)) == 2


def test_pump_says_when_the_next_message_is_due_and_none_when_the_heap_is_empty():
    sink, port, clock = port_sink()
    assert sink._pump(clock.now) is None
    sink.add(score(120.0, Note(0.5, 0.2, 440.0, 0.5, "lead", 0)), 0.0)
    nxt = sink._pump(clock.now)
    assert nxt is not None and abs(nxt - (clock.now + 0.5 - 0.0)) < 0.2  # the bend is due at 0.5 as well


def test_close_ends_notes_that_are_still_sounding_and_resets_each_channel_it_used():
    sink, port, clock = port_sink()
    sink.add(score(120.0, Note(0.1, 1.5, 440.0, 0.5, "lead", 0), Note(0.1, 1.5, 220.0, 0.5, "ttquantus", 0)), 0.0)
    sink._pump(clock.now + 0.2)
    before = len(port.sent)
    sink.close()
    tail = port.sent[before:]
    assert bytes([0x80, 69, 0]) in tail and bytes([0x81, 57, 0]) in tail
    assert bytes([0xB0, 123, 0]) in tail and bytes([0xB1, 123, 0]) in tail
    assert bytes([0xB2, 123, 0]) not in tail  # a channel nothing used is left alone
    assert tail.index(bytes([0x80, 69, 0])) < tail.index(bytes([0xB0, 123, 0]))
    assert port.closed == 1


def test_close_sends_no_off_for_a_note_that_already_ended_and_drops_what_is_not_due():
    sink, port, clock = port_sink()
    sink.add(score(120.0, Note(0.1, 0.1, 440.0, 0.5, "lead", 0), Note(1.8, 0.1, 880.0, 0.5, "lead", 0)), 0.0)
    sink._pump(clock.now + 0.5)
    sink.close()
    assert port.sent.count(bytes([0x80, 69, 0])) == 1
    assert bytes([0x90, 81, 64]) not in port.sent and not any(m[:2] == bytes([0x80, 81]) for m in port.sent)


def test_close_is_safe_to_call_twice_and_closes_the_port_once():
    sink, port, clock = port_sink()
    sink.close()
    sink.close()
    assert port.closed == 1


def test_a_send_that_raises_is_reported_once_and_everything_after_is_dropped():
    errors = []
    sink, port, clock = port_sink(port=FakePort(fail_on=2), errors=errors)
    sink.add(score(120.0, Note(0.1, 0.1, 440.0, 0.5, "lead", 0), Note(0.3, 0.1, 880.0, 0.5, "lead", 0)), 0.0)
    sink._pump(clock.now + 1.0)
    sink._pump(clock.now + 2.0)
    sink.add(score(120.0, Note(0.1, 0.1, 440.0, 0.5, "lead", 0)), 3.0)  # still returns
    sink._pump(clock.now + 9.0)
    sink.close()
    assert len(errors) == 1 and isinstance(errors[0], OSError)
    assert len(port.sent) == 1  # nothing went out after the failure, not even the close's resets
    assert port.closed == 1


def test_add_never_raises_even_for_a_score_that_cannot_be_routed():
    errors = []
    sink, port, clock = port_sink(errors=errors)
    sink.add(score(120.0, Note(0.1, 0.1, 440.0, 0.5, "nonesuch", 0)), 0.0)
    assert len(errors) == 1


def test_the_default_error_handler_does_nothing():
    sink = MidiPortSink("fake", open_output=lambda n: FakePort(fail_on=1), clock=FakeClock(), sleep=lambda s: None, threaded=False)
    sink.add(score(120.0, Note(0.0, 0.1, 440.0, 0.5, "lead", 0)), 0.0)
    sink._pump(1e9)


def test_the_summary_counts_what_was_added():
    sink, port, clock = port_sink()
    sink.add(score(120.0, Note(0.1, 0.1, 440.0, 0.5, "lead", 0), Trigger(0.2, 0.5, "gong", 1), Control(0.0, "power", 0.5, 0)), 0.0)
    assert sink.summary() == (1, 1, 1)


def test_the_scheduler_thread_sends_in_real_time_and_close_joins_it():
    port = FakePort()
    sink = MidiPortSink("fake", open_output=lambda n: port)
    sink.add(score(120.0, Note(0.05, 0.05, 440.0, 0.5, "lead", 0)), 0.0)
    deadline = time.monotonic() + 3.0
    while not ons(port) and time.monotonic() < deadline:
        time.sleep(0.01)
    assert len(ons(port)) == 1
    thread = sink._thread
    sink.close()
    assert not thread.is_alive() and port.closed == 1


def test_a_sink_with_no_adds_has_no_thread_to_leave_behind():
    port = FakePort()
    sink = MidiPortSink("fake", open_output=lambda n: port)
    sink.close()
    assert port.closed == 1


# --- the default opener ---------------------------------------------------------------------------------------------

def fake_mido(names=("Synth 1", "Synth 2")):
    mod = types.ModuleType("mido")
    mod.backend = None
    mod.opened = []
    mod.set_backend = lambda name: setattr(mod, "backend", name)
    mod.get_output_names = lambda: list(names)

    class Port:
        def __init__(self):
            self.sent, self.closed = [], 0

        def send(self, msg):
            self.sent.append(msg)

        def close(self):
            self.closed += 1

    def open_output(name, virtual=False):
        mod.opened.append((name, virtual))
        mod.port = Port()
        return mod.port

    mod.open_output = open_output
    mod.Message = types.SimpleNamespace(from_bytes=lambda data: ("msg", tuple(data)))
    return mod


def test_a_missing_library_is_one_line_that_names_the_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "mido", None)
    with pytest.raises(MidiPortError) as e:
        MidiPortSink("Synth 1")
    assert "pip install -e .[midi]" in str(e.value) and "\n" not in str(e.value)
    assert isinstance(e.value, RuntimeError)


def test_a_named_port_is_opened_with_the_rtmidi_backend_and_bytes_become_messages(monkeypatch):
    mod = fake_mido()
    monkeypatch.setitem(sys.modules, "mido", mod)
    sink = MidiPortSink("Synth 2", clock=FakeClock(), sleep=lambda s: None, threaded=False)
    assert mod.backend == "mido.backends.rtmidi" and mod.opened == [("Synth 2", False)]
    sink.add(score(120.0, Trigger(0.0, 0.5, "gong", 0)), 0.0)
    sink._pump(1e9)
    assert ("msg", (0x99, 49, 64)) in mod.port.sent
    sink.close()
    assert mod.port.closed == 1


def test_virtual_opens_a_virtual_output_called_tensixinet(monkeypatch):
    mod = fake_mido()
    monkeypatch.setitem(sys.modules, "mido", mod)
    MidiPortSink("virtual", threaded=False)
    assert mod.opened == [("tensixinet", True)]


def test_a_port_that_does_not_exist_names_itself_and_the_ports_there_are(monkeypatch):
    monkeypatch.setitem(sys.modules, "mido", fake_mido())
    with pytest.raises(MidiPortError) as e:
        MidiPortSink("Nope")
    msg = str(e.value)
    assert "Nope" in msg and "Synth 1" in msg and "Synth 2" in msg and "\n" not in msg


def test_no_ports_at_all_says_so(monkeypatch):
    monkeypatch.setitem(sys.modules, "mido", fake_mido(names=()))
    with pytest.raises(MidiPortError) as e:
        MidiPortSink("Nope")
    assert "none" in str(e.value).lower()


# --- the tee and closing any sink -----------------------------------------------------------------------------------

def test_a_tee_has_the_summary_and_path_of_its_first_sink_that_has_them(tmp_path):
    p = tmp_path / "x.mid"
    sink, port, clock = port_sink()
    tee = TeeScoreSink(sink, MidiFileSink(p))
    tee.add(score(120.0, Note(0.1, 0.1, 440.0, 0.5, "lead", 0)), 0.0)
    assert tee.summary() == (1, 0, 0) and tee.path == p


def test_a_tee_of_sinks_with_no_summary_reports_zeros_and_no_path():
    class Bare:
        def add(self, score, start_s): pass
        def close(self): pass
    tee = TeeScoreSink(Bare())
    assert tee.summary() == (0, 0, 0) and tee.path is None


def test_a_tee_closes_every_sink_even_when_one_fails_and_then_raises_the_first_failure():
    closed = []

    class Bad:
        def close(self):
            closed.append("bad")
            raise RuntimeError("boom")

    class Good:
        def close(self):
            closed.append("good")
    tee = TeeScoreSink(Bad(), Good())
    with pytest.raises(RuntimeError):
        tee.close()
    assert closed == ["bad", "good"]


def test_a_later_add_keeps_the_base_of_the_first_even_when_the_clock_has_moved_on():
    sink, port, clock = port_sink()
    base = clock.now
    s = score(120.0, Note(0.5, 0.2, 440.0, 0.5, "lead", 0))
    sink.add(s, 0.0)
    clock.now += 3.0
    sink.add(s, 2.0)  # due at base + 2.5, not at (base + 3) + 2.5
    sink._pump(base + 2.6)
    assert len(ons(port)) == 2
