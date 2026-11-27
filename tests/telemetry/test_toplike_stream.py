# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""ToplikeSource: read tt-toplike's /telemetry WebSocket instead of the chips again.

Every socket here is fake. The tests drive the reader loop synchronously through an
injectable `wait`, so nothing depends on thread timing or on a real server.
"""
import json
import struct
from pathlib import Path

import pytest

from tensixinet.telemetry.snapshot import parse_snapshot
from tensixinet.telemetry.toplike_stream import ToplikeSource, WS_MAX_FRAME_BYTES

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "tt_smi_snapshot.json"
BUS = ["0000:01:00.0", "0000:02:00.0", "0000:03:00.0", "0000:04:00.0"]
URL = "ws://127.0.0.1:8770/telemetry"
HANDSHAKE = (
    b"HTTP/1.1 101 Switching Protocols\r\n"
    b"Upgrade: websocket\r\n"
    b"Connection: Upgrade\r\n"
    b"Sec-WebSocket-Accept: deadbeef\r\n\r\n"
)


@pytest.fixture(scope="module")
def snapshot():
    return json.loads(FIXTURE.read_text())


def fast_entries(values=None):
    """One fast reading per fixture chip, by bus address, in either order.

    A bare scalar means "use it for all four readings", which keeps tests that
    only care about one field short.
    """
    values = values or [(61.0, 101.0, 0.51, 11.0), (62.0, 102.0, 0.52, 12.0),
                        (63.0, 103.0, 0.53, 13.0), (64.0, 104.0, 0.54, 14.0)]
    rows = [(v, v, v, v) if not isinstance(v, (list, tuple)) else v for v in values]
    return [dict(bus_id=bus, temp_c=t, power_w=p, voltage_v=v, current_a=a)
            for bus, (t, p, v, a) in zip(BUS, rows)]


def base_frame(snapshot, *, fast="default", view=None):
    """A /telemetry frame as tt-toplike publishes it."""
    frame = dict(snapshot)  # in the real frame these are the bare tt-smi keys
    frame["tt_toplike"] = {"schema": 1, "focus": "defrag"}
    if fast != "absent":
        ext = {"schema": 1, "fast": fast_entries() if fast == "default" else fast}
        if view is not None:
            ext["view"] = {"name": view[0], "seq": view[1]}
        frame["tensixinet"] = ext
    return frame


def text_frame(text):
    """An unmasked server->client text frame (what the broker sends)."""
    payload = text.encode()
    n = len(payload)
    if n < 126:
        head = bytes([0x81, n])
    elif n < 65536:
        head = bytes([0x81, 126]) + struct.pack(">H", n)
    else:
        head = bytes([0x81, 127]) + struct.pack(">Q", n)
    return head + payload


class FakeSocket:
    def __init__(self, chunks=()):
        self.data = b"".join(chunks)
        self.sent = []
        self.closed = False

    def recv(self, n):
        if not self.data:
            return b""
        out, self.data = self.data[:n], self.data[n:]
        return out

    def sendall(self, data):
        self.sent.append(bytes(data))

    def close(self):
        self.closed = True


class FakeConnect:
    """Return queued sockets, recording the (host, port) asked for."""

    def __init__(self, sockets):
        self.sockets = list(sockets)
        self.calls = []

    def __call__(self, host, port, **kw):
        self.calls.append((host, port))
        if not self.sockets:
            raise OSError("no more sockets")
        return self.sockets.pop(0)


class FakeWait:
    """Record backoff delays and end the loop after `stop_after` waits."""

    def __init__(self, stop_after):
        self.delays = []
        self.stop_after = stop_after

    def __call__(self, delay):
        self.delays.append(delay)
        return len(self.delays) >= self.stop_after


class Recorder:
    def __init__(self):
        self.lines = []

    def __call__(self, what, why, **fields):
        self.lines.append((what, why))


def source(snapshot, *, wait=None, log=None, **kw):
    return ToplikeSource(URL, wait=wait, log=log, first_wait_s=0, **kw)


def ready(snapshot, *, fast="default", view=None, **kw):
    """A connected source with one good frame already accepted."""
    src = source(snapshot, **kw)
    src._accept(json.dumps(base_frame(snapshot, fast=fast, view=view)))
    src._connected = True
    return src


def test_connect_sends_the_websocket_upgrade_request(snapshot):
    sock = FakeSocket([HANDSHAKE])
    connect = FakeConnect([sock])
    src = source(snapshot, connect=connect)

    got = src._connect()

    assert got is sock
    assert connect.calls == [("127.0.0.1", 8770)]
    request = b"".join(sock.sent).decode()
    assert request.startswith("GET /telemetry HTTP/1.1\r\n")
    assert "Host: 127.0.0.1:8770\r\n" in request
    assert "Upgrade: websocket\r\n" in request
    assert "Connection: Upgrade\r\n" in request
    assert "Sec-WebSocket-Version: 13\r\n" in request
    assert "Sec-WebSocket-Key: " in request
    assert sock.data == b""  # no handshake bytes left over into frame reading


def test_a_frame_becomes_telemetry_with_fast_fields_overlaid_by_bus_id(snapshot):
    src = ready(snapshot)
    out = src.sample()
    slow = parse_snapshot(snapshot)

    assert src.fast is True
    assert "tt-toplike" in src.note
    assert [t.asic_temperature for t in out] == [61.0, 62.0, 63.0, 64.0]
    assert [t.power for t in out] == [101.0, 102.0, 103.0, 104.0]
    assert [t.current for t in out] == [11.0, 12.0, 13.0, 14.0]
    assert [t.voltage for t in out] == pytest.approx([0.51, 0.52, 0.53, 0.54])
    for got, was in zip(out, slow):
        assert got.aiclk == was.aiclk
        assert got.gddr_temps == was.gddr_temps
        assert got.corr_errs == was.corr_errs


def test_fast_entries_are_matched_by_bus_address_not_position(snapshot):
    src = source(snapshot)
    src._accept(json.dumps(base_frame(snapshot, fast=list(reversed(fast_entries())))))
    src._connected = True

    assert [t.power for t in src.sample()] == [101.0, 102.0, 103.0, 104.0]


def test_two_frames_in_a_row_sample_only_the_newest(snapshot):
    src = source(snapshot)
    src._accept(json.dumps(base_frame(snapshot, fast=fast_entries([1, 1, 1, 1]))))
    newest = fast_entries([9, 9, 9, 9])
    src._accept(json.dumps(base_frame(snapshot, fast=newest)))
    src._connected = True

    assert [t.power for t in src.sample()] == [9.0, 9.0, 9.0, 9.0]


def test_a_bad_frame_is_ignored_and_the_last_good_one_kept(snapshot):
    src = source(snapshot)
    src._accept(json.dumps(base_frame(snapshot)))
    src._connected = True
    src._accept("this is not json")
    wrong_schema = base_frame(snapshot)
    wrong_schema["tensixinet"]["schema"] = 99
    src._accept(json.dumps(wrong_schema))
    src._accept(json.dumps({"device_info": []}))  # no chips

    assert [t.power for t in src.sample()] == [101.0, 102.0, 103.0, 104.0]


def test_no_tensixinet_key_falls_back_to_the_base_tt_smi_frame(snapshot):
    src = ready(snapshot, fast="absent")

    assert src.fast is False
    assert src.note != ""
    assert "tt-smi" in src.note
    assert src.sample() == parse_snapshot(snapshot)


def test_no_frame_yet_returns_an_empty_sample(snapshot):
    assert source(snapshot).sample() == []


def test_a_drop_returns_empty_so_run_fast_replays_the_last_reading(snapshot):
    src = ready(snapshot)
    src._connected = False

    assert src.sample() == []


def test_backoff_doubles_and_resets_after_a_good_frame(snapshot):
    first = FakeSocket()  # connect works, but no handshake -> a failed attempt
    good = FakeSocket([HANDSHAKE, text_frame(json.dumps(base_frame(snapshot)))])
    third = FakeSocket()  # fails again
    wait = FakeWait(stop_after=4)
    src = source(snapshot, connect=FakeConnect([first, good, third]), wait=wait)

    src._reader_loop()

    assert wait.delays == [0.25, 0.25, 0.5, 1.0]


def test_loss_and_recovery_are_logged_once_each(snapshot):
    bad = FakeSocket()
    good = FakeSocket([HANDSHAKE, text_frame(json.dumps(base_frame(snapshot)))])
    log = Recorder()
    wait = FakeWait(stop_after=3)
    src = source(snapshot, connect=FakeConnect([bad, good]), wait=wait, log=log)

    src._reader_loop()

    losses = [what for what, _ in log.lines if "lost" in what.lower()]
    recoveries = [what for what, _ in log.lines if "back" in what.lower()]
    assert len(losses) == 1
    assert len(recoveries) == 1


def test_drain_views_yields_each_new_seq_once_in_order(snapshot):
    src = source(snapshot)
    for name, seq in [("defrag", 1), ("defrag", 1), ("scope", 2), ("gong", 3)]:
        src._accept(json.dumps(base_frame(snapshot, view=(name, seq))))

    assert src.drain_views() == [("defrag", 1), ("scope", 2), ("gong", 3)]
    assert src.drain_views() == []


def test_view_returns_the_newest_name_and_seq(snapshot):
    src = source(snapshot)
    src._accept(json.dumps(base_frame(snapshot, view=("defrag", 7))))

    assert src.view() == ("defrag", 7)


def test_a_reconnect_does_not_replay_an_old_view_seq(snapshot):
    src = source(snapshot)
    src._accept(json.dumps(base_frame(snapshot, view=("scope", 5))))
    assert src.drain_views() == [("scope", 5)]

    src._connected = False  # dropped and came back
    src._connected = True
    src._accept(json.dumps(base_frame(snapshot, view=("scope", 5))))  # replayed frame
    src._accept(json.dumps(base_frame(snapshot, view=("defrag", 6))))

    assert src.drain_views() == [("defrag", 6)]


def test_a_127_bit_length_frame_is_read(snapshot):
    frame = base_frame(snapshot)
    frame["padding_for_the_length_test"] = "x" * 70000  # > 65535 -> 8-byte length
    sock = FakeSocket([HANDSHAKE, text_frame(json.dumps(frame))])
    src = source(snapshot, connect=FakeConnect([sock]), wait=FakeWait(stop_after=2))

    src._reader_loop()
    src._connected = True  # this test is about frame framing, not the reconnect state

    assert [t.power for t in src.sample()] == [101.0, 102.0, 103.0, 104.0]


def test_an_absurd_frame_length_is_rejected(snapshot):
    huge = bytes([0x81, 127]) + struct.pack(">Q", WS_MAX_FRAME_BYTES + 1)
    src = source(snapshot)

    with pytest.raises(ValueError):
        src._read_text_frame(FakeSocket([huge]))


def test_a_ping_gets_a_masked_pong_reply(snapshot):
    ping = bytes([0x89, 0x00])
    sock = FakeSocket([HANDSHAKE, ping])
    src = source(snapshot, connect=FakeConnect([sock]), wait=FakeWait(stop_after=2))

    src._reader_loop()

    replies = b"".join(sock.sent)
    assert b"\x8a\x80" in replies  # pong, FIN, masked
    assert len(replies) > len(b"".join(sock.sent[:0]))  # a reply was actually sent


def test_stop_closes_the_socket_and_ends_the_loop(snapshot):
    sock = FakeSocket([HANDSHAKE, text_frame(json.dumps(base_frame(snapshot)))])
    connect = FakeConnect([sock])
    src = source(snapshot, connect=connect, wait=FakeWait(stop_after=2))
    src._reader_loop()
    assert not sock.closed  # still open while blocked in the wait

    src.stop()

    assert sock.closed
    assert src._stop.is_set()


def test_the_default_connector_opens_a_real_socket():
    """The injected fakes hid a wrong call to `socket.create_connection` (it takes one
    address tuple, not host and port). Use the production default against a real
    loopback listener, not a fake."""
    import socket

    from tensixinet.telemetry import toplike_stream

    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]
        src = ToplikeSource(f"ws://127.0.0.1:{port}/telemetry")
        sock = src._connect_socket("127.0.0.1", port, timeout=1.0)
        try:
            assert sock.getpeername()[1] == port
        finally:
            sock.close()


def test_sample_waits_for_the_first_frame_when_the_reader_is_still_connecting(snapshot):
    """`run_fast` calls sample() right after start(), before the reader thread has even
    connected. That must wait up to first_wait_s for the first frame; returning [] there
    made a healthy stream look like 'no chips found' and ended the run."""
    import threading
    import time

    src = ToplikeSource(URL, first_wait_s=2.0)
    assert not src._connected  # the reader has not connected yet

    def late_first_frame():
        time.sleep(0.1)
        src._connected = True
        src._accept(json.dumps(base_frame(snapshot)))

    threading.Thread(target=late_first_frame, daemon=True).start()
    assert len(src.sample()) == 4


def test_sample_is_empty_after_the_stream_drops(snapshot):
    """Once a frame has been seen, a dropped connection still returns [] at once (run_fast replays)."""
    src = ToplikeSource(URL, first_wait_s=2.0)
    src._accept(json.dumps(base_frame(snapshot)))
    src._connected = False
    assert src.sample() == []
