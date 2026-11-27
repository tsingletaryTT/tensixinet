# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Read tt-toplike's ``/telemetry`` WebSocket instead of reading the chips again.

`ToplikeSource` has the same surface `run_fast` expects from `FastSource`:
``start()``, ``stop()``, ``sample()``, ``fast`` and ``note`` (plus ``log``, which
`run_fast` sets). It also has ``view()`` and ``drain_views()`` for Phase B, where
a view change in tt-toplike will change the music.

The reader is deliberately small and uses only the standard library. A WebSocket
here is just a TCP socket with an HTTP upgrade handshake, followed by unmasked
server-to-client text frames; subscribing needs no client frames at all. We send
one masked pong when the server pings us so the broker does not close us as dead.

Rules:
* Only the newest frame matters for sound. `sample()` lays ``tensixinet.fast``
  over the base ``tt-smi`` frame by PCI bus address, never by position.
* A frame with a wrong schema, a missing key, or bad types is ignored; the last
  good frame stays. Omitting ``fast`` means the producer does not stream fast
  readings, so the base ``tt-smi`` frame is used, not zeroed chips.
* Before the first frame, and whenever the socket is down, `sample()` returns
  ``[]``: `run_fast` then replays its last reading with no movement. We never
  invent chips.
* Reconnects start at 0.25 s and double up to 5 s, reset by a good frame. A loss
  and a recovery are each logged once, not once per retry.
"""
from __future__ import annotations

import base64
import json
import os
import socket
import struct
import threading
from dataclasses import replace
from typing import Callable, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from .snapshot import Telemetry, parse_snapshot, snapshot_bus_ids

WS_MAX_FRAME_BYTES = 1 << 20  # a telemetry frame is a few KB; anything larger is a mistake
FIRST_RETRY_S = 0.25
MAX_RETRY_S = 5.0

_OP_TEXT = 0x1
_OP_BINARY = 0x2
_OP_CLOSE = 0x8
_OP_PING = 0x9
_OP_PONG = 0xA


def _tcp_connect(host: str, port: int, timeout: float = 5.0):
    """The production connector: `socket.create_connection` takes one `(host, port)`
    tuple, while the injectable `connect(host, port, timeout=...)` takes them apart."""
    return socket.create_connection((host, port), timeout=timeout)


class ToplikeSource:
    """The newest frame from tt-toplike's telemetry stream, in `FastSource`'s shape."""

    def __init__(
        self,
        url: str,
        *,
        connect: Optional[Callable] = None,
        wait: Optional[Callable[[float], bool]] = None,
        first_wait_s: float = 3.0,
        timeout_s: float = 5.0,
        log: Optional[Callable[..., None]] = None,
    ):
        """`url` is `ws[s]://host:port/telemetry`.

        `connect(host, port, **kw)` and `wait(delay)` are injectable so tests never
        open a socket or sleep. `wait` returns True when the reader should stop;
        in production it is the stop Event's `wait`, so `stop()` ends a backoff
        immediately. `log(what, why)` hears about a lost or recovered stream.
        """
        self.url = url
        self._connect_socket = connect or _tcp_connect
        self._wait = wait
        self.timeout_s = timeout_s
        self.log = log  # run_fast replaces this with the run log, as it does for FastSource
        self.first_wait_s = first_wait_s

        self.fast = False
        self.note = "Waiting for tt-toplike's telemetry stream."
        self._lock = threading.Lock()
        self._have_frame = threading.Event()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._sock = None
        self._buffer = bytearray()  # bytes left over from the handshake, then partial frames

        self._connected = False
        self._readings: List[Telemetry] = []
        self._bus_ids: List[str] = []
        self._fast: Dict[str, Tuple[float, float, float, float]] = {}
        self._view: Tuple[Optional[str], int] = (None, 0)
        self._last_seq = 0
        self._pending: List[Tuple[str, int]] = []

    # -- the FastSource surface ------------------------------------------------
    def start(self) -> None:
        """Start the daemon reader thread. Calling twice is harmless."""
        if self._thread is not None:
            return
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._reader_loop, name="tensixinet-toplike", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """End the reader and close the socket it is using."""
        self._stop.set()
        sock = self._sock
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass
        thread = self._thread
        if thread is not None:
            thread.join(timeout=0.5)
        self._thread = None

    def sample(self) -> List[Telemetry]:
        """The newest frame's readings with the four fast fields overlaid, or [] when there is no live frame."""
        # Before the first frame the reader may still be connecting: wait for it. After a frame has been seen,
        # a dropped connection returns [] at once so run_fast replays the last reading.
        if not self._have_frame.is_set() and not self._have_frame.wait(self.first_wait_s):
            return []
        if not self._connected:
            return []
        with self._lock:
            readings, bus_ids, fast = self._readings, self._bus_ids, self._fast
        out = []
        for tel, bus in zip(readings, bus_ids):
            vals = fast.get(bus)
            if vals is None:
                out.append(tel)
            else:
                out.append(
                    replace(
                        tel,
                        asic_temperature=vals[0],
                        power=vals[1],
                        voltage=vals[2],
                        current=vals[3],
                    )
                )
        return out

    def view(self) -> Tuple[Optional[str], int]:
        """The newest ``(name, seq)`` tt-toplike has reported, or ``(None, 0)`` before any view frame."""
        with self._lock:
            return self._view

    def drain_views(self) -> List[Tuple[str, int]]:
        """New view changes since the last call, in order, each `seq` once."""
        with self._lock:
            out = list(self._pending)
            self._pending.clear()
        return out

    # -- connection -----------------------------------------------------------
    def _connect(self):
        """Open the WebSocket: TCP, HTTP upgrade handshake, then return the socket with any pipelined bytes kept."""
        parsed = urlparse(self.url)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or (443 if parsed.scheme == "wss" else 80)
        path = parsed.path or "/"
        if parsed.query:
            path += "?" + parsed.query
        sock = self._connect_socket(host, port, timeout=self.timeout_s)
        key = base64.b64encode(os.urandom(16)).decode("ascii")
        request = (
            f"GET {path} HTTP/1.1\r\n"
            f"Host: {host}:{port}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n\r\n"
        )
        sock.sendall(request.encode("ascii"))

        head = bytearray()
        while b"\r\n\r\n" not in head:
            chunk = sock.recv(4096)
            if not chunk:
                raise ValueError("the WebSocket handshake closed before it finished")
            head.extend(chunk)
            if len(head) > 65536:
                raise ValueError("the WebSocket handshake was unreasonably large")
        response, _, leftover = bytes(head).partition(b"\r\n\r\n")
        status = response.split(b"\r\n", 1)[0]
        if b" 101 " not in status and not status.endswith(b" 101"):
            raise ValueError(f"the WebSocket upgrade was refused: {status!r}")
        # A server may put its first frame in the same TCP segment as the handshake.
        # Keep those bytes rather than losing the first reading.
        self._buffer = bytearray(leftover)
        return sock

    def _need(self, sock, n: int) -> bool:
        """Buffer at least `n` bytes, reading more from the socket as needed. False on EOF."""
        while len(self._buffer) < n:
            chunk = sock.recv(65536)
            if not chunk:
                return False
            self._buffer.extend(chunk)
        return True

    def _read_text_frame(self, sock) -> Optional[Tuple[int, str]]:
        """One server-to-client frame as ``(opcode, text)``, or None at EOF.

        Server frames are unmasked; we still handle a mask if one appears, since
        a few brokers send them. A length beyond the cap raises ValueError and
        ends this connection rather than allocating from a bogus header.
        """
        if not self._need(sock, 2):
            return None
        first, second = self._buffer[0], self._buffer[1]
        opcode = first & 0x0F
        masked = bool(second & 0x80)
        length = second & 0x7F
        offset = 2
        if length == 126:
            if not self._need(sock, offset + 2):
                return None
            length = struct.unpack_from(">H", self._buffer, offset)[0]
            offset += 2
        elif length == 127:
            if not self._need(sock, offset + 8):
                return None
            length = struct.unpack_from(">Q", self._buffer, offset)[0]
            offset += 8
        if length > WS_MAX_FRAME_BYTES:
            raise ValueError(f"a WebSocket frame of {length} bytes is too large")

        mask_len = 4 if masked else 0
        if not self._need(sock, offset + mask_len + length):
            return None
        payload = bytes(self._buffer[offset + mask_len : offset + mask_len + length])
        if masked:
            mask = bytes(self._buffer[offset : offset + 4])
            payload = bytes(byte ^ mask[i % 4] for i, byte in enumerate(payload))
        del self._buffer[: offset + mask_len + length]
        return opcode, payload.decode("utf-8", errors="replace")

    def _send_pong(self, sock) -> None:
        """Answer a ping with a masked, empty pong so the broker keeps the stream open."""
        try:
            sock.sendall(bytes([0x80 | _OP_PONG, 0x80]) + os.urandom(4))
        except OSError:
            pass

    # -- frame parsing --------------------------------------------------------
    @staticmethod
    def _is_number(value) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    def _accept(self, text: str) -> bool:
        """Take one frame into the latest state. False for a malformed frame (ignored)."""
        try:
            frame = json.loads(text)
        except (TypeError, ValueError):
            return False
        if not isinstance(frame, dict):
            return False

        ext = frame.get("tensixinet")
        fast: Dict[str, Tuple[float, float, float, float]] = {}
        view: Optional[Tuple[str, int]] = None
        if ext is not None:
            if not isinstance(ext, dict) or ext.get("schema") != 1:
                return False  # fail closed on a schema we do not understand
            raw_fast = ext.get("fast")
            if raw_fast is not None:
                if not isinstance(raw_fast, list):
                    return False
                for entry in raw_fast:
                    if not isinstance(entry, dict):
                        return False
                    bus = entry.get("bus_id")
                    vals = (
                        entry.get("temp_c"),
                        entry.get("power_w"),
                        entry.get("voltage_v"),
                        entry.get("current_a"),
                    )
                    if not isinstance(bus, str) or not all(self._is_number(v) for v in vals):
                        return False
                    fast[bus] = tuple(float(v) for v in vals)  # type: ignore[assignment]
            raw_view = ext.get("view")
            if raw_view is not None:
                if not isinstance(raw_view, dict):
                    return False
                name, seq = raw_view.get("name"), raw_view.get("seq")
                if not isinstance(name, str) or not isinstance(seq, int) or isinstance(seq, bool):
                    return False
                view = (name, seq)

        try:
            readings = parse_snapshot(frame)
            bus_ids = snapshot_bus_ids(frame)
        except (TypeError, ValueError, KeyError, AttributeError):
            return False
        if not readings:
            return False

        with self._lock:
            self._readings, self._bus_ids = readings, bus_ids
            self._fast = fast
            self._have_frame.set()
            if view is not None:
                name, seq = view
                self._view = view
                if seq > self._last_seq:
                    self._last_seq = seq
                    self._pending.append(view)
            self.fast = bool(fast)
            self.note = (
                f"Readings come from tt-toplike's stream ({len(fast)} chips), with the base tt-smi frame for the rest."
                if fast
                else "Readings come from the base tt-smi frame carried on tt-toplike's stream."
            )
        return True

    # -- reconnect loop -------------------------------------------------------
    def _say(self, what: str, why: str) -> None:
        if self.log is not None:
            self.log(what, why)

    def _set_connected(self, connected: bool) -> None:
        """Announce a loss or recovery once, on the transition, not on every retry."""
        if self._connected == connected:
            return
        self._connected = connected
        if connected:
            self._say(
                "connection to tt-toplike's stream is back",
                "the synth follows the stream's readings again",
            )
        else:
            self._say(
                "connection to tt-toplike's stream lost",
                "retries double the wait each time, up to a cap",
            )

    def _reader_loop(self) -> None:
        """Connect, read frames until the stream ends, then back off and reconnect."""
        backoff = FIRST_RETRY_S
        while not self._stop.is_set():
            try:
                sock = self._connect()
                self._sock = sock
                self._set_connected(True)
                while not self._stop.is_set():
                    frame = self._read_text_frame(sock)
                    if frame is None:
                        break  # the server closed the stream
                    opcode, text = frame
                    if opcode == _OP_CLOSE:
                        break
                    if opcode == _OP_PING:
                        self._send_pong(sock)
                        continue
                    if opcode == _OP_PONG:
                        continue
                    if opcode in (_OP_TEXT, _OP_BINARY) and self._accept(text):
                        backoff = FIRST_RETRY_S  # a good frame resets the backoff
            except (OSError, ValueError):
                pass  # reconnect below with a longer wait
            finally:
                self._set_connected(False)

            if self._stop.is_set():
                break
            wait = self._wait or self._stop.wait
            if wait(backoff):
                break
            backoff = min(backoff * 2, MAX_RETRY_S)
