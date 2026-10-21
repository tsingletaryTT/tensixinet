# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Which session the page is showing.

Every Play (and every change of a setting while playing, which plays again) calls `begin`, which ends the session that was playing and returns a
token. The generator that streams to the page checks `is_current(token)` each step and says nothing once a newer begin or a Stop has passed, so
an old stream cannot write over a new one. `attach` hands over the session once it has been built (building can take a while: the simulator
imports torch); it refuses if a Stop or a newer begin came first, and the caller then ends that session itself.

`config` is what the playing session was started with, so a change that leaves the settings as they were (tabbing out of a field) does not
restart it. A page that has stopped, or whose session ended on its own, waits for Play: a change does not start it.
"""
from __future__ import annotations

import threading
from typing import Any, Optional


class Player:
    def __init__(self):
        self._lock = threading.Lock()
        self._epoch = 0
        self._session: Any = None
        self._active = False
        self._config: Any = None

    @property
    def active(self) -> bool:
        return self._active

    @property
    def config(self) -> Any:
        return self._config

    def claim(self, config: Any, only_if_changed: bool = False):
        """Take the page for a new session, without ending the old one yet: returns (token, the session that was playing). The caller can show
        that it is starting and then stop the old one, which can take seconds. With `only_if_changed` (a change of a setting), the claim is
        refused as (None, None) unless something is playing with other settings; the check and the claim are one step, so a Stop that comes
        after it is not undone."""
        with self._lock:
            if only_if_changed and not (self._active and config != self._config):
                return None, None
            self._epoch += 1
            old, self._session = self._session, None
            self._active, self._config = True, config
            return self._epoch, old

    def begin(self, config: Any) -> int:
        token, old = self.claim(config)
        if old is not None:
            old.stop()
        return token

    def attach(self, token: int, session: Any) -> bool:
        with self._lock:
            if token != self._epoch:
                return False
            self._session = session
            return True

    def is_current(self, token: int) -> bool:
        return token == self._epoch

    def stop(self) -> None:
        with self._lock:
            self._epoch += 1
            old, self._session = self._session, None
            self._active, self._config = False, None
        if old is not None:
            old.stop()

    def finish(self, token: int) -> None:
        """The stream for `token` ended on its own (an error, or the session ran out): idle, unless a newer one has begun."""
        with self._lock:
            if token == self._epoch:
                self._active, self._config, self._session = False, None, None


class Busy(Exception):
    """Every place is taken: `playing` visitors are playing and the limit is `limit`."""

    def __init__(self, playing: int, limit: int):
        super().__init__(f"{playing} of {limit} places are taken")
        self.playing, self.limit = playing, limit


class Visitors:
    """One `Player` for each person on the page, so one person's Play, Stop and changes never touch another's session.

    Every session runs a render thread (and, in the simulator mode, chip workers), so only `limit` visitors may be playing at once; a visitor who
    would be the next is refused with `Busy`. Only visitors that are playing count, not everyone who has the page open. The count and the claim are
    one step under the registry's lock, so a crowd pressing Play together still ends up with `limit` sessions and no more. A visitor who is already
    playing can always play again or change a setting: that replaces their own session and takes no further place."""

    def __init__(self, limit: int):
        if limit < 1:
            raise ValueError("a limit of at least one visitor is needed")
        self.limit = limit
        self._lock = threading.Lock()
        self._players: dict = {}

    def __len__(self) -> int:
        return len(self._players)

    def player(self, key: str) -> Player:
        with self._lock:
            return self._players.setdefault(key, Player())

    def playing(self) -> int:
        return sum(p.active for p in list(self._players.values()))

    def claim(self, key: str, config: Any, only_if_changed: bool = False):
        """(that visitor's player, token, the session that was playing for them). Raises `Busy` when the visitor is not playing and the places are taken."""
        with self._lock:
            p = self._players.setdefault(key, Player())
            if not only_if_changed and not p.active and self.playing() >= self.limit:
                raise Busy(self.playing(), self.limit)
            token, old = p.claim(config, only_if_changed)
        return p, token, old

    def drop(self, key: str) -> None:
        """The visitor has left: end their session and forget them."""
        with self._lock:
            p = self._players.pop(key, None)
        if p is not None:
            p.stop()

    def reset(self) -> None:
        for key in list(self._players):
            self.drop(key)
