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
