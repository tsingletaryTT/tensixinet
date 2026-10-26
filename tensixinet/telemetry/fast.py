# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Fast telemetry: hwmon's four live readings laid over the slower `tt-smi -s` reading.

`tt-smi -s` supplies everything (clocks, GDDR temperatures, masks, error counts) but takes
a quarter of a second or more, and sometimes hangs. hwmon (see `hwmon.py`) supplies only
temperature, power, voltage and current, in microseconds. `FastSource` keeps the latest
`tt-smi` reading, refreshed on a background thread, and `sample()` returns it with those
four fields replaced by a fresh hwmon read.

Rules:
* `sample()` never runs `tt-smi` and never waits on one. Only before the first slow reading
  exists does it wait, up to `first_wait_s`, and then it returns [].
* Chips are paired by PCI bus address. If hwmon is missing, finds a different number of
  chips, or any address differs from tt-smi's, no fast reading is used at all: pairing by
  position could give one chip another chip's power.
* A chip whose hwmon read fails keeps its last fast reading, or the slow value if it never
  had one. The first failure of a stretch is logged with the chip's index and bus address.
* After `REFIND_AFTER` failed reads in a row (a chip reset can move its hwmon files), the
  chips are searched for again, and the chip is read once more at whatever path the search
  gave it. While it still fails, its four fields come from `tt-smi`, `fast` is False and the
  note says which chip. The search is repeated after every `REFIND_AFTER` more failures. A
  chip that reads again is logged once and returns to hwmon.
"""
from __future__ import annotations

import threading
from dataclasses import replace
from typing import Callable, Dict, List, Optional, Set, Tuple

from . import hwmon
from .hwmon import FastReading
from .snapshot import Telemetry, fetch_snapshot, parse_snapshot, snapshot_bus_ids

REFIND_AFTER = 10  # failed hwmon reads in a row of one chip before the chips are searched for again and that chip falls back to tt-smi


class FastSource:
    """The latest slow reading with hwmon's fast readings laid over it."""

    def __init__(
        self,
        find: Callable = hwmon.find_chips,
        read: Callable = hwmon.read_chip,
        fetch: Callable = fetch_snapshot,
        slow_every_s: float = 1.5,
        first_wait_s: float = 3.0,
        log: Optional[Callable[..., None]] = None,
    ):
        """`log(what, why, chip=index)`, when given, hears about a chip whose hwmon reads fail and come back (None: silent). `run_fast` sets it
        to the run log."""
        self._find = find
        self.log = log
        self._read = read
        self._fetch = fetch
        self.slow_every_s = slow_every_s
        self.first_wait_s = first_wait_s
        self.fast = False
        self.note = "Readings come from tt-smi (the fast source has not started)."
        self._lock = threading.Lock()
        self._slow: Optional[Tuple[List[Telemetry], List[str]]] = None  # readings and their bus addresses
        self._have_slow = threading.Event()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._chips: list = []
        self._last_fast: Dict[str, FastReading] = {}
        self._fails: Dict[str, int] = {}  # failed hwmon reads in a row, by bus address
        self._on_slow: Set[str] = set()  # chips whose four fast fields come from tt-smi until a read works again

    def start(self) -> None:
        """Find the hwmon chips and start the thread that refreshes the slow reading."""
        if self._thread is not None:
            return
        try:
            self._chips = list(self._find())
        except Exception:  # a finder that raises leaves no fast source
            self._chips = []
        # Each run gets its own stop Event and starts with no slow reading. A thread abandoned
        # inside a hung fetch keeps its own (set) Event, so it ends when the fetch returns and
        # never writes a reading into the new run.
        self._stop = threading.Event()
        with self._lock:
            self._slow = None
        self._have_slow.clear()
        self._thread = threading.Thread(
            target=self._refresh_loop, args=(self._stop,), name="tensixinet-slow", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """End the thread. A fetch that is hanging is abandoned (the thread is a daemon)."""
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=0.5)
        self._thread = None

    def _refresh_loop(self, stop: threading.Event) -> None:
        while not stop.is_set():
            try:
                snapshot = self._fetch()
                readings = parse_snapshot(snapshot) if snapshot else []
            except Exception:
                readings = []
                snapshot = {}
            if readings and not stop.is_set():
                with self._lock:
                    self._slow = (readings, snapshot_bus_ids(snapshot))
                self._have_slow.set()
            stop.wait(self.slow_every_s)

    def _decide(self, bus_ids: List[str]) -> bool:
        """Set `fast` and `note` from the chips found and the addresses tt-smi gave."""
        hw = [c.bus_id for c in self._chips]
        if not hw:
            self.fast, self.note = False, "No hwmon chips found, so readings come at tt-smi's pace."
        elif len(hw) != len(bus_ids):
            self.fast = False
            self.note = (
                f"hwmon found {len(hw)} chips but tt-smi found {len(bus_ids)}, "
                "so readings come at tt-smi's pace."
            )
        elif sorted(hw) != sorted(bus_ids):
            self.fast = False
            self.note = "hwmon and tt-smi list different bus addresses, so readings come at tt-smi's pace."
        else:
            self.fast, self.note = True, f"Readings come from hwmon ({len(hw)} chips), with tt-smi for the rest."
        return self.fast

    def sample(self) -> List[Telemetry]:
        """The latest slow reading with the fast fields replaced, or [] if none has arrived."""
        if not self._have_slow.wait(self.first_wait_s):
            return []
        with self._lock:
            readings, bus_ids = self._slow
        if not self._decide(bus_ids):
            return list(readings)
        out = []
        for index, (tel, bus) in enumerate(zip(readings, bus_ids)):
            got = self._read_one(bus)
            if got is None:
                got = self._failed(index, bus)
            elif self._fails.get(bus):
                self._recovered(index, bus)
            if got is not None:
                self._last_fast[bus] = got
            got = None if bus in self._on_slow else self._last_fast.get(bus)  # a failed read keeps the last fast reading, for a while
            if got is None:
                out.append(tel)
            else:
                out.append(
                    replace(
                        tel,
                        asic_temperature=got.asic_temperature,
                        power=got.power,
                        voltage=got.voltage,
                        current=got.current,
                    )
                )
        if self._on_slow:
            on_slow = [(i, b) for i, b in enumerate(bus_ids) if b in self._on_slow]
            names = ", ".join(f"chip {i} ({b})" for i, b in on_slow)
            self.fast = False
            self.note = (f"Readings come from hwmon ({len(bus_ids)} chips), with tt-smi for the rest; {names} could not be read from hwmon, "
                         f"so {'its' if len(on_slow) == 1 else 'their'} temperature, power, voltage and current come from tt-smi.")
        return out

    def _read_one(self, bus: str) -> Optional[FastReading]:
        """One hwmon read of the chip at `bus`; None when it fails or the chip is not among those found."""
        chip = next((c for c in self._chips if c.bus_id == bus), None)
        if chip is None:
            return None
        try:
            return self._read(chip)
        except (OSError, ValueError):
            return None

    def _say(self, what: str, why: str, index: int) -> None:
        if self.log is not None:
            self.log(what, why, chip=index)

    def _failed(self, index: int, bus: str) -> Optional[FastReading]:
        """Count a failed read. The first of a stretch is logged; every `REFIND_AFTER` in a row the chips are searched for again and the chip is
        read once more at its new path. Returns that read when it works, else None."""
        n = self._fails.get(bus, 0) + 1
        self._fails[bus] = n
        if n == 1:
            self._say(f"chip {index} ({bus}): its hwmon read failed",
                      f"its last hwmon reading is played until a read works; after {REFIND_AFTER} failures in a row the chips are searched for "
                      "again and this chip's temperature, power, voltage and current come from tt-smi", index)
        if n % REFIND_AFTER:
            return None
        try:
            found = {c.bus_id: c for c in self._find()}
        except Exception:  # a search that fails leaves the chips as they were
            found = {}
        if bus in found:  # a reset can move a chip's hwmon files; only this chip's entry is replaced, so the pairing with tt-smi stands
            self._chips = [found[bus] if c.bus_id == bus else c for c in self._chips]
        got = self._read_one(bus)
        if got is not None:
            self._recovered(index, bus)
            return got
        if bus not in self._on_slow:
            self._on_slow.add(bus)
            self._say(f"chip {index} ({bus}) failed {n} hwmon reads in a row: its temperature, power, voltage and current now come from tt-smi",
                      "a searched-for chip that still cannot be read would otherwise play one frozen reading for the rest of the run; "
                      "tt-smi's values arrive about every 1.5 s", index)
        return None

    def _recovered(self, index: int, bus: str) -> None:
        """A read worked after a stretch of failures: log it once and read the chip from hwmon again."""
        self._fails[bus] = 0
        self._on_slow.discard(bus)
        self._say(f"chip {index} ({bus}) reads from hwmon again", "its temperature, power, voltage and current follow the chip at every block once more",
                  index)
