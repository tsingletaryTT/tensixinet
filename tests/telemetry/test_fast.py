# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""FastSource: hwmon readings laid over a slow tt-smi reading. Fakes only; no hardware, no real /sys."""
import json
import threading
import time
from pathlib import Path

import pytest

from tensixinet.telemetry.fast import FastSource
from tensixinet.telemetry.hwmon import FastReading, HwmonChip
from tensixinet.telemetry.snapshot import parse_snapshot

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "tt_smi_snapshot.json"
BUS = ["0000:01:00.0", "0000:02:00.0", "0000:03:00.0", "0000:04:00.0"]


@pytest.fixture(scope="module")
def snapshot():
    return json.loads(FIXTURE.read_text())


def chips(bus_ids):
    return [HwmonChip(path=Path(f"/fake/{b}"), bus_id=b) for b in bus_ids]


def fast_value(chip):
    """A distinctive reading per chip, so a mix-up shows."""
    n = int(chip.bus_id.split(":")[1])
    return FastReading(asic_temperature=60.0 + n, power=100.0 + n, voltage=0.5 + n / 100, current=10.0 + n)


@pytest.fixture
def started():
    sources = []

    def make(snapshot, find=None, read=fast_value, fetch=None, **kw):
        src = FastSource(
            find=find or (lambda: chips(BUS)),
            read=read,
            fetch=fetch or (lambda: snapshot),
            slow_every_s=0.01,
            **kw,
        )
        sources.append(src)
        src.start()
        return src

    yield make
    for s in sources:
        s.stop()


def test_fast_fields_replace_the_slow_ones_and_slow_fields_stay(snapshot, started):
    src = started(snapshot)
    out = src.sample()
    slow = parse_snapshot(snapshot)
    assert src.fast is True
    assert [t.asic_temperature for t in out] == [61.0, 62.0, 63.0, 64.0]
    assert [t.power for t in out] == [101.0, 102.0, 103.0, 104.0]
    assert [t.current for t in out] == [11.0, 12.0, 13.0, 14.0]
    assert [t.voltage for t in out] == pytest.approx([0.51, 0.52, 0.53, 0.54])
    for got, was in zip(out, slow):
        assert got.aiclk == was.aiclk
        assert got.gddr_temps == was.gddr_temps
        assert got.ddr_status == was.ddr_status


def test_chips_are_matched_by_bus_address_when_the_orders_differ(snapshot, started):
    # hwmon lists the chips in reverse; chip 0 of tt-smi still gets the 01:00.0 readings.
    src = started(snapshot, find=lambda: list(reversed(chips(BUS))))
    out = src.sample()
    assert src.fast is True
    assert [t.power for t in out] == [101.0, 102.0, 103.0, 104.0]


def test_addresses_that_disagree_turn_fast_off_even_when_the_counts_match(snapshot, started):
    other = ["0000:01:00.0", "0000:02:00.0", "0000:03:00.0", "0000:09:00.0"]
    src = started(snapshot, find=lambda: chips(other))
    out = src.sample()
    assert src.fast is False
    assert "tt-smi" in src.note
    assert out == parse_snapshot(snapshot)


def test_a_different_number_of_chips_turns_fast_off_and_says_so(snapshot, started):
    src = started(snapshot, find=lambda: chips(BUS[:3]))
    out = src.sample()
    assert src.fast is False
    assert "tt-smi" in src.note
    assert "3" in src.note and "4" in src.note
    assert out == parse_snapshot(snapshot)


def test_no_hwmon_turns_fast_off(snapshot, started):
    src = started(snapshot, find=lambda: [])
    out = src.sample()
    assert src.fast is False
    assert "tt-smi" in src.note
    assert out == parse_snapshot(snapshot)


def test_the_note_names_the_source_in_use(snapshot, started):
    src = started(snapshot)
    src.sample()
    assert "hwmon" in src.note


def test_a_fetch_that_never_returns_does_not_stop_sample(snapshot, started):
    release = threading.Event()
    calls = []

    def fetch():
        calls.append(1)
        if len(calls) == 1:
            return snapshot
        release.wait(30)
        return snapshot

    src = started(snapshot, fetch=fetch)
    assert len(src.sample()) == 4
    time.sleep(0.1)  # the second fetch is now hanging
    try:
        for _ in range(5):
            t0 = time.monotonic()
            out = src.sample()
            assert time.monotonic() - t0 < 0.05
            assert len(out) == 4
    finally:
        release.set()


def test_sample_never_calls_fetch_on_the_calling_thread(snapshot, started):
    callers = set()

    def fetch():
        callers.add(threading.current_thread())
        return snapshot

    src = started(snapshot, fetch=fetch)
    src.sample()
    src.sample()
    assert threading.current_thread() not in callers


def test_a_chip_whose_read_fails_keeps_its_last_fast_reading(snapshot, started):
    broken = set()

    def read(chip):
        if chip.bus_id in broken:
            return None
        return fast_value(chip)

    src = started(snapshot, read=read)
    first = src.sample()
    broken.add("0000:02:00.0")
    out = src.sample()
    assert out[1].power == first[1].power == 102.0
    assert out[0].power == 101.0


def test_a_chip_that_never_read_keeps_the_slow_value(snapshot, started):
    src = started(snapshot, read=lambda chip: None if chip.bus_id == "0000:03:00.0" else fast_value(chip))
    out = src.sample()
    slow = parse_snapshot(snapshot)
    assert out[2].power == slow[2].power
    assert out[2].asic_temperature == slow[2].asic_temperature
    assert out[1].power == 102.0


def test_fast_values_are_read_on_every_sample(snapshot, started):
    n = [0]

    def read(chip):
        n[0] += 1
        return FastReading(float(n[0]), 1.0, 1.0, 1.0)

    src = started(snapshot, read=read)
    a = src.sample()[0].asic_temperature
    b = src.sample()[0].asic_temperature
    assert b > a


def test_a_failed_slow_reading_keeps_the_last_good_one(snapshot, started):
    calls = [0]

    def fetch():
        calls[0] += 1
        return snapshot if calls[0] == 1 else {}

    src = started(snapshot, fetch=fetch)
    src.sample()
    time.sleep(0.1)
    assert len(src.sample()) == 4


def test_no_slow_reading_ever_gives_an_empty_list_after_the_wait(snapshot):
    src = FastSource(find=lambda: [], read=fast_value, fetch=lambda: {}, slow_every_s=0.01, first_wait_s=0.05)
    src.start()
    try:
        t0 = time.monotonic()
        assert src.sample() == []
        assert 0.04 <= time.monotonic() - t0 < 1.0
    finally:
        src.stop()


def test_the_first_wait_is_three_seconds_by_default():
    assert FastSource().first_wait_s == 3.0


def test_stop_ends_the_thread_promptly_and_it_is_a_daemon(snapshot):
    src = FastSource(find=lambda: chips(BUS), read=fast_value, fetch=lambda: snapshot, slow_every_s=30)
    src.start()
    src.sample()
    thread = src._thread
    assert thread.daemon is True
    t0 = time.monotonic()
    src.stop()
    assert time.monotonic() - t0 < 1.0
    assert not thread.is_alive()


def test_stop_without_start_is_harmless():
    FastSource().stop()


def test_restart_while_a_fetch_hangs_leaves_one_refresh_thread(snapshot):
    release = threading.Event()
    lock = threading.Lock()
    threads = []
    first = [True]

    def fetch():
        with lock:
            hang = first[0]
            first[0] = False
        if hang:
            release.wait(30)  # the abandoned thread's fetch
        else:
            with lock:
                threads.append(threading.current_thread())
        return snapshot

    src = FastSource(find=lambda: chips(BUS), read=fast_value, fetch=fetch, slow_every_s=0.01)
    src.start()
    time.sleep(0.05)
    src.stop()  # the first thread is still inside fetch
    src.start()
    try:
        time.sleep(0.1)
        release.set()  # the old fetch returns; a revived thread would now loop and fetch again
        time.sleep(0.2)
        with lock:
            names = {t.ident for t in threads}
        assert len(names) == 1
    finally:
        release.set()
        src.stop()


def test_after_a_restart_sample_waits_for_a_new_slow_reading(snapshot):
    ok = threading.Event()
    ok.set()

    def fetch():
        ok.wait(30)
        return snapshot

    src = FastSource(find=lambda: chips(BUS), read=fast_value, fetch=fetch, slow_every_s=0.01, first_wait_s=0.05)
    src.start()
    assert len(src.sample()) == 4
    src.stop()
    ok.clear()
    src.start()
    try:
        assert src.sample() == []
    finally:
        ok.set()
        src.stop()


def test_a_reader_that_raises_counts_as_a_failed_read(snapshot, started):
    broken = set()

    def read(chip):
        if chip.bus_id in broken:
            raise OSError("gone")
        return fast_value(chip)

    src = started(snapshot, read=read)
    src.sample()
    broken.add("0000:02:00.0")
    out = src.sample()
    assert out[1].power == 102.0
    broken.add("0000:03:00.0")
    src2 = started(snapshot, read=read)
    assert src2.sample()[2].power == parse_snapshot(snapshot)[2].power


# --- a chip whose hwmon reads keep failing, for example after a reset (final review, I3) ------------------------------------------------

def _flaky(broken):
    """A reader that fails for every chip whose bus address is in `broken`, and reads `fast_value` for the rest."""
    def read(chip):
        return None if chip.bus_id in broken else fast_value(chip)
    return read


def test_the_first_failed_read_of_a_stretch_is_logged_once_with_the_chip_and_its_address(snapshot, started):
    lines = []
    broken = set()
    src = started(snapshot, read=_flaky(broken), log=lambda what, why, **fields: lines.append((what, why, fields)))
    src.sample()
    broken.add("0000:02:00.0")
    for _ in range(3):
        src.sample()
    (failed,) = lines
    assert "chip 1" in failed[0] and "0000:02:00.0" in failed[0] and failed[2] == {"chip": 1}
    broken.clear()
    src.sample()
    assert len(lines) == 2 and "chip 1" in lines[1][0] and "hwmon again" in lines[1][0]  # the recovery, once
    src.sample()
    broken.add("0000:02:00.0")
    src.sample()
    assert len(lines) == 3 and "0000:02:00.0" in lines[2][0]  # a new stretch of failures is logged again


def test_ten_failed_reads_in_a_row_search_again_and_use_tt_smis_values_and_say_so(snapshot, started):
    lines, finds = [], []
    broken = set()

    def find():
        finds.append(1)
        return chips(BUS)

    src = started(snapshot, find=find, read=_flaky(broken), log=lambda what, why, **fields: lines.append(what))
    slow = parse_snapshot(snapshot)
    assert src.sample()[2].power == 103.0
    broken.add("0000:03:00.0")
    for n in range(1, 10):
        out = src.sample()
        assert out[2].power == 103.0 and src.fast  # up to nine failures in a row: the last hwmon reading
    assert len(finds) == 1
    out = src.sample()  # the tenth in a row
    assert len(finds) == 2  # searched again
    assert out[2].power == slow[2].power and out[2].asic_temperature == slow[2].asic_temperature  # tt-smi's values for that chip
    assert out[1].power == 102.0  # the others still read hwmon
    assert not src.fast and "chip 2" in src.note and "0000:03:00.0" in src.note and "tt-smi" in src.note
    assert any("chip 2" in what and "tt-smi" in what for what in lines)
    for _ in range(9):
        src.sample()
    assert len(finds) == 2  # the next search is after ten more
    src.sample()
    assert len(finds) == 3


def test_a_chip_found_again_at_a_new_path_returns_to_hwmon_and_logs_it_once(snapshot, started):
    lines = []
    moved = HwmonChip(path=Path("/fake/after-reset"), bus_id="0000:03:00.0")
    finds = []

    def find():
        finds.append(1)
        return chips(BUS) if len(finds) == 1 else [c if c.bus_id != moved.bus_id else moved for c in chips(BUS)]

    def read(chip):
        if chip.bus_id == moved.bus_id and chip.path != moved.path:
            return None  # the old hwmon path is gone after the reset
        if chip.path == moved.path:
            return FastReading(asic_temperature=70.0, power=150.0, voltage=0.8, current=20.0)
        return fast_value(chip)

    src = started(snapshot, find=find, read=read, log=lambda what, why, **fields: lines.append(what))
    for _ in range(10):
        out = src.sample()
    assert len(finds) == 2 and out[2].power == 150.0  # the search found the chip's new path and it read at once
    assert src.fast and "hwmon (4 chips)" in src.note
    assert len([w for w in lines if "hwmon again" in w]) == 1
    src.sample()
    assert len([w for w in lines if "hwmon again" in w]) == 1


def test_without_a_log_the_source_is_silent_and_the_old_arguments_still_work(snapshot):
    src = FastSource(lambda: chips(BUS), _flaky({"0000:02:00.0"}), lambda: snapshot, 0.01, 3.0)  # positional, as before
    src.start()
    try:
        for _ in range(12):
            assert len(src.sample()) == 4
    finally:
        src.stop()


def test_a_chip_on_tt_smis_values_returns_to_hwmon_when_an_ordinary_read_works(snapshot, started):
    broken = set()
    src = started(snapshot, read=_flaky(broken))
    src.sample()
    broken.add("0000:03:00.0")
    for _ in range(10):
        src.sample()
    assert not src.fast  # on tt-smi's values now
    broken.clear()
    out = src.sample()
    assert out[2].power == 103.0 and src.fast and "could not be read" not in src.note


def test_a_search_that_misses_the_failing_chip_leaves_the_other_chips_on_hwmon(snapshot, started):
    finds = []

    def find():  # after the reset the failing chip has no hwmon files at all yet
        finds.append(1)
        return chips(BUS) if len(finds) == 1 else [c for c in chips(BUS) if c.bus_id != "0000:03:00.0"]

    broken = set()
    src = started(snapshot, find=find, read=_flaky(broken))
    src.sample()
    broken.add("0000:03:00.0")
    for _ in range(11):  # the tenth searches; the eleventh is read with what the search left
        out = src.sample()
    assert len(finds) == 2
    assert [t.power for t in (out[0], out[1], out[3])] == [101.0, 102.0, 104.0]  # still hwmon: one chip missing from the search changes no pairing
    assert out[2].power == parse_snapshot(snapshot)[2].power and "chip 2" in src.note
