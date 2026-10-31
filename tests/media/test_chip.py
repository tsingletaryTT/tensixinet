# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The chip worker and its client: a subprocess that holds a leased chip and runs the kernels on request.

These tests start the worker in its host mode, which runs each kernel's formula in numpy and needs no chip, no lease and no
TT-Lang. The kernels themselves are checked against the devices in the simulator and on the chips (`kernels/README.md`)."""
import dataclasses
import sys

import numpy as np
import pytest

from tensixinet.kernels import packing as pk
from tensixinet.media.chip import CHIP_OPS, ChipError, ChipWorker, host_command
from tensixinet.music.devices import gong_parameters, seconds_since, strike_times
from tensixinet.music.patch import DEFAULT_PATCH
from tensixinet.telemetry import FALLBACK

SR = 8000
N = 3 * pk.TILE_SAMPLES + 123  # a few tiles and a ragged end


def _inputs(op, n=N):
    """Realistic packed inputs for each op, built the way the kernel scripts build them."""
    rng = np.random.default_rng(3)
    t = dataclasses.replace(FALLBACK, aiclk=1350.0, heartbeat=3)
    since = seconds_since(np.arange(0.2, n / SR, 0.9), n, SR)
    if op == "ttshock":
        return pk.shock_inputs(since, 3.0, 0.8)
    if op == "ttshimmer":
        return pk.shimmer_inputs(since, 12000.0, 12040.0, 0.7, n, SR)
    if op == "ttgong":
        g = gong_parameters(t, SR, DEFAULT_PATCH)
        return pk.gong_inputs(pk.strike_ages(np.arange(0.2, n / SR, 2.0), n, SR, g.length), g, SR)
    if op == "ttavdog":
        return pk.avdog_inputs(since, 3.0, 0.8, 110.0, 0.3, n, SR)
    if op == "ttmix":
        return pk.mix_inputs([rng.uniform(-1, 1, n) for _ in range(3)], [30.0, 40.0, 50.0], n)
    if op == "ttchoir":  # five harmonics of a 220 Hz voice whose levels move, so three slots are silent
        phase = 2 * np.pi * 220.0 * np.arange(n) / SR
        return pk.choir_inputs(phase, [rng.uniform(0, 0.2) * (1 + np.sin(np.arange(n) / 900 + j)) for j in range(5)], [1, 2, 3, 5, 8], n)
    return pk.coco_inputs([rng.uniform(-1, 1, n) for _ in range(5)], n)


EXPECTED = {
    "ttshock": lambda tiles, n: pk.shock_on_host(*tiles, n),
    "ttshimmer": lambda tiles, n: pk.shimmer_on_host(*tiles, n),
    "ttgong": pk.gong_on_host,
    "ttavdog": pk.avdog_on_host,
    "ttmix": pk.mix_on_host,
    "ttcoco": pk.coco_on_host,
    "ttchoir": pk.choir_on_host,
}


@pytest.fixture
def worker():
    with ChipWorker(host_command()) as w:
        yield w


def test_the_worker_knows_the_seven_kernels():
    assert set(CHIP_OPS) == set(EXPECTED) == {"ttshock", "ttshimmer", "ttgong", "ttavdog", "ttmix", "ttcoco", "ttchoir"}
    assert CHIP_OPS["ttshock"] == 3 and CHIP_OPS["ttshimmer"] == 5 and CHIP_OPS["ttgong"] == 16
    assert CHIP_OPS["ttavdog"] == 10 and CHIP_OPS["ttmix"] == 4 and CHIP_OPS["ttcoco"] == 8
    assert CHIP_OPS["ttchoir"] == 2 * pk.CHOIR_GROUP == 16


@pytest.mark.parametrize("op", sorted(EXPECTED))
def test_the_worker_returns_what_the_kernels_formula_gives_for_each_op(worker, op):
    tiles = _inputs(op)
    got = worker.run(op, tiles, N)
    assert got.shape == (N,) and got.dtype == np.float64
    assert np.allclose(got, EXPECTED[op](tiles, N), atol=1e-6)
    assert np.abs(got).max() > 0  # the check is not on silence


@pytest.mark.parametrize("op", sorted(EXPECTED))
def test_a_chunk_smaller_than_the_input_gives_the_same_answer(op):
    """The chip runs fixed-size chunks (so each kernel compiles once). The last is padded with zeros and trimmed."""
    tiles = _inputs(op)
    with ChipWorker(host_command(chunk_tiles=1)) as small:
        got = small.run(op, tiles, N)
    assert np.allclose(got, EXPECTED[op](tiles, N), atol=1e-6)


def test_an_input_shorter_than_one_chunk_is_padded_and_trimmed():
    tiles = _inputs("ttshock", 500)
    with ChipWorker(host_command(chunk_tiles=8)) as w:
        got = w.run("ttshock", tiles, 500)
    assert got.shape == (500,) and np.allclose(got, EXPECTED["ttshock"](tiles, 500), atol=1e-6)


def test_an_unknown_op_or_the_wrong_number_of_inputs_is_an_error_the_worker_survives(worker):
    with pytest.raises(ChipError, match="ttbogus"):
        worker.run("ttbogus", _inputs("ttshock"), N)
    with pytest.raises(ChipError, match="takes 3"):
        worker.run("ttshock", _inputs("ttshock")[:2], N)
    tiles = _inputs("ttshock")
    assert np.allclose(worker.run("ttshock", tiles, N), EXPECTED["ttshock"](tiles, N), atol=1e-6)


def test_a_worker_that_has_died_is_an_error_not_a_hang():
    w = ChipWorker(host_command())
    w.start()
    w.process.kill()
    w.process.wait()
    with pytest.raises(ChipError):
        w.run("ttshock", _inputs("ttshock"), N)
    w.close()


def test_a_worker_that_cannot_start_says_so():
    with pytest.raises(ChipError, match="could not start"):
        ChipWorker(["/no/such/worker"]).start()
    bad = ChipWorker([sys.executable, "-c", "import sys; sys.exit(7)"])
    with pytest.raises(ChipError):
        bad.start()
        bad.run("ttshock", _inputs("ttshock"), N)
    bad.close()


def test_close_ends_the_process_and_is_safe_to_repeat():
    w = ChipWorker(host_command())
    w.start()
    process = w.process
    w.close()
    assert process.poll() is not None and w.process is None
    w.close()


def test_the_default_command_asks_gozer_for_one_chip_and_names_who_and_why():
    from tensixinet.media.chip import chip_command
    cmd = chip_command()
    assert cmd[0] == "gozer" and cmd[1] == "run" and cmd[cmd.index("--chips") + 1] == "1"
    assert "--who" in cmd and "--reason" in cmd and cmd[cmd.index("--") + 1] == "python3"
    assert cmd[-2].endswith("kernels/pool.py") and cmd[-1] == "--devices=1"


# --- closing a worker under gozer: the exit resets the chips, which takes about 40 s, and must not be cut short -----------

def test_a_chip_worker_is_given_long_enough_to_reset_the_chips_and_a_host_worker_is_not():
    from tensixinet.media.chip import CHIP_CLOSE_TIMEOUT_S, HOST_CLOSE_TIMEOUT_S, chip_command
    assert CHIP_CLOSE_TIMEOUT_S >= 120 and HOST_CLOSE_TIMEOUT_S <= 60   # gozer's reset took 44 s on this box
    assert ChipWorker(chip_command()).close_timeout == CHIP_CLOSE_TIMEOUT_S
    assert ChipWorker(host_command()).close_timeout == HOST_CLOSE_TIMEOUT_S
    assert ChipWorker(host_command(), close_timeout=5).close_timeout == 5


def _slow_worker(seconds, close_timeout):
    """A worker that is ready at once, and after its input closes takes `seconds` to finish, as gozer does."""
    code = ("import sys, time\nfrom tensixinet.media.chip import write_frame\nimport numpy as np\n"
            "write_frame(sys.stdout.buffer, {'ready': np.array(1)})\nsys.stdin.buffer.read()\ntime.sleep(%s)\nsys.exit(0)" % seconds)
    return ChipWorker([sys.executable, "-c", code], close_timeout=close_timeout)


def test_close_waits_for_a_slow_exit_instead_of_killing_the_worker():
    w = _slow_worker(1.5, close_timeout=20)
    w.start()
    process = w.process
    w.close()
    assert process.returncode == 0           # it finished on its own: gozer's release would have run


def test_close_kills_a_worker_only_after_the_whole_timeout():
    w = _slow_worker(30, close_timeout=1)
    w.start()
    process = w.process
    w.close()
    assert process.returncode not in (0, None)


# --- several chips: one worker, one lease, one open device per chip ------------------------------------------------------

def test_a_worker_can_hold_several_devices_and_each_request_names_one():
    tiles = _inputs("ttshock")
    with ChipWorker(host_command(devices=4), devices=4) as w:
        assert w.devices == 4
        for device in range(4):
            got = w.run("ttshock", tiles, N, device=device)
            assert np.allclose(got, EXPECTED["ttshock"](tiles, N), atol=1e-6)
        with pytest.raises(ChipError, match="device 4"):
            w.run("ttshock", tiles, N, device=4)
        with pytest.raises(ChipError, match="device -1"):
            w.run("ttshock", tiles, N, device=-1)


def test_the_worker_itself_refuses_a_device_it_did_not_open():
    """The client checks too, but a worker started with fewer devices than the client thinks must say so, not use device 0."""
    with ChipWorker(host_command(devices=2), devices=4) as w:
        assert np.abs(w.run("ttshock", _inputs("ttshock"), N, device=1)).max() > 0
        with pytest.raises(ChipError, match="refused ttshock.*device 3"):
            w.run("ttshock", _inputs("ttshock"), N, device=3)


def test_the_commands_ask_for_as_many_chips_as_devices():
    from tensixinet.media.chip import chip_command
    cmd = chip_command(4)
    assert cmd[cmd.index("--chips") + 1] == "4" and cmd[-1] == "--devices=4" and cmd[cmd.index("--") + 1] == "python3"
    assert chip_command()[chip_command().index("--chips") + 1] == "1" and chip_command()[-1] == "--devices=1"
    assert "--devices=3" in host_command(devices=3) and "--devices=1" in host_command()


# --- gozer queued the request: say so, and do not leave the ticket behind ----------------------------------------------------

QUEUED = ("import sys\nprint('gozer: queued: ticket ab12', file=sys.stderr)\nsys.exit(10)")


def test_a_queued_request_cancels_its_own_ticket_and_says_the_chips_are_busy():
    cancelled = []
    w = ChipWorker([sys.executable, "-c", QUEUED])
    w.cancel = cancelled.append
    with pytest.raises(ChipError, match="chips are busy.*cancelled ticket ab12"):
        w.start()
    assert cancelled == ["ab12"]


def test_only_a_gozer_queue_exit_cancels_a_ticket():
    cancelled = []
    for code in ("import sys; print('ticket ab12', file=sys.stderr); sys.exit(3)",):
        w = ChipWorker([sys.executable, "-c", code])
        w.cancel = cancelled.append
        with pytest.raises(ChipError):
            w.start()
    assert cancelled == []


def test_the_default_cancel_asks_gozer_to_cancel_that_ticket(monkeypatch):
    import subprocess
    seen = []
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: seen.append(cmd))
    ChipWorker([sys.executable, "-c", "pass"]).cancel("ab12")
    assert seen == [["gozer", "cancel", "ab12"]]


# --- no orphans: a worker that will not exit takes its children with it --------------------------------------------------------

def test_a_close_that_times_out_ends_the_workers_children_too(tmp_path):
    """gozer run is the process we hold; the pool and the workers are its children. Killing only gozer once left a worker
    holding the chips for minutes, so the whole process group goes."""
    import os
    import time
    pidfile = tmp_path / "child.pid"
    code = ("import subprocess, sys, time\nimport numpy as np\nfrom tensixinet.media.chip import write_frame\n"
            f"child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\nopen({str(pidfile)!r}, 'w').write(str(child.pid))\n"
            "write_frame(sys.stdout.buffer, {'ready': np.array(1)})\ntime.sleep(60)")
    w = ChipWorker([sys.executable, "-c", code], close_timeout=1)
    w.start()
    child = int(pidfile.read_text())
    w.close()
    for _ in range(50):
        try:
            os.kill(child, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    else:
        os.kill(child, 9)
        raise AssertionError("the worker's child survived the close")


def test_a_worker_that_never_becomes_ready_is_ended_at_once_not_after_the_close_wait():
    import time
    w = ChipWorker([sys.executable, "-c", "import time; time.sleep(60)"], start_timeout=1, close_timeout=120)
    started = time.monotonic()
    with pytest.raises(ChipError, match="not ready after"):
        w.start()
    assert time.monotonic() - started < 40 and w.process is None


# --- one worker per board: the two chips of a board are one tt-metal cluster -------------------------------------------------

def test_chips_are_dealt_two_to_a_board_in_the_leases_order():
    from tensixinet.media.chip import CHIPS_PER_BOARD, board_of, boards_needed
    assert CHIPS_PER_BOARD == 2
    assert [board_of(c) for c in range(5)] == [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)]
    assert [boards_needed(n) for n in (1, 2, 3, 4, 5)] == [1, 1, 2, 2, 3]


def test_a_board_worker_is_shown_both_chips_of_its_board_and_no_others():
    """Showing a worker one chip of a two-chip board made tt-metal call the cluster CUSTOM and refuse to start; showing all
    four to four workers made them wait on each other's locks. A board's two chips, one worker per board, runs in parallel."""
    from tensixinet.media.chip import board_visible_devices
    lease = "0000:01:00.0,0000:02:00.0,0000:03:00.0,0000:04:00.0"
    assert board_visible_devices(lease, 0, 4) == "0000:01:00.0,0000:02:00.0"
    assert board_visible_devices(lease, 1, 4) == "0000:03:00.0,0000:04:00.0"
    assert board_visible_devices(" 0000:01:00.0 , 0000:02:00.0 ", 0, 2) == "0000:01:00.0,0000:02:00.0"        # spaces around commas


def test_a_lease_with_too_few_chips_or_none_is_refused_with_a_reason():
    from tensixinet.media.chip import board_visible_devices
    with pytest.raises(ChipError, match="board 1.*2 chips"):
        board_visible_devices("0000:01:00.0,0000:02:00.0", 1, 4)
    with pytest.raises(ChipError, match="TT_VISIBLE_DEVICES"):
        board_visible_devices(None, 0, 1)
    with pytest.raises(ChipError, match="TT_VISIBLE_DEVICES"):
        board_visible_devices("", 0, 1)


def _children(pid):
    import subprocess
    out = subprocess.run(["ps", "--ppid", str(pid), "-o", "pid="], capture_output=True, text=True).stdout.split()
    return [int(x) for x in out]


@pytest.mark.parametrize("chips, boards", [(1, 1), (2, 1), (3, 2), (4, 2)])
def test_the_pool_starts_one_worker_process_per_board(chips, boards):
    with ChipWorker(host_command(devices=chips), devices=chips) as w:
        assert len(_children(w.process.pid)) == boards


def test_every_chip_of_a_pool_answers_and_the_request_reaches_the_right_board():
    tiles = _inputs("ttshock")
    with ChipWorker(host_command(devices=4), devices=4) as w:
        for chip in range(4):
            assert np.allclose(w.run("ttshock", tiles, N, device=chip), EXPECTED["ttshock"](tiles, N), atol=1e-6)


# --- the TT-Lang simulator as a worker (no chip, no lease) ------------------------------------------------------------------

def test_the_simulator_command_runs_the_pool_with_the_simulator_and_the_chip_count():
    from tensixinet.media.chip import WORKER, simulator_command
    cmd = simulator_command(4, exe="/venv/bin/tt-lang-sim", chunk_tiles=16)
    assert cmd[:2] == [sys.executable, str(WORKER)] and "--sim=/venv/bin/tt-lang-sim" in cmd and "--devices=4" in cmd
    assert "--chunk-tiles=16" in cmd and "--host" not in cmd and "gozer" not in cmd  # no lease: the simulator opens no chip


def test_the_simulator_is_found_from_the_environment_then_the_path_then_beside_python(tmp_path):
    from tensixinet.media.chip import find_simulator
    named = tmp_path / "named"
    named.write_text("")
    beside = tmp_path / "bin" / "tt-lang-sim"
    beside.parent.mkdir()
    beside.write_text("")
    py = str(beside.parent / "python")
    assert find_simulator({"TT_LANG_SIM": str(named)}, lambda n: "/usr/bin/tt-lang-sim", py) == str(named)
    assert find_simulator({}, lambda n: "/usr/bin/tt-lang-sim", py) == "/usr/bin/tt-lang-sim"
    assert find_simulator({}, lambda n: None, py) == str(beside)
    assert find_simulator({"TT_LANG_SIM": str(tmp_path / "gone")}, lambda n: None, str(tmp_path / "nowhere" / "python")) is None


def test_no_simulator_is_an_error_that_says_how_to_get_one(monkeypatch):
    import tensixinet.media.chip as chip
    monkeypatch.setattr(chip, "find_simulator", lambda *a, **k: None)
    with pytest.raises(ChipError, match="tt-lang-sim"):
        chip.simulator_command(2)


def test_a_simulator_worker_gets_the_quick_timeouts_of_a_host_worker_not_the_chip_ones():
    from tensixinet.media.chip import SIM_START_TIMEOUT_S, simulator_command
    w = ChipWorker(simulator_command(2, exe="x"))
    assert w.close_timeout == 30.0 and w.start_timeout == SIM_START_TIMEOUT_S < 300.0
