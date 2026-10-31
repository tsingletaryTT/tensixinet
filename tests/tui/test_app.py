# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The screen driver: it draws what the playhead shows on its own thread, redraws only changed cells, follows a resize, ends the
run on `q`, never holds up the loop when the terminal blocks, and on every exit puts the terminal back before it prints.

No test needs a real terminal except the one that checks `TtyTerminal` restores the mode it found, which uses a pty."""
import os
import queue
import re
import signal
import termios
import threading
import time

import pytest

from tensixinet.music.patch import DEFAULT_PATCH
from tensixinet.telemetry import mock_telemetry
from tensixinet.tui.app import ERROR_RECORDS, FRAME_S, REDRAW_S, TtyTerminal, TuiApp
from tensixinet.tui.playhead import Playhead
from tensixinet.ui.flow import build_flow

from .test_frame import Screen, strip

WAIT_S = 5.0  # every wait in these tests gives up after this long, so a broken driver fails the test and never hangs it
TEST_FRAME_S = 0.01


class Clock:
    def __init__(self, t=100.0):
        self.t = t

    def __call__(self):
        return self.t


class FakeTerminal:
    """Records every call in order. `write` waits on `block` when one is set (a terminal that has stopped reading); the text
    is recorded when the write begins, as a real terminal would have received it."""

    def __init__(self, size=(80, 24)):
        self.size_ = size
        self.keys = queue.Queue()
        self.events = []  # ("enter" | "leave" | "write" | "flush", text)
        self.block = None
        self.fail_next = False
        self.size_block = None
        self.entered = self.left = 0
        self._lock = threading.Lock()

    def _log(self, kind, text=""):
        with self._lock:
            self.events.append((kind, text))

    def write(self, text):
        if self.fail_next:
            self.fail_next = False
            raise OSError("the terminal went away for a moment")  # nothing reached the screen
        self._log("write", text)
        # Only the drawing thread's writes get stuck: the test is about what the rest of the program does meanwhile.
        if self.block is not None and threading.current_thread().name == "tensixinet-tui":
            self.block.wait(WAIT_S)

    def flush(self):
        self._log("flush")

    def size(self):
        # `size_block` holds the drawing thread here, before it has written anything for the frame.
        if self.size_block is not None and threading.current_thread().name == "tensixinet-tui":
            self.size_block.wait(WAIT_S)
        return self.size_

    def enter(self):
        self.entered += 1
        self._log("enter")

    def leave(self):
        self.left += 1
        self._log("leave")

    def read_key(self, timeout):
        try:
            return self.keys.get(timeout=timeout)
        except queue.Empty:
            return None

    # -- what the tests read --------------------------------------------------------------------------------------------------

    def snapshot(self):
        with self._lock:
            return list(self.events)

    @property
    def calls(self):
        return [kind for kind, _ in self.snapshot()]

    @property
    def writes(self):
        return [text for kind, text in self.snapshot() if kind == "write"]

    def writes_from(self, index):
        return [text for kind, text in self.snapshot()[index:] if kind == "write"]


def screen(term, width=None, height=None):
    """The text a real terminal would show after every frame written so far (exit lines included)."""
    w, h = term.size_ if width is None else (width, height)
    s = Screen()
    for text in term.writes:
        s.feed(text)
    return "\n".join(s.lines(w, h))


def after_leave(term):
    events = term.snapshot()
    i = max(n for n, (kind, _) in enumerate(events) if kind == "leave")
    return "".join(text for kind, text in events[i:] if kind == "write")


def max_column_written(texts):
    """The rightmost column (counted from 1) any of `texts` puts a character in."""
    best = 0
    s = Screen(400, 200)
    for text in texts:
        s.feed(text)
    for row in s.grid:
        line = "".join(row).rstrip()
        best = max(best, len(line))
    return best


def wait_until(predicate, what="the condition"):
    end = time.monotonic() + WAIT_S
    while time.monotonic() < end:
        if predicate():
            return
        time.sleep(0.002)
    raise AssertionError(f"timed out waiting for {what}")


def wait_frames(app, n):
    """Wait until `n` whole frames have been drawn after this call (one more than `n` is counted, because a frame may have been
    half-way through when this was called and read the playhead before the change the test just made)."""
    start = app.frames
    wait_until(lambda: app.frames >= start + n + 1, f"{n} frames")


def state(segment, n=4, soloist=2):
    return build_flow(mock_telemetry(n), DEFAULT_PATCH, segment=segment, soloist_slot=soloist, seconds=4.0)


def rec(what, why=""):
    return {"t": 1.0, "kind": "note", "what": what, "why": why}


class Rig:
    """A fake terminal, a playhead on a hand-moved clock, and a TuiApp drawing between them."""

    def __init__(self, colour=True, size=(80, 24), playhead=None, log_dir="logs"):
        self.term = FakeTerminal(size)
        self.clock = Clock()
        self.ph = playhead if playhead is not None else Playhead(clock=self.clock)
        self.interrupts = []
        self.installed = {signal.SIGWINCH: "the old handler"}
        self.log_dir = log_dir
        self.app = TuiApp(
            self.term, self.ph, colour=colour, truecolor=True, unit="segment", log_dir=log_dir, frame_s=TEST_FRAME_S,
            interrupt=lambda: self.interrupts.append(1), install_signal=self._install, clock=self.clock,
        )

    def _install(self, signum, handler):
        old = self.installed.get(signum)
        self.installed[signum] = handler
        return old


@pytest.fixture
def rig():
    r = Rig()
    yield r
    if r.term.block is not None:
        r.term.block.set()
    r.app.close(0)


# --- constants --------------------------------------------------------------------------------------------------------------

def test_the_frame_period_and_the_error_record_count_are_pinned():
    assert FRAME_S == 0.1 and ERROR_RECORDS == 5


# --- entering, drawing, leaving ---------------------------------------------------------------------------------------------

def test_start_enters_the_alternate_screen_and_close_leaves_it(rig):
    rig.app.start()
    rig.app.close(0)
    calls = rig.term.calls
    assert rig.term.entered == 1 and rig.term.left == 1 and calls.index("leave") > calls.index("enter")


def test_the_drawing_runs_on_its_own_daemon_thread(rig):
    rig.app.start()
    wait_frames(rig.app, 1)
    names = {t.name: t for t in threading.enumerate()}
    assert "tensixinet-tui" in names and names["tensixinet-tui"].daemon
    assert rig.app.running


def test_the_screen_shows_the_state_the_playhead_has_reached(rig):
    rig.ph.hand_over(4.0, state(0))
    rig.ph.started()
    rig.ph.hand_over(4.0, state(1))
    rig.clock.t += 1.0
    rig.app.start()
    wait_frames(rig.app, 2)
    shown = screen(rig.term)
    assert "segment 0" in shown and "segment 1" not in shown
    rig.clock.t += 4.0
    wait_frames(rig.app, 2)
    shown = screen(rig.term)
    assert "segment 1" in shown and "segment 0" not in shown


def test_the_log_pane_shows_the_records_the_listener_has_reached(rig):
    rig.ph.note(rec("the opening line"))
    rig.app.start()
    wait_frames(rig.app, 1)
    assert "the opening line" in screen(rig.term)


def test_before_any_audio_the_heading_says_it_is_waiting(rig):
    rig.app.start()
    wait_frames(rig.app, 1)
    assert "waiting for the first segment to play" in screen(rig.term)


def test_only_changed_cells_are_redrawn(rig):
    rig.ph.hand_over(4.0, state(0))
    rig.ph.started()
    rig.app.start()
    wait_frames(rig.app, 2)
    mark = len(rig.term.snapshot())
    rig.ph.note(rec("a new line for the log"))  # tagged at 4.0, where the audio handed over so far ends
    rig.clock.t += 4.5  # the listener reaches it; the state is still segment 0, the only one handed over
    wait_frames(rig.app, 2)
    later = "".join(rig.term.writes_from(mark))
    assert "a new line for the log" in screen(rig.term)
    assert "new" in strip(later) and len(strip(later)) < 200  # a few changed cells, far from the 1920 of a full screen
    assert "\x1b[2J" not in later and "segment 0" not in strip(later)  # the heading did not change, so it was not written again


def test_a_frame_that_did_not_change_writes_nothing(rig):
    rig.ph.hand_over(4.0, state(0))
    rig.ph.started()
    rig.app.start()
    wait_frames(rig.app, 2)
    before = len(rig.term.writes)
    wait_frames(rig.app, 5)
    assert len(rig.term.writes) == before


def test_without_colour_no_colour_codes_are_written():
    r = Rig(colour=False)
    r.ph.hand_over(4.0, state(0))
    r.ph.started()
    r.app.start()
    wait_frames(r.app, 1)
    r.app.close(0)
    text = "".join(r.term.writes)
    assert "segment 0" in strip(text)
    assert not re.search(r"\x1b\[[0-9;]*38;", text)


def test_too_small_shows_one_line_and_keeps_going(rig):
    rig.term.size_ = (50, 10)
    rig.app.start()
    wait_frames(rig.app, 1)
    assert "too small" in screen(rig.term) and rig.app.running


# --- resize -----------------------------------------------------------------------------------------------------------------

def test_start_installs_a_sigwinch_handler(rig):
    rig.app.start()
    assert callable(rig.installed[signal.SIGWINCH])


def test_a_resize_clears_and_redraws_at_the_new_size(rig):
    rig.ph.hand_over(4.0, state(0))
    rig.ph.started()
    rig.app.start()
    wait_frames(rig.app, 2)
    handler = rig.installed[signal.SIGWINCH]
    mark = len(rig.term.snapshot())
    rig.term.size_ = (100, 30)
    handler(signal.SIGWINCH, None)
    wait_frames(rig.app, 2)
    after = rig.term.writes_from(mark)
    assert any(t.startswith("\x1b[2J") for t in after)
    assert max_column_written(after) > 80


def test_a_resize_redraws_everything_even_when_the_size_did_not_change(rig):
    # A terminal that was resized and resized back (or that cleared itself) may show anything: the signal alone forces a full redraw.
    rig.ph.hand_over(4.0, state(0))
    rig.ph.started()
    rig.app.start()
    wait_frames(rig.app, 2)
    mark = len(rig.term.snapshot())
    rig.installed[signal.SIGWINCH](signal.SIGWINCH, None)
    wait_frames(rig.app, 2)
    after = "".join(rig.term.writes_from(mark))
    assert after.startswith("\x1b[2J") and "segment 0" in strip(after)


def test_close_puts_back_the_previous_sigwinch_handler(rig):
    rig.app.start()
    assert rig.installed[signal.SIGWINCH] != "the old handler"
    rig.app.close(0)
    assert rig.installed[signal.SIGWINCH] == "the old handler"


# --- keys -------------------------------------------------------------------------------------------------------------------

def test_q_ends_the_run_the_way_ctrl_c_does_and_a_second_q_sends_again(rig):
    rig.app.start()
    rig.term.keys.put("q")
    wait_until(lambda: rig.interrupts == [1], "the interrupt")
    rig.term.keys.put("Q")  # pressed again: the run is playing out what it made, and a second press says "stop now"
    wait_until(lambda: rig.interrupts == [1, 1], "the second interrupt")


STOPPING = "stopping: playing out what is already made, Ctrl-C to stop now"


def test_q_shows_at_once_that_the_run_is_stopping(rig):
    rig.ph.hand_over(4.0, state(0))
    rig.ph.started()
    rig.app.start()
    wait_frames(rig.app, 2)
    assert STOPPING not in screen(rig.term)
    rig.term.keys.put("q")
    wait_until(lambda: rig.interrupts == [1], "the interrupt")
    wait_frames(rig.app, 2)
    assert STOPPING in screen(rig.term).splitlines()[0]  # in the heading, while the queued segments still play


def test_q_is_sent_from_a_thread_that_blocks_sigint_and_sigwinch():
    # The signal is meant for the main thread: the TUI's own thread must not be the one that takes it.
    masks = []
    term = FakeTerminal()
    app = TuiApp(term, Playhead(clock=Clock()), colour=False, truecolor=False, unit="segment", log_dir="logs", frame_s=TEST_FRAME_S,
                 interrupt=lambda: masks.append(signal.pthread_sigmask(signal.SIG_BLOCK, [])), install_signal=lambda s, h: None)
    app.start()
    term.keys.put("q")
    wait_until(lambda: masks, "the interrupt")
    app.close(130)
    assert signal.SIGINT in masks[0] and signal.SIGWINCH in masks[0]


def test_q_wakes_a_main_thread_blocked_in_a_wait_in_under_a_second():
    # The real default `interrupt`, and a real SIGINT. `_thread.interrupt_main` only sets a flag, which a main thread blocked in a C-level
    # wait (the live loop's `room.acquire()`) sees when the wait ends, seconds later; a signal wakes the wait at once.
    assert threading.current_thread() is threading.main_thread()
    term = FakeTerminal()
    app = TuiApp(term, Playhead(clock=Clock()), colour=False, truecolor=False, unit="segment", log_dir="logs", frame_s=TEST_FRAME_S,
                 install_signal=lambda s, h: None)
    previous = signal.signal(signal.SIGINT, signal.default_int_handler)
    blocked = threading.Semaphore(0)
    try:
        app.start()
        threading.Timer(0.3, lambda: term.keys.put("q")).start()
        start = time.monotonic()
        try:
            blocked.acquire(timeout=3)
            time.sleep(0.01)  # a flag set by interrupt_main is raised here, after the whole wait
            elapsed = None
        except KeyboardInterrupt:
            elapsed = time.monotonic() - start
        if elapsed is None:  # the interrupt arrived after the wait, or not at all
            elapsed = time.monotonic() - start
        assert elapsed < 1.0, f"q took {elapsed:.2f} s to reach the main thread"
    finally:
        app.close(130)
        signal.signal(signal.SIGINT, previous)


def test_capital_q_also_ends_the_run(rig):
    rig.app.start()
    rig.term.keys.put("Q")
    wait_until(lambda: rig.interrupts == [1], "the interrupt")


def test_other_keys_do_nothing(rig):
    rig.app.start()
    for key in "xw \n":
        rig.term.keys.put(key)
    wait_frames(rig.app, 6)
    assert rig.interrupts == [] and rig.app.running


# --- a slow terminal --------------------------------------------------------------------------------------------------------

def test_a_blocked_terminal_never_blocks_the_loop(rig):
    rig.term.block = threading.Event()
    rig.ph.hand_over(0.1, state(0))
    rig.ph.started()
    rig.app.start()
    wait_until(lambda: len(rig.term.writes) >= 1, "the first write")  # the drawing thread is now stuck in write
    t = time.monotonic()
    s0 = state(0)
    for i in range(200):
        rig.ph.hand_over(0.1, s0)
        rig.ph.note(rec(str(i)))
    assert time.monotonic() - t < 0.5  # the loop's side returned while the terminal was stuck
    rig.term.block.set()
    rig.app.close(0)


def test_a_frame_that_was_half_made_when_close_began_is_never_written(rig):
    # The drawing thread is held after it began a frame and before it wrote any of it. close gives up waiting for it, leaves the
    # alternate screen, and only then is the thread let go: it must not draw that frame over the restored screen.
    rig.ph.hand_over(4.0, state(0))
    rig.ph.started()
    rig.term.size_block = threading.Event()
    rig.app.start()
    time.sleep(5 * TEST_FRAME_S)  # the thread is now waiting in size()
    rig.app.close(0)
    rig.term.size_block.set()
    time.sleep(10 * TEST_FRAME_S)
    assert not any("\x1b[" in text for text in rig.term.writes)


def test_every_render_happens_on_the_tui_thread(rig, monkeypatch):
    # A render takes well under a millisecond, so a loop that rendered would still pass the timing test above. This one sees it.
    import tensixinet.tui.app as app_module
    import tensixinet.tui.layout as layout_module

    threads = []
    real = layout_module.render

    def spy(*args, **kwargs):
        threads.append(threading.current_thread().name)
        return real(*args, **kwargs)

    monkeypatch.setattr(layout_module, "render", spy)
    monkeypatch.setattr(app_module, "render", spy)
    rig.app.start()
    for i in range(20):
        rig.ph.hand_over(0.1, state(0))
        rig.ph.note(rec(str(i)))
    rig.ph.started()
    wait_frames(rig.app, 2)
    assert threads and set(threads) == {"tensixinet-tui"}


def test_close_returns_while_the_terminal_is_still_blocked_and_no_frame_follows_the_leave(rig):
    rig.term.block = threading.Event()
    rig.ph.hand_over(4.0, state(0))
    rig.ph.started()
    rig.app.start()
    wait_until(lambda: len(rig.term.writes) >= 1, "the first write")
    done = []
    closer = threading.Thread(target=lambda: done.append(rig.app.close(0)), daemon=True)
    closer.start()
    closer.join(1.0)
    assert done, "close waited on a terminal that had stopped reading"
    rig.term.block.set()  # the stuck write returns now; the drawing thread must not draw over the restored screen
    time.sleep(10 * TEST_FRAME_S)
    events = rig.term.snapshot()
    leave = events.index(("leave", ""))
    frames_after = [text for kind, text in events[leave:] if kind == "write" and "\x1b[" in text]
    assert frames_after == []


# --- exit -------------------------------------------------------------------------------------------------------------------

def test_a_normal_end_prints_only_the_last_line_and_the_log_directory(rig):
    rig.app.start()
    rig.ph.note(rec("the first line"))
    rig.ph.note(rec("wrote out/chorus.wav", why="the run ended"))
    lines = rig.app.close(0)
    assert lines == ["tensixinet: wrote out/chorus.wav; the log is in logs"]
    assert after_leave(rig.term) == "tensixinet: wrote out/chorus.wav; the log is in logs\n"


def test_ctrl_c_code_130_is_a_normal_end(rig):
    rig.app.start()
    for i in range(8):
        rig.ph.note(rec(f"line {i}", why=f"because {i}"))
    assert rig.app.close(130) == ["tensixinet: line 7; the log is in logs"]


def test_an_error_code_prints_the_last_five_records_after_leaving(rig):
    rig.app.start()
    for i in range(8):
        rig.ph.note(rec(f"line {i}", why=f"because {i}"))
    lines = rig.app.close(3)
    assert [l for l in lines if l.startswith("line")] == [f"line {i}" for i in range(3, 8)]
    assert [l for l in lines if l.startswith("  why:")] == [f"  why: because {i}" for i in range(3, 8)]
    assert lines[-1] == "tensixinet: line 7; the log is in logs"
    printed = after_leave(rig.term)
    assert "line 3\n  why: because 3\n" in printed and printed.endswith("tensixinet: line 7; the log is in logs\n")


def test_an_error_record_ahead_of_the_listener_is_still_printed(rig):
    rig.app.start()
    rig.ph.hand_over(4.0, state(0))
    rig.ph.started()
    rig.ph.hand_over(4.0, state(1))
    rig.ph.note(rec("the audio output stopped: broken pipe"))  # logged ahead of the listener, who is still at 0
    lines = rig.app.close(1)
    assert "the audio output stopped: broken pipe" in lines
    assert lines[-1] == "tensixinet: the audio output stopped: broken pipe; the log is in logs"


def test_a_record_without_a_why_prints_no_why_line(rig):
    rig.app.start()
    rig.ph.note(rec("no reason given"))
    assert rig.app.close(2) == ["no reason given", "tensixinet: no reason given; the log is in logs"]


def test_a_run_with_no_records_still_names_the_log_directory(rig):
    rig.app.start()
    lines = rig.app.close(0)
    assert len(lines) == 1 and lines[0].startswith("tensixinet: ") and lines[0].endswith("; the log is in logs")


def test_the_terminal_is_left_before_the_exit_lines_are_written(rig):
    rig.app.start()
    rig.ph.note(rec("the last line"))
    rig.app.close(3)
    events = rig.term.snapshot()
    leave = events.index(("leave", ""))
    exit_writes = [n for n, (kind, text) in enumerate(events) if kind == "write" and "the last line" in text]
    assert exit_writes and all(n > leave for n in exit_writes)


def test_an_exception_in_the_run_still_restores_the_terminal_and_prints_the_error(rig):
    with pytest.raises(RuntimeError):
        with rig.app:
            rig.ph.note(rec("the audio output stopped: broken pipe"))
            raise RuntimeError("x")
    printed = after_leave(rig.term)
    assert rig.term.left == 1 and "broken pipe" in printed and "the log is in logs" in printed
    assert "RuntimeError: x" in printed
    assert rig.installed[signal.SIGWINCH] == "the old handler" and not rig.app.running


def test_leaving_the_block_without_an_exception_leaves_the_close_to_the_caller(rig):
    with rig.app as app:
        assert app is rig.app
    assert rig.term.left == 0 and rig.app.running
    rig.app.close(0)
    assert rig.term.left == 1


def test_close_twice_leaves_once(rig):
    rig.app.start()
    rig.ph.note(rec("the end"))
    first = rig.app.close(0)
    second = rig.app.close(3)
    assert rig.term.left == 1 and first == second
    assert after_leave(rig.term).count("the log is in logs") == 1


def test_close_without_start_only_writes_the_exit_lines(rig):
    rig.ph.note(rec("discover found no chips", why="tt-smi listed none"))
    lines = rig.app.close(2)
    assert rig.term.entered == 0 and rig.term.left == 0
    assert lines == ["discover found no chips", "  why: tt-smi listed none",
                     "tensixinet: discover found no chips; the log is in logs"]
    assert "".join(rig.term.writes) == "".join(line + "\n" for line in lines)
    assert rig.installed[signal.SIGWINCH] == "the old handler"


def test_close_stops_the_drawing_thread(rig):
    rig.app.start()
    wait_frames(rig.app, 1)
    rig.app.close(0)
    assert not rig.app.running
    wait_until(lambda: "tensixinet-tui" not in {t.name for t in threading.enumerate()}, "the thread to end")


# --- an error inside the drawing --------------------------------------------------------------------------------------------

class FlakyPlayhead(Playhead):
    """A playhead whose `shown` raises the first time it is called."""

    def __init__(self, clock):
        super().__init__(clock=clock)
        self.failed = False

    def shown(self):
        if not self.failed:
            self.failed = True
            raise ValueError("the layout fell over")
        return super().shown()


def test_a_drawing_error_is_reported_at_close_and_never_raised_into_the_loop():
    clock = Clock()
    r = Rig(playhead=FlakyPlayhead(clock))
    r.app.start()
    wait_until(lambda: r.ph.failed, "the drawing error")
    r.ph.hand_over(4.0, state(0))  # the loop's side keeps working
    r.ph.note(rec("still logging"))
    r.ph.started()
    wait_frames(r.app, 2)
    assert r.app.running and "segment 0" in screen(r.term)  # and the drawing carried on after the error
    lines = r.app.close(0)
    assert any("ValueError: the layout fell over" in line for line in lines)
    assert lines[-1] == "tensixinet: still logging; the log is in logs"


# --- the real terminal ------------------------------------------------------------------------------------------------------

def drain(fd):
    """Everything waiting on `fd`: a pty can hand its output over in more than one read."""
    import select

    chunks = []
    while select.select([fd], [], [], 0.2)[0]:
        chunk = os.read(fd, 4096)
        if not chunk:
            break
        chunks.append(chunk)
    return b"".join(chunks).decode()


def test_tty_terminal_restores_the_mode_it_found():
    master, slave = os.openpty()
    try:
        out = os.fdopen(os.dup(slave), "w")
        inp = os.fdopen(os.dup(slave), "r")
        before = termios.tcgetattr(slave)
        t = TtyTerminal(out=out, inp=inp)
        t.enter()
        assert termios.tcgetattr(slave) != before  # cbreak is on
        assert termios.tcgetattr(slave)[3] & termios.ISIG  # Ctrl-C still sends SIGINT
        assert not termios.tcgetattr(slave)[0] & termios.IXON  # Ctrl-S is a key, and cannot freeze the output
        os.write(master, b"q")  # cbreak: one key arrives without a newline
        assert t.read_key(1.0) == "q"
        assert t.read_key(0.01) is None
        t.leave()
        t.leave()  # safe twice
        assert termios.tcgetattr(slave) == before
        written = drain(master)
        assert written.startswith("\x1b[?1049h\x1b[?25l")
        assert written.count("\x1b[?1049l") == 1 and written.index("\x1b[?25h") < written.index("\x1b[?1049l")
        out.close()
        inp.close()
    finally:
        os.close(master)
        os.close(slave)


def test_tty_terminal_reads_no_keys_and_changes_no_mode_when_input_is_not_a_terminal(tmp_path):
    with open(tmp_path / "out", "w") as out, open(os.devnull) as inp:
        t = TtyTerminal(out=out, inp=inp)
        t.enter()
        start = time.monotonic()
        assert t.read_key(0.05) is None and time.monotonic() - start >= 0.04
        t.leave()
    text = (tmp_path / "out").read_text()
    assert text.startswith("\x1b[?1049h") and text.endswith("\x1b[?1049l")


def test_tty_terminal_size_is_columns_then_rows(monkeypatch):
    monkeypatch.setenv("COLUMNS", "123")
    monkeypatch.setenv("LINES", "45")
    assert TtyTerminal().size() == (123, 45)


def test_a_frame_that_did_not_change_is_not_even_diffed(rig, monkeypatch):
    import tensixinet.tui.app as app_module

    calls = []
    real = app_module.diff
    monkeypatch.setattr(app_module, "diff", lambda *a, **k: calls.append(1) or real(*a, **k))
    rig.ph.hand_over(4.0, state(0))
    rig.ph.started()
    rig.app.start()
    wait_frames(rig.app, 2)
    before = len(calls)
    wait_frames(rig.app, 5)
    assert before >= 1 and len(calls) == before


def test_after_a_failed_write_the_next_frame_redraws_the_whole_screen(rig):
    rig.ph.hand_over(4.0, state(0))
    rig.ph.started()
    rig.term.fail_next = True
    rig.app.start()
    wait_until(lambda: not rig.term.fail_next, "the failed write")
    wait_frames(rig.app, 2)
    assert rig.term.writes[0].startswith("\x1b[2J") and "segment 0" in screen(rig.term)
    assert any("the terminal went away" in line for line in rig.app.close(0))


def test_ctrl_c_or_q_leaving_the_block_is_a_normal_end(rig):
    with pytest.raises(KeyboardInterrupt):
        with rig.app:
            for i in range(8):
                rig.ph.note(rec(f"line {i}", why=f"because {i}"))
            raise KeyboardInterrupt
    assert rig.term.left == 1 and rig.app.close(0) == ["tensixinet: line 7; the log is in logs"]


# --- a frame with nothing new costs nothing ------------------------------------------------------------------------------------

def _count_calls(monkeypatch):
    """Spies on the two expensive steps of a frame, as the driver calls them."""
    import tensixinet.tui.app as app_module

    calls = {"render": 0, "log_rows": 0}
    real_render, real_rows = app_module.render, app_module.log_rows

    def render(*a, **k):
        calls["render"] += 1
        return real_render(*a, **k)

    def log_rows(*a, **k):
        calls["log_rows"] += 1
        return real_rows(*a, **k)

    monkeypatch.setattr(app_module, "render", render)
    monkeypatch.setattr(app_module, "log_rows", log_rows)
    return calls


def _steady(rig, monkeypatch):
    calls = _count_calls(monkeypatch)
    rig.ph.hand_over(4.0, state(0))
    rig.ph.started()
    rig.ph.hand_over(4.0, state(1))
    rig.ph.note(rec("the first line"))
    rig.app.start()
    wait_frames(rig.app, 3)
    return calls


def test_a_frame_with_nothing_new_neither_renders_nor_lays_out_the_log_again(rig, monkeypatch):
    calls = _steady(rig, monkeypatch)
    before = dict(calls)
    wait_frames(rig.app, 6)
    assert before["render"] >= 1 and before["log_rows"] >= 1 and calls == before


def test_a_new_record_lays_out_the_log_and_renders_again(rig, monkeypatch):
    calls = _steady(rig, monkeypatch)
    before = dict(calls)
    rig.ph.note(rec("a second line"))  # tagged at 8.0; the listener is at 0
    rig.clock.t += 8.5  # the listener reaches it, and segment 1
    wait_frames(rig.app, 2)
    assert calls["log_rows"] > before["log_rows"] and calls["render"] > before["render"]
    assert "a second line" in screen(rig.term)


def test_a_new_state_with_the_same_records_renders_again_but_keeps_the_log_rows(rig, monkeypatch):
    calls = _steady(rig, monkeypatch)
    before = dict(calls)
    rig.clock.t += 4.5  # the listener reaches segment 1; no new record
    wait_frames(rig.app, 2)
    assert calls["render"] > before["render"] and calls["log_rows"] == before["log_rows"]
    assert "segment 1" in screen(rig.term)


def test_a_new_size_without_a_resize_signal_renders_again_at_the_new_width(rig, monkeypatch):
    calls = _steady(rig, monkeypatch)
    before = dict(calls)
    mark = len(rig.term.snapshot())
    rig.term.size_ = (100, 30)
    wait_frames(rig.app, 2)
    assert calls["render"] > before["render"] and calls["log_rows"] > before["log_rows"]
    assert max_column_written(rig.term.writes_from(mark)) > 80


def test_a_record_that_replaces_the_oldest_still_shows_though_the_count_is_the_same(monkeypatch):
    # The playhead keeps a fixed number of records: once it is full, a new one leaves the count as it was.
    clock = Clock()
    r = Rig(playhead=Playhead(clock=clock, keep=2))
    r.ph.note(rec("one"))
    r.ph.note(rec("two"))
    r.app.start()
    try:
        wait_frames(r.app, 2)
        r.ph.note(rec("three"))
        wait_frames(r.app, 2)
        assert "three" in screen(r.term)
    finally:
        r.app.close(0)


# --- the screen repairs itself ---------------------------------------------------------------------------------------------------

def test_redraw_period_is_pinned():
    assert REDRAW_S == 5.0


def test_ctrl_l_redraws_the_whole_screen(rig):
    rig.ph.hand_over(4.0, state(0))
    rig.ph.started()
    rig.app.start()
    wait_frames(rig.app, 2)
    mark = len(rig.term.snapshot())
    rig.term.keys.put("\x0c")
    wait_frames(rig.app, 2)
    after = "".join(rig.term.writes_from(mark))
    assert after.startswith("\x1b[2J") and "segment 0" in strip(after) and rig.interrupts == []


def test_the_whole_screen_is_redrawn_every_five_seconds_and_not_more_often(rig):
    # Something else (a warning on stderr) may have written over the screen; diff-only drawing would never put it right.
    rig.ph.hand_over(60.0, state(0))
    rig.ph.started()
    rig.app.start()
    wait_frames(rig.app, 2)
    mark = len(rig.term.snapshot())
    rig.clock.t += 4.9
    wait_frames(rig.app, 3)
    assert not any(t.startswith("\x1b[2J") for t in rig.term.writes_from(mark))
    rig.clock.t += 0.2
    wait_frames(rig.app, 3)
    full = [t for t in rig.term.writes_from(mark) if t.startswith("\x1b[2J")]
    assert len(full) == 1 and "segment 0" in strip(full[0])


# --- the race between q and close -----------------------------------------------------------------------------------------------

class _StaleStop(threading.Event):
    """A stop flag whose check on the drawing thread, just after a key, says "not stopped" only after `close` has finished: the
    window between the loop's check and its interrupt, held open."""

    def __init__(self):
        super().__init__()
        self.checked = threading.Event()
        self.closed = threading.Event()
        self.hold = False

    def is_set(self):
        if self.hold and threading.current_thread().name == "tensixinet-tui":
            self.hold = False
            self.checked.set()
            self.closed.wait(WAIT_S)
            return False
        return super().is_set()


def test_a_q_that_loses_the_race_with_close_sends_nothing(rig):
    stop = _StaleStop()
    rig.app._stop = stop
    rig.app.start()
    wait_frames(rig.app, 1)
    stop.hold = True
    rig.term.keys.put("q")
    assert stop.checked.wait(WAIT_S)
    rig.app.close(0)  # sets the flags while the drawing thread sits between its check and its interrupt
    stop.closed.set()
    time.sleep(10 * TEST_FRAME_S)
    assert rig.interrupts == [] and rig.term.left == 1


def test_an_interrupt_that_lands_inside_close_still_restores_the_terminal(rig):
    class Interrupted(threading.Event):
        raised = False

        def set(self):
            super().set()
            if not self.raised:  # once: the fixture's own close at the end must not be interrupted too
                self.raised = True
                raise KeyboardInterrupt  # a q sent a moment before close began, raised on the main thread here

    rig.app._stop = Interrupted()
    rig.app.start()
    wait_frames(rig.app, 1)
    with pytest.raises(KeyboardInterrupt):
        rig.app.close(0)
    assert rig.term.left == 1 and rig.installed[signal.SIGWINCH] == "the old handler"


def test_after_disarm_q_sends_nothing(rig):
    # The run loop has returned; a q must not interrupt what follows (the gozer release takes 40 s).
    rig.app.start()
    rig.app.disarm()
    rig.term.keys.put("q")
    wait_frames(rig.app, 5)
    assert rig.interrupts == [] and rig.app.running


# --- a frozen terminal cannot hang the exit ----------------------------------------------------------------------------------

class FakeTermios:
    """Just enough of `termios` to see the modes `TtyTerminal` sets and how it sets them."""

    TCSANOW, TCSADRAIN, TCSAFLUSH = termios.TCSANOW, termios.TCSADRAIN, termios.TCSAFLUSH
    ECHO, ICANON, ISIG, IXON, ICRNL = termios.ECHO, termios.ICANON, termios.ISIG, termios.IXON, termios.ICRNL
    VMIN, VTIME = termios.VMIN, termios.VTIME

    def __init__(self):
        self.mode = [self.IXON | self.ICRNL, 0, 0, self.ECHO | self.ICANON | self.ISIG, 0, 0, [b"\x00"] * 32]
        self.set = []

    def tcgetattr(self, fd):
        return [list(x) if isinstance(x, list) else x for x in self.mode]

    def tcsetattr(self, fd, when, mode):
        self.set.append((when, [list(x) if isinstance(x, list) else x for x in mode]))


class FakeTty:
    def isatty(self):
        return True

    def fileno(self):
        return 99


def test_enter_turns_off_flow_control_and_leave_restores_without_waiting_for_output(tmp_path):
    fake = FakeTermios()
    original = fake.tcgetattr(99)
    with open(tmp_path / "out", "w") as out:
        t = TtyTerminal(out=out, inp=FakeTty(), termios_module=fake)
        t.enter()
        _, cbreak = fake.set[-1]
        assert not cbreak[0] & fake.IXON  # Ctrl-S cannot freeze the output
        assert cbreak[3] & fake.ISIG and not cbreak[3] & (fake.ECHO | fake.ICANON)  # cbreak, with Ctrl-C still a signal
        t.leave()
    when, restored = fake.set[-1]
    assert when == fake.TCSANOW and restored == original  # TCSADRAIN would wait on a terminal that is not reading


def test_a_record_dropped_from_a_full_playhead_leaves_the_screen_though_the_newest_shown_is_the_same():
    # A full playhead drops its oldest record when a new one arrives. When the new one is still ahead of the listener, the records
    # shown lose one at the front and keep the same newest: only their number says the screen is out of date.
    clock = Clock()
    r = Rig(playhead=Playhead(clock=clock, keep=2))
    r.ph.note(rec("the oldest line"))
    r.ph.note(rec("the middle line"))
    r.ph.hand_over(4.0, state(0))  # the state does not change during the test
    r.ph.started()
    r.app.start()
    try:
        wait_frames(r.app, 2)
        assert "the oldest line" in screen(r.term)
        r.ph.note(rec("a line ahead of the listener"))  # tagged at 4.0; the listener is at 0
        wait_frames(r.app, 2)
        shown = screen(r.term)
        assert "the oldest line" not in shown and "the middle line" in shown and "ahead of the listener" not in shown
    finally:
        r.app.close(0)
