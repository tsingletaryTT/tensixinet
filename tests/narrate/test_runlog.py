"""RunLog: JSONL for programs, a wrapped left-bar text view for people."""
import io
import json

from tensixinet.narrate.runlog import RunLog


# --- RunLog --------------------------------------------------------------------

def make_log(clock):
    stream = io.StringIO()
    jsonl = io.StringIO()
    return RunLog(jsonl=jsonl, stream=stream, clock=clock), jsonl, stream


def test_runlog_writes_one_json_object_per_line_with_elapsed_seconds():
    ticks = iter([100.0, 100.0, 103.5])
    log, jsonl, _ = make_log(lambda: next(ticks))  # first tick = start
    log.event("discover", what="found 4 chips", why="tt-smi -s lists them")
    log.event("render", what="rendering", why="because")
    lines = [json.loads(l) for l in jsonl.getvalue().splitlines()]
    assert [l["event"] for l in lines] == ["discover", "render"]
    assert lines[0]["t"] == 0.0 and lines[1]["t"] == 3.5
    assert lines[0]["what"] == "found 4 chips" and lines[0]["why"] == "tt-smi -s lists them"


def test_runlog_keeps_extra_fields_for_machine_readers():
    log, jsonl, _ = make_log(lambda: 0.0)
    log.event("chip", what="w", why="y", chip=2, aiclk=1350.0)
    rec = json.loads(jsonl.getvalue())
    assert rec["chip"] == 2 and rec["aiclk"] == 1350.0


def test_runlog_human_lines_show_what_and_why_with_no_right_border():
    log, _, stream = make_log(lambda: 0.0)
    log.event("discover", what="found 4 chips", why="tt-smi -s lists them")
    out = stream.getvalue()
    assert "found 4 chips" in out and "tt-smi -s lists them" in out
    assert out.lstrip().startswith("║")  # left bar only, per house style
    assert not any(line.rstrip().endswith(("║", "│", "┃")) for line in out.splitlines())


# --- wrapping to the pane width -------------------------------------------------

LONG_WHY = "each DDR field adds a polyrhythm lane of 3, 4, 5 or 7 pulses per bar; status 0 mutes it"


def wrapped_log(width):
    stream = io.StringIO()
    log = RunLog(jsonl=io.StringIO(), stream=stream, clock=lambda: 0.0, width=width)
    log.event("chip", what="DDR status (1, 1, 1, 1) -> lanes (3, 4, 5, 7)", why=LONG_WHY)
    return stream.getvalue()


def test_wrapped_lines_fit_the_width_and_keep_the_left_bar():
    lines = wrapped_log(50).splitlines()
    assert len(lines) > 2  # it really did wrap
    assert all(l.startswith("║") and len(l) <= 50 for l in lines)


def test_wrapping_loses_no_words_and_continuation_lines_are_indented_under_the_text():
    out = wrapped_log(50)
    words = [w for l in out.splitlines() for w in l.lstrip("║").split()]
    assert " ".join(words).endswith(LONG_WHY)
    cont = out.splitlines()[-1]
    assert cont.startswith("║" + " " * 15) and cont[16] != " "  # hanging indent past "why: "


def test_no_width_means_no_wrapping():
    assert len(wrapped_log(None).splitlines()) == 2


def test_wrapping_never_touches_the_jsonl():
    jsonl = io.StringIO()
    log = RunLog(jsonl=jsonl, stream=io.StringIO(), clock=lambda: 0.0, width=30)
    log.event("chip", what="w", why=LONG_WHY)
    assert json.loads(jsonl.getvalue())["why"] == LONG_WHY


# --- repeated reasons are shown once on screen, kept every time in the JSONL ----

def dedupe_log():
    jsonl, stream = io.StringIO(), io.StringIO()
    return RunLog(jsonl=jsonl, stream=stream, clock=lambda: 0.0, dedupe_why=True), jsonl, stream


def test_a_repeated_why_is_printed_once_but_every_what_is_printed():
    log, _, stream = dedupe_log()
    log.event("chip", what="chip 0 power 20 W -> volume 0.32", why="0-200 W maps onto volume 0.3-1.0", chip=0)
    log.event("chip", what="chip 1 power 44 W -> volume 0.36", why="0-200 W maps onto volume 0.3-1.0", chip=1)
    out = stream.getvalue()
    assert out.count("0-200 W maps onto volume 0.3-1.0") == 1
    assert "chip 0 power 20 W" in out and "chip 1 power 44 W" in out


def test_dedupe_never_drops_a_why_from_the_jsonl():
    log, jsonl, _ = dedupe_log()
    for i in range(3):
        log.event("chip", what=f"w{i}", why="same reason", chip=i)
    assert [json.loads(l)["why"] for l in jsonl.getvalue().splitlines()] == ["same reason"] * 3


def test_dedupe_is_off_by_default():
    log, _, stream = make_log(lambda: 0.0)
    log.event("a", what="w1", why="same")
    log.event("b", what="w2", why="same")
    assert stream.getvalue().count("same") == 2
