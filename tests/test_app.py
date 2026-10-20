# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The tt-discolike manifest and the Gradio app that it launches. The page itself is checked by hand in a browser; what is here is what a
test can say without one."""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _manifest():
    fields = {}
    for line in (ROOT / ".disco" / "app.yaml").read_text().splitlines():
        key, _, value = line.partition(":")
        fields[key.strip()] = value.strip()
    return fields


def test_the_discolike_manifest_names_the_app_the_way_discolike_requires_and_launches_a_file_that_exists():
    m = _manifest()
    assert re.fullmatch(r"[A-Za-z0-9_.-]+", m["name"]) and m["description"] and 1024 < int(m["port"]) < 65536
    launch = m["launch"].split()
    assert launch[0] == ".venv-app/bin/python" and (ROOT / launch[1]).is_file()
    assert "chips" not in m  # it reads the chips with tt-smi and needs no lease


def test_the_apps_port_in_the_manifest_is_the_port_the_app_defaults_to():
    source = (ROOT / "app" / "gradio_app.py").read_text()
    assert f'"PORT", "{_manifest()["port"]}"' in source


def test_the_app_keeps_its_virtual_environment_out_of_git():
    assert ".venv-app/" in (ROOT / ".gitignore").read_text().splitlines()


def test_the_app_builds_its_page_and_every_choice_it_offers_is_one_the_forms_accept():
    pytest.importorskip("gradio", exc_type=ImportError, reason="gradio is installed in .venv-app, and the TT venv's copy does not import")
    import sys
    sys.path.insert(0, str(ROOT / "app"))
    try:
        import gradio_app
    finally:
        sys.path.pop(0)
    demo = gradio_app.build()
    assert demo is not None
    from tensixinet.ui.forms import DRAW, SOURCES, new_session, patch_choices, style_choices
    for name in patch_choices(gradio_app.PATCH_DIRS):
        for style in style_choices():
            new_session(SOURCES[0], style, 1, 4, 0, name, gradio_app.PATCH_DIRS, lambda s: None)


# --- Play, Stop and a change of setting ---------------------------------------------------------------------------------------------

def _app():
    pytest.importorskip("gradio", exc_type=ImportError, reason="gradio is installed in .venv-app, and the TT venv's copy does not import")
    import sys
    sys.path.insert(0, str(ROOT / "app"))
    try:
        import gradio_app
    finally:
        sys.path.pop(0)
    gradio_app.player.stop()
    return gradio_app


class _Session:
    """Stands in for ui.session.Session: hands out one chunk of audio per call, until stopped."""
    sample_rate = 8000

    def __init__(self):
        import numpy as np
        self.chunk, self.started, self.stopped = np.zeros(80, dtype=float), False, 0

    def start(self):
        self.started = True

    def stop(self):
        self.stopped += 1

    def next_audio(self, timeout=0.0):
        return None if self.stopped else self.chunk

    def snapshot(self):
        from tensixinet.ui.session import Snapshot
        return Snapshot(None, (), (), -1, self.started and not self.stopped, None)


class _Patch:
    style, seed = "enya", 3


def _take(stream, n):
    """At most `n` steps of a stream. A stream that wrongly never ends then fails the test it is in instead of hanging the run."""
    import itertools
    return list(itertools.islice(stream, n))


def _args(**over):
    base = dict(source="synthetic chips", style="plain draw", seed=4, chips=4, load_s=0, patch_name="draw from the seed and the style",
                kernels="CPU only", stereo=True, evict=True, cables=True)
    base.update(over)
    return tuple(base.values())


def _fake_new_session(app, monkeypatch, made):
    def new_session(source, style, seed, chips, load_s, patch_name, dirs, sleep, **kw):
        s = _Session()
        made.append((s, dict(source=source, style=style, seed=seed, chips=chips, **kw)))
        return s, _Patch()
    monkeypatch.setattr(app, "new_session", new_session)


def test_play_empties_the_page_first_so_nothing_of_the_old_session_is_taken_for_the_new_one(monkeypatch):
    app, made = _app(), []
    _fake_new_session(app, monkeypatch, made)
    g = app.play(*_args())
    audio, chart, log, status = next(g)
    assert audio is None and chart == app.STARTING and status == "starting" and "starting" in chart
    assert not made   # nothing has been built yet: the page cleared before the slow part
    audio, chart, log, status = next(g)
    assert made[0][0].started and status.startswith("playing enya (seed 3)")
    g.close()
    assert made[0][0].stopped >= 1


def test_a_change_while_playing_plays_again_with_the_change_and_ends_the_old_session(monkeypatch):
    app, made = _app(), []
    _fake_new_session(app, monkeypatch, made)
    g1 = app.play(*_args(style="enya"))
    next(g1), next(g1)
    g2 = app.restart(*_args(style="blues"))
    outs = [next(g2), next(g2)]
    assert made[0][0].stopped >= 1 and made[1][1]["style"] == "blues" and made[1][0].started
    assert outs[0][0] is None   # the audio player is emptied before the new session
    assert _take(g1, 3) == []   # the old stream says nothing more: the page is the new session's
    g2.close()


def test_a_change_with_nothing_playing_leaves_the_page_alone(monkeypatch):
    app, made = _app(), []   # first: it skips where gradio does not import
    import gradio as gr
    _fake_new_session(app, monkeypatch, made)
    out = _take(app.restart(*_args(style="blues")), 3)
    assert len(out) == 1 and all(isinstance(o, type(gr.skip())) or o == gr.skip() for o in out[0]) and not made


def test_a_change_that_leaves_the_settings_as_they_were_does_not_restart(monkeypatch):
    app, made = _app(), []
    _fake_new_session(app, monkeypatch, made)
    g = app.play(*_args())
    next(g), next(g)
    assert len(_take(app.restart(*_args()), 3)) == 1   # a field lost focus and nothing changed
    assert len(made) == 1 and not made[0][0].stopped
    g.close()


def test_stop_empties_the_audio_player_and_a_change_afterwards_does_not_start_anything(monkeypatch):
    app, made = _app(), []
    _fake_new_session(app, monkeypatch, made)
    g = app.play(*_args())
    next(g), next(g)
    audio, chart, status = app.stop()
    assert audio is None and status == "stopped" and chart == app.STOPPED and made[0][0].stopped >= 1
    assert len(_take(app.restart(*_args(style="blues")), 3)) == 1
    assert len(made) == 1
    last = _take(g, 3)
    assert len(last) == 1 and last[0][0] is None and last[0][1] == app.STOPPED and last[0][3] == "stopped"   # it finishes the page itself


def test_a_stop_while_the_session_is_still_being_built_means_it_never_starts(monkeypatch):
    app, built = _app(), []

    def slow_new_session(*a, **kw):
        app.stop()   # the user presses Stop while the simulator is still importing torch
        s = _Session()
        built.append(s)
        return s, _Patch()
    monkeypatch.setattr(app, "new_session", slow_new_session)
    outs = _take(app.play(*_args()), 3)
    assert [o[1] for o in outs] == [app.STARTING, app.STOPPED]   # it cleared the page, then ended it as stopped
    assert built and not built[0].started


def test_the_page_wires_every_setting_to_restart():
    app = _app()
    cfg = app.build().config
    deps = cfg["dependencies"]
    streams = [d for d in deps if d["api_name"] == "play" or d["api_name"].startswith("restart")]
    assert len(streams) == 12                                  # Play, plus a change of each of ten settings (the seed on blur and on Enter)
    assert not any(d["cancels"] for d in deps)   # a cancel cannot recall a frame already sent; each stream ends the page itself instead
    kinds = sorted(t[1] for d in streams for t in d["targets"])
    assert kinds.count("input") == 7 and kinds.count("release") == 2 and kinds.count("blur") == 1 and kinds.count("submit") == 1
    assert kinds.count("click") == 1
    assert all(d["trigger_mode"] == "multiple" for d in streams)   # the default holds a second change back while a stream is running


def test_a_stream_superseded_while_it_waits_for_audio_does_not_yield_that_audio(monkeypatch):
    app, made = _app(), []
    _fake_new_session(app, monkeypatch, made)
    g = app.play(*_args())
    next(g), next(g)
    session = made[0][0]
    session.next_audio = lambda timeout=0.0: (app.player.begin("a newer play"), session.chunk)[1]   # a newer Play arrives during the wait
    assert _take(g, 3) == []


def test_a_stream_superseded_between_steps_asks_for_no_more_audio(monkeypatch):
    app, made = _app(), []
    _fake_new_session(app, monkeypatch, made)
    g = app.play(*_args())
    next(g), next(g)
    session, asked = made[0][0], []
    session.next_audio = lambda timeout=0.0: asked.append(1) or session.chunk
    app.player.begin("a newer play")
    assert _take(g, 3) == [] and asked == []


def test_a_stop_between_a_changes_claim_and_its_build_is_not_undone_by_the_restart(monkeypatch):
    app, made = _app(), []
    _fake_new_session(app, monkeypatch, made)
    g1 = app.play(*_args(style="enya"))
    next(g1), next(g1)
    g2 = app.restart(*_args(style="blues"))
    assert next(g2)[1] == app.STARTING          # the change has claimed the page and shown that it is starting
    app.stop()                                  # and now the user presses Stop
    assert _take(g2, 3) == [app.STOPPED_FRAME] and not app.player.active   # the stopped page is the last thing it says
    assert all(not s.started for s, kw in made if kw["style"] == "blues")


def test_a_stop_while_the_stream_waits_for_audio_ends_the_page_as_stopped_without_that_audio(monkeypatch):
    app, made = _app(), []
    _fake_new_session(app, monkeypatch, made)
    g = app.play(*_args())
    next(g), next(g)
    session = made[0][0]
    session.next_audio = lambda timeout=0.0: (app.player.stop(), session.chunk)[1]   # Stop is pressed during the wait
    assert _take(g, 3) == [app.STOPPED_FRAME]


def test_the_page_cuts_the_browsers_buffered_audio_on_stop_play_and_a_real_change_before_the_server_answers():
    app = _app()
    deps = app.build().config["dependencies"]
    by_name = {d["api_name"]: d for d in deps}
    assert by_name["stop"]["js"] and "pause()" in by_name["stop"]["js"]
    assert "pause()" in by_name["play"]["js"]
    restarts = [d for d in deps if d["api_name"].startswith("restart")]
    assert len(restarts) == 11 and all("pause()" in d["js"] and "tensixinetPlaying" in d["js"] for d in restarts)
    assert "!==" in app.JS_CHANGE   # only a change from what is playing silences it: tabbing out of a field must not cut the sound


def test_the_app_asks_for_the_lead_so_the_browsers_player_has_audio_in_hand(monkeypatch):
    from tensixinet.ui.session import PREROLL_S
    app, made = _app(), []
    _fake_new_session(app, monkeypatch, made)
    g = app.play(*_args())
    next(g), next(g)
    g.close()
    assert made[0][1]["lead_s"] == PREROLL_S and PREROLL_S >= 8


def _starting_snapshot(*states):
    from tensixinet.ui.progress import Progress, Step
    from tensixinet.ui.session import Snapshot
    labels = ["start the TT-Lang simulator", "render the first segment", "fill the buffer"]
    steps = tuple(Step(l, st, 3.0) for l, st in zip(labels, states))
    return Snapshot(None, (), (), -1, True, None, Progress(steps, None, 9.0))


def test_before_the_first_segment_the_page_shows_the_steps_of_the_start_and_names_the_active_one(monkeypatch):
    app, made = _app(), []
    _fake_new_session(app, monkeypatch, made)
    g = app.play(*_args())
    next(g), next(g)   # the cleared page, then the first step of the loop (the session is built by now)
    made[0][0].snapshot = lambda: _starting_snapshot("done", "active", "todo")
    audio, chart, log, status = next(g)
    assert "start the TT-Lang simulator" in chart and "fill the buffer" in chart and 'data-state="active"' in chart
    assert status == "starting: render the first segment, 3 s"
    g.close()


def test_once_the_start_is_over_the_status_says_what_is_playing_and_the_chart_is_left_alone(monkeypatch):
    app, made = _app(), []   # first: it skips where gradio does not import
    import gradio as gr
    _fake_new_session(app, monkeypatch, made)
    g = app.play(*_args())
    next(g)
    audio, chart, log, status = next(g)   # the stand-in session has no progress and no state
    assert status.startswith("playing enya") and chart == gr.skip()
    g.close()
