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
    gradio_app.visitors.reset()
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
    session.next_audio = lambda timeout=0.0: (app.visitors.player(app.LOCAL).begin("a newer play"), session.chunk)[1]   # a newer Play arrives during the wait
    assert _take(g, 3) == []


def test_a_stream_superseded_between_steps_asks_for_no_more_audio(monkeypatch):
    app, made = _app(), []
    _fake_new_session(app, monkeypatch, made)
    g = app.play(*_args())
    next(g), next(g)
    session, asked = made[0][0], []
    session.next_audio = lambda timeout=0.0: asked.append(1) or session.chunk
    app.visitors.player(app.LOCAL).begin("a newer play")
    assert _take(g, 3) == [] and asked == []


def test_a_stop_between_a_changes_claim_and_its_build_is_not_undone_by_the_restart(monkeypatch):
    app, made = _app(), []
    _fake_new_session(app, monkeypatch, made)
    g1 = app.play(*_args(style="enya"))
    next(g1), next(g1)
    g2 = app.restart(*_args(style="blues"))
    assert next(g2)[1] == app.STARTING          # the change has claimed the page and shown that it is starting
    app.stop()                                  # and now the user presses Stop
    assert _take(g2, 3) == [app.STOPPED_FRAME] and not app.visitors.player(app.LOCAL).active   # the stopped page is the last thing it says
    assert all(not s.started for s, kw in made if kw["style"] == "blues")


def test_a_stop_while_the_stream_waits_for_audio_ends_the_page_as_stopped_without_that_audio(monkeypatch):
    app, made = _app(), []
    _fake_new_session(app, monkeypatch, made)
    g = app.play(*_args())
    next(g), next(g)
    session = made[0][0]
    session.next_audio = lambda timeout=0.0: (app.visitors.player(app.LOCAL).stop(), session.chunk)[1]   # Stop is pressed during the wait
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


# --- one session for each visitor ---------------------------------------------------------------------------------------------------

class _Req:
    """Stands in for gradio's Request: all the app reads is the visitor's session hash."""
    def __init__(self, who):
        self.session_hash = who


def test_two_visitors_play_at_once_and_one_stopping_does_not_stop_the_other(monkeypatch):
    app, made = _app(), []
    _fake_new_session(app, monkeypatch, made)
    a, b = _Req("a"), _Req("b")
    ga, gb = app.play(*_args(style="enya"), request=a), app.play(*_args(style="blues"), request=b)
    next(ga), next(ga), next(gb), next(gb)
    assert len(made) == 2 and made[0][1]["style"] == "enya" and made[1][1]["style"] == "blues"
    app.stop(request=a)
    assert made[0][0].stopped >= 1 and made[1][0].stopped == 0
    audio, chart, log, status = next(gb)
    assert audio is not None and status.startswith("playing")         # b goes on, untouched
    assert _take(ga, 3) == [app.STOPPED_FRAME]                              # a's own page says stopped
    gb.close()


def test_a_new_play_by_one_visitor_does_not_end_anothers_session(monkeypatch):
    app, made = _app(), []
    _fake_new_session(app, monkeypatch, made)
    ga = app.play(*_args(), request=_Req("a"))
    next(ga), next(ga)
    gb = app.play(*_args(style="enya"), request=_Req("b"))
    next(gb), next(gb)
    assert made[0][0].stopped == 0 and next(ga)[0] is not None
    ga.close(), gb.close()


def test_a_change_of_setting_restarts_only_that_visitors_session(monkeypatch):
    app, made = _app(), []
    _fake_new_session(app, monkeypatch, made)
    ga, gb = app.play(*_args(style="enya"), request=_Req("a")), app.play(*_args(style="blues"), request=_Req("b"))
    next(ga), next(ga), next(gb), next(gb)
    g2 = app.restart(*_args(style="enya", seed=9), request=_Req("a"))
    next(g2), next(g2)
    assert made[0][0].stopped >= 1 and made[1][0].stopped == 0 and len(made) == 3
    assert made[2][1]["seed"] == 9
    g2.close(), gb.close()


def test_a_visitor_over_the_limit_is_told_the_page_is_busy_and_nothing_is_built_for_them(monkeypatch):
    app, made = _app(), []
    _fake_new_session(app, monkeypatch, made)
    from tensixinet.ui.player import Visitors
    monkeypatch.setattr(app, "visitors", Visitors(limit=1))
    ga = app.play(*_args(), request=_Req("a"))
    next(ga), next(ga)
    out = _take(app.play(*_args(), request=_Req("b")), 3)
    assert len(out) == 1 and "busy" in out[0][3] and "1" in out[0][3] and out[0][1] == app.WAITING and len(made) == 1
    assert next(ga)[0] is not None                                           # the first visitor is untouched
    app.stop(request=_Req("a"))
    g = app.play(*_args(), request=_Req("b"))                                # and when they stop, the place is free
    next(g), next(g)
    assert len(made) == 2
    g.close()


def test_a_visitor_who_leaves_the_page_has_their_session_ended_and_is_forgotten(monkeypatch):
    app, made = _app(), []
    _fake_new_session(app, monkeypatch, made)
    ga = app.play(*_args(), request=_Req("a"))
    next(ga), next(ga)
    app.leave(_Req("a"))
    assert made[0][0].stopped >= 1 and len(app.visitors) == 0
    app.leave(_Req("a"))                                                     # leaving twice is harmless
    ga.close()


def test_with_no_request_at_all_the_app_still_works_as_one_local_visitor(monkeypatch):
    app, made = _app(), []
    _fake_new_session(app, monkeypatch, made)
    g = app.play(*_args())
    next(g), next(g)
    assert app.visitors.player(app.LOCAL).active
    app.stop()
    assert not app.visitors.player(app.LOCAL).active
    g.close()


def test_the_limit_comes_from_the_environment_with_a_default_and_the_page_unloads_a_visitor_when_they_go(monkeypatch):
    app = _app()
    assert app.MAX_PLAYERS >= 1 and app.visitors.limit == app.MAX_PLAYERS
    assert app._limit_from({"TENSIXINET_MAX_PLAYERS": "7"}) == 7
    assert app._limit_from({}) == 3 and app._limit_from({"TENSIXINET_MAX_PLAYERS": "nonsense"}) == 3
    assert app._limit_from({"TENSIXINET_MAX_PLAYERS": "0"}) == 3
    demo = app.build()
    assert any(t[1] == "unload" for fn in demo.fns.values() for t in fn.targets)


def test_gradio_hands_each_handler_the_request_so_visitors_can_be_told_apart():
    """The unit tests above pass `request=` themselves. This asks Gradio's own argument handling whether it would: a handler that names no
    `request` parameter, or hides it behind *args, gets every visitor as the same one."""
    app = _app()
    from gradio.helpers import special_args
    values = list(_args())
    sentinel = _Req("a visitor")
    for handler in (app.play, app.restart):
        filled = special_args(handler, list(values), sentinel)[0]
        assert filled[-1] is sentinel and filled[:-1] == values
    for handler in (app.stop, app.leave):
        assert special_args(handler, [], sentinel)[0] == [sentinel]


# --- the Play tab -------------------------------------------------------------------------------------------------------------------

class _Instrument:
    """Stands in for ui.instrument.InstrumentSession."""
    sample_rate = 8000

    def __init__(self, knobs, **settings):
        import numpy as np
        self.knobs, self.settings = knobs, settings
        self.chunk, self.started, self.stopped = np.zeros(80, dtype=float), False, 0

    def start(self):
        self.started = True

    def stop(self):
        self.stopped += 1

    def next_audio(self, timeout=0.0):
        return None if self.stopped else self.chunk

    def snapshot(self):
        from tensixinet.music.live import Tracker
        from tensixinet.ui.instrument import InstrumentSnapshot
        from tensixinet.ui.knobs import Knobs, to_telemetry
        return InstrumentSnapshot(self.started and not self.stopped, None, 0, tuple(to_telemetry(Knobs(2), Tracker(), 0.0)), ("gong",), ())


def _fake_new_instrument(app, monkeypatch, made):
    def new_instrument(scope, style, seed, chips, stereo, cables, evict, knobs, **kw):
        s = _Instrument(knobs, scope=scope, style=style, seed=seed, chips=chips, stereo=stereo, cables=cables, evict=evict, **kw)
        made.append(s)
        return s, _Patch()
    monkeypatch.setattr(app, "new_instrument", new_instrument)


def _iargs(**over):
    base = dict(scope="system", style="plain draw", seed=4, chips=4, stereo=True, cables=True, evict=True)
    base.update(over)
    return tuple(base.values())


def test_the_page_has_two_tabs_named_what_is_playing_and_play():
    app = _app()
    tabs = [c["props"]["label"] for c in app.build().config["components"] if c["type"] == "tabitem"]
    assert tabs == ["What is playing", "Play"]


def test_play_instrument_clears_the_page_then_streams_audio_a_readout_and_a_status(monkeypatch):
    app, made = _app(), []
    _fake_new_instrument(app, monkeypatch, made)
    g = app.instrument_play(*_iargs(scope="device:gong", style="enya", seed=3, chips=2))
    audio, readout, status = next(g)
    assert audio is None and readout == app.READOUT_WAITING and status == "starting" and not made
    audio, readout, status = next(g)
    assert made[0].started and audio is not None and audio[0] == 8000 and audio[1].dtype.name == "int16"
    assert "chip 0" in readout and "gong" in readout and status.startswith("playing") and "device: gong" in status and "2 chips" in status
    assert (made[0].settings["scope"], made[0].settings["chips"], made[0].settings["style"]) == ("device:gong", 2, "enya")
    g.close()
    assert made[0].stopped >= 1


def test_the_instrument_is_given_the_apps_lead_and_real_sleep(monkeypatch):
    import time
    app, made = _app(), []
    _fake_new_instrument(app, monkeypatch, made)
    g = app.instrument_play(*_iargs())
    next(g), next(g)
    g.close()
    assert made[0].settings["lead_s"] == app.INSTRUMENT_LEAD_S and 1.0 <= app.INSTRUMENT_LEAD_S <= 6.0
    assert made[0].settings["sleep"] is time.sleep


def test_a_knob_set_before_play_is_in_the_knobs_the_session_gets(monkeypatch):
    app, made = _app(), []
    _fake_new_instrument(app, monkeypatch, made)
    app.knob_setter(0, "power")(120.0)
    g = app.instrument_play(*_iargs())
    next(g), next(g)
    assert made[0].knobs.get(0, "power") == 120.0
    g.close()


def test_a_knob_moved_while_playing_reaches_the_running_session_and_does_not_restart_it(monkeypatch):
    app, made = _app(), []
    _fake_new_instrument(app, monkeypatch, made)
    g = app.instrument_play(*_iargs())
    next(g), next(g)
    app.knob_setter(1, "temperature")(90.0)
    app.event_trigger(1, "memory_error")()
    assert made[0].knobs.get(1, "temperature") == 90.0 and made[0].knobs.take_events() == [(1, "memory_error")]
    assert len(_take(app.instrument_restart(*_iargs()), 3)) == 1  # a field that lost focus and changed nothing
    assert len(made) == 1 and not made[0].stopped
    g.close()


def test_a_change_of_scope_while_playing_restarts_with_the_new_scope_and_the_same_knobs(monkeypatch):
    app, made = _app(), []
    _fake_new_instrument(app, monkeypatch, made)
    g1 = app.instrument_play(*_iargs(scope="system"))
    next(g1), next(g1)
    app.knob_setter(0, "power")(111.0)
    g2 = app.instrument_restart(*_iargs(scope="module:melody"))
    outs = [next(g2), next(g2)]
    assert made[0].stopped >= 1 and made[1].settings["scope"] == "module:melody" and made[1].started and outs[0][0] is None
    assert made[1].knobs is made[0].knobs and made[1].knobs.get(0, "power") == 111.0
    assert _take(g1, 3) == []
    g2.close()


def test_a_change_with_nothing_playing_leaves_the_play_tab_alone(monkeypatch):
    app, made = _app(), []
    _fake_new_instrument(app, monkeypatch, made)
    out = _take(app.instrument_restart(*_iargs(scope="module:melody")), 3)
    assert len(out) == 1 and not made


def test_playing_in_the_play_tab_ends_what_the_same_visitor_played_in_the_first_tab(monkeypatch):
    app, watched, made = _app(), [], []
    _fake_new_session(app, monkeypatch, watched)
    _fake_new_instrument(app, monkeypatch, made)
    g1 = app.play(*_args())
    next(g1), next(g1)
    g2 = app.instrument_play(*_iargs())
    next(g2), next(g2)
    assert watched[0][0].stopped >= 1 and made[0].started
    said = _take(g1, 3)  # the first tab's stream says one more thing: the visitor is on the other tab now
    assert len(said) == 1 and said[0][0] is None and said[0][3] == "stopped: you started playing in the Play tab"
    g2.close()


def test_playing_in_the_first_tab_ends_the_play_tab_session(monkeypatch):
    app, watched, made = _app(), [], []
    _fake_new_session(app, monkeypatch, watched)
    _fake_new_instrument(app, monkeypatch, made)
    g2 = app.instrument_play(*_iargs())
    next(g2), next(g2)
    g1 = app.play(*_args())
    next(g1), next(g1)
    assert made[0].stopped >= 1 and watched[0][0].started
    said = _take(g2, 3)
    assert len(said) == 1 and said[0][0] is None and said[0][2] == "stopped: you started playing in the first tab"
    g1.close()


def test_a_setting_changed_in_the_first_tab_does_not_start_playing_there_while_the_play_tab_plays(monkeypatch):
    app, watched, made = _app(), [], []
    _fake_new_session(app, monkeypatch, watched)
    _fake_new_instrument(app, monkeypatch, made)
    g2 = app.instrument_play(*_iargs())
    next(g2), next(g2)
    assert len(_take(app.restart(*_args(style="blues")), 3)) == 1 and not watched and not made[0].stopped
    g2.close()


def test_and_a_setting_changed_in_the_play_tab_does_not_start_playing_there_while_the_first_tab_plays(monkeypatch):
    app, watched, made = _app(), [], []
    _fake_new_session(app, monkeypatch, watched)
    _fake_new_instrument(app, monkeypatch, made)
    g1 = app.play(*_args())
    next(g1), next(g1)
    assert len(_take(app.instrument_restart(*_iargs(scope="device:gong")), 3)) == 1 and not made and not watched[0][0].stopped
    g1.close()


def test_stop_empties_the_play_tab_and_a_change_afterwards_starts_nothing(monkeypatch):
    app, made = _app(), []
    _fake_new_instrument(app, monkeypatch, made)
    g = app.instrument_play(*_iargs())
    next(g), next(g)
    audio, readout, status = app.instrument_stop()
    assert audio is None and readout == app.READOUT_WAITING and status == "stopped" and made[0].stopped >= 1
    assert len(_take(app.instrument_restart(*_iargs(scope="device:gong")), 3)) == 1 and len(made) == 1
    last = _take(g, 3)
    assert len(last) == 1 and last[0][0] is None and last[0][2] == "stopped"


def test_a_stop_while_the_instrument_is_still_being_built_means_it_never_starts(monkeypatch):
    app, built = _app(), []

    def slow(scope, style, seed, chips, stereo, cables, evict, knobs, **kw):
        app.instrument_stop()
        s = _Instrument(knobs)
        built.append(s)
        return s, _Patch()
    monkeypatch.setattr(app, "new_instrument", slow)
    outs = _take(app.instrument_play(*_iargs()), 3)
    assert [o[2] for o in outs] == ["starting", "stopped"] and built and not built[0].started


def test_a_scope_that_cannot_be_played_says_why_and_leaves_the_visitor_idle(monkeypatch):
    app = _app()
    monkeypatch.setattr(app, "new_instrument", lambda *a, **k: (_ for _ in ()).throw(ValueError("unknown scope 'x'")))
    outs = _take(app.instrument_play(*_iargs(scope="x")), 4)
    assert outs[-1][2].startswith("could not start") and "unknown scope" in outs[-1][2]
    assert not app.visitors.player(app.LOCAL).active


def test_a_session_that_stops_with_an_error_says_so(monkeypatch):
    from tensixinet.ui.instrument import InstrumentSnapshot
    app, made = _app(), []
    _fake_new_instrument(app, monkeypatch, made)
    g = app.instrument_play(*_iargs())
    next(g), next(g)
    made[0].snapshot = lambda: InstrumentSnapshot(False, "RuntimeError: no sound today", 0, (), (), ())
    out = next(g)
    assert out[2] == "stopped: RuntimeError: no sound today" and not app.visitors.player(app.LOCAL).active


def test_two_visitors_have_their_own_knobs_and_sessions_and_a_third_is_told_the_places_are_taken(monkeypatch):
    app, made = _app(), []
    _fake_new_instrument(app, monkeypatch, made)
    from tensixinet.ui.player import Visitors
    monkeypatch.setattr(app, "visitors", Visitors(limit=2))
    a, b, c = _Req("a"), _Req("b"), _Req("c")
    app.knob_setter(0, "power")(50.0, request=a)
    app.knob_setter(0, "power")(140.0, request=b)
    ga, gb = app.instrument_play(*_iargs(), request=a), app.instrument_play(*_iargs(), request=b)
    next(ga), next(ga), next(gb), next(gb)
    assert made[0].knobs.get(0, "power") == 50.0 and made[1].knobs.get(0, "power") == 140.0 and made[0].knobs is not made[1].knobs
    app.knob_setter(0, "power")(60.0, request=a)
    assert made[0].knobs.get(0, "power") == 60.0 and made[1].knobs.get(0, "power") == 140.0
    out = _take(app.instrument_play(*_iargs(), request=c), 3)
    assert len(out) == 1 and "busy" in out[0][2] and len(made) == 2
    ga.close(), gb.close()


def test_a_visitor_who_leaves_loses_their_session_and_their_knobs(monkeypatch):
    app, made = _app(), []
    _fake_new_instrument(app, monkeypatch, made)
    app.knob_setter(0, "power")(50.0, request=_Req("a"))
    g = app.instrument_play(*_iargs(), request=_Req("a"))
    next(g), next(g)
    app.leave(_Req("a"))
    assert made[0].stopped >= 1 and len(app.visitors) == 0 and "a" not in app.knob_store
    app.leave(_Req("a"))  # twice is harmless
    g.close()


def test_load_chips_returns_a_status_then_the_value_of_every_slider(monkeypatch):
    from tensixinet.telemetry import mock_telemetry
    app = _app()
    seen = []
    monkeypatch.setattr(app, "read_chip_knobs", lambda knobs: (seen.append(knobs), "loaded 4 chips from tt-smi")[1])
    out = app.load_chips(request=_Req("a"))
    assert out[0] == "loaded 4 chips from tt-smi" and len(out) == 1 + 4 * len(app.KNOB_ORDER)
    knobs = app.knob_store["a"]
    assert seen == [knobs]
    knobs.set(2, "aiclk", 1000.0)
    again = app.load_chips(request=_Req("a"))
    index = 1 + 2 * len(app.KNOB_ORDER) + app.KNOB_ORDER.index("aiclk")
    assert again[index] == 1000.0  # the slider values come from the knobs, chip by chip, in KNOB_ORDER


def test_the_sliders_on_the_page_are_the_knobs_ranges_four_times_over():
    from tensixinet.ui.knobs import KNOB_RANGES
    app = _app()
    sliders = [(c["props"]["minimum"], c["props"]["maximum"], c["props"]["value"]) for c in app.build().config["components"]
               if c["type"] == "slider" and str(c["props"].get("label", "")).startswith("chip ")]
    expected = [KNOB_RANGES[name] for _ in range(4) for name in app.KNOB_ORDER]
    assert sliders == expected


def test_the_play_tab_wires_every_setting_every_knob_and_every_event_and_never_cancels():
    app = _app()
    deps = app.build().config["dependencies"]
    names = [d["api_name"] for d in deps]
    assert len(set(names)) == len(names)  # Gradio would rename a clash, and the old tests count by name
    play = [d for d in deps if d["api_name"] == "instrument_play" or d["api_name"].startswith("instrument_restart")]
    assert len(play) == 1 + 7 + 1  # Play, a change of each of seven settings, and the chip count's slider on release
    assert all(d["trigger_mode"] == "multiple" for d in play)
    assert len([n for n in names if n.startswith("knob_")]) == 4 * len(app.KNOB_ORDER)
    assert len([n for n in names if n.startswith("event_")]) == 4 * 3
    assert not any(d["cancels"] for d in deps)
    for d in play:
        assert "pause()" in d["js"]
    assert "pause()" in next(d for d in deps if d["api_name"] == "instrument_stop")["js"]


def test_gradio_hands_every_new_handler_the_request_so_visitors_can_be_told_apart():
    app = _app()
    from gradio.helpers import special_args
    sentinel = _Req("a visitor")
    values = list(_iargs())
    for handler in (app.instrument_play, app.instrument_restart):
        filled = special_args(handler, list(values), sentinel)[0]
        assert filled[-1] is sentinel and filled[:-1] == values
    for handler in (app.instrument_stop, app.load_chips):
        assert special_args(handler, [], sentinel)[0] == [sentinel]
    for handler in (app.knob_setter(0, "power"), app.event_trigger(0, "memory_error")):
        values = [1.0] if handler.__name__ == "set_knob" else []
        assert special_args(handler, list(values), sentinel)[0] == values + [sentinel]


def test_the_status_says_one_chip_in_the_singular(monkeypatch):
    app, made = _app(), []
    _fake_new_instrument(app, monkeypatch, made)
    g = app.instrument_play(*_iargs(chips=1))
    next(g)
    assert next(g)[2].endswith("on 1 chip")
    g.close()


def test_a_session_that_has_run_out_ends_the_stream_and_frees_the_place(monkeypatch):
    app, made = _app(), []
    _fake_new_instrument(app, monkeypatch, made)
    g = app.instrument_play(*_iargs())
    next(g), next(g)
    made[0].next_audio = lambda timeout=0.0: None
    made[0].snapshot = lambda: type("S", (), {"running": False, "error": None, "telemetry": (), "playing": (), "notes": ()})()
    assert _take(g, 3) != [] and not app.visitors.player(app.LOCAL).active
