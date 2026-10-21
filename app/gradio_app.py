# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""What is playing, drawn as it plays: a live flow chart of the chips, the devices dealt to them, the cables between the devices and the
stereo field, with the run log beside it. Pick a source (synthetic or the real chips), a style or a saved patch, and press Play.

This is not a synth panel. Nothing here changes the sound while it plays; it shows how the sound is being made. Everything it draws comes from
`tensixinet.ui` (flow state, chart, log, session), which the tests cover; this file only lays the page out and moves data between them.

    .venv-app/bin/python app/gradio_app.py        # then open http://localhost:7880

It reads the real chips with tt-smi (read-only, no device open), so it needs no gozer lease.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import gradio as gr
import numpy as np

from tensixinet.music.patch import save_patch
from tensixinet.ui.chart import flow_svg
from tensixinet.ui.forms import DRAW, SOURCES, kernel_choices, new_session, patch_choices, source_choices, style_choices
from tensixinet.ui.logview import log_html
from tensixinet.ui.player import Busy, Visitors
from tensixinet.ui.progress import ACTIVE, progress_html
from tensixinet.ui.session import PREROLL_S

ROOT = Path(__file__).resolve().parent.parent
PATCH_DIRS = [ROOT / "demo", ROOT / "patches"]  # patches/ is optional: drop a *.patch.json there to see it in the list
PORT = int(os.environ.get("PORT", "7880"))
INTRO = ("Chip telemetry played as a band of Ciat-Lonbarde-style devices. The chart follows the segment that is playing: each chip's readings, "
         "the devices it plays, the patch cables bending one device with another, and where every voice sits in the stereo field. "
         "Wear headphones.")
WAITING = '<div style="padding:40px;color:#7d9a98;font:14px ui-monospace,monospace">press Play: the chart draws what is playing as it plays</div>'

STARTING = '<div style="padding:40px;color:#7d9a98;font:14px ui-monospace,monospace">starting: ending what was playing and building the new session</div>'
STOPPED = '<div style="padding:40px;color:#7d9a98;font:14px ui-monospace,monospace">stopped: press Play, or change a setting while it plays and it restarts with the change</div>'

# The browser plays on what it had already buffered after the server has stopped sending, for about ten seconds. These run in the page before
# the server hears anything: they cut the audio on Stop, on Play, and on a change that alters the settings of what is playing (the server
# still decides whether to restart; if it does not, the stream goes on and the audio comes back with its next segment).
SILENCE = "document.querySelectorAll('audio').forEach(a => { a.pause(); });"
JS_PLAY = "(...a) => { %s window.tensixinetPlaying = JSON.stringify(a); return a; }" % SILENCE
JS_CHANGE = ("(...a) => { if (window.tensixinetPlaying && window.tensixinetPlaying !== JSON.stringify(a)) "
             "{ %s window.tensixinetPlaying = JSON.stringify(a); } return a; }" % SILENCE)
JS_STOP = "() => { %s window.tensixinetPlaying = null; }" % SILENCE

DEFAULT_MAX_PLAYERS = 3  # a small machine (a Space's two CPUs) can render a few sessions at once
LOCAL = "local"  # the visitor when there is no request (tests, or a call that did not come through the page)


def _limit_from(env) -> int:
    """How many visitors may play at once: TENSIXINET_MAX_PLAYERS, or the default when it is missing, not a number or below one."""
    try:
        limit = int(env.get("TENSIXINET_MAX_PLAYERS", DEFAULT_MAX_PLAYERS))
    except ValueError:
        return DEFAULT_MAX_PLAYERS
    return limit if limit >= 1 else DEFAULT_MAX_PLAYERS


MAX_PLAYERS = _limit_from(os.environ)
visitors = Visitors(MAX_PLAYERS)  # which session each visitor's page shows (see ui/player.py); their audio, chart and log all belong to it


def _key(request) -> str:
    """The visitor: Gradio gives every open page its own session hash."""
    return getattr(request, "session_hash", None) or LOCAL


def _config(source, style, seed, chips, load_s, patch_name, kernels, stereo, evict, cables):
    return (source, style, int(seed or 0), int(chips), float(load_s), patch_name, kernels, bool(stereo), bool(evict), bool(cables))


STOPPED_FRAME = (None, STOPPED, gr.skip(), "stopped")


def _ended(player, token):
    """What a stream that is no longer the current one says last. After a Stop it puts the page in the stopped state itself: a frame it sent
    just before can reach the browser after Stop's own result, and Gradio's cancel cannot recall it. After a newer Play it says nothing."""
    if not player.active and not player.is_current(token):
        yield STOPPED_FRAME


def _chart(snap):
    """The flow chart of the segment that is playing; before the first one, the steps of the start (so a slow start is not a blank page)."""
    if snap.state is not None:
        return flow_svg(snap.state)
    return progress_html(snap.progress) if snap.progress is not None else gr.skip()


def _status(snap, playing):
    """The status line: the step the start is on while it starts, then what is playing."""
    if snap.state is None and snap.progress is not None:
        step = next(s for s in snap.progress.steps if s.state == ACTIVE)
        return f"starting: {step.label}, {step.seconds:.0f} s"
    return playing


def _stream(key, config, only_if_changed):
    """Claim the page for `config`, show that it is starting, end the session that was playing, then play. The claim comes first and is
    one step with the check that a change really is one: a Stop after it is honoured, because everything below asks the player if it is
    still current. Starting clears the page, so nothing of the old session is mistaken for the new one."""
    source, style, seed, chips, load_s, patch_name, kernels, stereo, evict, cables = config
    try:
        player, token, old = visitors.claim(key, config, only_if_changed)
    except Busy as e:
        yield gr.skip(), WAITING, gr.skip(), (f"busy: {e.playing} of {e.limit} places are taken by other listeners. "
                                              "Try again in a minute, or run it yourself (see the README)")
        return
    if token is None:
        yield gr.skip(), gr.skip(), gr.skip(), gr.skip()
        return
    yield None, STARTING, log_html([]), "starting"  # None empties the audio player, which would otherwise play on what it had buffered
    if old is not None:
        old.stop()  # can take seconds: it waits for a segment that is being rendered
    try:
        session, patch = new_session(source, style, seed, chips, load_s, patch_name, PATCH_DIRS, time.sleep,
                                     stereo=stereo, evict=evict, cables=cables, kernels=kernels, lead_s=PREROLL_S)
    except ValueError as e:
        if player.is_current(token):
            player.finish(token)
            yield gr.skip(), WAITING, log_html([]), f"could not start: {e}"
        return
    if not player.attach(token, session):  # a Stop or a newer Play came while it was being built
        yield from _ended(player, token)
        return
    session.start()
    status = f"playing {patch.style or 'a plain patch'} (seed {patch.seed}) from {source}"
    try:
        while player.is_current(token):
            chunk = session.next_audio(timeout=0.5)
            if not player.is_current(token):
                yield from _ended(player, token)  # the page belongs to a newer session or to Stop
                return
            snap = session.snapshot()
            if snap.error:
                player.finish(token)
                yield gr.skip(), flow_svg(snap.state) if snap.state else WAITING, log_html(snap.records), f"stopped: {snap.error}"
                return
            audio = gr.skip() if chunk is None else (session.sample_rate, (np.clip(chunk, -1, 1) * 32767).astype(np.int16))
            yield audio, _chart(snap), log_html(snap.records), _status(snap, status)
            if not snap.running and chunk is None:
                player.finish(token)
                return
        yield from _ended(player, token)  # the loop ended because the page moved on
    finally:
        session.stop()


# The handlers name every setting so that Gradio finds the `request` parameter (it does not look for one after a *args), which is how each
# visitor is told from the next.
def play(source, style, seed, chips, load_s, patch_name, kernels, stereo, evict, cables, request: gr.Request = None):
    """Play with the page's settings, ending whatever this visitor was playing."""
    yield from _stream(_key(request), _config(source, style, seed, chips, load_s, patch_name, kernels, stereo, evict, cables), only_if_changed=False)


def restart(source, style, seed, chips, load_s, patch_name, kernels, stereo, evict, cables, request: gr.Request = None):
    """A setting changed. While this visitor is playing and the settings differ, play again with them; otherwise leave the page as it is until Play."""
    yield from _stream(_key(request), _config(source, style, seed, chips, load_s, patch_name, kernels, stereo, evict, cables), only_if_changed=True)


def stop(request: gr.Request = None):
    """Stop this visitor's session, and empty their audio player: the browser would go on playing what it had already buffered."""
    visitors.player(_key(request)).stop()
    return None, STOPPED, "stopped"


def leave(request: gr.Request = None):
    """The visitor closed the page or lost the connection: end their session and give the place back."""
    visitors.drop(_key(request))


def build() -> gr.Blocks:
    with gr.Blocks(title="tensixinet: what is playing") as demo:
        gr.Markdown(f"## tensixinet: what is playing\n{INTRO}")
        with gr.Row():
            sources, kernel_options = source_choices(), kernel_choices()  # the real chips need tt-smi, the simulator needs tt-lang-sim
            source = gr.Radio(sources, value=sources[0], label="Source")
            style = gr.Dropdown(style_choices(), value=style_choices()[0], label="Style")
            patch_name = gr.Dropdown(patch_choices(PATCH_DIRS), value=DRAW, label="Patch (a saved one ignores the style and the seed)")
        with gr.Row():
            seed = gr.Number(value=4, precision=0, label="Seed")
            chips = gr.Slider(1, 8, value=4, step=1, label="Synthetic chips")
            load_s = gr.Slider(0, 120, value=60, step=5, label="Load cycle, seconds (0 = none; a chip that unloads evicts)")
        with gr.Row():
            stereo = gr.Checkbox(True, label="Stereo")
            kernels = gr.Radio(kernel_options, value=kernel_options[0], label="Where the gong, shimmer, drone ripple, AVDog, delay mix and mixer run")
            evict = gr.Checkbox(True, label="Eviction")
            cables = gr.Checkbox(True, label="Patch cables")
            play_btn = gr.Button("Play", variant="primary")
            stop_btn = gr.Button("Stop")
        status = gr.Markdown("")
        with gr.Row():
            with gr.Column(scale=3):
                chart = gr.HTML(WAITING, label="Flow")
                audio = gr.Audio(streaming=True, autoplay=True, label="What the chips play", interactive=False)
            with gr.Column(scale=2):
                log = gr.HTML(log_html([]), label="Log")
        inputs = [source, style, seed, chips, load_s, patch_name, kernels, stereo, evict, cables]
        outputs = [audio, chart, log, status]
        # trigger_mode "multiple": Gradio holds an input event back while the last one is still running, and these never finish (they stream),
        # so a second change would wait for ever and the page would keep the first.
        # No concurrency limit: a new Play must be able to start and end the one that is running, and every stream ends when it is no longer
        # the current one. Stop ends them all by moving the player on. A change of setting while playing plays again with it (Player.should_restart decides).
        streams = [play_btn.click(play, inputs, outputs, concurrency_limit=None, trigger_mode="multiple", js=JS_PLAY)]
        for control, event in ((source, "input"), (style, "input"), (patch_name, "input"), (kernels, "input"), (stereo, "input"),
                               (evict, "input"), (cables, "input"), (chips, "release"), (load_s, "release"), (seed, "blur"), (seed, "submit")):
            streams.append(getattr(control, event)(restart, inputs, outputs, concurrency_limit=None, trigger_mode="multiple", js=JS_CHANGE))
        demo.unload(leave)
        stop_btn.click(stop, None, [audio, chart, status], js=JS_STOP)  # every stream ends itself within half a second of the player moving on
    return demo


if __name__ == "__main__":
    build().queue().launch(server_name="0.0.0.0", server_port=PORT)
