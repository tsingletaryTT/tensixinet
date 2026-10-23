# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""What is playing, drawn as it plays: a live flow chart of the chips, the devices dealt to them, the cables between the devices and the
stereo field, with the run log beside it. Pick a source (synthetic or the real chips), a style or a saved patch, and press Play.

The first tab is not a synth panel: nothing there changes the sound while it plays, it shows how the sound is being made. The second tab, Play, is
the instrument: pick one device, a module or the whole system, move knobs that stand in for each chip's readings (temperature, power, aiclk, GDDR
spread, voltage), press buttons that inject a memory error, a thermal trip or a load burst, and hear the result a few seconds later. Everything
either tab draws or plays comes from `tensixinet.ui` (flow state, chart, log, session, knobs, scopes, instrument, readout), which the tests cover;
this file only lays the page out and moves data between them. A visitor plays in one tab at a time: Play in either ends what they had in the other.

    .venv-app/bin/python app/gradio_app.py        # then open http://localhost:7880

It reads the real chips with tt-smi (read-only, no device open), so it needs no gozer lease.
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import gradio as gr
import numpy as np

from tensixinet.music.patch import save_patch
from tensixinet.ui.chart import flow_svg
from tensixinet.ui.forms import (DRAW, SOURCES, kernel_choices, new_instrument, new_session, patch_choices, read_chip_knobs, source_choices,
                                 style_choices)
from tensixinet.ui.knobs import EVENTS, KNOB_RANGES, MAX_CHIPS, Knobs
from tensixinet.ui.logview import log_html
from tensixinet.ui.player import Busy, Visitors
from tensixinet.ui.progress import ACTIVE, progress_html
from tensixinet.ui.readout import READOUT_WAITING, readout_html
from tensixinet.ui.scopes import scope_choices
from tensixinet.ui.session import PREROLL_S

ROOT = Path(__file__).resolve().parent.parent
PATCH_DIRS = [ROOT / "demo", ROOT / "patches"]  # patches/ is optional: drop a *.patch.json there to see it in the list
PORT = int(os.environ.get("PORT", "7880"))
INTRO = ("Chip telemetry played as a band of Ciat-Lonbarde-style devices. The chart follows the segment that is playing: each chip's readings, "
         "the devices it plays, the patch cables bending one device with another, and where every voice sits in the stereo field. "
         "Wear headphones.")
PLAY_INTRO = ("Play the instruments with the readings. Pick one device, a module or the whole system; move a chip's sliders (temperature sets the pitch, "
              "aiclk the tempo, power the volume and with temperature the pressure, GDDR spread and voltage the colour of several devices) and press its "
              "buttons to inject a memory error, a thermal trip or a load burst. A move is heard a few seconds later. Wear headphones.")
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
INSTRUMENT = "instrument"  # the first item of a Play-tab session's config, so the two tabs' configs can be told apart
INSTRUMENT_LEAD_S = 3.0  # how far ahead of the listener the Play tab's session may run: the browser's player stalls on less
KNOB_ORDER = tuple(KNOB_RANGES)  # the order of a chip's sliders on the page
KNOB_LABELS = {"temperature": ("temperature", "C", 1), "power": ("power", "W", 1), "aiclk": ("aiclk", "MHz", 10),
               "gddr_spread": ("GDDR spread", "C", 0.5), "voltage": ("voltage", "V", 0.01)}  # label, unit, step
EVENT_LABELS = {"memory_error": "memory error", "thermal_trip": "thermal trip", "load_burst": "load burst"}


def _limit_from(env) -> int:
    """How many visitors may play at once: TENSIXINET_MAX_PLAYERS, or the default when it is missing, not a number or below one."""
    try:
        limit = int(env.get("TENSIXINET_MAX_PLAYERS", DEFAULT_MAX_PLAYERS))
    except ValueError:
        return DEFAULT_MAX_PLAYERS
    return limit if limit >= 1 else DEFAULT_MAX_PLAYERS


MAX_PLAYERS = _limit_from(os.environ)
visitors = Visitors(MAX_PLAYERS)  # which session each visitor's page shows (see ui/player.py); their audio, chart and log all belong to it


knob_store: dict = {}  # visitor -> the Play tab's Knobs, made when they first touch a knob or press Play, dropped when they leave
_knob_lock = threading.Lock()


def _knobs(key: str) -> Knobs:
    """That visitor's knobs. The page's sliders and buttons write to them as they are moved, whether or not anything is playing."""
    with _knob_lock:
        return knob_store.setdefault(key, Knobs(MAX_CHIPS))


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
    elif player.active and _is_instrument_config(player.config):  # the visitor pressed Play in the other tab
        yield None, gr.skip(), gr.skip(), "stopped: you started playing in the Play tab"


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


def _is_instrument_config(config) -> bool:
    return isinstance(config, tuple) and config[:1] == (INSTRUMENT,)


def _plays_instrument(key: str) -> bool:
    """Whether this visitor's running session is the Play tab's."""
    return _is_instrument_config(visitors.player(key).config)


def restart(source, style, seed, chips, load_s, patch_name, kernels, stereo, evict, cables, request: gr.Request = None):
    """A setting changed. While this visitor is playing and the settings differ, play again with them; otherwise leave the page as it is until Play.
    A visitor who is playing in the Play tab is not playing here, so a change here leaves them alone."""
    if _plays_instrument(_key(request)):
        yield gr.skip(), gr.skip(), gr.skip(), gr.skip()
        return
    yield from _stream(_key(request), _config(source, style, seed, chips, load_s, patch_name, kernels, stereo, evict, cables), only_if_changed=True)


def stop(request: gr.Request = None):
    """Stop this visitor's session, and empty their audio player: the browser would go on playing what it had already buffered."""
    visitors.player(_key(request)).stop()
    return None, STOPPED, "stopped"


def leave(request: gr.Request = None):
    """The visitor closed the page or lost the connection: end their session and give the place back."""
    visitors.drop(_key(request))
    with _knob_lock:
        knob_store.pop(_key(request), None)


# --- the Play tab -------------------------------------------------------------------------------------------------------------------

JS_PLAY_INSTRUMENT = "(...a) => { %s window.tensixinetInstrument = JSON.stringify(a); return a; }" % SILENCE
JS_CHANGE_INSTRUMENT = ("(...a) => { if (window.tensixinetInstrument && window.tensixinetInstrument !== JSON.stringify(a)) "
                        "{ %s window.tensixinetInstrument = JSON.stringify(a); } return a; }" % SILENCE)
JS_STOP_INSTRUMENT = "() => { %s window.tensixinetInstrument = null; }" % SILENCE
INSTRUMENT_STOPPED = (None, READOUT_WAITING, "stopped")
SCOPE_LABELS = {value: label for label, value in scope_choices()}


def _instrument_config(scope, style, seed, chips, stereo, cables, evict):
    return (INSTRUMENT, scope, style, int(seed or 0), int(chips), bool(stereo), bool(cables), bool(evict))


def _instrument_ended(player, token):
    """What a Play-tab stream that is no longer the current one says last (see `_ended`)."""
    if not player.active and not player.is_current(token):
        yield INSTRUMENT_STOPPED
    elif player.active and not _is_instrument_config(player.config):  # the visitor pressed Play in the first tab
        yield None, gr.skip(), "stopped: you started playing in the first tab"


def _instrument_stream(key, config, only_if_changed):
    """The Play tab's `_stream`: claim the page, show that it is starting, end what was playing, build the instrument over this visitor's knobs,
    then stream its audio and its readout until it is no longer the current session. The knobs are not part of `config`: moving one never restarts."""
    _, scope, style, seed, chips, stereo, cables, evict = config
    if only_if_changed and visitors.player(key).active and not _plays_instrument(key):
        yield gr.skip(), gr.skip(), gr.skip()  # this visitor is playing in the first tab; a change here is not a Play
        return
    try:
        player, token, old = visitors.claim(key, config, only_if_changed)
    except Busy as e:
        yield gr.skip(), READOUT_WAITING, (f"busy: {e.playing} of {e.limit} places are taken by other listeners. "
                                           "Try again in a minute, or run it yourself (see the README)")
        return
    if token is None:
        yield gr.skip(), gr.skip(), gr.skip()
        return
    yield None, READOUT_WAITING, "starting"
    if old is not None:
        old.stop()
    try:
        session, patch = new_instrument(scope, style, seed, chips, stereo, cables, evict, _knobs(key), sleep=time.sleep, lead_s=INSTRUMENT_LEAD_S)
    except ValueError as e:
        if player.is_current(token):
            player.finish(token)
            yield gr.skip(), READOUT_WAITING, f"could not start: {e}"
        return
    if not player.attach(token, session):
        yield from _instrument_ended(player, token)
        return
    session.start()
    status = (f"playing {SCOPE_LABELS.get(scope, scope)} ({patch.style or 'plain patch'}, seed {patch.seed}) "
              f"on {chips} chip{'s' if chips != 1 else ''}")
    shown = None
    try:
        while player.is_current(token):
            chunk = session.next_audio(timeout=0.5)
            if not player.is_current(token):
                yield from _instrument_ended(player, token)
                return
            snap = session.snapshot()
            readout = readout_html(snap)
            if snap.error:
                player.finish(token)
                yield gr.skip(), readout, f"stopped: {snap.error}"
                return
            audio = gr.skip() if chunk is None else (session.sample_rate, (np.clip(chunk, -1, 1) * 32767).astype(np.int16))
            yield audio, gr.skip() if readout == shown else readout, status
            shown = readout
            if not snap.running and chunk is None:
                player.finish(token)
                return
        yield from _instrument_ended(player, token)
    finally:
        session.stop()


def instrument_play(scope, style, seed, chips, stereo, cables, evict, request: gr.Request = None):
    """Play the instrument with the page's settings, ending whatever this visitor was playing (in either tab)."""
    yield from _instrument_stream(_key(request), _instrument_config(scope, style, seed, chips, stereo, cables, evict), only_if_changed=False)


def instrument_restart(scope, style, seed, chips, stereo, cables, evict, request: gr.Request = None):
    """A Play-tab setting changed. While this visitor is playing here and the settings differ, play again with them; otherwise leave the page alone."""
    yield from _instrument_stream(_key(request), _instrument_config(scope, style, seed, chips, stereo, cables, evict), only_if_changed=True)


def instrument_stop(request: gr.Request = None):
    """Stop this visitor's session and empty their audio player, as `stop` does for the first tab."""
    visitors.player(_key(request)).stop()
    return None, READOUT_WAITING, "stopped"


def knob_setter(chip: int, name: str):
    """The handler for one slider: it writes the slider's value to this visitor's knobs (a running session reads them at its next block)."""
    def set_knob(value, request: gr.Request = None):
        _knobs(_key(request)).set(chip, name, value)
    return set_knob


def event_trigger(chip: int, event: str):
    """The handler for one event button: it queues the event on this visitor's knobs for the next block."""
    def trigger_event(request: gr.Request = None):
        _knobs(_key(request)).trigger(chip, event)
    return trigger_event


def load_chips(request: gr.Request = None):
    """Copy one real `tt-smi` reading into this visitor's knobs. Returns the status line and then every slider's new value (chip by chip, in KNOB_ORDER)."""
    knobs = _knobs(_key(request))
    status = read_chip_knobs(knobs)
    return (status, *[knobs.get(chip, name) for chip in range(MAX_CHIPS) for name in KNOB_ORDER])


def show_chips(chips):
    """Show the controls of the chips that are being played and hide the rest (their values are kept)."""
    return [gr.update(visible=i < int(chips)) for i in range(MAX_CHIPS)]



def _build_watch_tab() -> None:
    """The first tab: Play draws what is playing."""
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
    stop_btn.click(stop, None, [audio, chart, status], js=JS_STOP)  # every stream ends itself within half a second of the player moving on


def _build_play_tab() -> None:
    """The second tab: the instrument. The sliders and buttons write to the visitor's knobs as they are moved and never restart the stream; a change
    of scope, style, seed, chip count or a switch plays again with it."""
    gr.Markdown(PLAY_INTRO)
    with gr.Row():
        scope = gr.Dropdown(scope_choices(), value="system", label="Play")
        style = gr.Dropdown(style_choices(), value=style_choices()[0], label="Style")
        seed = gr.Number(value=4, precision=0, label="Seed")
        chips = gr.Slider(1, MAX_CHIPS, value=MAX_CHIPS, step=1, label="Chips")
    with gr.Row():
        stereo = gr.Checkbox(True, label="Stereo")
        cables = gr.Checkbox(True, label="Patch cables")
        evict = gr.Checkbox(True, label="Eviction (system only)")
        play_btn = gr.Button("Play", variant="primary")
        stop_btn = gr.Button("Stop")
        load_btn = gr.Button("Load from the chips") if len(source_choices()) > 1 else None  # it reads tt-smi, so it needs tt-smi
    status = gr.Markdown("")
    with gr.Row():
        with gr.Column(scale=2):
            readout = gr.HTML(READOUT_WAITING, label="Readout")
            audio = gr.Audio(streaming=True, autoplay=True, label="What you play", interactive=False)
        with gr.Column(scale=3):
            columns, sliders = [], []
            for chip in range(MAX_CHIPS):
                with gr.Column(visible=True) as column:
                    columns.append(column)
                    with gr.Row():
                        for name in KNOB_ORDER:
                            label, unit, step = KNOB_LABELS[name]
                            low, high, default = KNOB_RANGES[name]
                            slider = gr.Slider(low, high, value=default, step=step, label=f"chip {chip}: {label} ({unit})")
                            sliders.append(slider)
                            slider.input(knob_setter(chip, name), slider, None, api_name=f"knob_{chip}_{name}", show_progress="hidden",
                                         concurrency_limit=None, trigger_mode="always_last")
                    with gr.Row():
                        for event in EVENTS:
                            button = gr.Button(f"chip {chip}: {EVENT_LABELS[event]}", size="sm")
                            button.click(event_trigger(chip, event), None, None, api_name=f"event_{chip}_{event}", show_progress="hidden",
                                         concurrency_limit=None)
    inputs = [scope, style, seed, chips, stereo, cables, evict]
    outputs = [audio, readout, status]
    play_btn.click(instrument_play, inputs, outputs, concurrency_limit=None, trigger_mode="multiple", js=JS_PLAY_INSTRUMENT)
    for control, event in ((scope, "input"), (style, "input"), (stereo, "input"), (cables, "input"), (evict, "input"),
                           (chips, "release"), (seed, "blur"), (seed, "submit")):
        getattr(control, event)(instrument_restart, inputs, outputs, concurrency_limit=None, trigger_mode="multiple", js=JS_CHANGE_INSTRUMENT)
    chips.input(show_chips, chips, columns, show_progress="hidden", concurrency_limit=None)
    stop_btn.click(instrument_stop, None, outputs, js=JS_STOP_INSTRUMENT)
    if load_btn is not None:
        load_btn.click(load_chips, None, [status] + sliders, concurrency_limit=None)


def build() -> gr.Blocks:
    with gr.Blocks(title="tensixinet: what is playing") as demo:
        gr.Markdown(f"## tensixinet: what is playing\n{INTRO}")
        with gr.Tabs():
            with gr.Tab("What is playing"):
                _build_watch_tab()
            with gr.Tab("Play"):
                _build_play_tab()
        demo.unload(leave)
    return demo


if __name__ == "__main__":
    build().queue().launch(server_name="0.0.0.0", server_port=PORT)
