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
import threading
import time
from pathlib import Path

import gradio as gr
import numpy as np

from tensixinet.music.patch import save_patch
from tensixinet.ui.chart import flow_svg
from tensixinet.ui.forms import DRAW, SOURCES, new_session, patch_choices, style_choices
from tensixinet.ui.logview import log_html

ROOT = Path(__file__).resolve().parent.parent
PATCH_DIRS = [ROOT / "demo", ROOT / "patches"]  # patches/ is optional: drop a *.patch.json there to see it in the list
PORT = int(os.environ.get("PORT", "7880"))
INTRO = ("Chip telemetry played as a band of Ciat-Lonbarde-style devices. The chart follows the segment that is playing: each chip's readings, "
         "the devices it plays, the patch cables bending one device with another, and where every voice sits in the stereo field. "
         "Wear headphones.")
WAITING = '<div style="padding:40px;color:#7d9a98;font:14px ui-monospace,monospace">press Play: the chart draws what is playing as it plays</div>'

_current = None  # the one session that is playing; starting another ends it
_lock = threading.Lock()


def _stop_current() -> None:
    global _current
    with _lock:
        old, _current = _current, None
    if old is not None:
        old.stop()


def play(source, style, seed, chips, load_s, patch_name, stereo, evict, cables):
    """Start a session and stream it: each step yields the next segment's audio with the chart and the log as they stand."""
    global _current
    _stop_current()
    try:
        session, patch = new_session(source, style, int(seed), int(chips), float(load_s), patch_name, PATCH_DIRS, time.sleep,
                                     stereo=stereo, evict=evict, cables=cables)
    except ValueError as e:
        yield gr.skip(), WAITING, log_html([]), f"could not start: {e}"
        return
    with _lock:
        _current = session
    session.start()
    status = f"playing {patch.style or 'a plain patch'} (seed {patch.seed}) from {source}"
    try:
        while True:
            chunk = session.next_audio(timeout=0.5)
            snap = session.snapshot()
            if snap.error:
                yield gr.skip(), flow_svg(snap.state) if snap.state else WAITING, log_html(snap.records), f"stopped: {snap.error}"
                return
            audio = gr.skip() if chunk is None else (session.sample_rate, (np.clip(chunk, -1, 1) * 32767).astype(np.int16))
            yield audio, flow_svg(snap.state) if snap.state else WAITING, log_html(snap.records), status
            if not snap.running and chunk is None:
                return
    finally:
        session.stop()


def stop():
    _stop_current()
    return "stopped"


def build() -> gr.Blocks:
    with gr.Blocks(title="tensixinet: what is playing") as demo:
        gr.Markdown(f"## tensixinet: what is playing\n{INTRO}")
        with gr.Row():
            source = gr.Radio(SOURCES, value=SOURCES[0], label="Source")
            style = gr.Dropdown(style_choices(), value=style_choices()[0], label="Style")
            patch_name = gr.Dropdown(patch_choices(PATCH_DIRS), value=DRAW, label="Patch (a saved one ignores the style and the seed)")
        with gr.Row():
            seed = gr.Number(value=4, precision=0, label="Seed")
            chips = gr.Slider(1, 8, value=4, step=1, label="Synthetic chips")
            load_s = gr.Slider(0, 120, value=60, step=5, label="Load cycle, seconds (0 = none; a chip that unloads evicts)")
        with gr.Row():
            stereo = gr.Checkbox(True, label="Stereo")
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
        inputs = [source, style, seed, chips, load_s, patch_name, stereo, evict, cables]
        # No concurrency limit: a new Play must be able to start and end the one that is running, and Stop cancels the stream.
        playing = play_btn.click(play, inputs, [audio, chart, log, status], concurrency_limit=None)
        stop_btn.click(stop, None, status, cancels=[playing])
    return demo


if __name__ == "__main__":
    build().queue().launch(server_name="0.0.0.0", server_port=PORT)
