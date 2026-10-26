# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Command line: prompt the LLM, sample the chips, play the chorus, log why."""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Optional, Sequence

from .kernels.backend import KERNEL_DEVICES, KernelBackend
from .fast import PLAYER_LATENCY_MS, SampledSource, run_fast
from .live import run_live
from .media.chip import ChipError, ChipWorker, chip_command, host_command, simulator_command
from .music.chorus import chip_devices, device_names
from .media import llm
from .media import toplike
from .media.player import PLAYERS, PlayerError, make_sink as default_make_sink
from .music.patch import DEFAULT_PATCH, draw_patch, load_patch, save_patch, seed_from_telemetry
from .music.styles import STYLES, draw_styled_patch
from .run import close_score_sink, run
from .score.routing import DEFAULT_ROUTES, load_routes
from .score.sinks import MidiFileSink, MidiPortError, MidiPortSink, TeeScoreSink
from .narrate.color import Palette, banner, detect, legend
from .narrate.runlog import RunLog
from .telemetry import fetch_snapshot, host_fingerprint, mock_telemetry, parse_snapshot
from .telemetry.fast import FastSource




def _positive_int(text: str) -> int:
    n = int(text)
    if n < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return n


INTERVAL_MS = (10, 1000)  # --interval's range, as tt-toplike's
DEFAULT_SAMPLE_RATE = 44100  # --sample-rate when it is not given
FAST_SAMPLE_RATE = 22050  # --sample-rate when it is not given with --interval: the Play tab's rate; the EQ already falls from 8 kHz


def _interval_ms(text: str) -> int:
    n = int(text)
    if not INTERVAL_MS[0] <= n <= INTERVAL_MS[1]:
        raise argparse.ArgumentTypeError(f"must be between {INTERVAL_MS[0]} and {INTERVAL_MS[1]} ms, got {n}")
    return n


def build_parser() -> argparse.ArgumentParser:
    # No abbreviations: `--with-tt` must not stand for `--with-tt-toplike`, because the synth pane's flags are rebuilt from the full spellings.
    p = argparse.ArgumentParser(prog="tensixinet", description="Play your Tenstorrent chips as a chorus.", allow_abbrev=False)
    p.add_argument("prompt", nargs="?", help="send this to the LLM first, so its load shows up in the telemetry")
    p.add_argument("--duration", type=float, default=60.0,
                   help="seconds of audio (default 60; with --live, 0 plays until Ctrl-C)")
    p.add_argument("--sample-rate", type=int, default=None,
                   help=f"samples a second (default {DEFAULT_SAMPLE_RATE}; with --interval, {FAST_SAMPLE_RATE}, which renders in half the time)")
    p.add_argument("--out", help="WAV path (default audio/tensixinet_<time>.wav)")
    p.add_argument("--log-dir", default="logs", help="where the run log goes (default logs/)")
    p.add_argument("--mock-chips", type=_positive_int, metavar="N", help="play N synthetic chips; reads no hardware")
    p.add_argument("--seed", type=int, default=0, help="seed for --mock-chips")
    p.add_argument("--mock-faults", action="store_true",
                   help="with --mock-chips, give some chips memory errors and thermal trips so the event sounds play")
    p.add_argument("--patch", metavar="FILE", help="replay a saved patch (patch_*.json from an earlier run's log dir)")
    p.add_argument("--style", choices=tuple(STYLES), metavar="NAME",
                   help=f"draw the patch in a style, a sketch of a sound that narrows the draw and sets the band's balance "
                        f"({', '.join(STYLES)}); the system and the seed still decide the rest, so no two runs match")
    p.add_argument("--patch-seed", type=int, metavar="N", help="draw the patch from this seed instead of from the chips and entropy")
    p.add_argument("--no-normalise", action="store_true",
                   help="write the render at the level the chips' power gives it, with no loudness gain")
    p.add_argument("--only", metavar="DEVICES",
                   help="audition: play only these devices, comma separated (%s). The rest are silent and there is no turn-taking"
                        % ", ".join(device_names()))
    p.add_argument("--no-eq", action="store_true",
                   help="leave the final mix unshaped (by default an EQ lifts the lows and takes down the shimmer)")
    p.add_argument("--no-cables", action="store_true",
                   help="leave out the patch cables (by default five to eight cables let one device's sound bend, crunch, "
                        "swell or fold another's, drawn from the patch seed)")
    p.add_argument("--live", action="store_true",
                   help="read the chips again every turn and play as it goes (the default renders one file from one reading)")
    p.add_argument("--player", choices=("auto", "none") + PLAYERS, metavar="NAME",
                   help="with --live, where the sound goes: auto (the first of %s that is installed), none (no sound, "
                        "at the speed of the music), or one of those players. --out still records what is played"
                        % ", ".join(PLAYERS))
    p.add_argument("--kernels", nargs="?", const="chip", choices=("chip", "chips", "host", "sim"), metavar="MODE",
                   help="run the devices that have TT-Lang kernels (the gong, shimmer, drone ripple, AVDog, ttcoco mix and the mixer) "
                        "on chips: 'chip' (the default) takes a gozer lease on one chip for the run, 'chips' on as many chips as "
                        "the run plays (each chip's devices run on that chip), 'host' runs the same kernels' formulas in numpy "
                        "with no chip, 'sim' runs the real kernels in the TT-Lang simulator (tt-lang-sim, no chip and no lease); the other devices stay on "
                        "the CPU")
    p.add_argument("--color", choices=("auto", "always", "never"), default="auto",
                   help="colour the screen with the Ciat-Lonbarde jack colours: 'auto' (the default) colours a terminal unless NO_COLOR "
                        "is set, 'always' colours a recording that is not a terminal; the text log and the JSONL are never coloured")
    p.add_argument("--mono", action="store_true",
                   help="play one channel, the sound as it was before stereo. By default every voice is placed in a stereo field: a "
                        "home for each chip, an offset for each device, an orbit that follows the chip's chaos and clock, depth, and "
                        "movements from inside the devices")
    p.add_argument("--mock-load", type=float, metavar="SECONDS",
                   help="with --mock-chips and --live: take each synthetic chip round a load cycle of this many seconds (rest, a load, rest "
                        "again, the chips staggered), so a chip unloads and the eviction can be heard without a model")
    p.add_argument("--no-evict", action="store_true",
                   help="in live mode, do not dissolve a chip's voices when its model unloads (power back to rest after a load, as "
                        "tt-toplike's Defrag EVICT reads it)")
    p.add_argument("--midi-file", metavar="PATH",
                   help="also write the devices' notes, triggers and controls as a Standard MIDI File (one track per chip); "
                        "the audio is unchanged. With --live the file holds every segment, written when the run ends")
    p.add_argument("--midi-port", metavar="NAME",
                   help="with --live, play the events on this MIDI output as they happen ('virtual' makes an output called tensixinet); "
                        "needs the extra: pip install -e .[midi]")
    p.add_argument("--midi-offset-ms", metavar="MS", type=float, default=0.0,
                   help="with --midi-port, send every message this many milliseconds later (negative: earlier) to line it up with the sound")
    p.add_argument("--midi-map", metavar="FILE",
                   help="a JSON file that changes the MIDI routes (channels, drum notes, controller numbers); needs --midi-file or --midi-port")
    p.add_argument("--interval", type=_interval_ms, metavar="MS",
                   help="with --live: read the chips every MS milliseconds (10 to 1000) and play a change soon after it (the calculated delay is "
                        "220 ms at the worst when the last phrase began over a second earlier; a change inside that second waits for it; a block "
                        "is the interval, at least 100 ms). Temperature, power, voltage and current come from hwmon; tt-smi fills the "
                        "other fields about every 1.5 s. Every 5 s the log reports the delay the run calculates from its own timings. Without it, live mode reads the "
                        "chips once a turn")
    p.add_argument("--with-tt-toplike", action="store_true",
                   help="with --live: start a tmux session with two panes, this synth on the left and tt-toplike on the right, both "
                        "reading the chips at the same --interval (100 ms when --interval is not given). Inside tmux it splits the "
                        "current window instead. Ctrl-C in the synth pane ends both; so does quitting tt-toplike. Refused with "
                        "--mock-chips and --kernels")
    p.add_argument("--toplike-args", metavar="ARGS",
                   help='with --with-tt-toplike: extra arguments for tt-toplike, as one quoted string (for example "--mode arcade")')
    p.add_argument("--no-banner", action="store_true", help="start with the first log line instead of the art and the colour legend")
    p.add_argument("--endpoint", default="http://127.0.0.1:8011", help="OpenAI-compatible LLM endpoint")
    p.add_argument("--model", help="model name (default: whatever the endpoint serves)")
    return p


def default_make_worker(mode: str, chips: int = 1) -> ChipWorker:
    """The chip worker for `--kernels MODE`: on `chips` leased chips, or the same kernels' formulas in numpy."""
    command = host_command(devices=chips) if mode == "host" else simulator_command(chips) if mode == "sim" else chip_command(chips)
    return ChipWorker(command, devices=chips)


def _with_toplike(parser, a, raw_argv, which, run) -> int:
    """`--with-tt-toplike`: refuse what cannot work, then run tmux. This process reads no chips and writes no log; the panes do."""
    if os.environ.get(toplike.IN_SESSION):
        parser.error("--with-tt-toplike was started inside its own session; it cannot start another")
    if not a.live:
        parser.error("--with-tt-toplike needs --live: tt-toplike shows the chips as they are now, so the synth has to follow them")
    if a.mock_chips:
        parser.error("--with-tt-toplike cannot be used with --mock-chips: the two would show different chips (tt-toplike the real ones, the synth the synthetic ones)")
    if a.kernels:
        parser.error("--with-tt-toplike cannot be used with --kernels: the kernels hold the chips, and tt-toplike and the fast sampler read them")
    missing = toplike.check_available(which)
    if missing:
        parser.error(missing)
    inside_tmux = bool(os.environ.get("TMUX"))
    if not inside_tmux and toplike.session_exists(toplike.SESSION, run):
        parser.error(f"a tmux session named {toplike.SESSION} is already running; end it with `tmux kill-session -t {toplike.SESSION}`, "
                     f"or attach to it with `tmux attach -t {toplike.SESSION}`")
    interval = a.interval if a.interval is not None else toplike.DEFAULT_INTERVAL_MS
    # A detached session starts at tmux's 80x24 and the synth sizes its log to that; give it the terminal it will be attached to.
    size = None if inside_tmux else tuple(shutil.get_terminal_size())
    commands = toplike.build_session(raw_argv, interval, toplike.split_toplike_args(a.toplike_args), inside_tmux=inside_tmux,
                                     cwd=os.getcwd(), environ=os.environ, toplike_path=which("tt-toplike"), size=size)
    return toplike.run_session(commands, run)


def _check_arguments(parser, a) -> tuple:
    """The checks every run makes of its arguments (a parser error for each problem). Returns the `--only` device list (or None) and the MIDI routes."""
    only = None
    if a.only:
        only = [name.strip() for name in a.only.split(",") if name.strip()]
        unknown = [name for name in only if name not in device_names()]
        if unknown or not only:
            parser.error(f"--only: unknown device {', '.join(unknown) or '(none given)'}; the devices are {', '.join(device_names())}")
    if a.style and a.patch:
        parser.error("--style draws a new patch and --patch replays a saved one; use one of them (a saved patch keeps its own style)")
    if a.mock_load is not None:
        if not a.mock_chips:
            parser.error("--mock-load needs --mock-chips: real chips load and unload when the work does")
        if not a.live:
            parser.error("--mock-load needs --live: a load cycle plays out over time, and an offline run reads the chips once")
        if a.mock_load <= 0:
            parser.error("--mock-load needs a positive number of seconds")
    if a.interval is not None:
        if not a.live:
            parser.error("--interval needs --live: it sets how often a live run reads the chips")
        if a.kernels:
            parser.error("--interval cannot be used with --kernels: the kernels hold the chips and the render budget is too small")
    routes = DEFAULT_ROUTES
    if a.midi_map:
        if not a.midi_file and not a.midi_port:
            parser.error("--midi-map needs a MIDI output flag (--midi-file or --midi-port): it only changes how events are routed")
        try:
            routes = load_routes(a.midi_map)
        except (OSError, ValueError) as e:  # ValueError covers bad JSON and every out-of-range key
            parser.error(f"--midi-map: cannot read {a.midi_map}: {e}")
    if a.player and not a.live:
        parser.error("--player needs --live: an offline run writes a file and plays nothing")
    if a.mock_faults and not a.mock_chips:
        parser.error("--mock-faults needs --mock-chips: real chips report their own errors")
    if a.midi_port and not a.live:
        parser.error("--midi-port needs --live: an offline run is written to a file (--midi-file)")
    if a.midi_offset_ms and not a.midi_port:
        parser.error("--midi-offset-ms needs --midi-port: it only shifts what goes to a live port")
    return only, routes


def main(
    argv: Optional[Sequence[str]] = None,
    ask: Optional[Callable] = None,
    sample: Optional[Callable] = None,
    entropy: Optional[Callable] = None,
    make_sink: Optional[Callable] = None,
    make_worker: Optional[Callable] = None,
    clock: Optional[Callable[[], float]] = None,
    which: Callable = shutil.which,
    tmux_run: Callable = subprocess.call,
    open_midi: Optional[Callable] = None,
) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)  # --with-tt-toplike rebuilds the synth pane's command line from this
    parser = build_parser()
    a = parser.parse_args(raw_argv)
    if a.toplike_args is not None and not a.with_tt_toplike:
        parser.error("--toplike-args needs --with-tt-toplike: it gives arguments to the tt-toplike that option starts")
    only, routes = _check_arguments(parser, a)  # before --with-tt-toplike, so its errors show here and not inside the synth pane
    if a.with_tt_toplike:
        return _with_toplike(parser, a, raw_argv, which, tmux_run)
    rate_given = a.sample_rate is not None
    if not rate_given:  # resolved here, once, so every use below sees the same rate
        a.sample_rate = FAST_SAMPLE_RATE if a.interval is not None else DEFAULT_SAMPLE_RATE
    file_sink = MidiFileSink(a.midi_file, routes) if a.midi_file else None
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = Path(a.out or f"audio/tensixinet_{stamp}.wav")
    out.parent.mkdir(parents=True, exist_ok=True)
    log_dir = Path(a.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    # Fit the pane we are in; one column short so a full line never wraps the terminal itself.
    width = shutil.get_terminal_size().columns - 1
    palette = detect(sys.stdout, a.color)
    log = RunLog(
        jsonl=open(log_dir / f"tensixinet_{stamp}.jsonl", "w"),
        stream=sys.stdout,
        plain_stream=open(log_dir / f"tensixinet_{stamp}.txt", "w"),
        palette=palette,
        width=width,
        dedupe_why=True,  # each reason once on screen; the JSONL keeps every one
    )
    if not a.no_banner:
        art = banner(width, palette) + [""] + legend(width, palette) + [""]
        plain_art = banner(width, Palette(False)) + [""] + legend(width, Palette(False)) + [""]
        log.raw("\n".join(art) + "\n", "\n".join(plain_art) + "\n")

    ask = ask or (lambda prompt, endpoint: llm.ask(prompt, endpoint, model=a.model))
    host = [0]  # fingerprint of the machine tt-smi describes; 0 when the readings are not from tt-smi
    fast_source = None  # with --interval: what each block reads
    if a.interval is not None and sample is None and not a.mock_chips:
        def fetch():
            snapshot = fetch_snapshot()
            host[0] = host_fingerprint(snapshot)
            return snapshot
        fast_source = FastSource(fetch=fetch)
        fast_source.start()  # its first tt-smi reading starts now, and the patch below is drawn from it
        sample = fast_source.sample
        source = "hwmon and tt-smi -s"
    elif sample is None:
        if a.mock_chips:
            started = (clock or time.monotonic)()
            sample = lambda: mock_telemetry(a.mock_chips, a.seed, faults=a.mock_faults, load_period=a.mock_load,
                                            elapsed=(clock or time.monotonic)() - started)
            source = "--mock-chips (synthetic, no hardware is read)"
        else:
            def sample():
                snapshot = fetch_snapshot()
                host[0] = host_fingerprint(snapshot)
                return parse_snapshot(snapshot)
            source = "tt-smi -s"
    else:
        source = "the supplied telemetry source"

    # The port is opened here, after the logs exist and before any chip is read or any audio starts; nothing that can
    # fail sits between this and the `try` below, whose `finally` closes it.
    port_sink = None
    if a.midi_port:
        # Opened before any chip is read and before any audio starts, so a missing library or port ends the run at once.
        try:
            port_sink = MidiPortSink(a.midi_port, routes=routes, offset_ms=a.midi_offset_ms, **({"open_output": open_midi} if open_midi else {}))
        except MidiPortError as e:
            log.close()
            parser.error(str(e))
    sinks = [x for x in (file_sink, port_sink) if x is not None]
    score_sink = sinks[0] if len(sinks) == 1 else TeeScoreSink(*sinks) if sinks else None
    worker = None
    score_closed = False
    held_chips = False  # true once the worker is ready, so a failed start does not claim to release anything
    try:
        replay = None
        if a.patch:
            try:
                replay = load_patch(a.patch)
            except ValueError as e:
                log.event("patch", f"cannot read patch {a.patch}", f"{e}; nothing was played")
                return 2
        if a.prompt:
            try:
                reply = ask(a.prompt, a.endpoint)
                log.event("llm", f"LLM replied: {reply[:80]!r}",
                          "LLM activity loads the chips, which changes the telemetry sampled next")
            except llm.LLMError as e:
                log.event("llm", f"LLM unavailable: {e}",
                          "continuing with telemetry only; the chorus does not need the LLM to play")
        try:
            chips = sample()
            draw = (lambda seed, c: draw_styled_patch(seed, c, a.style)) if a.style else draw_patch
            in_style = f", drawn in the {a.style} style" if a.style else ""
            if not chips:  # run() and run_live() log that there is nothing to play and raise
                patch, patch_source = DEFAULT_PATCH, "no chips, so no patch was drawn"
            elif replay is not None:
                patch, patch_source = replay, f"replayed from {a.patch}, so this run sounds like the one that saved it"
            elif a.patch_seed is not None:
                patch = draw(a.patch_seed, chips)
                patch_source = f"seed {a.patch_seed} given with --patch-seed{in_style}"
            else:
                noise = (entropy or time.time_ns)()
                seed = seed_from_telemetry(chips, noise + host[0])
                patch = draw(seed, chips)
                host_text = " and the host description" if host[0] else ""
                patch_source = (f"seed {seed} comes from every chip reading{host_text} plus entropy {noise}, "
                                f"so no two runs match{in_style}")
            if chips:
                save_patch(patch, log_dir / f"patch_{stamp}.json")
            backend = None
            if a.kernels:
                held = len(chips) if a.kernels in ("chips", "host", "sim") else 1  # chips (or pretend chips) the worker holds open
                worker = (make_worker or default_make_worker)(a.kernels, held)
                where = ("the host (numpy, no chip)" if a.kernels == "host"
                         else f"the TT-Lang simulator (no chip; {held} pretend chip{'' if held == 1 else 's'})" if a.kernels == "sim"
                         else f"{held} leased chip{'' if held == 1 else 's'}")
                try:
                    worker.start()
                    held_chips = True
                except ChipError as e:
                    log.event("backend", f"the kernels could not start: {e}",
                              "a run with --kernels ends when the kernels cannot run, so nothing was played")
                    return 3
                backend = KernelBackend(worker, log=lambda note: log.event(
                    "backend", note, "the kernels' formulas run on the CPU from here on, so the sound does not stop"))
                log.event("backend", f"TT-Lang kernels on {where} ({a.kernels} mode)",
                          f"the {', '.join(KERNEL_DEVICES)} devices and the mixer do their per-sample arithmetic in the kernels, "
                          "and the rest stay on the CPU: ttrollz, ttrax, ttquantus, the lead, ttsteam and the choir")
                if held > 1:
                    for index in range(len(chips)):  # which of this chip's devices run on its own kernels
                        mine = [d.__name__ for d in chip_devices(index, len(chips)) if d.__name__ in KERNEL_DEVICES]
                        mixer = ", and the mixer" if index == 0 else ""
                        log.event("backend", f"chip {index} runs {', '.join(mine) or 'nothing'}{mixer} on its own kernels",
                                  f"the devices dealt to chip {index} do their arithmetic on {'simulated' if a.kernels == 'sim' else 'leased'} chip {index % held}, so all "
                                  f"{held} chips are working at once")
            if a.live and a.interval is not None:
                # fast mode's own first line says how often the chips are read (and whether --interval was raised to the shortest block)
                if fast_source is None:
                    note = ("The readings are synthetic (--mock-chips): no hardware is read." if a.mock_chips
                            else f"The readings are from {source}.")
                    fast_source = SampledSource(sample, note)
                try:
                    sink, where = (make_sink or default_make_sink)(a.player or "auto", a.sample_rate, out, channels=1 if a.mono else 2,
                                                                   latency_ms=PLAYER_LATENCY_MS)
                except PlayerError as e:
                    log.event("player", f"the audio output could not be used: {e}", "a live run ends when it has nowhere to play")
                    return 3
                rate_why = (f"the sound is made at {a.sample_rate} Hz, as --sample-rate asked" if rate_given else
                            f"the sound is made at {a.sample_rate} Hz, fast mode's default: it renders in half the time of {DEFAULT_SAMPLE_RATE} Hz, "
                            f"and --sample-rate {DEFAULT_SAMPLE_RATE} gives the fuller rate")
                log.event("live", f"live mode: {where}", f"each block is played as soon as it is made; {rate_why}")
                try:
                    run_fast(fast_source, sink, log, a.sample_rate, a.interval / 1000.0, a.duration or None, patch, patch_source,
                             normalise=not a.no_normalise, eq=not a.no_eq, only=only, cables=not a.no_cables, stereo=not a.mono,
                             evict=not a.no_evict, player_name=a.player or "auto", latency_ms=PLAYER_LATENCY_MS, label=source,
                             **({"clock": clock} if clock else {}))
                except PlayerError:
                    return 3  # run_fast has logged what stopped
            elif a.live:
                try:
                    sink, where = (make_sink or default_make_sink)(a.player or "auto", a.sample_rate, out, channels=1 if a.mono else 2)
                except PlayerError as e:
                    log.event("player", f"the audio output could not be used: {e}", "a live run ends when it has nowhere to play")
                    return 3
                log.event("live", f"live mode: {where}",
                          "the chips are read again every turn and each turn is played as soon as it is made")
                try:
                    run_live(sample, sink, log, a.sample_rate, duration=a.duration or None, source=source, patch=patch,
                             patch_source=patch_source, normalise=not a.no_normalise, eq=not a.no_eq, only=only, cables=not a.no_cables,
                             backend=backend, stereo=not a.mono, evict=not a.no_evict, score_sink=score_sink, **({"clock": clock} if clock else {}))
                except PlayerError:
                    return 3  # run_live has logged what stopped
                finally:
                    if score_sink is not None:
                        score_closed = True
                        try:
                            close_score_sink(score_sink, log)
                        except Exception as e:  # a sink that cannot close must not change how the run ends
                            log.event("midi", f"the MIDI output could not be closed: {e}", "the run's own result is unchanged")
            else:
                run(chips, a.duration, out, log, a.sample_rate, source=source, patch=patch, patch_source=patch_source,
                    normalise=not a.no_normalise, eq=not a.no_eq, only=only, cables=not a.no_cables, backend=backend, stereo=not a.mono, score_sink=score_sink)
        except ValueError:
            return 2
        return 0
    except KeyboardInterrupt:
        return 130  # Ctrl-C before the run started; once a live run is playing it stops itself
    finally:
        if port_sink is not None and not score_closed:  # the run ended before the live loop did: do not leave notes hanging
            try:
                port_sink.close()
            except Exception:
                pass
        if fast_source is not None:
            fast_source.stop()  # also stopped by run_fast; this covers a run that ended before it started
        if worker is not None:
            if held_chips and a.kernels in ("chip", "chips"):
                log.event("backend", "releasing the chips",
                          "gozer resets the chips before it ends the lease, which takes about 40 s; the run is finished")
            worker.close()  # ends the worker, which releases the chip's lease
        log.close()
