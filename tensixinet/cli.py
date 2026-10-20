# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Command line: prompt the LLM, sample the chips, play the chorus, log why."""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path
from typing import Callable, Optional, Sequence

from .kernels.backend import KERNEL_DEVICES, KernelBackend
from .live import run_live
from .media.chip import ChipError, ChipWorker, chip_command, host_command
from .music.chorus import chip_devices, device_names
from .media import llm
from .media.player import PLAYERS, PlayerError, make_sink as default_make_sink
from .music.patch import DEFAULT_PATCH, draw_patch, load_patch, save_patch, seed_from_telemetry
from .music.styles import STYLES, draw_styled_patch
from .run import run
from .narrate.color import Palette, banner, detect, legend
from .narrate.runlog import RunLog
from .telemetry import fetch_snapshot, host_fingerprint, mock_telemetry, parse_snapshot




def _positive_int(text: str) -> int:
    n = int(text)
    if n < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return n


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tensixinet", description="Play your Tenstorrent chips as a chorus.")
    p.add_argument("prompt", nargs="?", help="send this to the LLM first, so its load shows up in the telemetry")
    p.add_argument("--duration", type=float, default=60.0,
                   help="seconds of audio (default 60; with --live, 0 plays until Ctrl-C)")
    p.add_argument("--sample-rate", type=int, default=44100)
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
    p.add_argument("--kernels", nargs="?", const="chip", choices=("chip", "chips", "host"), metavar="MODE",
                   help="run the devices that have TT-Lang kernels (the gong, shimmer, drone ripple, AVDog, ttcoco mix and the mixer) "
                        "on chips: 'chip' (the default) takes a gozer lease on one chip for the run, 'chips' on as many chips as "
                        "the run plays (each chip's devices run on that chip), 'host' runs the same kernels' formulas in numpy "
                        "with no chip; the other devices stay on the CPU")
    p.add_argument("--color", choices=("auto", "always", "never"), default="auto",
                   help="colour the screen with the Ciat-Lonbarde jack colours: 'auto' (the default) colours a terminal unless NO_COLOR "
                        "is set, 'always' colours a recording that is not a terminal; the text log and the JSONL are never coloured")
    p.add_argument("--no-banner", action="store_true", help="start with the first log line instead of the art and the colour legend")
    p.add_argument("--endpoint", default="http://127.0.0.1:8011", help="OpenAI-compatible LLM endpoint")
    p.add_argument("--model", help="model name (default: whatever the endpoint serves)")
    return p


def default_make_worker(mode: str, chips: int = 1) -> ChipWorker:
    """The chip worker for `--kernels MODE`: on `chips` leased chips, or the same kernels' formulas in numpy."""
    return ChipWorker(host_command(devices=chips) if mode == "host" else chip_command(chips), devices=chips)


def main(
    argv: Optional[Sequence[str]] = None,
    ask: Optional[Callable] = None,
    sample: Optional[Callable] = None,
    entropy: Optional[Callable] = None,
    make_sink: Optional[Callable] = None,
    make_worker: Optional[Callable] = None,
) -> int:
    parser = build_parser()
    a = parser.parse_args(argv)
    only = None
    if a.only:
        only = [name.strip() for name in a.only.split(",") if name.strip()]
        unknown = [name for name in only if name not in device_names()]
        if unknown or not only:
            parser.error(f"--only: unknown device {', '.join(unknown) or '(none given)'}; the devices are {', '.join(device_names())}")
    if a.style and a.patch:
        parser.error("--style draws a new patch and --patch replays a saved one; use one of them (a saved patch keeps its own style)")
    if a.player and not a.live:
        parser.error("--player needs --live: an offline run writes a file and plays nothing")
    if a.mock_faults and not a.mock_chips:
        parser.error("--mock-faults needs --mock-chips: real chips report their own errors")
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
    if sample is None:
        if a.mock_chips:
            sample = lambda: mock_telemetry(a.mock_chips, a.seed, faults=a.mock_faults)
            source = "--mock-chips (synthetic, no hardware is read)"
        else:
            def sample():
                snapshot = fetch_snapshot()
                host[0] = host_fingerprint(snapshot)
                return parse_snapshot(snapshot)
            source = "tt-smi -s"
    else:
        source = "the supplied telemetry source"

    worker = None
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
                held = len(chips) if a.kernels in ("chips", "host") else 1  # chips (or pretend chips) the worker holds open
                worker = (make_worker or default_make_worker)(a.kernels, held)
                where = ("the host (numpy, no chip)" if a.kernels == "host"
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
                                  f"the devices dealt to chip {index} do their arithmetic on leased chip {index % held}, so all "
                                  f"{held} chips are working at once")
            if a.live:
                try:
                    sink, where = (make_sink or default_make_sink)(a.player or "auto", a.sample_rate, out)
                except PlayerError as e:
                    log.event("player", f"the audio output could not be used: {e}", "a live run ends when it has nowhere to play")
                    return 3
                log.event("live", f"live mode: {where}",
                          "the chips are read again every turn and each turn is played as soon as it is made")
                try:
                    run_live(sample, sink, log, a.sample_rate, duration=a.duration or None, source=source, patch=patch,
                             patch_source=patch_source, normalise=not a.no_normalise, eq=not a.no_eq, only=only, cables=not a.no_cables,
                             backend=backend)
                except PlayerError:
                    return 3  # run_live has logged what stopped
            else:
                run(chips, a.duration, out, log, a.sample_rate, source=source, patch=patch, patch_source=patch_source,
                    normalise=not a.no_normalise, eq=not a.no_eq, only=only, cables=not a.no_cables, backend=backend)
        except ValueError:
            return 2
        return 0
    except KeyboardInterrupt:
        return 130  # Ctrl-C before the run started; once a live run is playing it stops itself
    finally:
        if worker is not None:
            if held_chips and a.kernels in ("chip", "chips"):
                log.event("backend", "releasing the chips",
                          "gozer resets the chips before it ends the lease, which takes about 40 s; the run is finished")
            worker.close()  # ends the worker, which releases the chip's lease
        log.close()
