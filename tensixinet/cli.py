"""Command line: prompt the LLM, sample the chips, play the chorus, log why."""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path
from typing import Callable, Optional, Sequence

from .media import llm
from .run import run
from .narrate.runlog import RunLog
from .telemetry import fetch_snapshot, mock_telemetry, parse_snapshot


class _Tee:
    """Write to several streams at once (terminal and a log file)."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, s):
        for st in self._streams:
            st.write(s)

    def flush(self):
        for st in self._streams:
            st.flush()

    def close(self):
        for st in self._streams[1:]:  # never close the terminal
            st.close()


def _positive_int(text: str) -> int:
    n = int(text)
    if n < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return n


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tensixinet", description="Play your Tenstorrent chips as a chorus.")
    p.add_argument("prompt", nargs="?", help="send this to the LLM first, so its load shows up in the telemetry")
    p.add_argument("--duration", type=float, default=60.0, help="seconds of audio (default 60)")
    p.add_argument("--sample-rate", type=int, default=44100)
    p.add_argument("--out", help="WAV path (default audio/tensixinet_<time>.wav)")
    p.add_argument("--log-dir", default="logs", help="where the run log goes (default logs/)")
    p.add_argument("--mock-chips", type=_positive_int, metavar="N", help="play N synthetic chips; reads no hardware")
    p.add_argument("--seed", type=int, default=0, help="seed for --mock-chips")
    p.add_argument("--endpoint", default="http://127.0.0.1:8011", help="OpenAI-compatible LLM endpoint")
    p.add_argument("--model", help="model name (default: whatever the endpoint serves)")
    return p


def main(
    argv: Optional[Sequence[str]] = None,
    ask: Optional[Callable] = None,
    sample: Optional[Callable] = None,
) -> int:
    a = build_parser().parse_args(argv)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = Path(a.out or f"audio/tensixinet_{stamp}.wav")
    out.parent.mkdir(parents=True, exist_ok=True)
    log_dir = Path(a.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log = RunLog(
        jsonl=open(log_dir / f"tensixinet_{stamp}.jsonl", "w"),
        stream=_Tee(sys.stdout, open(log_dir / f"tensixinet_{stamp}.txt", "w")),
        # Fit the pane we are in; one column short so a full line never wraps the terminal itself.
        width=shutil.get_terminal_size().columns - 1,
        dedupe_why=True,  # each reason once on screen; the JSONL keeps every one
    )

    ask = ask or (lambda prompt, endpoint: llm.ask(prompt, endpoint, model=a.model))
    if sample is None:
        if a.mock_chips:
            sample = lambda: mock_telemetry(a.mock_chips, a.seed)
            source = "--mock-chips (synthetic, no hardware is read)"
        else:
            sample = lambda: parse_snapshot(fetch_snapshot())
            source = "tt-smi -s"
    else:
        source = "the supplied telemetry source"

    try:
        if a.prompt:
            try:
                reply = ask(a.prompt, a.endpoint)
                log.event("llm", f"LLM replied: {reply[:80]!r}",
                          "LLM activity loads the chips, which changes the telemetry sampled next")
            except llm.LLMError as e:
                log.event("llm", f"LLM unavailable: {e}",
                          "continuing with telemetry only; the chorus does not need the LLM to play")
        try:
            run(sample(), a.duration, out, log, a.sample_rate, source=source)
        except ValueError:
            return 2
        return 0
    finally:
        log.close()
