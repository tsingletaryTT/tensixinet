#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Run every kernel on every chip this process can see, and keep what each chip computed.

Run it under a lease. One lease covers a board, which is two chips, so a whole box takes one lease per board:

    gozer run --exact 0,1 --who "claude:tensixinet" --reason "cycle the kernels" -- python3 kernels/cycle.py --out runs/board0
    gozer run --exact 2,3 --who "claude:tensixinet" --reason "cycle the kernels" -- python3 kernels/cycle.py --out runs/board1

Each kernel runs as its own process (a device is opened and closed for each), with `--device N` for each leased chip and
`--save` so the chip's audio lands in `<out>/chip<N>/` next to the numpy reference. One JSON line per run goes to
`<out>/results.jsonl`, and a table is printed. Exits 0 when every run was within tolerance.

This script opens no device itself and imports no `ttnn`, so the lease guard in each kernel script is what protects the chip.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
KERNELS = ["ttshock.py", "ttshimmer.py", "ttgong.py", "ttavdog.py", "ttmix.py", "ttcoco.py"]


def last_json(text: str):
    lines = [l for l in text.splitlines() if l.startswith("{") and '"kernel"' in l]
    return json.loads(lines[-1]) if lines else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", required=True, help="where the results and the saved audio go")
    parser.add_argument("--chips", type=int, default=2, help="how many leased chips to cycle through (default 2, one board)")
    parser.add_argument("--atol", type=float, help="override every kernel's tolerance")
    parser.add_argument("--only", help="comma-separated kernel names to run (for example ttshock,ttgong)")
    args = parser.parse_args()
    wanted = [k for k in KERNELS if not args.only or k[:-3] in args.only.split(",")]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, failed = [], 0
    with open(out / "results.jsonl", "a") as log:
        for chip in range(args.chips):
            for script in wanted:
                cmd = [sys.executable, str(HERE / script), "--device", str(chip), "--save", str(out / f"chip{chip}")]
                if args.atol is not None:
                    cmd += ["--atol", str(args.atol)]
                done = subprocess.run(cmd, capture_output=True, text=True)
                result = last_json(done.stdout) or last_json(done.stderr)
                if result is None:
                    result = {"kernel": script[:-3], "device": chip, "ok": False, "error": (done.stderr or done.stdout)[-300:]}
                result["exit"] = done.returncode
                failed += 0 if result["ok"] else 1
                log.write(json.dumps(result) + "\n")
                log.flush()
                rows.append(result)
                print(f"chip {chip}  {result['kernel']:<10} {'ok  ' if result['ok'] else 'FAIL'} worst={result.get('worst', float('nan')):.2e}  run={result.get('run_ms', float('nan')):8.2f} ms", flush=True)
    print(f"{len(rows) - failed} of {len(rows)} runs within tolerance")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
