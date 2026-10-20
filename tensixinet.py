#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""tensixinet: the one command to type.

    ./tensixinet.py --live          play your chips as it goes, until Ctrl-C
    ./tensixinet.py --app           open the page that draws what is playing (http://localhost:7880)
    ./tensixinet.py --doctor        check what is installed
    ./tensixinet.py --setup         build the environment now (a run builds it the first time anyway)

Everything else goes to the command line unchanged, so `./tensixinet.py --mock-chips 4 --duration 20` and
`./tensixinet.py --help` work as `python3 -m tensixinet` does. This file uses only the standard library.

What it adds to the command line:
  * It finds a Python that has numpy and soundfile. If the one running it does not, it makes `.venv` in this
    folder and installs the repo there once, then runs in it.
  * With no `tt-smi` on the PATH and no chosen source, it plays four synthetic chips and says so.
  * `--live` with no `--duration` plays until Ctrl-C (the command line's own default is 60 s).
"""
from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODES = {"--setup": "setup", "--doctor": "doctor", "--app": "app"}
MIN_PYTHON = (3, 10)
SYNTHETIC_CHIPS = "4"
PLAYERS = ("paplay", "aplay", "ffplay")


def split_args(argv):
    """(mode, the rest): a launcher flag in first place picks a mode; any other command line goes to the package."""
    if argv and argv[0] in MODES:
        return MODES[argv[0]], list(argv[1:])
    return None, list(argv)


def _has(args, flag):
    return any(a == flag or a.startswith(flag + "=") for a in args)


def with_defaults(args, tt_smi):
    """The command line to run and the notes to print first."""
    args, notes = list(args), []
    if "-h" in args or "--help" in args:
        return args, notes
    if "--live" in args and not _has(args, "--duration"):
        args += ["--duration", "0"]
    if not tt_smi and not _has(args, "--mock-chips") and not _has(args, "--patch"):
        args += ["--mock-chips", SYNTHETIC_CHIPS]
        notes.append(f"no tt-smi on the PATH: playing {SYNTHETIC_CHIPS} synthetic chips (--mock-chips)")
    return args, notes


def deps_present(find=importlib.util.find_spec):
    return all(find(name) is not None for name in ("numpy", "soundfile"))


def venv_python(root, app=False):
    return str(Path(root) / (".venv-app" if app else ".venv") / "bin" / "python")


def pick_python(root, current_ok):
    """The interpreter to run in: this one if it has the dependencies, else the repo's .venv, else None."""
    if current_ok:
        return sys.executable
    py = venv_python(root)
    return py if Path(py).exists() else None


def setup_env(root, app=False, run=subprocess.call):
    """Make `.venv` (or `.venv-app`, with Gradio) and install the repo in it. Returns its python."""
    root, py = Path(root), venv_python(root, app)
    env_dir = str(Path(py).parent.parent)
    extra = ["gradio>=5"] if app else []
    print(f"tensixinet: building {env_dir} (once)", file=sys.stderr)
    for step, cmd in (("venv", [sys.executable, "-m", "venv", env_dir]),
                      ("install", [py, "-m", "pip", "install", *extra, "-e", str(root)])):
        if run(cmd) != 0:
            raise SystemExit(f"tensixinet: setup failed at the {step} step: {' '.join(cmd)}")
    return py


def doctor(root, which=shutil.which, deps=None, version=sys.version_info[:2]):
    """(ok, lines): what is installed. Python and the dependencies are needed; the rest changes what you can do."""
    deps = deps_present() if deps is None else deps
    lines, ok = [], True
    py_ok = tuple(version[:2]) >= MIN_PYTHON
    lines.append(f"{'ok' if py_ok else 'FAIL'}  python {version[0]}.{version[1]} (needs {MIN_PYTHON[0]}.{MIN_PYTHON[1]} or newer)")
    if deps:
        lines.append("ok    numpy and soundfile")
    elif Path(venv_python(root)).exists():
        lines.append("ok    numpy and soundfile are in .venv (this python lacks them; a run uses .venv)")
        deps = True
    else:
        lines.append("FAIL  numpy and soundfile: run ./tensixinet.py --setup")
    ok = py_ok and deps
    smi = which("tt-smi")
    lines.append(f"ok    tt-smi at {smi}: your chips are read" if smi else
                 "note  no tt-smi: the run plays synthetic chips (install tt-smi to play real ones)")
    found = [(p, which(p)) for p in PLAYERS if which(p)]
    lines.append(f"ok    sound out via {found[0][1]}" if found else
                 f"note  no {'/'.join(PLAYERS)}: nothing can play sound live (--out still records a WAV)")
    ffmpeg = which("ffmpeg")
    lines.append(f"ok    ffmpeg at {ffmpeg}" if ffmpeg else "note  no ffmpeg: --app needs it to stream audio to the browser")
    return ok, lines


def main(argv=None, root=ROOT, which=shutil.which, deps=None, run=subprocess.call, execv=os.execv):
    mode, rest = split_args(sys.argv[1:] if argv is None else argv)
    deps = deps_present() if deps is None else deps
    if mode == "doctor":
        ok, lines = doctor(root, which=which, deps=deps)
        print("\n".join(lines))
        return 0 if ok else 1
    if mode == "setup":
        print(f"ready: {setup_env(root, run=run)}")
        return 0
    os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, [str(root), os.environ.get("PYTHONPATH", "")]))
    if mode == "app":
        py = venv_python(root, app=True)
        if not Path(py).exists():
            py = setup_env(root, app=True, run=run)
        execv(py, [py, str(Path(root) / "app" / "gradio_app.py"), *rest])
        return 0
    py = pick_python(root, deps) or setup_env(root, run=run)
    args, notes = with_defaults(rest, tt_smi=bool(which("tt-smi")))
    for note in notes:
        print(f"tensixinet: {note}", file=sys.stderr)
    execv(py, [py, "-m", "tensixinet", *args])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
