#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""tensixinet: the one command to type.

    ./tensixinet.py --live          play your chips as it goes, until Ctrl-C
    ./tensixinet.py --app           open the page that draws what is playing (http://localhost:7880)
    ./tensixinet.py --doctor        check what is installed
    ./tensixinet.py --setup         build the environment now (a run builds it the first time anyway)
    ./tensixinet.py --rtpmidi-setup get rtpmidid, which sends tensixinet's MIDI port to a Mac or iPad (nothing is installed system-wide)

Everything else goes to the command line unchanged, so `./tensixinet.py --mock-chips 4 --duration 20` and
`./tensixinet.py --help` work as `python3 -m tensixinet` does. This file uses only the standard library.

What it adds to the command line:
  * It finds a Python that has numpy and soundfile. If the one running it does not, it makes `.venv` in this
    folder and installs the repo there once, then runs in it.
  * With no `tt-smi` on the PATH and no chosen source, it plays four synthetic chips and says so (not with
    `--with-tt-toplike`, which needs the real chips; it says that instead).
  * `--live` with no `--duration` plays until Ctrl-C (the command line's own default is 60 s).
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODES = {"--setup": "setup", "--doctor": "doctor", "--app": "app", "--rtpmidi-setup": "rtpmidi"}
MIN_PYTHON = (3, 10)
SYNTHETIC_CHIPS = "4"
PLAYERS = ("paplay", "aplay", "ffplay")
RTPMIDID_RELEASE = "https://api.github.com/repos/davidmoreno/rtpmidid/releases/latest"
RTPMIDID_ARCH = {"x86_64": "amd64", "aarch64": "arm64"}


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
    if not tt_smi and "--with-tt-toplike" in args:
        # tt-toplike shows the real chips, so synthetic ones would be refused for a flag the user never typed
        notes.append("--with-tt-toplike needs the real chips; tt-smi was not found")
    elif not tt_smi and not _has(args, "--mock-chips") and not _has(args, "--patch"):
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


def rtpmidid_dir(home=None):
    """Where `--rtpmidi-setup` unpacks rtpmidid: the user's own folder, so nothing needs sudo."""
    return Path(home or Path.home()) / ".local" / "share" / "tensixinet" / "rtpmidid"


def find_rtpmidid(which=shutil.which, home=None):
    """The rtpmidid to run: one on the PATH, else the copy `--rtpmidi-setup` unpacked, else None."""
    on_path = which("rtpmidid")
    if on_path:
        return on_path
    unpacked = rtpmidid_dir(home) / "usr" / "bin" / "rtpmidid"
    return str(unpacked) if unpacked.exists() else None


def ubuntu_version(os_release_text):
    """The VERSION_ID of an Ubuntu /etc/os-release, else None (the release builds are for Ubuntu)."""
    fields = dict(line.split("=", 1) for line in os_release_text.splitlines() if "=" in line)
    if fields.get("ID", "").strip('"') != "ubuntu":
        return None
    return fields.get("VERSION_ID", "").strip('"') or None


def pick_rtpmidid_asset(names, version, machine):
    """The release's `.deb` built for this Ubuntu and CPU, else None."""
    arch = RTPMIDID_ARCH.get(machine)
    if not version or not arch:
        return None
    prefix, suffix = f"rtpmidid-ubuntu-{version.replace('.', '_')}-{arch}-", ".deb"
    return next((n for n in names if n.startswith(prefix) and n.endswith(suffix)), None)


def _fetch_json(url):
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.load(r)


def _download(url, dest):
    with urllib.request.urlopen(url, timeout=120) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)


def _output(cmd, **kw):
    return subprocess.check_output(cmd, text=True, **kw)


def setup_rtpmidi(home=None, which=shutil.which, fetch=_fetch_json, download=_download, run=_output, os_release=None, machine=None):
    """Get rtpmidid's latest release into `rtpmidid_dir()`, without installing it: download the `.deb` for this system, check that the package
    is rtpmidid, and unpack it with `dpkg -x`. Returns the path of the executable. Does nothing if rtpmidid is already there."""
    have = find_rtpmidid(which, home)
    if have:
        return have
    if os_release is None:
        os_release = Path("/etc/os-release").read_text() if Path("/etc/os-release").exists() else ""
    machine = machine or os.uname().machine
    release = fetch(RTPMIDID_RELEASE)
    assets = {a["name"]: a["browser_download_url"] for a in release.get("assets", [])}
    name = pick_rtpmidid_asset(list(assets), ubuntu_version(os_release), machine)
    if name is None:
        raise SystemExit(f"tensixinet: no rtpmidid build for this system ({machine}); see https://github.com/davidmoreno/rtpmidid/releases")
    dest = rtpmidid_dir(home)
    dest.mkdir(parents=True, exist_ok=True)
    deb = dest / name
    print(f"tensixinet: downloading {name} ({release.get('tag_name', '')})", file=sys.stderr)
    download(assets[name], str(deb))
    package = run(["dpkg-deb", "-f", str(deb), "Package"]).strip()
    if package != "rtpmidid":
        deb.unlink()
        raise SystemExit(f"tensixinet: {name} is not rtpmidid (its package name is {package!r}); nothing was unpacked")
    run(["dpkg", "-x", str(deb), str(dest)])
    deb.unlink()
    return str(dest / "usr" / "bin" / "rtpmidid")


def doctor(root, which=shutil.which, deps=None, version=sys.version_info[:2], home=None):
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
    rtp = find_rtpmidid(which, home)
    lines.append(f"ok    rtpmidid at {rtp}: the MIDI port can go to a Mac or iPad" if rtp else
                 "note  no rtpmidid: ./tensixinet.py --rtpmidi-setup gets it, to send the MIDI port to a Mac or iPad")
    return ok, lines


def main(argv=None, root=ROOT, which=shutil.which, deps=None, run=subprocess.call, execv=os.execv):
    mode, rest = split_args(sys.argv[1:] if argv is None else argv)
    deps = deps_present() if deps is None else deps
    if mode == "doctor":
        ok, lines = doctor(root, which=which, deps=deps)
        print("\n".join(lines))
        return 0 if ok else 1
    if mode == "rtpmidi":
        exe = setup_rtpmidi(which=which)
        print(f"rtpmidid: {exe}\n\n"
              f"  {exe} --name tensixinet-net &        start it\n"
              "  ./tensixinet.py --live --midi-port virtual\n"
              "  aconnect -l                          find the numbers of tensixinet's port and of rtpmidid's\n"
              "  aconnect TENSIXINET_PORT RTPMIDID_PORT\n\n"
              "Then pick the session in Audio MIDI Setup (Network) on a Mac. Steps and what was tested: docs/MIDI.md.")
        return 0
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
