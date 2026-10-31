# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""`--with-tt-toplike`: the synth and tt-toplike side by side in two tmux panes, reading the chips at the same interval.

This module only builds the tmux commands (lists of words) and runs them through a `run` function that a test can replace.
Nothing here starts tmux or tt-toplike by itself.

How the two panes end each other: each pane's shell command starts by tagging its own pane with a tmux pane option
(`@tensixinet_role` = `synth` or `toplike`) and ends by looking up the other pane in its window by that tag.
  * When the synth ends (Ctrl-C there, or `--duration`), the pane tagged `toplike` is closed with `tmux kill-pane`.
  * When tt-toplike ends (`q` there), the pane tagged `synth` gets `tmux send-keys C-c`, so the synth ends the way it does for a
    person: the log and the WAV are written out.
  * With `remain-on-exit off`, an ended pane closes, and a window with no panes left closes its session.
No `pane-exited` hook is used: a hook runs for either pane and would have to work out which one is left. Relative targets such as
`{right-of}` are not used either: tmux resolves them against the window's active pane, not the pane a command runs in (seen in a
real session, where the synth's `kill-pane -t '{right-of}'` found nothing).
"""
from __future__ import annotations

import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, List, Mapping, Optional, Sequence, Tuple

DEFAULT_INTERVAL_MS = 100  # tt-toplike's own default, used for both panes when --interval is not given
SESSION = "tensixinet"
SOCKET = "tensixinet"  # the tmux socket the session lives on when it is not inside the owner's own tmux
CONF = Path(__file__).with_name("tmux-toplike.conf")
TMUX = ("tmux",)


def own_tmux() -> Tuple[str, ...]:
    """The tmux command for a session of our own: its own socket and the config next to this file (status bar off, truecolor).
    A dedicated socket is needed because tmux honours `-f` only when it starts a new server; on the default socket a running server of the
    owner's would ignore it."""
    return ("tmux", "-L", SOCKET, "-f", str(CONF))


def split_toplike_args(text: Optional[str]) -> List[str]:
    """`--toplike-args "--mode arcade"` as the two words a shell would make of it."""
    return shlex.split(text) if text else []


def synth_flags(argv: Sequence[str], interval_ms: int) -> List[str]:
    """The user's own arguments for the synth pane: the launcher-only flags are dropped and `--interval N` is set to the shared value.

    Both spellings of a value are handled (`--toplike-args X` and `--toplike-args=X`; likewise `--interval`).
    """
    kept: List[str] = []
    skip_next = False
    for word in argv:
        if skip_next:
            skip_next = False
        elif word == "--with-tt-toplike":
            pass
        elif word in ("--toplike-args", "--interval"):
            skip_next = True
        elif word.startswith(("--toplike-args=", "--interval=")):
            pass
        else:
            kept.append(word)
    return kept + ["--interval", str(interval_ms)]


ROLE_OPTION = "@tensixinet_role"
IN_SESSION = "TENSIXINET_IN_SESSION"  # set in the synth pane; `--with-tt-toplike` refuses to run where it is set
PASS_ENV = ("PATH", "PYTHONPATH", "VIRTUAL_ENV", "TT_VISIBLE_DEVICES", "NO_COLOR")  # the caller's variables the panes need
EXIT_CTRL_C = 130  # a run that Ctrl-C ended before it started; it is a normal end, so the pane does not wait


def log_dir_of(argv: Sequence[str]) -> str:
    """The `--log-dir` the synth was given (either spelling), or the CLI's default `logs`."""
    found = "logs"
    for i, word in enumerate(argv):
        if word == "--log-dir" and i + 1 < len(argv):
            found = argv[i + 1]
        elif word.startswith("--log-dir="):
            found = word.split("=", 1)[1]
    return found


def _pane_command(words: Sequence[str], role: str, other_role: str, action: str, log_dir: Optional[str] = None) -> str:
    """The shell command tmux starts in a pane: `sh -c <script>`, so the user's own shell (fish, say) is never asked to parse it.

    The script tags this pane with `role`, runs the program, and closes the pane carrying `other_role` with the tmux command `action`
    (`kill-pane` or `send-keys ... C-c`). The closing step is an EXIT trap, so it runs however the program ended. `trap : INT` makes
    the wrapper survive a Ctrl-C that reaches its whole process group (the program still dies of it: a trapped signal is reset to the
    default in a child). With `log_dir`, a program that ends with a code other than 0 or 130 leaves the pane open, naming the code and
    the log directory, until Enter is pressed; otherwise the error text would vanish with the pane.
    """
    lines = [
        "close() {",
        f"  tmux list-panes -t \"$TMUX_PANE\" -F '#{{pane_id}} #{{{ROLE_OPTION}}}' | while read id role; do",
        f'    if [ "$role" = {other_role} ]; then tmux {action}; fi',
        "  done",
        "}",
        f'tmux set-option -p -t "$TMUX_PANE" {ROLE_OPTION} {role}',
        "trap : INT",
        "trap close EXIT",
        shlex.join(words),
        "code=$?",
    ]
    if log_dir is not None:
        lines += [
            f'if [ "$code" -ne 0 ] && [ "$code" -ne {EXIT_CTRL_C} ]; then',
            "  printf '\\ntensixinet ended with exit code %s; its log is in %s. Press Enter to close this pane.\\n' "
            f'"$code" {shlex.quote(log_dir)}',
            "  read answer",
            "fi",
        ]
    return f"sh -c {shlex.quote(chr(10).join(lines))}"


def build_session(
    synth_argv: Sequence[str],
    interval_ms: int,
    toplike_args: Sequence[str],
    session: str = SESSION,
    inside_tmux: bool = False,
    cwd: Optional[str] = None,
    environ: Optional[Mapping[str, str]] = None,
    toplike_path: str = "tt-toplike",
    tmux: Sequence[str] = TMUX,
    size: Optional[Tuple[int, int]] = None,
    attach: bool = True,
    tui: bool = False,
) -> List[List[str]]:
    """The tmux commands, in order, that make the two panes.

    `synth_argv` is the user's command line (the launcher-only flags are removed here). `cwd` is given to each pane with `-c`, so a
    relative `--out` or `--log-dir` means the same there. A tmux server that is already running gives a pane its own environment, not the
    caller's, so the caller's `PASS_ENV` variables found in `environ` are given to each pane with `-e`, and tt-toplike is started by
    `toplike_path` (the absolute path the caller found). The synth pane also gets `TENSIXINET_IN_SESSION=1`, which makes a
    `--with-tt-toplike` that reaches it refuse to start a session of its own.
    `tmux` is the program and any socket flags (a test uses `-L name` to stay clear of the owner's sessions). `size` gives a detached
    session a window size and `attach=False` leaves it detached.

    Outside tmux: a detached session whose first pane runs the synth, a split to its right for tt-toplike, `remain-on-exit off`, then
    an attach (its exit code is the run's). Inside tmux ($TMUX set): the current window is split twice, so the user's own pane, the
    synth and tt-toplike sit side by side (both splits divide the pane this command runs in; the panes find each other by their tag).
    Inside tmux with `tui` (the synth pane runs `--tui`): a third of the window is too narrow for the full-screen view, so a new window
    named `tensixinet` is opened for the synth and split in two for tt-toplike, and the user's own pane is left as it was. The split
    and the option target `:`, the session's current window, which is the new one; the pane tags work as before, since `close` lists
    the panes of the window it runs in.
    """
    tmux = list(tmux)
    synth = _pane_command([sys.executable, "-m", "tensixinet", *synth_flags(synth_argv, interval_ms)],
                          "synth", "toplike", 'kill-pane -t "$id"', log_dir=log_dir_of(synth_argv))
    top = _pane_command([toplike_path, "--interval", str(interval_ms), *toplike_args],
                        "toplike", "synth", 'send-keys -t "$id" C-c')
    where = ["-c", cwd] if cwd else []
    caller = [f"{name}={environ[name]}" for name in PASS_ENV if environ and name in environ]
    env = [x for pair in caller for x in ("-e", pair)]
    synth_env = env + ["-e", f"{IN_SESSION}=1"]
    if inside_tmux and tui:
        return [
            tmux + ["new-window", "-n", session, *where, *synth_env, synth],
            tmux + ["split-window", "-h", "-t", ":", *where, *env, top],
            tmux + ["set-option", "-w", "-t", ":", "remain-on-exit", "off"],
        ]
    if inside_tmux:
        return [
            tmux + ["split-window", "-h", *where, *synth_env, synth],
            tmux + ["split-window", "-h", *where, *env, top],
            tmux + ["set-option", "-w", "remain-on-exit", "off"],
        ]
    commands = [
        tmux + ["new-session", "-d", "-s", session, *(["-x", str(size[0]), "-y", str(size[1])] if size else []), *where, *synth_env, synth],
        tmux + ["split-window", "-h", "-t", session, *where, *env, top],
        tmux + ["set-option", "-t", session, "remain-on-exit", "off"],
    ]
    if attach:
        commands.append(tmux + ["attach-session", "-t", session])
    return commands


def check_available(which: Callable[[str], Optional[str]] = shutil.which) -> Optional[str]:
    """A plain sentence when tmux or tt-toplike is missing, `None` when both are on the PATH."""
    if not which("tmux"):
        return "--with-tt-toplike needs tmux, and there is no tmux on the PATH"
    if not which("tt-toplike"):
        return "--with-tt-toplike needs tt-toplike, and there is no tt-toplike on the PATH"
    return None


def session_exists(name: str, run: Callable = subprocess.call, tmux: Sequence[str] = TMUX) -> bool:
    """True when a tmux session of this name is running (`=` asks for the exact name, not a prefix)."""
    return run(list(tmux) + ["has-session", "-t", f"={name}"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0


def run_session(commands: Sequence[Sequence[str]], run: Callable = subprocess.call) -> int:
    """Run the commands in order. A command that fails ends the run with its code; otherwise the last command's code is returned."""
    code = 0
    for command in commands:
        code = run(list(command))
        if code != 0:
            return code
    return code
