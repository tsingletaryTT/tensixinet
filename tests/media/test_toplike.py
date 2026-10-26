# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""--with-tt-toplike: the tmux commands that put the synth and tt-toplike side by side. No test runs tmux or tt-toplike."""
import shlex
import sys

import pytest

from tensixinet.media.toplike import (
    DEFAULT_INTERVAL_MS,
    IN_SESSION,
    PASS_ENV,
    build_session,
    check_available,
    run_session,
    session_exists,
    split_toplike_args,
    synth_flags,
)

SYNTH = ["--live", "--with-tt-toplike", "--player", "none", "--toplike-args", "--mode arcade"]


def pane_commands(commands):
    """The shell script of each new pane, in order. Each pane command must be `sh -c <script>`, whatever the user's default shell is."""
    scripts = []
    for c in commands:
        if c[1] in ("new-session", "split-window"):
            words = shlex.split(c[-1])
            assert words[:2] == ["sh", "-c"] and len(words) == 3
            scripts.append(words[2])
    return scripts


def test_outside_tmux_the_session_has_two_panes_with_the_same_interval():
    commands = build_session(["--live", "--player", "none"], 100, ["--mode", "arcade"], cwd="/w")
    verbs = [c[1] for c in commands]
    assert verbs == ["new-session", "split-window", "set-option", "attach-session"]  # attach is last
    synth, toplike = pane_commands(commands)
    assert f"{sys.executable} -m tensixinet --live --player none --interval 100" in synth
    assert "--interval 100" in synth and "--interval 100" in toplike
    assert "tt-toplike" in toplike and "--mode arcade" in toplike
    assert commands[0][:5] == ["tmux", "new-session", "-d", "-s", "tensixinet"]
    assert commands[1][commands[1].index("-h")] == "-h" and "tensixinet" in commands[1]  # splits the session's one pane
    assert commands[2][-2:] == ["remain-on-exit", "off"]
    assert commands[3] == ["tmux", "attach-session", "-t", "tensixinet"]


def test_the_two_panes_end_each_other():
    synth, toplike = pane_commands(build_session(["--live"], 100, []))
    # each pane tags itself, runs its program, then acts on the pane carrying the other tag
    assert 'tmux set-option -p -t "$TMUX_PANE" @tensixinet_role synth' in synth
    assert 'tmux set-option -p -t "$TMUX_PANE" @tensixinet_role toplike' in toplike
    assert '[ "$role" = toplike ]' in synth and 'tmux kill-pane -t "$id"' in synth  # the synth ends: tt-toplike's pane goes
    assert '[ "$role" = synth ]' in toplike and 'tmux send-keys -t "$id" C-c' in toplike  # tt-toplike ends: the synth gets Ctrl-C
    assert "{right-of}" not in synth + toplike  # relative targets look at the active pane, not this one


def env_of(command):
    return [command[i + 1] for i, word in enumerate(command) if word == "-e"]


def test_the_working_directory_and_the_callers_environment_carry_into_both_panes():
    caller = {"PATH": "/a:/b", "PYTHONPATH": "/repo", "VIRTUAL_ENV": "/v", "TT_VISIBLE_DEVICES": "0,1", "NO_COLOR": "1", "HOME": "/h"}
    commands = build_session(["--live"], 100, [], cwd="/work dir", environ=caller)
    for c in (commands[0], commands[1]):
        assert c[c.index("-c") + 1] == "/work dir"
        passed = env_of(c)
        for name in PASS_ENV:
            assert f"{name}={caller[name]}" in passed
        assert not any(e.startswith("HOME=") for e in passed)  # only the named variables are passed on


def test_a_variable_the_caller_has_not_set_is_not_passed():
    commands = build_session(["--live"], 100, [], environ={"PATH": "/a"})
    for c in (commands[0], commands[1]):
        assert [e for e in env_of(c) if not e.startswith("TENSIXINET")] == ["PATH=/a"]


def test_the_pass_through_variables_are_the_named_ones():
    assert PASS_ENV == ("PATH", "PYTHONPATH", "VIRTUAL_ENV", "TT_VISIBLE_DEVICES", "NO_COLOR")


def test_the_synth_pane_is_marked_so_it_cannot_start_a_session_of_its_own():
    commands = build_session(["--live"], 100, [], environ={})
    assert f"{IN_SESSION}=1" in env_of(commands[0]) and IN_SESSION == "TENSIXINET_IN_SESSION"
    assert not any(e.startswith(IN_SESSION) for e in env_of(commands[1]))


def test_tt_toplike_is_started_by_the_path_that_was_found():
    _, toplike = pane_commands(build_session(["--live"], 100, ["--mode", "arcade"], toplike_path="/opt/tt/bin/tt-toplike"))
    assert "/opt/tt/bin/tt-toplike --interval 100 --mode arcade" in toplike


def test_every_pane_script_closes_the_other_pane_however_the_program_ended():
    for script in pane_commands(build_session(["--live"], 100, [])):
        assert "trap : INT" in script  # the shell stays alive when Ctrl-C reaches its whole process group
        assert "trap close EXIT" in script  # the closing step does not depend on how the program died


def _pause_line(script):
    (line,) = [l for l in script.splitlines() if l.startswith("  printf")]
    return line


def test_a_synth_that_fails_keeps_its_pane_open_with_the_exit_code_and_log_directory():
    synth, toplike = pane_commands(build_session(["--live", "--log-dir", "my logs"], 100, []))
    assert 'if [ "$code" -ne 0 ] && [ "$code" -ne 130 ]; then' in synth  # 130: Ctrl-C before the run started is a normal end
    assert "read answer" in synth and "read answer" not in toplike
    assert shlex.split(_pause_line(synth))[-2:] == ["$code", "my logs"]
    assert shlex.split(_pause_line(pane_commands(build_session(["--live"], 100, []))[0]))[-1] == "logs"  # the CLI's own default
    assert shlex.split(_pause_line(pane_commands(build_session(["--live", "--log-dir=elsewhere"], 100, []))[0]))[-1] == "elsewhere"


def test_inside_tmux_it_splits_the_current_window_and_does_not_attach():
    commands = build_session(["--live"], 250, [], inside_tmux=True)
    assert [c[1] for c in commands] == ["split-window", "split-window", "set-option"]
    assert not any(c[1] in ("new-session", "attach-session") for c in commands)
    synth, toplike = pane_commands(commands)
    assert "--interval 250" in synth and "--interval 250" in toplike
    assert "-t" not in commands[1]  # both splits divide the pane that ran the command


def test_a_socket_prefix_goes_in_front_of_every_command():
    commands = build_session(["--live"], 100, [], tmux=("tmux", "-L", "scratch"))
    assert all(c[:3] == ["tmux", "-L", "scratch"] for c in commands)
    inside = build_session(["--live"], 100, [], tmux=("tmux", "-L", "scratch"), inside_tmux=True)
    assert all(c[:3] == ["tmux", "-L", "scratch"] for c in inside)


def test_a_size_and_no_attach_make_a_session_a_test_can_look_at():
    commands = build_session(["--live"], 100, [], size=(200, 50), attach=False)
    assert commands[0][commands[0].index("-x") + 1] == "200" and commands[0][commands[0].index("-y") + 1] == "50"
    assert all(c[1] != "attach-session" for c in commands)
    assert "-x" not in build_session(["--live"], 100, [])[0]  # no size unless asked


def test_the_toplike_args_are_split_like_a_shell_would():
    assert split_toplike_args("--mode arcade") == ["--mode", "arcade"]
    assert split_toplike_args("--mode 'two words'") == ["--mode", "two words"]
    assert split_toplike_args(None) == []
    _, toplike = pane_commands(build_session(["--live"], 100, ["--mode", "two words"]))
    assert "--interval 100 --mode 'two words'" in toplike  # quoted again for the shell tmux starts


def test_the_flags_for_the_synth_pane_drop_the_launcher_only_flags():
    assert synth_flags(SYNTH, 100) == ["--live", "--player", "none", "--interval", "100"]
    assert synth_flags(["--live", "--toplike-args=--mode arcade", "--with-tt-toplike"], 100) == ["--live", "--interval", "100"]
    # a given interval is replaced by the resolved one, so the two panes cannot differ
    assert synth_flags(["--live", "--interval", "50"], 50) == ["--live", "--interval", "50"]
    assert synth_flags(["--live", "--interval=50"], 50) == ["--live", "--interval", "50"]
    assert synth_flags(["--live", "--with-tt-toplike", "--style", "enya"], 100)[-2:] == ["--interval", "100"]
    assert synth_flags(["--live", "--interval", "50"], 100) == ["--live", "--interval", "100"]


def test_the_default_interval_is_tt_toplikes_own():
    assert DEFAULT_INTERVAL_MS == 100


def found(*names):
    return lambda name: f"/bin/{name}" if name in names else None


def test_tools_present_means_no_complaint():
    assert check_available(found("tmux", "tt-toplike")) is None


def test_a_missing_tmux_or_toplike_is_named_in_a_sentence():
    assert "no tmux on the PATH" in check_available(found("tt-toplike"))
    assert "no tt-toplike on the PATH" in check_available(found("tmux"))


def test_an_existing_session_is_found_through_has_session():
    seen = []
    def run(cmd, **kw):
        seen.append(cmd)
        return 0
    assert session_exists("tensixinet", run) is True
    assert seen[0][1] == "has-session" and "=tensixinet" in seen[0]
    assert session_exists("tensixinet", lambda cmd, **kw: 1) is False
    assert session_exists("x", lambda cmd, **kw: 0, tmux=("tmux", "-L", "s")) is True


def test_a_run_returns_the_exit_code_of_the_attach():
    commands = build_session(["--live"], 100, [])
    ran = []
    def run(cmd, **kw):
        ran.append(cmd)
        return 7 if cmd[1] == "attach-session" else 0
    assert run_session(commands, run=run) == 7 and ran == commands


def test_a_failed_setup_command_stops_the_run_with_its_code():
    commands = build_session(["--live"], 100, [])
    ran = []
    def run(cmd, **kw):
        ran.append(cmd[1])
        return 1 if cmd[1] == "split-window" else 0
    assert run_session(commands, run=run) == 1 and ran == ["new-session", "split-window"]


def test_inside_tmux_the_synth_pane_is_marked_too():
    commands = build_session(["--live"], 100, [], inside_tmux=True, environ={})
    assert f"{IN_SESSION}=1" in env_of(commands[0]) and not any(e.startswith(IN_SESSION) for e in env_of(commands[1]))


def test_the_synth_script_run_with_a_failing_program_waits_for_enter_then_closes_the_toplike_pane(tmp_path):
    """Runs the built script under `sh` with a stand-in `tmux` (a file that logs its arguments); no real tmux or chip is used."""
    import os
    import pathlib
    import subprocess
    root = pathlib.Path(__file__).resolve().parents[2]
    fake = tmp_path / "bin"
    fake.mkdir()
    log = tmp_path / "tmux.log"
    (fake / "tmux").write_text(f'#!/bin/sh\nif [ "$1" = list-panes ]; then echo "%9 toplike"; echo "%8 synth"; fi\necho "$@" >> {log}\n')
    (fake / "tmux").chmod(0o755)
    synth, _ = pane_commands(build_session(["--live", "--interval", "5", "--log-dir", "the logs"], 100, []))
    # --interval 5 is replaced by 100 in the real command, so run the script with the program swapped for a command that fails with 2
    script = synth.replace(f"{sys.executable} -m tensixinet --live --log-dir 'the logs' --interval 100", "sh -c 'exit 2'")
    assert script != synth
    done = subprocess.run(["sh", "-c", script], input="\n", capture_output=True, text=True, timeout=30,
                          env={"PATH": f"{fake}:{os.environ['PATH']}", "TMUX_PANE": "%8", "PYTHONPATH": str(root)})
    assert "exit code 2; its log is in the logs" in done.stdout
    calls = log.read_text().splitlines()
    assert "kill-pane -t %9" in calls and not any("%8" in c and c.startswith("kill-pane") for c in calls)
    assert calls[0].startswith("set-option -p -t %8 @tensixinet_role synth")
