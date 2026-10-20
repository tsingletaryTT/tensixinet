# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The top-level `tensixinet.py`: `./tensixinet.py --live` should be all anyone types. It finds or builds an environment, falls back to synthetic
chips on a box with no Tenstorrent tools, and hands everything else to the command line. It sits beside the `tensixinet/` package, so the test
loads it by path under another name."""
import importlib.util
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("tensixinet_launcher", ROOT / "tensixinet.py")
L = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(L)


def test_the_launcher_is_executable_and_starts_with_a_python_shebang():
    f = ROOT / "tensixinet.py"
    assert f.read_text().startswith("#!/usr/bin/env python3\n") and os.access(f, os.X_OK)


# --- which mode ------------------------------------------------------------------

def test_launcher_flags_are_taken_out_and_everything_else_is_passed_on():
    assert L.split_args(["--live", "--duration", "5"]) == (None, ["--live", "--duration", "5"])
    assert L.split_args(["--setup"]) == ("setup", [])
    assert L.split_args(["--doctor"]) == ("doctor", [])
    assert L.split_args(["--app"]) == ("app", [])
    assert L.split_args(["--app", "--other"]) == ("app", ["--other"])


# --- defaults --------------------------------------------------------------------

def test_live_with_no_duration_plays_until_ctrl_c():
    args, _ = L.with_defaults(["--live", "--mock-chips", "2"], tt_smi=True)
    assert args[args.index("--duration") + 1] == "0"


def test_an_explicit_duration_is_kept_in_either_spelling():
    assert L.with_defaults(["--live", "--duration", "30"], tt_smi=True)[0] == ["--live", "--duration", "30"]
    assert L.with_defaults(["--live", "--duration=30"], tt_smi=True)[0] == ["--live", "--duration=30"]


def test_a_run_that_is_not_live_keeps_the_command_lines_own_duration():
    args, _ = L.with_defaults(["--mock-chips", "4"], tt_smi=True)
    assert "--duration" not in args


def test_with_no_tt_smi_and_no_source_the_chips_are_synthetic_and_it_says_so():
    args, notes = L.with_defaults(["--live"], tt_smi=False)
    assert args[args.index("--mock-chips") + 1] == "4"
    assert any("tt-smi" in n and "synthetic" in n for n in notes)


def test_with_tt_smi_the_real_chips_are_read_and_nothing_is_added():
    args, notes = L.with_defaults(["--live"], tt_smi=True)
    assert "--mock-chips" not in args and not notes


def test_a_chosen_source_is_never_replaced():
    for given in (["--mock-chips", "2"], ["--mock-chips=2"], ["--patch", "p.json"]):
        args, notes = L.with_defaults(given, tt_smi=False)
        assert args.count("--mock-chips") + sum(a.startswith("--mock-chips=") for a in args) == (0 if "--patch" in given else 1)
        assert not notes


def test_help_is_passed_through_untouched():
    args, notes = L.with_defaults(["--help"], tt_smi=False)
    assert args == ["--help"] and not notes


# --- which python ----------------------------------------------------------------

def test_the_running_python_is_used_when_it_has_what_is_needed():
    assert L.pick_python(Path("/nope"), current_ok=True) == sys.executable


def test_otherwise_the_repo_venv_is_used_if_it_exists(tmp_path):
    py = tmp_path / ".venv" / "bin" / "python"
    py.parent.mkdir(parents=True)
    py.write_text("")
    assert L.pick_python(tmp_path, current_ok=False) == str(py)


def test_with_neither_there_is_nothing_to_use_yet(tmp_path):
    assert L.pick_python(tmp_path, current_ok=False) is None


def test_the_running_python_is_not_ok_when_a_dependency_cannot_be_found():
    assert L.deps_present(find=lambda name: object()) is True
    assert L.deps_present(find=lambda name: None if name == "soundfile" else object()) is False
    assert L.deps_present(find=lambda name: None if name == "numpy" else object()) is False


# --- setup -----------------------------------------------------------------------

def test_setup_creates_the_venv_and_installs_the_repo_in_it(tmp_path):
    calls = []
    py = L.setup_env(tmp_path, run=lambda cmd, **kw: calls.append(cmd) or 0)
    assert py == str(tmp_path / ".venv" / "bin" / "python")
    assert calls[0][:3] == [sys.executable, "-m", "venv"] and calls[0][-1] == str(tmp_path / ".venv")
    assert calls[1][:5] == [py, "-m", "pip", "install", "-e"] and calls[1][5] == str(tmp_path)


def test_setup_stops_and_says_which_step_failed(tmp_path):
    with pytest.raises(SystemExit) as e:
        L.setup_env(tmp_path, run=lambda cmd, **kw: 1)
    assert "venv" in str(e.value)


def test_the_app_environment_is_its_own_venv_with_gradio(tmp_path):
    calls = []
    py = L.setup_env(tmp_path, app=True, run=lambda cmd, **kw: calls.append(cmd) or 0)
    assert py == str(tmp_path / ".venv-app" / "bin" / "python")
    assert any("gradio" in " ".join(c) for c in calls)


# --- doctor ----------------------------------------------------------------------

def _no_build(cmd, **kw):
    raise AssertionError(f"a test started a real build: {cmd}")   # a broken launcher must fail here, not make a venv in the repo


def _doctor(which, deps=True, version=(3, 12)):
    return L.doctor(Path("/nonexistent"), which=which, deps=deps, version=version)


def test_doctor_passes_with_python_and_dependencies_and_notes_what_is_optional():
    ok, lines = _doctor(which=lambda n: None)
    text = "\n".join(lines)
    assert ok and "tt-smi" in text and "synthetic" in text and "paplay" in text


def test_doctor_fails_on_an_old_python_or_missing_dependencies():
    assert not _doctor(which=lambda n: None, version=(3, 9))[0]
    ok, lines = _doctor(which=lambda n: None, deps=False)
    assert not ok and any("--setup" in l for l in lines)


def test_doctor_reports_the_tools_it_finds():
    ok, lines = _doctor(which=lambda n: f"/usr/bin/{n}")
    text = "\n".join(lines)
    assert "/usr/bin/tt-smi" in text and "/usr/bin/paplay" in text


# --- main ------------------------------------------------------------------------

def test_a_run_hands_the_command_line_to_the_package_with_the_repo_on_the_path(tmp_path):
    got = {}
    L.main(["--live", "--mock-chips", "2"], root=ROOT, which=lambda n: None, deps=True, run=_no_build,
           execv=lambda py, argv: got.update(py=py, argv=argv))
    assert got["py"] == sys.executable
    assert got["argv"][:3] == [sys.executable, "-m", "tensixinet"] and got["argv"][3:5] == ["--live", "--mock-chips"]
    assert str(ROOT) in os.environ["PYTHONPATH"].split(os.pathsep)


def test_a_run_with_no_environment_builds_one_first_then_runs_in_it(tmp_path):
    got, built = {}, []
    L.main(["--mock-chips", "2"], root=tmp_path, which=lambda n: None, deps=False,
           run=lambda cmd, **kw: built.append(cmd) or 0, execv=lambda py, argv: got.update(py=py, argv=argv))
    assert built and got["py"] == str(tmp_path / ".venv" / "bin" / "python")


def test_setup_and_doctor_modes_do_not_start_a_run(tmp_path, capsys):
    got = {}
    assert L.main(["--doctor"], root=Path("/nonexistent"), which=lambda n: None, deps=True, run=_no_build, execv=lambda *a: got.update(ran=1)) == 0
    assert "ok" in capsys.readouterr().out and not got
    assert L.main(["--doctor"], root=Path("/nonexistent"), which=lambda n: None, deps=False, run=_no_build, execv=lambda *a: got.update(ran=1)) == 1


def test_the_app_mode_starts_the_gradio_page_in_its_own_environment(tmp_path):
    got = {}
    (tmp_path / "app").mkdir()
    L.main(["--app"], root=tmp_path, which=lambda n: None, deps=True,
           run=lambda cmd, **kw: 0, execv=lambda py, argv: got.update(py=py, argv=argv))
    assert got["py"] == str(tmp_path / ".venv-app" / "bin" / "python")
    assert got["argv"][1] == str(tmp_path / "app" / "gradio_app.py")


def test_the_notes_are_printed_before_the_run(capsys):
    L.main(["--live"], root=ROOT, which=lambda n: None, deps=True, run=_no_build, execv=lambda *a: None)
    assert "synthetic" in capsys.readouterr().err
