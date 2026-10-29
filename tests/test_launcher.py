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


def test_an_interval_is_passed_through_unchanged_with_the_open_ended_duration():
    args, notes = L.with_defaults(["--interval", "100", "--live"], tt_smi=True)
    assert args == ["--interval", "100", "--live", "--duration", "0"] and notes == []


def test_with_tt_toplike_is_passed_through():
    args, notes = L.with_defaults(["--live", "--with-tt-toplike", "--interval", "50"], tt_smi=True)
    assert args == ["--live", "--with-tt-toplike", "--interval", "50", "--duration", "0"] and notes == []


def test_with_tt_toplike_and_no_tt_smi_adds_no_synthetic_chips_and_says_why():
    # synthetic chips would be refused by --with-tt-toplike, for a flag the user never typed; the command line's own refusal names tt-toplike
    args, notes = L.with_defaults(["--live", "--with-tt-toplike"], tt_smi=False)
    assert "--mock-chips" not in args and args == ["--live", "--with-tt-toplike", "--duration", "0"]
    assert notes == ["--with-tt-toplike needs the real chips; tt-smi was not found"]


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


# --- --rtpmidi-setup: get rtpmidid (RTP-MIDI to a Mac or iPad) without installing anything system-wide ---

ASSETS = ["rtpmidid-debian-trixie-amd64-26.01.deb", "rtpmidid-ubuntu-25_10-amd64-26.01.deb", "rtpmidid-ubuntu-24_04-amd64-26.01.deb",
          "rtpmidid-ubuntu-24_04-arm64-26.01.deb", "rtpmidid-26.01.tar.gz"]


def test_the_asset_for_this_ubuntu_and_cpu_is_picked():
    assert L.pick_rtpmidid_asset(ASSETS, "24.04", "x86_64") == "rtpmidid-ubuntu-24_04-amd64-26.01.deb"
    assert L.pick_rtpmidid_asset(ASSETS, "25.10", "x86_64") == "rtpmidid-ubuntu-25_10-amd64-26.01.deb"
    assert L.pick_rtpmidid_asset(ASSETS, "24.04", "aarch64") == "rtpmidid-ubuntu-24_04-arm64-26.01.deb"


def test_no_asset_is_picked_for_a_system_it_was_not_built_for():
    assert L.pick_rtpmidid_asset(ASSETS, "22.04", "x86_64") is None
    assert L.pick_rtpmidid_asset(ASSETS, "24.04", "riscv64") is None
    assert L.pick_rtpmidid_asset(ASSETS, None, "x86_64") is None


def test_the_ubuntu_version_is_read_from_os_release():
    text = 'NAME="Ubuntu"\nID=ubuntu\nVERSION_ID="24.04"\n'
    assert L.ubuntu_version(text) == "24.04"
    assert L.ubuntu_version('ID=debian\nVERSION_ID="12"\n') is None
    assert L.ubuntu_version("") is None


def test_rtpmidid_on_the_path_is_used_as_it_is(tmp_path):
    assert L.find_rtpmidid(lambda n: "/usr/bin/rtpmidid", tmp_path) == "/usr/bin/rtpmidid"


def test_an_unpacked_copy_is_found_when_nothing_is_on_the_path(tmp_path):
    assert L.find_rtpmidid(lambda n: None, tmp_path) is None
    exe = L.rtpmidid_dir(tmp_path) / "usr" / "bin" / "rtpmidid"
    exe.parent.mkdir(parents=True)
    exe.write_text("")
    assert L.find_rtpmidid(lambda n: None, tmp_path) == str(exe)


def _release(name=ASSETS[2]):
    return {"tag_name": "v26.01", "assets": [{"name": n, "browser_download_url": f"https://example.invalid/{n}"} for n in ASSETS]}


def _setup(tmp_path, **over):
    calls = {"dl": [], "run": []}

    def download(url, dest):
        calls["dl"].append(url)
        Path(dest).write_text("deb")

    def run(cmd, **kw):
        calls["run"].append(cmd)
        if cmd[:2] == ["dpkg-deb", "-f"]:
            return over.get("package", "rtpmidid")
        if cmd[:2] == ["dpkg", "-x"]:
            exe = Path(cmd[3]) / "usr" / "bin" / "rtpmidid"
            exe.parent.mkdir(parents=True)
            exe.write_text("")
        return ""

    args = dict(home=tmp_path, which=lambda n: None, fetch=lambda url: _release(), download=download, run=run,
                os_release='ID=ubuntu\nVERSION_ID="24.04"\n', machine="x86_64")
    args.update({k: v for k, v in over.items() if k != "package"})
    return L.setup_rtpmidi(**args), calls


def test_setup_downloads_checks_and_unpacks_into_the_users_own_folder(tmp_path):
    exe, calls = _setup(tmp_path)
    assert exe == str(L.rtpmidid_dir(tmp_path) / "usr" / "bin" / "rtpmidid")
    assert calls["dl"] == ["https://example.invalid/rtpmidid-ubuntu-24_04-amd64-26.01.deb"]
    # the package is asked what it is before it is unpacked, and unpacking is dpkg -x (nothing is installed, no sudo)
    kinds = [c[:2] for c in calls["run"]]
    assert kinds.index(["dpkg-deb", "-f"]) < kinds.index(["dpkg", "-x"])
    assert not any("sudo" in c or c[:2] == ["dpkg", "-i"] for c in calls["run"])


def test_setup_stops_when_the_package_is_not_rtpmidid(tmp_path):
    with pytest.raises(SystemExit, match="not rtpmidid"):
        _setup(tmp_path, package="something-else")
    assert not (L.rtpmidid_dir(tmp_path) / "usr").exists()


def test_setup_does_nothing_when_rtpmidid_is_already_there(tmp_path):
    exe, calls = _setup(tmp_path, which=lambda n: "/usr/bin/rtpmidid")
    assert exe == "/usr/bin/rtpmidid" and not calls["dl"]


def test_setup_says_when_there_is_no_build_for_this_system(tmp_path):
    with pytest.raises(SystemExit, match="no rtpmidid build"):
        _setup(tmp_path, machine="riscv64")


def test_the_rtpmidi_setup_mode_prints_how_to_use_it(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(L, "setup_rtpmidi", lambda **kw: "/x/rtpmidid")
    got = {}
    assert L.main(["--rtpmidi-setup"], root=tmp_path, which=lambda n: None, deps=True, run=_no_build, execv=lambda *a: got.update(ran=1)) == 0
    out = capsys.readouterr().out
    assert "/x/rtpmidid" in out and "--midi-port virtual" in out and "aconnect" in out and not got


def test_doctor_notes_whether_rtpmidid_is_there():
    _, lines = L.doctor(Path("/nonexistent"), which=lambda n: None, deps=True, home=Path("/nonexistent"))
    assert any("rtpmidid" in l and "--rtpmidi-setup" in l for l in lines)
    _, lines = L.doctor(Path("/nonexistent"), which=lambda n: "/usr/bin/rtpmidid" if n == "rtpmidid" else None, deps=True, home=Path("/nonexistent"))
    assert any(l.startswith("ok") and "rtpmidid" in l for l in lines)


def test_only_a_deb_is_picked_even_when_a_checksum_file_shares_the_name():
    assert L.pick_rtpmidid_asset(["rtpmidid-ubuntu-24_04-amd64-26.01.sha256"], "24.04", "x86_64") is None


def test_the_downloaded_package_is_removed_once_unpacked(tmp_path):
    _setup(tmp_path)
    assert not list(L.rtpmidid_dir(tmp_path).glob("*.deb"))
