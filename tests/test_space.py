# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The Hugging Face Space: synthetic chips and the kernels in the TT-Lang simulator. These check the files that make the Space and the folder
that `space/assemble.sh` builds from them. Whether the image builds and the app answers inside it was checked by building it (see CLAUDE.md)."""
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPACE = ROOT / "space"


def _front_matter():
    text = (SPACE / "README.md").read_text()
    assert text.startswith("---\n")
    head = text.split("---\n")[1]
    return dict(line.split(": ", 1) for line in head.strip().splitlines())


def test_the_readme_front_matter_makes_a_docker_space_on_the_port_the_app_listens_on():
    fm = _front_matter()
    assert fm["sdk"] == "docker" and fm["app_port"] == "7860" and fm["title"]
    assert len(fm["short_description"]) <= 60  # the Hub refuses a longer one
    assert re.search(r"^ENV .*\bPORT=7860\b", (SPACE / "Dockerfile").read_text(), re.M)


def test_the_dockerfile_runs_as_user_1000_and_starts_the_app():
    d = (SPACE / "Dockerfile").read_text()
    assert re.search(r"useradd .*-u 1000", d) and re.search(r"^USER user$", d, re.M)
    assert re.search(r'^CMD \["python", "app/gradio_app.py"\]$', d, re.M)


def test_the_dockerfile_installs_ffmpeg_as_root_before_dropping_to_the_user():
    # Gradio's streaming audio shells out to ffmpeg; without it every Play fails after the first segment.
    d = (SPACE / "Dockerfile").read_text()
    apt = re.search(r"apt-get install[^\n]*\bffmpeg\b", d)
    assert apt and apt.start() < re.search(r"^USER user$", d, re.M).start()


def test_the_dockerfile_pins_what_it_installs_and_installs_no_ttnn():
    d = (SPACE / "Dockerfile").read_text()
    for pkg in ("tt-lang-sim", "gradio", "torch"):
        assert re.search(rf"{pkg}==\d", d), pkg
    assert "download.pytorch.org/whl/cpu" in d  # the CPU wheel: no CUDA in a Space with no GPU
    assert not re.search(r"pip install[^\n]*\bttnn\b", d)  # tt-lang-sim must be the only provider of ttnn (kernels/README.md)


def test_the_assembled_folder_has_what_the_space_runs_and_nothing_else(tmp_path):
    dest = tmp_path / "space"
    done = subprocess.run(["bash", str(SPACE / "assemble.sh"), str(dest)], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    for name in ("Dockerfile", "README.md", "pyproject.toml", "LICENSE", "tensixinet/__init__.py", "tensixinet/ui/session.py",
                 "tensixinet/ui/knobs.py", "tensixinet/ui/scopes.py", "tensixinet/ui/instrument.py", "tensixinet/ui/readout.py",   # the Play tab
                 "kernels/pool.py", "kernels/worker.py", "kernels/_common.py", "app/gradio_app.py", "demo/chorus.patch.json"):
        assert (dest / name).is_file(), name
    for kernel in ("ttshock", "ttshimmer", "ttgong", "ttavdog", "ttmix", "ttcoco"):
        assert (dest / "kernels" / f"{kernel}.py").is_file(), kernel
    for name in ("tests", "docs", "site", "demo/assets", ".venv-app", ".git", ".disco"):
        assert not (dest / name).exists(), name
    assert not list(dest.rglob("__pycache__"))
    assert sum(f.stat().st_size for f in dest.rglob("*") if f.is_file()) < 3_000_000  # code only: no video, audio or screenshots
    assert (dest / "README.md").read_text() == (SPACE / "README.md").read_text()


def test_assembling_twice_into_the_same_folder_replaces_it(tmp_path):
    dest = tmp_path / "space"
    for _ in range(2):
        assert subprocess.run(["bash", str(SPACE / "assemble.sh"), str(dest)], capture_output=True).returncode == 0
    (dest / "stale.txt").write_text("x")
    assert subprocess.run(["bash", str(SPACE / "assemble.sh"), str(dest)], capture_output=True).returncode == 0
    assert not (dest / "stale.txt").exists()
