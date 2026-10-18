# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The project site (site/index.html) quotes the code: device names and cable wiring. It must not drift from it."""
import json
import re
from pathlib import Path

import pytest

from tensixinet.music.cables import draw_cables
from tensixinet.music.chorus import device_names

SITE = Path(__file__).resolve().parent.parent / "site"
HTML = (SITE / "index.html").read_text()


def embedded(name):
    m = re.search(rf"const {name} = (.*?);\n", HTML, re.S)
    assert m, f"index.html has no {name}"
    return json.loads(m.group(1))


def test_the_device_list_on_the_site_is_the_code_list():
    assert embedded("DEVICES") == device_names()


def test_every_patch_drawn_on_the_site_is_what_the_code_draws_for_that_seed():
    seeds = embedded("SEEDS")
    assert len(seeds) >= 4
    for seed, cables in seeds.items():
        assert cables == [[c.source, c.target, c.kind, c.depth] for c in draw_cables(int(seed), device_names())], seed


def test_every_local_file_the_page_loads_exists():
    refs = re.findall(r'(?:src|href)="(assets/[^"]+)"', HTML)
    assert refs and all((SITE / r).is_file() for r in refs), [r for r in refs if not (SITE / r).is_file()]


def test_the_band_cards_name_every_device():
    for name in device_names():
        shown = name.replace("ultrasonic_shimmer", "ultrasonic shimmer")
        assert f'"{shown}"' in HTML, name


def test_the_four_cable_kinds_are_all_explained_on_the_page():
    for kind in ("fm", "crunch", "inflate", "fold"):
        assert f"<code>{kind}</code>" in HTML


def test_the_page_credits_ciat_lonbarde_and_says_it_is_not_a_recreation():
    assert "Ciat-Lonbarde" in HTML and "Josh Singer" in HTML and "does not claim to recreate" in HTML


def test_the_deploy_workflow_publishes_the_site_folder():
    flow = (SITE.parent / ".github" / "workflows" / "pages.yml").read_text()
    assert "path: site/" in flow and "deploy-pages" in flow


def test_every_file_the_page_loads_is_tracked_by_git_so_the_deploy_has_it():
    """A .gitignore rule once swallowed site/assets/audio, so the files existed here and 404ed on the site."""
    import shutil
    import subprocess
    if shutil.which("git") is None or not (SITE.parent / ".git").exists():
        pytest.skip("not a git checkout")
    tracked = set(subprocess.run(["git", "ls-files", "site"], cwd=SITE.parent, capture_output=True, text=True, check=True).stdout.split())
    refs = set(re.findall(r'(?:src|href)="(assets/[^"]+)"', HTML)) | {"index.html"}
    assert {f"site/{r}" for r in refs} <= tracked, sorted({f"site/{r}" for r in refs} - tracked)


# --- the mapping sandbox: its JS constants must match tensixinet/music/mapping.py and patch.py ----------------------

def js_number(name):
    m = re.search(rf"\b{name}\s*=\s*([-\d.]+)", HTML)
    assert m, f"index.html has no {name}"
    return float(m.group(1))


def js_array(name):
    m = re.search(rf"\b{name}\s*=\s*\[([^\]]*)\]", HTML)
    assert m, f"index.html has no {name}"
    return [float(x) for x in m.group(1).split(",")]


def test_the_sandboxs_pitch_and_tempo_constants_match_mapping_py():
    from tensixinet.music.mapping import AICLK_RANGE, BPM_RANGE, C3_HZ, PITCH_SEMITONES, TEMP_RANGE
    assert js_number("C3_HZ") == C3_HZ
    assert js_number("PITCH_SEMITONES") == PITCH_SEMITONES
    assert js_array("TEMP_RANGE") == list(TEMP_RANGE)
    assert js_array("AICLK_RANGE") == list(AICLK_RANGE)
    assert js_array("BPM_RANGE") == list(BPM_RANGE)


def test_the_sandboxs_pressure_constants_match_mapping_py():
    from tensixinet.music.mapping import PRESSURE_POWER_FLOOR, PRESSURE_TEMP_FLOOR_C
    assert js_number("PRESSURE_TEMP_FLOOR_C") == PRESSURE_TEMP_FLOOR_C
    assert js_number("PRESSURE_POWER_FLOOR") == PRESSURE_POWER_FLOOR


def test_the_sandboxs_scale_step_and_order_match_patch_py():
    from tensixinet.music.patch import DEGREES_PER_SCALE_STEP_C, SCALE_NAMES
    assert js_number("DEGREES_PER_SCALE_STEP_C") == DEGREES_PER_SCALE_STEP_C
    m = re.search(r'const SCALE_NAMES = \[(.*?)\];', HTML)
    shown = [s.strip().strip('"') for s in m.group(1).split(",")]
    assert shown == [n.replace("_", " ") for n in SCALE_NAMES]


def test_the_page_links_to_tenstorrent_hardware_and_the_github_org():
    assert 'href="https://tenstorrent.com"' in HTML
    assert 'href="https://github.com/tenstorrent"' in HTML
