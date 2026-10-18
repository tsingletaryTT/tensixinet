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
