# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The tt-discolike manifest and the Gradio app that it launches. The page itself is checked by hand in a browser; what is here is what a
test can say without one."""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _manifest():
    fields = {}
    for line in (ROOT / ".disco" / "app.yaml").read_text().splitlines():
        key, _, value = line.partition(":")
        fields[key.strip()] = value.strip()
    return fields


def test_the_discolike_manifest_names_the_app_the_way_discolike_requires_and_launches_a_file_that_exists():
    m = _manifest()
    assert re.fullmatch(r"[A-Za-z0-9_.-]+", m["name"]) and m["description"] and 1024 < int(m["port"]) < 65536
    launch = m["launch"].split()
    assert launch[0] == ".venv-app/bin/python" and (ROOT / launch[1]).is_file()
    assert "chips" not in m  # it reads the chips with tt-smi and needs no lease


def test_the_apps_port_in_the_manifest_is_the_port_the_app_defaults_to():
    source = (ROOT / "app" / "gradio_app.py").read_text()
    assert f'"PORT", "{_manifest()["port"]}"' in source


def test_the_app_keeps_its_virtual_environment_out_of_git():
    assert ".venv-app/" in (ROOT / ".gitignore").read_text().splitlines()


def test_the_app_builds_its_page_and_every_choice_it_offers_is_one_the_forms_accept():
    pytest.importorskip("gradio", exc_type=ImportError, reason="gradio is installed in .venv-app, and the TT venv's copy does not import")
    import sys
    sys.path.insert(0, str(ROOT / "app"))
    try:
        import gradio_app
    finally:
        sys.path.pop(0)
    demo = gradio_app.build()
    assert demo is not None
    from tensixinet.ui.forms import DRAW, SOURCES, new_session, patch_choices, style_choices
    for name in patch_choices(gradio_app.PATCH_DIRS):
        for style in style_choices():
            new_session(SOURCES[0], style, 1, 4, 0, name, gradio_app.PATCH_DIRS, lambda s: None)
