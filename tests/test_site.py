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


# --- the rhythm section: its numbers must be the code's -----------------------------------------------------------

def test_the_rhythm_section_names_every_steam_route_and_the_library_sizes_the_code_has():
    from tensixinet.music.patch import GONG_TUNINGS, SCALES, STEAM_ROUTES
    assert 'id="rhythm"' in HTML and 'href="#rhythm"' in HTML
    for route in STEAM_ROUTES:
        assert f"<code>{route}</code>" in HTML, route
    assert f"{len(SCALES)} scales" in HTML and f"{len(GONG_TUNINGS)} gong tunings" in HTML


def test_the_rhythm_section_states_the_roll_model_with_the_codes_ranges():
    from tensixinet.music.ttrolzer import DOWNBEAT_GAIN, DRIFT_RANGE, LINK_PULL_RANGE, MAX_LINKS, MIN_LINKS, SLIP
    section = HTML[HTML.index('id="rhythm"'): HTML.index('id="band"')]
    assert "Sandrode" in section and "Brown" in section
    assert f"{MIN_LINKS} to {MAX_LINKS} links" in section
    assert f"{LINK_PULL_RANGE[0]:.0%} to {LINK_PULL_RANGE[1]:.0%}" in section
    assert f"{DRIFT_RANGE[0]:.1%} to {DRIFT_RANGE[1]:.0%}" in section
    assert f"{SLIP:.0%} of a cycle" in section and f"{DOWNBEAT_GAIN:g} of the lane's level" in section
    from tensixinet.music.ttrolzer import BARRE_GEOMETRY, GONG_ROLL_ODDS
    assert "a 40-roll chip gets 6 to 12" in section and f"the real Rolzer has {len(BARRE_GEOMETRY) * 7}" in section
    assert GONG_ROLL_ODDS == 0.75 and "three chips in four" in section.lower()  # the gong is patched to a roll this often


def test_no_page_links_to_the_lapsed_ciat_lonbarde_domain():
    """ciat-lonbarde.net expired and now serves an unrelated football-streaming site. Peter Blasser's own page and Patch Point,
    which calls itself the official home of Ciat-Lonbarde, are where the links go."""
    readme = (SITE.parent / "README.md").read_text()
    for name, text in (("site/index.html", HTML), ("README.md", readme)):
        assert "ciat-lonbarde.net" not in text, name
    assert 'href="https://pblasser.github.io/"' in HTML and "(https://pblasser.github.io/)" in readme
    assert 'href="https://patch-point.com"' in HTML


# --- the kernels recording, the colour, and the styles -------------------------------------------------------------------------

def test_the_hero_is_the_kernels_recording_and_the_older_every_feature_video_is_still_on_the_page():
    hero = HTML[HTML.index('<header class="hero">'): HTML.index("</header>")]
    assert 'src="assets/kernels.mp4"' in hero and 'poster="assets/kernels-frame.png"' in hero
    for name in ("kernels.mp4", "kernels-frame.png", "features.mp4"):
        assert (SITE / "assets" / name).is_file(), name
    assert 'src="assets/features.mp4"' in HTML and "assets/features.mp4" not in hero
    caption = re.search(r'<p class="caption">(.*?)</p>', hero, re.S).group(1)
    assert "four chips" in caption and "kernels" in caption and "colour" in caption


def test_the_page_explains_the_jack_colours_with_the_guides_own_meanings():
    from tensixinet.narrate.color import JACK_MEANING, LOG_MEANING
    section = HTML[HTML.index('id="colour"'): HTML.index('id="rhythm"')]
    for name in ("blue", "green", "purple", "white", "orange", "red", "yellow", "brown", "grey"):
        assert f"<code>{name}</code>" in section and LOG_MEANING[name] in section, name
    assert JACK_MEANING["purple"] in section or "innards" in section


def test_the_styles_section_carries_every_styles_name_and_description_from_the_code():
    from html import escape
    from tensixinet.music.styles import STYLES
    section = HTML[HTML.index('id="styles"'): HTML.index('id="band"')]
    assert 'href="#styles"' in HTML
    for name, style in STYLES.items():
        assert f"<code>--style {name}</code>" in section, name
        assert escape(style.blurb, quote=False) in section or style.blurb in section, name


def test_the_kernels_section_says_the_whole_soundtrack_can_run_on_all_four_chips():
    section = HTML[HTML.index('id="kernels"'): HTML.index('id="hardware"')]
    assert "<code>--kernels chips</code>" in section and "one gozer lease" in section and "0.0034" in section


def test_the_space_section_states_the_stereo_and_evict_numbers_the_code_uses():
    from tensixinet.music import evict, space, stereo
    section = HTML[HTML.index('id="space"'): HTML.index('id="band"')]
    assert 'href="#space"' in HTML
    for text in (f"{space.SPREAD}", f"{evict.LOAD_SURGE_FACTOR}", f"{evict.EVICT_MARGIN}", f"{evict.POWER_IDLE_W:g} W",
                 f"{evict.EVICT_SECONDS:g} s", f"{round(evict.RUN_TDP_FRACTION * 100)}%", f"{stereo.TURN_DEPTH}",
                 "<code>--mono</code>", "<code>--mock-load</code>", "<code>--no-evict</code>"):
        assert text in section, text
    assert 'src="assets/stereo.mp4"' in section and (SITE / "assets" / "stereo.mp4").is_file()
    assert 'poster="assets/stereo-frame.png"' in section and (SITE / "assets" / "stereo-frame.png").is_file()
    assert 'src="assets/audio/stereo.mp3"' in section and (SITE / "assets" / "audio" / "stereo.mp3").is_file()
    assert "Qwen3-Coder-Next" in section and "real chips" in section  # the recording is of real load; a synthetic one must not take its place


def test_the_app_section_shows_the_screenshot_gives_the_commands_and_says_it_does_not_run_on_the_page():
    import re as _re
    section = HTML[HTML.index('id="app"'): HTML.index('id="band"')]
    assert 'href="#app"' in HTML and 'src="assets/app.png"' in section and (SITE / "assets" / "app.png").is_file()
    port = _re.search(r"^port:\s*(\d+)", (SITE.parent / ".disco" / "app.yaml").read_text(), _re.M).group(1)
    assert f"localhost:{port}" in section and ".venv-app/bin/python app/gradio_app.py" in section
    assert "does not run on this page" in section and "tt-discolike" in section


# --- the snare's odds, the delay bus and the rests: their numbers must be the code's ---------------------------------------------

def test_the_rhythm_section_gives_the_route_and_snare_odds_the_seed_uses():
    from tensixinet.music.patch import SNARE_LEVELS, SNARE_ODDS, STEAM_ROUTE_COUNT_ODDS, STEAM_ROUTE_WEIGHTS
    section = HTML[HTML.index('id="rhythm"'): HTML.index('id="bus"')]
    names = ("no route", "one", "two", "three")
    for name, odds in zip(names, STEAM_ROUTE_COUNT_ODDS):
        assert f"{name} ({odds:.0%})" in section, name
    assert f"drawn {STEAM_ROUTE_WEIGHTS[0]:g} times as often" in section
    assert [f"({o:.0%})" for o in SNARE_ODDS] == ["(40%)", "(30%)", "(20%)", "(10%)"]
    for text in ("none of the register's hits (40%)", "20% of them (30%)", "half of them (20%)", "all of them (10%)"):
        assert text in section, text
    assert SNARE_LEVELS == (0.0, 0.2, 0.5, 1.0)
    assert "25%), one (50%)" not in section and "half as often" not in section   # the old odds are gone


def test_the_delay_and_rests_section_states_the_buss_numbers_and_the_rests_ranges_the_code_has():
    from tensixinet.music import cocobus
    from tensixinet.music.devices import REST_FADE_S
    from tensixinet.music.patch import COCO_REVERB_RANGE, COCO_SEND_RANGE, LEAD_RESTS_RANGE, RESTS_RANGE, REVERB_S_RANGE, ROLL_RESTS_RANGE
    section = HTML[HTML.index('id="bus"'): HTML.index('id="styles"')]
    assert 'href="#bus"' in HTML
    pct = lambda r: f"{r[0]:.0%} to {r[1]:.0%}"
    assert cocobus.BUS_TAPS == 4   # the page says "four"
    for text in ("four 8-bit delay taps", f"from {cocobus.MIN_FEEDBACK:g} to 0.8", f"up to {cocobus.SWAY_DEPTH:.0%}",
                 f"{REVERB_S_RANGE[0]:g} to {REVERB_S_RANGE[1]:g} s", f"{cocobus.REVERB_CUTOFF_HZ / 1000:g} kHz",
                 f"{COCO_SEND_RANGE[0]:.0%} to {COCO_SEND_RANGE[1]:.0%}", pct(COCO_REVERB_RANGE), f"{REST_FADE_S * 1000:g} ms",
                 f"{RESTS_RANGE[0]:.0%} to {RESTS_RANGE[1]:.0%}", f"{LEAD_RESTS_RANGE[0]:.0%} to {LEAD_RESTS_RANGE[1]:.0%}",
                 f"{ROLL_RESTS_RANGE[0]:.0%} to {ROLL_RESTS_RANGE[1]:.0%}", ">coco_send, coco_reverb, reverb_s<",
                 ">rests<", ">lead_rests, roll_rests<", "first note of a phrase"):
        assert text in section, text
    assert "the real unit has none" in section   # the reverb is an addition, and the page says so


def test_each_style_has_its_own_clip_and_the_styles_table_carries_its_snare_bus_and_rest_values():
    from tensixinet.music.styles import STYLES
    section = HTML[HTML.index('id="styles"'): HTML.index('id="space"')]
    for name, st in STYLES.items():
        assert f'src="assets/audio/{name}.mp3"' in section and (SITE / "assets" / "audio" / f"{name}.mp3").is_file(), name
        row = re.search(rf"<tr><td><code>{name}</code></td>(.*?)</tr>", section).group(1)
        cells = re.findall(r"<td>(.*?)</td>", row)
        assert cells == [f"{st.snare:g}", f"{st.bus[0]:g}", f"{st.bus[1]:g}", f"{st.bus[2]:g} s", f"{st.rests:g}", f"{st.pulse_rests[0]:g}", f"{st.pulse_rests[1]:g}", f"{st.mod:g}"], name


def test_every_audio_clip_on_the_page_exists():
    for src in re.findall(r'<audio[^>]*src="(assets/[^"]+)"', HTML):
        assert (SITE / src).is_file(), src


def test_the_app_section_says_how_many_visitors_play_at_once():
    import importlib.util
    section = HTML[HTML.index('id="app"'): HTML.index('id="band"')]
    assert "<code>TENSIXINET_MAX_PLAYERS</code> (3 by default)" in section
    assert 'DEFAULT_MAX_PLAYERS = 3' in (SITE.parent / "app" / "gradio_app.py").read_text()


def test_the_reset_clip_says_how_it_was_made_and_what_the_loop_did_while_the_chips_were_gone():
    card = HTML[HTML.index("<h3>A chip reset</h3>"): HTML.index("assets/audio/reset.mp3")]
    assert (SITE / "assets" / "audio" / "reset.mp3").is_file()
    for text in ("gozer release", "tt-smi -r", "no chips in this reading", "plays the last readings that worked again", "on the CPU"):
        assert text in card, text
    # the loop really does say that when a reading fails, and the script that made the clip releases a lease rather than running a reset by hand
    assert "no chips in this reading" in (SITE.parent / "tensixinet" / "live.py").read_text()
    script = (SITE.parent / "demo" / "make_reset_clip.sh").read_text()
    assert 'gozer release "$LEASE"' in script and "tt-smi -r" not in script.replace("the same reset `tt-smi -r` makes", "")


def test_the_pitch_movement_card_states_the_numbers_the_code_uses():
    from tensixinet.music.devices import QUANTUS_SHIFT_DEGREES, VIBRATO_HZ, VIBRATO_MAX_ST
    from tensixinet.music.patch import QUANTUS_MOD_RANGE
    section = HTML[HTML.index('id="bus"'): HTML.index('id="styles"')]
    card = section[section.index("<h3>A line that moves</h3>"):]
    card = card[: card.index("</div>")]
    for text in (f"up to {QUANTUS_SHIFT_DEGREES} scale degrees", f"{VIBRATO_HZ:g} Hz vibrato", f"up to {round(VIBRATO_MAX_ST * 100)} cents",
                 f"{QUANTUS_MOD_RANGE[0]:.0%} to {QUANTUS_MOD_RANGE[1]:.0%}", ">quantus_mod<", "ttcoco"):
        assert text in card, text


def test_the_pitch_movement_has_a_held_and_a_moving_clip_from_the_same_patch_and_readings():
    listen = HTML[HTML.index('id="listen"'): HTML.index('id="new"')]
    for name in ("quantus_fixed", "quantus_moving"):
        assert f'src="assets/audio/{name}.mp3"' in listen and (SITE / "assets" / "audio" / f"{name}.mp3").is_file(), name
    assert "same patch and the same synthetic readings" in listen and listen.index("quantus_fixed") < listen.index("quantus_moving")
    script = (SITE.parent / "demo" / "make_clips.sh").read_text()
    assert 'd["quantus_mod"]=0.0' in script and "--mock-chips 4 --seed 3" in script    # the held clip is the moving one's patch with the modulation off


# --- the VCV Rack voice on top of the MIDI --------------------------------------------------------------------------------------

def test_the_voice_section_has_the_recording_its_poster_and_the_patchs_own_numbers():
    from tensixinet.score import vcv
    from tensixinet.score.routing import DEFAULT_ROUTES
    section = HTML[HTML.index('id="voice"'): HTML.index('id="bio"')]
    assert 'href="#voice"' in HTML and HTML.index('id="space"') < HTML.index('id="voice"') < HTML.index('id="bio"')
    assert 'src="assets/voice.mp4"' in section and 'poster="assets/voice-frame.png"' in section
    for name in ("voice.mp4", "voice-frame.png"):
        assert (SITE / "assets" / name).is_file(), name
    assert (SITE.parent / "vcv" / "tensixinet-voice.vcv").is_file() and "vcv/tensixinet-voice.vcv" in section
    for text in (f"MIDI channel {DEFAULT_ROUTES.note_channels['lead'] + 1}", f"cutoff is at {vcv.CUTOFF:.2f}", f"resonance at {vcv.RESONANCE:.2f}",
                 f"by {DEFAULT_ROUTES.bend_range:g} semitones", "--midi-port virtual", "--device auto", "--midi-offset-ms"):
        assert text in section, text


def test_the_voice_section_says_what_was_measured_and_what_was_not_built():
    section = HTML[HTML.index('id="voice"'): HTML.index('id="bio"')]
    for text in ("440.0 Hz", "493.8 Hz", "392.0 Hz", "30 ms", "first guess", "demo/vcv_record.sh", "is not built"):
        assert text in section, text
    assert 'href="https://github.com/tsingletaryTT/tt-cv-agency"' in section and "docs/MIDI.md" in section


# --- the tt-bio-demo recording ------------------------------------------------------------------------------------------------

def test_the_bio_section_has_the_recording_its_poster_and_the_patch_it_replays_with_the_patchs_own_numbers():
    from tensixinet.music.patch import load_patch
    section = HTML[HTML.index('id="bio"'): HTML.index('id="app"')]
    assert 'href="#bio"' in HTML and HTML.index('id="space"') < HTML.index('id="bio"') < HTML.index('id="app"')
    assert 'src="assets/bio.mp4"' in section and 'poster="assets/bio-frame.png"' in section
    for name in ("bio.mp4", "bio-frame.png"):
        assert (SITE / "assets" / name).is_file(), name
    patch = load_patch(SITE.parent / "demo" / "bio.patch.json")
    for text in ("--patch demo/bio.patch.json", patch.scale_name, patch.gong_tuning.replace("_", " "), f"snare at {patch.snare:g}",
                 f"{patch.coco_send:.0%} of the band", f"{patch.rests:.0%} of the coco's notes", f"pitch movement {patch.quantus_mod:g}"):
        assert text in section, text
    assert [r for r in patch.steam_routes] == ["avdog", "pitch"] and "<code>avdog</code> and <code>pitch</code>" in section


def test_the_bio_section_says_what_ran_where_and_links_to_both_projects():
    section = HTML[HTML.index('id="bio"'): HTML.index('id="app"')]
    for text in ("all four chips", "tt-bio 0.21.0", "tt-smi", "on the CPU", "the booth holds the chips"):
        assert text in section, text
    assert 'href="https://github.com/tsingletaryTT/tt-bio-demo"' in section and 'href="https://github.com/moritztng/tt-bio"' in section
    assert "no questions" in section.lower()   # the affinity-question mode was not part of this recording


def test_no_file_the_site_serves_is_over_what_github_accepts():
    limit = 95 * 1024 * 1024   # GitHub refuses a file over 100 MB
    big = [str(p.relative_to(SITE)) for p in SITE.rglob("*") if p.is_file() and p.stat().st_size > limit]
    assert not big, big


def test_the_app_section_describes_the_play_tab_with_its_screenshot_and_the_numbers_the_code_has():
    from tensixinet.ui.knobs import KNOB_RANGES, MAX_CHIPS
    section = HTML[HTML.index('id="app"'): HTML.index('id="band"')]
    assert 'src="assets/play-tab.png"' in section and (SITE / "assets" / "play-tab.png").is_file()
    for text in ("Play tab", "flow chart", "memory error", "thermal trip", "load burst", "Load from the chips", f"{MAX_CHIPS} chips", f"{len(KNOB_RANGES)} sliders"):
        assert text in section, text
    assert "rhythm section" in section and "delay and echo" in section  # the modules a visitor can pick
