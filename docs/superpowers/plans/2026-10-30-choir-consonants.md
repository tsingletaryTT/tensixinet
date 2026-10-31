# Choir Consonants Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `ttchoir` sings syllables (onset consonant, vowel, sometimes a closing consonant) that start on a trigger from a source the patch names. It also gets a rare self-oscillation howl, a pressure-driven speech-chip crush, drifting range, and a consonant and vowel pick in the Play tab.

**Architecture:** Trigger sources live in `music/devices.py` beside `strike_times`. Each source is recomputed from the choir's own chip with the function the source device uses, and returns `(times, strengths)`. Consonant synthesis is a set of pure functions in a new `music/consonants.py`. `ttchoir.py` keeps the old vowel choir verbatim as `_vowel_choir`. With `patch.choir_onset == ""` it is called and nothing else runs. Otherwise `_syllable_choir` plans syllables from the triggers, renders the voiced part on the existing frame grid (additive formants) and adds consonant noise, the howl and the crush. New `Patch` fields come from their own random stream. Saved patches without them replay byte for byte.

**Tech Stack:** Python 3.10+, numpy only (no scipy; filters are FFT masks or closed forms), pytest. Gradio only in `app/` (`.venv-app`).

**Spec:** `docs/superpowers/specs/2026-10-30-choir-consonants-design.md` (read it whole, including "Decisions from the builder check", which bind this plan).

## Global Constraints

* The owner's rule: **nothing is committed**. Every task ends with the changes left uncommitted.
* `CLAUDE.md` rules: test first and watch it fail for the right reason; then a mutation check (`PYTHONDONTWRITEBYTECODE=1`, confirm the test was green before mutating, and confirm each scripted mutation changed the file). Every new `.py` file gets the SPDX header (`# SPDX-License-Identifier: Apache-2.0` / `# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.`) and a test file in the matching `tests/` folder. Never import `ttnn`, `ttl`, `ttml`, `torch` or `sounddevice`. Docs that name numbers change in the same task as the code.
* Writing: plain English, one idea per sentence, no "X, not Y" framings, no aphoristic closers. Sweep docs for `, not `, `rather than`, `instead of`, `reads as`, `honest`, `the one`, `which is why`, `This is what`.
* "No words, no word list, no text-to-speech, and no language model." The code holds no word list.
* "No trigger, no syllable." With a trigger source on, the voice sings a syllable on each trigger and is silent between them, apart from the last vowel dying away over its release.
* "The new fields off" means no source (`choir_onset == ""`): the old choir, on its bar clock, exactly as before, byte for byte. With no onset source every other choir field is ignored.
* Retrigger: a syllable is cut at the next trigger, its vowel crossfading into the next onset over **20 ms**. Its length is clamped to the gap to the next trigger. The drifting syllable length is the longest it may be.
* Strength: a source with none (the register, the roll pulses) gives a constant **0.5**.
* Self-oscillation: time cap **4 seconds**, peak limited to **-1 dBFS** (`10 ** (-1 / 20)` = 0.8913), feedback gain cut when the level passes the ceiling. It may fire during a syllable and replaces it. It is logged when it fires.
* Per chip: a source is recomputed from the choir chip's own readings. The choir does not follow another chip's device.
* The bit-crush is inside the choir's render (cables see the crushed voice) and is off at rest.
* Segments: triggers come from the source's own segment (`seed // SEGMENT_SEED_STRIDE`). Syllable state does not carry over.
* `DEVICE_INPUTS["ttchoir"]` lists the choir's own fields plus those of every source a patch can draw. The inputs test runs once per source.
* The Play tab picks the next consonant and vowel. It has no fire button.
* Not a change to any other device's sound, and not a TT-Lang kernel. Measured only; nobody has listened.
* The working tree already holds uncommitted changes from other work (among them the enya rebalance in `styles.py`, `tests/music/test_styles.py`, `docs/SOUND.md`, `README.md`, `site/index.html` and `CLAUDE.md`). Edit on top of them and never revert or stash them. Task 9's replay check compares `ttchoir` only, so these changes do not affect it.
* Tasks 1 to 3 must not touch `tensixinet/music/ttchoir.py`. Task 4's characterization depends on it being as it is at `HEAD`.

## Review Focus

1. **A source that never fires on this chip.** `FALLBACK` has every Tensix column enabled, so the register source gives no triggers. A chip with no lanes up has no heard roll. A 1.2 s Play-tab slice often has no gong strike. Expected: silence of the right length and dtype, no exception, and a log line that says the choir is waiting on its source. Test in Task 4 (`test_a_source_with_no_triggers_gives_silence_of_the_right_length`).
2. **Low sample rates.** Fast mode renders at 22050 Hz, tests at 8000. "s" is centred at 6500 Hz, which is past Nyquist at 8000. Expected: every band is clamped under 0.45 of the rate and nothing is sung above 0.46 of it. Tests in Task 3 (`test_bands_are_clamped_under_the_nyquist_limit`) and Task 4 (`test_nothing_is_sung_above_the_nyquist_limit_with_every_source`).
3. **Dense triggers.** A 14-bar ttrax at aiclk 1350, or a roll at a fast barre, gives triggers a few ms apart. Expected: triggers closer than 60 ms merge, every syllable is at least 60 ms long, the output is finite, and a 20 s render at 16 kHz stays within a time budget. Test in Task 4 (`test_dense_triggers_merge_and_render_in_time`).
4. **Renders cut short or straddling the end.** Durations of 0, 1 ms, 50 ms and 0.31 s, a trigger on the last sample, a howl that fires 0.1 s before the end. Expected: the right length, finite, and no index error. Tests in Task 4 (`test_very_short_renders_with_a_source_are_the_right_length_and_finite`) and Task 5 (`test_a_howl_that_fires_near_the_end_is_cut_cleanly`).
5. **A howl firing while one is already sounding** (howl odds 1.0 on a dense source). Expected: one howl at a time, never past 4 s from its own fire, never above -1 dBFS, and the crush after it never raises the peak. Tests in Task 5 (`test_a_howl_on_every_trigger_still_respects_the_cap_and_the_ceiling`) and Task 6 (`test_the_crush_never_raises_a_peak`).

---

### Task 1: Trigger sources

**Files:**
- Modify: `tensixinet/music/devices.py` (beside `strike_times`), `tensixinet/music/ttrax.py` (beside `gesture_envelopes`), `tensixinet/music/patch.py` (only the `CHOIR_SOURCES` constant)
- Test: `tests/music/test_choir_triggers.py` (new; `devices.py` already has its test file, so this extra file is allowed), `tests/music/test_ttrax.py`

**Interfaces:**
- Consumes: `strike_times`, `roll_pulses`, `ttrollz_lanes`, `ttrollz_pulse_times`, `ttquantus_gate`, `_note_rate` (devices.py); `register.step_times`; `lead.lead_notes`; `ttrax.gesture_envelopes`; `SEGMENT_SEED_STRIDE`.
- Produces:
  - `patch.CHOIR_SOURCES: Tuple[str, ...] = ("gong", "rolls", "register", "quantus", "lead", "rax")`. It lives in `patch.py` so `_validate` can use it; `devices.py` re-exports it.
  - `devices.SOURCE_SLOTS: Dict[str, int] = {"quantus": 2, "lead": 4}`: the chorus slot of the source device whose seed the choir borrows, so its rests and petals match the device that is heard.
  - `devices.NO_STRENGTH = 0.5`
  - `devices.CHOIR_SOURCE_INPUTS: Dict[str, frozenset]` (filled in Task 7; an empty dict is fine here)
  - `ttrax.gesture_times(bpm: float, n_bars: int, duration: float) -> Tuple[np.ndarray, np.ndarray]`: (times sorted ascending, `is_press` booleans) for every barre's press and release before `duration`.
  - `devices.choir_roll(t: Telemetry) -> Optional[int]`: the heard lane the choir follows (`heard[personality(t) % len(heard)]`), or None when no lane is up.
  - `devices.ttquantus_onsets(t, duration: float, sample_rate: int, seed: int, patch: Patch = DEFAULT_PATCH) -> np.ndarray`: start times of the held notes that play.
  - `devices.choir_triggers(source: str, t: Telemetry, duration: float, sample_rate: int, segment: int, patch: Patch = DEFAULT_PATCH) -> Tuple[np.ndarray, np.ndarray]`: (times ascending in `[0, duration)`, strengths in `[0, 1]`, same length). `ValueError` naming `CHOIR_SOURCES` for an unknown source.

Pinned source rules (the "my call" values the tests fix):

| source | times | strength |
|---|---|---|
| gong | `strike_times(t, duration, patch, segment)` | 1.0 (the gong's own render strikes every time at full level) |
| rolls | `roll_pulses(...)[choir_roll(t)][0]`, the whole roll as the gong follows it; with no lane up, `ttrollz_pulse_times(bpm, duration)` | 0.5 |
| register | `register.step_times(t, duration)`: every step that fires. The snare share is a level on the sound, and the register keeps stepping | 0.5 |
| quantus | `ttquantus_onsets(t, duration, sr, segment * SEGMENT_SEED_STRIDE + SOURCE_SLOTS["quantus"], patch)` | 0.5 |
| lead | `lead_notes(t, duration, sr, segment * SEGMENT_SEED_STRIDE + SOURCE_SLOTS["lead"], patch)` starts | 1.0 on a phrase's first note, 0.5 otherwise |
| rax | `gesture_times(aiclk_to_bpm(t.aiclk), patch.rax_bars, duration)` | 1.0 on a press, 0.5 on a release |

- [ ] **Step 1: Write the failing tests**

```python
def test_the_sources_are_the_six_the_spec_names_and_the_borrowed_slots_match_the_chorus():
    assert CHOIR_SOURCES == ("gong", "rolls", "register", "quantus", "lead", "rax")
    from tensixinet.music.chorus import DEVICES
    names = [d.__name__ for d in DEVICES]
    assert SOURCE_SLOTS == {"quantus": names.index("ttquantus"), "lead": names.index("lead")}

def test_gong_triggers_are_the_strikes_at_full_strength():
    times, strengths = choir_triggers("gong", ROLLED, 15.0, SR, 3)
    assert np.array_equal(times, strike_times(ROLLED, 15.0, DEFAULT_PATCH, 3)) and (strengths == 1.0).all()

def test_register_and_roll_triggers_have_the_constant_strength_of_one_half():
    for source in ("register", "rolls"):
        _, s = choir_triggers(source, HARVESTED, 15.0, SR, 0)
        assert len(s) > 0 and (s == 0.5).all()

def test_register_triggers_ignore_the_snare_share():
    quiet = dataclasses.replace(DEFAULT_PATCH, snare=0.0)
    assert np.array_equal(choir_triggers("register", HARVESTED, 15.0, SR, 0, quiet)[0], step_times(HARVESTED, 15.0))

def test_quantus_onsets_are_the_notes_the_score_says_play():
    p = dataclasses.replace(DEFAULT_PATCH, rests=0.3)
    seed = 2 * SEGMENT_SEED_STRIDE + 2
    expected = [n.t for n in quantus_events(FALLBACK, 0, 8.0, SR, seed, p)]
    assert np.allclose(ttquantus_onsets(FALLBACK, 8.0, SR, seed, p), expected)
    assert np.allclose(choir_triggers("quantus", FALLBACK, 8.0, SR, 2, p)[0], expected)

def test_lead_triggers_are_its_notes_and_a_phrases_first_note_is_strong():
    times, strengths = choir_triggers("lead", FALLBACK, 8.0, SR, 0)
    assert np.allclose(times, [n[0] for n in lead_notes(FALLBACK, 8.0, SR, SOURCE_SLOTS["lead"], DEFAULT_PATCH)])
    assert strengths[0] == 1.0 and set(strengths.tolist()) == {0.5, 1.0}

def test_rax_gestures_fall_where_the_envelopes_peak():  # in tests/music/test_ttrax.py
    times, press = gesture_times(100.0, 7, 10.0)
    assert (np.diff(times) >= 0).all() and press.any() and (~press).any()
    # each press is where that barre's press envelope is 1, each release where its release envelope is 1

def test_rax_triggers_follow_the_patchs_bar_count():
    small = choir_triggers("rax", FALLBACK, 10.0, SR, 0, dataclasses.replace(DEFAULT_PATCH, rax_bars=4))[0]
    big = choir_triggers("rax", FALLBACK, 10.0, SR, 0, dataclasses.replace(DEFAULT_PATCH, rax_bars=14))[0]
    assert len(big) > len(small)

def test_a_chip_with_no_lane_up_falls_back_to_its_clock():
    bare = dataclasses.replace(FALLBACK, ddr_status=(0, 0, 0, 0), eth_mask=0)
    assert choir_roll(bare) is None
    assert np.allclose(choir_triggers("rolls", bare, 6.0, SR, 0)[0], ttrollz_pulse_times(aiclk_to_bpm(bare.aiclk), 6.0))

def test_the_segment_moves_the_roll_and_gong_triggers():
    assert not np.array_equal(choir_triggers("rolls", ROLLED, 15.0, SR, 0)[0], choir_triggers("rolls", ROLLED, 15.0, SR, 1)[0])

def test_every_source_gives_times_inside_the_render_and_matching_lengths():
    for source in CHOIR_SOURCES:
        times, strengths = choir_triggers(source, HARVESTED, 5.0, SR, 0)
        assert len(times) == len(strengths) and ((times >= 0) & (times < 5.0)).all() and (np.diff(times) >= 0).all()
        assert ((strengths >= 0) & (strengths <= 1)).all()

def test_an_unknown_source_is_refused_with_the_list():
    with pytest.raises(ValueError, match="gong, rolls, register"):
        choir_triggers("snare", FALLBACK, 1.0, SR, 0)
```

Fixtures: `SR = 8000`; `ROLLED = replace(FALLBACK, tensix_mask=0x3002, ddr_status=(1, 1, 1, 1))` (its gong follows a roll); `HARVESTED = replace(FALLBACK, tensix_mask=0x37F7, ddr_status=(1, 1, 1, 1))` (its register fires).

- [ ] **Step 2: Run them and see them fail**

Run: `python3 -m pytest tests/music/test_choir_triggers.py tests/music/test_ttrax.py -q`
Expected: FAIL on import (`cannot import name 'choir_triggers'`, `'gesture_times'`).

- [ ] **Step 3: Implement**

Write the signatures above in their files. `ttquantus_onsets` follows `score.collect.quantus_events` exactly: a note starts at `k / _note_rate(t)` and plays when `ttquantus_gate` at its middle sample is above 0.5. Leave `collect.py` alone, because the test ties the two together. The lead's phrase-first rule uses `lead_notes`' own grid: `eighth = round(30 / bpm * sr)` samples, `bar = eighth * len(patch.motif)`, and a note is first when `round(start * sr) % bar == 0`. Do not import `ttsteam` into `devices.py`: `ttsteam` imports `devices`. `register.step_times` is the source.

- [ ] **Step 4: Run them and see them pass**

Run: `python3 -m pytest tests/music/test_choir_triggers.py tests/music/test_ttrax.py tests/music/test_devices.py -q`
Expected: PASS.

- [ ] **Step 5: Mutation check**

Try each of these and confirm that a test goes red: gong strength 0.9; `NO_STRENGTH` 0.6; the register using `steam_hits` (the snare-thinned list); `SOURCE_SLOTS["lead"]` set to 5; `choir_roll` using `heard[0]`; the lead's first-note test dropping `% bar`; `gesture_times` leaving out releases; the rax source ignoring `patch.rax_bars` (fixed at 7); `segment` not passed to `roll_pulses`; the unknown-source check removed. For each survivor, add a test that changes only that one thing.

- [ ] **Step 6: Leave the changes uncommitted.**

---

### Task 2: Patch fields, styles and the patch log

**Files:**
- Modify: `tensixinet/music/patch.py`, `tensixinet/music/styles.py`, `tensixinet/narrate/explain.py` (`explain_patch`)
- Test: `tests/music/test_patch.py`, `tests/music/test_styles.py`, `tests/narrate/test_explain.py`

**Interfaces:**
- Consumes: `CHOIR_SOURCES` (Task 1).
- Produces (new `Patch` fields, all defaulted so `DEFAULT_PATCH` and old patches mean "off"):
  - `choir_onset: str = ""`: the source whose triggers start syllables; "" is the old choir
  - `choir_coda: str = ""`: the source of closing consonants; "" is none
  - `choir_howl: str = ""`: the source that may fire self-oscillation; "" is none
  - `choir_howl_odds: float = 0.0`: the share of that source's triggers that fire it
  - `choir_mix: Tuple[float, float, float] = (1.0, 1.0, 1.0)`: the family weights (plosive, fricative, nasal)
  - `choir_crush: float = 0.0`: the speech-chip crush at full pressure
  - `choir_drift: float = 0.0`: how far the voice type, register and syllable length drift
  - `patch._draw_choir(seed: int) -> Dict[str, object]`: those seven fields as keyword arguments, from `np.random.default_rng([seed, 0xC401])`
  - `Style.choir_mix: Tuple[float, float, float]`, `Style.choir_crush: float`

Pinned draw values:
* `CHOIR_ONSET_ODDS = (0.25, 0.2, 0.15, 0.15, 0.15, 0.1)` in `CHOIR_SOURCES` order. A drawn patch always has an onset source.
* Coda: none 40%, the onset's own source 30%, another source 30% (chosen evenly from the other five).
* Howl: a source chosen evenly from the six; `CHOIR_HOWL_ODDS_RANGE = (0.02, 0.08)`.
* `CHOIR_MIX_RANGE = (0.2, 1.0)` per family, `CHOIR_CRUSH_RANGE = (0.3, 1.0)`, `CHOIR_DRIFT_RANGE = (0.3, 1.0)`. Floats are rounded to 3 places, as the other draws are.

Pinned style values (sources still come from the seed through `_draw_choir`; the style overrides mix and crush):

| style | choir_mix (plosive, fricative, nasal) | choir_crush |
|---|---|---|
| chiptune | (0.5, 0.3, 0.2) | 1.0 |
| enya | (0.1, 0.3, 0.6) | 0.0 |
| blues | (0.7, 0.15, 0.15) | 0.3 |
| aphex | (0.4, 0.4, 0.2) | 0.8 |

Validation (`_validate`): each source field is "" or in `CHOIR_SOURCES`; `choir_coda` or `choir_howl` set while `choir_onset` is "" is refused ("a coda or howl needs an onset source"); `0 <= choir_howl_odds <= 1`; `choir_mix` has three finite non-negative weights with a positive sum; `0 <= choir_crush <= 1`; `0 <= choir_drift <= 1`. NaN fails each range check. `patch_from_dict` reads each field with the default above.

- [ ] **Step 1: Write the failing tests**

```python
def test_a_drawn_patch_always_has_an_onset_source_and_the_choir_fields_in_range():
    for seed in range(40):
        p = draw_patch(seed, CHIPS)
        assert p.choir_onset in CHOIR_SOURCES and p.choir_howl in CHOIR_SOURCES
        assert p.choir_coda in ("",) + CHOIR_SOURCES and 0.02 <= p.choir_howl_odds <= 0.08
        assert all(0.2 <= w <= 1.0 for w in p.choir_mix) and 0.3 <= p.choir_crush <= 1.0 and 0.3 <= p.choir_drift <= 1.0

def test_the_choir_draw_has_its_own_stream_so_no_other_field_moved():
    p = draw_patch(11, CHIPS)
    old = dataclasses.replace(p, **{f: getattr(DEFAULT_PATCH, f) for f in CHOIR_FIELDS})
    assert old == OLD_DRAW_11  # every field of draw_patch(11) as taken before this task, pinned as literals

def test_the_coda_odds_are_none_same_and_other():
    codas = [(p.choir_coda == "", p.choir_coda == p.choir_onset) for p in (draw_patch(s, CHIPS) for s in range(2000))]
    # none about 0.40, same about 0.30, another about 0.30, each within 0.04

def test_onset_sources_follow_their_odds():  # 4000 draws, each share within 0.03 of CHOIR_ONSET_ODDS

def test_an_old_saved_patch_has_no_choir_source_and_round_trips(tmp_path):
    p = patch_from_dict(OLD_DICT)  # a dict saved before this change, with no choir fields
    assert p.choir_onset == "" and p.choir_mix == (1.0, 1.0, 1.0) and p.choir_crush == 0.0
    drawn = draw_patch(5, CHIPS); save_patch(drawn, tmp_path / "p.json")
    assert load_patch(tmp_path / "p.json") == drawn

@pytest.mark.parametrize("bad", [{"choir_onset": "snare"}, {"choir_coda": "gong"}, {"choir_howl": "gong"},
    {"choir_onset": "gong", "choir_howl_odds": 1.5}, {"choir_onset": "gong", "choir_mix": [0, 0, 0]},
    {"choir_onset": "gong", "choir_mix": [1, 1]}, {"choir_onset": "gong", "choir_crush": float("nan")},
    {"choir_onset": "gong", "choir_drift": -0.1}])
def test_a_broken_choir_field_is_refused(bad): ...  # ValueError

def test_each_style_sets_the_choirs_consonant_mix_and_crush():  # in tests/music/test_styles.py
    assert {n: (s.choir_mix, s.choir_crush) for n, s in STYLES.items()} == {...the table above...}
    enya, blues = STYLES["enya"].choir_mix, STYLES["blues"].choir_mix
    assert enya[0] < enya[2] and blues[0] > max(blues[1:])  # enya leans soft, blues plosive
    for name in STYLES:
        p = draw_styled_patch(3, CHIPS, name)
        assert (p.choir_mix, p.choir_crush) == (STYLES[name].choir_mix, STYLES[name].choir_crush)
        assert p.choir_onset == draw_patch(3, CHIPS).choir_onset  # the sources still come from the seed

def test_the_patch_log_names_the_choirs_sources_and_numbers():  # in tests/narrate/test_explain.py
    p = dataclasses.replace(DEFAULT_PATCH, choir_onset="gong", choir_coda="rolls", choir_howl="register",
                            choir_howl_odds=0.05, choir_mix=(0.5, 0.3, 0.2), choir_crush=0.6, choir_drift=0.4)
    text = " | ".join(d.what for d in explain_patch(p, CHIPS, "test"))
    assert "choir syllables on the gong's strikes" in text and "codas on the rolls" in text
    assert "5% of the register's hits" in text and "4 s" in " ".join(d.why for d in explain_patch(p, CHIPS, "test"))
    off = " | ".join(d.what for d in explain_patch(DEFAULT_PATCH, CHIPS, "test"))
    assert "choir: vowels on the bar" in off
```

`CHOIR_FIELDS` is the tuple of the seven new field names, exported from `patch.py`. `OLD_DRAW_11` is the `Patch` that `draw_patch(11, CHIPS)` returns before this task, written out as literals in the test file. Take it before editing `patch.py`.

- [ ] **Step 2: Run them and see them fail**

Run: `python3 -m pytest tests/music/test_patch.py tests/music/test_styles.py tests/narrate/test_explain.py -q`
Expected: FAIL (`unexpected keyword 'choir_onset'`, `cannot import name 'CHOIR_FIELDS'`, a missing log line).

- [ ] **Step 3: Implement**

Add the fields after `quantus_mod`. Add `_draw_choir`, call it in `draw_patch` (`**_draw_choir(seed)`) and in `draw_styled_patch` (then override mix and crush). Extend `_validate`, `patch_from_dict` and `Style`. In `explain.py`, `_choir_decisions(patch) -> List[Decision]` goes into `explain_patch` after `_pitch_movement_decision`. With no onset it gives one line: `"choir: vowels on the bar, no consonants"`. Otherwise it gives one line each for the sources, the consonant mix (each family's share of the weights as a percentage), the howl (its share of the source's triggers, "at most 4 s, held under -1 dBFS"), the crush ("none at rest") and the drift. Each line's `why` says "set by the X style" or the drawn range, as `_rests_decision` does.

- [ ] **Step 4: Run them and see them pass**

Run: `python3 -m pytest tests/music tests/narrate -q`
Expected: PASS. Every existing pinned draw test (`test_patch.py`, `test_styles.py`) still passes unchanged.

- [ ] **Step 5: Mutation check**

Try: the choir draw taken from the main `rng` (the own-stream test must go red); `CHOIR_ONSET_ODDS` reordered; coda odds swapped; the howl odds range widened to (0.01, 0.08); `patch_from_dict` defaulting `choir_mix` to `(0, 0, 0)`; the onset-required check removed; the NaN check written as `p.choir_crush > 1` only; a style's mix not applied; the patch log missing the howl's percentage.

- [ ] **Step 6: Leave the changes uncommitted.**

---

### Task 3: Consonant synthesis

**Files:**
- Create: `tensixinet/music/consonants.py`
- Test: `tests/music/test_consonants.py`

**Interfaces:**
- Consumes: numpy only. This module imports nothing from the package, so `patch.py` can import from it in Task 8.
- Produces:
  - `Consonant(NamedTuple)`: `family: str` ("plosive", "fricative" or "nasal"), `voiced: bool`, `centre_hz: float`, `width_hz: float`, `level: float`
  - `CONSONANTS: Dict[str, Consonant]`, `PLOSIVES = ("b", "d", "g", "p", "t", "k")`, `FRICATIVES = ("h", "f", "sh", "s")`, `NASALS = ("m", "n")`. Each is ordered from soft to sharp.
  - `FAMILIES = ("plosive", "fricative", "nasal")` (the order of `choir_mix`)
  - `band_noise(n: int, centre_hz: float, width_hz: float, sweep: float, sample_rate: int, rng: np.random.Generator) -> np.ndarray`: unit RMS noise in a band whose centre moves by `sweep` times itself over the length
  - `fricative_noise(name: str, n: int, sample_rate: int, rng, rising: bool = True) -> np.ndarray`
  - `burst(name: str, sample_rate: int, rng) -> np.ndarray`: `BURST_S` long
  - `nasal_gain(freqs: np.ndarray, name: str) -> np.ndarray`
  - `channel_sharpness(gddr_temps: Sequence[float], channel: int) -> float`: 0 for the coolest channel, 1 for the hottest, 0.5 when the half spread is under 0.25 C (the rule in `ttchoir.vowel_sequence`)
  - `pick_consonant(sharpness: float, strength: float, mix: Tuple[float, float, float], u: float) -> str`

Pinned inventory (`centre_hz`, `width_hz`, `level`):
s fricative 6500, 2500, 0.35 · sh fricative 3200, 1600, 0.4 · f fricative 5000, 5000, 0.15 · h fricative 1500, 2500, 0.2 · p/b plosive 900, 1200, 0.5 · t/d plosive 4500, 3000, 0.5 · k/g plosive 2200, 1200, 0.5 · m nasal (notch) 1000, 300 · n nasal (notch) 1800, 400. The voiced members are b, d, g, m and n.

Pinned timing and shape: `FRICATIVE_S = 0.09`, `CODA_FRICATIVE_S = 0.12`, `FRICATIVE_SWEEP = 0.15`, `CLOSURE_S = 0.05`, `BURST_S = 0.015`, `BURST_ATTACK_S = 0.001`, `ONSET_GLIDE_S = 0.03`, `NASAL_S = 0.07`, `NASAL_DEPTH = 0.95`, `VOICE_BAR_LEVEL = 0.12`, `VOICE_BAR_CUTOFF_HZ = 500.0`, `EDGE_FADE_S = 0.005`. A band's centre is clamped to `0.42 * sample_rate`, and its width shrinks so that `centre + width / 2 <= 0.45 * sample_rate`.

The tests do not determine the band-noise algorithm, so it is fixed here. Low-pass white noise to `width / 2` with an rfft mask (a raised-cosine edge). Multiply it by `cos(2π ∫ centre(t) dt)`, with `centre(t)` linear from `centre` to `centre * (1 ± sweep)`. Normalise to unit RMS. No sample-by-sample loop.

`pick_consonant`: the family weights are `mix[0] * strength`, `mix[1] * (1 - strength)` and `mix[2] * (1 - strength)`. When all three are 0, use `mix` itself. The family is the first whose cumulative normalised weight exceeds `u`. The member is `family[round(sharpness * (len(family) - 1))]`.

- [ ] **Step 1: Write the failing tests**

```python
def spectral_centre(x, sr): ...  # power-weighted mean frequency of the rfft

def test_the_inventory_is_three_families_ordered_soft_to_sharp():
    assert set(CONSONANTS) == set(PLOSIVES + FRICATIVES + NASALS) and FAMILIES == ("plosive", "fricative", "nasal")
    assert {c for c, v in CONSONANTS.items() if v.voiced} == {"b", "d", "g", "m", "n"}
    assert CONSONANTS["s"][2:] == (6500.0, 2500.0, 0.35) and CONSONANTS["m"].centre_hz == 1000.0

def test_s_is_brighter_than_sh_and_each_fricative_sits_at_its_centre():
    sr, rng = 44100, np.random.default_rng(0)
    s, sh = fricative_noise("s", 4000, sr, rng), fricative_noise("sh", 4000, sr, rng)
    assert spectral_centre(s, sr) == pytest.approx(6500 * 1.075, rel=0.15)  # the centre sweeps up 15% over the sound
    assert spectral_centre(s, sr) > 1.6 * spectral_centre(sh, sr)

def test_a_band_moves_with_its_sweep():  # first quarter's centre below the last quarter's by at least 10% for sweep 0.15

def test_bands_are_clamped_under_the_nyquist_limit():
    for name in CONSONANTS:
        if CONSONANTS[name].family != "nasal":
            x = fricative_noise(name, 800, 8000, np.random.default_rng(1)) if name in FRICATIVES else burst(name, 8000, np.random.default_rng(1))
            assert band_share(x, 0.46 * 8000, 4000, 8000) < 0.02 and np.isfinite(x).all()

def test_a_burst_has_a_short_attack_so_it_does_not_click():
    b = burst("t", 44100, np.random.default_rng(2))
    assert len(b) == int(0.015 * 44100) and abs(b[0]) < 0.05 * np.abs(b).max()
    assert np.argmax(np.abs(b)) >= int(0.001 * 44100) - 1 and np.abs(b[-10:]).max() < 0.05 * np.abs(b).max()

def test_a_nasal_cuts_a_notch_and_leaves_the_rest():
    f = np.array([250.0, 1000.0, 2000.0, 4000.0])
    g = nasal_gain(f, "m")
    assert g[1] <= 0.1 and g[0] >= 0.9 and g[2] >= 0.9 and nasal_gain(np.array([1800.0]), "n")[0] <= 0.1

def test_a_hot_channel_and_a_strong_trigger_give_a_sharp_plosive():
    assert pick_consonant(1.0, 1.0, (1.0, 1.0, 1.0), 0.5) == "k"
    assert pick_consonant(0.0, 1.0, (1.0, 1.0, 1.0), 0.5) == "b"

def test_a_weak_trigger_leans_to_fricatives_and_nasals_and_a_cool_channel_to_soft_ones():
    picks = [pick_consonant(0.0, 0.2, (1.0, 1.0, 1.0), u) for u in np.linspace(0, 0.999, 200)]
    assert sum(c in FRICATIVES + NASALS for c in picks) > 0.8 * len(picks) and set(picks) <= {"b", "h", "m"}

def test_the_mix_weights_the_families():  # mix (0, 0, 1) always gives a nasal; (1, 0, 0) at strength 0 falls back to plosives

def test_channel_sharpness_follows_vowel_sequences_rule():
    assert channel_sharpness((40.0,) * 8, 3) == 0.5
    temps = (30.0, 35.0, 40.0, 45.0, 50.0, 38.0, 32.0, 48.0)
    assert channel_sharpness(temps, 0) == 0.0 and channel_sharpness(temps, 4) == 1.0
```

- [ ] **Step 2: Run them and see them fail**

Run: `python3 -m pytest tests/music/test_consonants.py -q`
Expected: FAIL (`No module named 'tensixinet.music.consonants'`).

- [ ] **Step 3: Implement `consonants.py`** with the interfaces and pinned values above. Add the SPDX header.

- [ ] **Step 4: Run them and see them pass**

Run: `python3 -m pytest tests/music/test_consonants.py tests/test_hygiene.py -q`
Expected: PASS.

- [ ] **Step 5: Mutation check**

Try: s and sh centres swapped; `BURST_ATTACK_S` set to 0; the burst's tail fade removed; the Nyquist clamp at 0.5 instead of 0.42; the width not shrunk; `NASAL_DEPTH` 0.5; the strength terms swapped between plosives and fricatives; the zero-weights fallback removed (a ZeroDivisionError must be caught by a test); `round` changed to `int` in the member choice; the sweep sign flipped.

- [ ] **Step 6: Leave the changes uncommitted.**

---

### Task 4: The event-gated syllable choir

**Files:**
- Modify: `tensixinet/music/ttchoir.py`
- Test: `tests/music/test_ttchoir.py`

**Interfaces:**
- Consumes: `choir_triggers`, `CHOIR_SOURCES` (Task 1); the `Patch` fields (Task 2); everything in `consonants.py` (Task 3); `rests.rest_flags`; `SEGMENT_SEED_STRIDE`.
- Produces:
  - `Syllable` (frozen dataclass): `start: float`, `length: float`, `onset: str` (a consonant name), `vowel: str` (a letter of `VOWEL_ORDER`), `coda: str` ("" for none), `coda_at: float` (seconds from the render's start; equal to `start + length` when there is no coda), `strength: float`, `channel: int`
  - `merge_triggers(times: np.ndarray, strengths: np.ndarray, min_gap: float = MIN_TRIGGER_GAP_S) -> Tuple[np.ndarray, np.ndarray]`: drops a trigger closer than `min_gap` to the last kept one; the kept one takes the larger strength
  - `split_alternate(times, strengths) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]`: even triggers start syllables, odd triggers close them. It is used when `choir_coda == choir_onset`, because with one source a closing trigger otherwise never lands inside a syllable.
  - `syllable_limit(t: Telemetry, seed: int, patch: Patch, at: np.ndarray) -> np.ndarray`: the longest length in seconds for a syllable starting at each time
  - `plan_syllables(t: Telemetry, duration: float, seed: int, patch: Patch, onsets: Tuple[np.ndarray, np.ndarray], codas: Optional[Tuple[np.ndarray, np.ndarray]]) -> List[Syllable]`
  - `ttchoir(...)` unchanged signature. It calls `_vowel_choir` (the old body, moved verbatim) when `patch.choir_onset == ""`, else `_syllable_choir`.

Pinned values: `XFADE_S = 0.02`, `RELEASE_S = 0.15`, `MIN_TRIGGER_GAP_S = 0.06`, `DRIFT_BARS = 3` (the drift walks have a knot every 3 bars), `SYLLABLE_SHRINK_OCTAVES = 3` (with drift 1 the limit wanders from 4 beats down to 0.5 beats; with drift 0 it is 4 beats, the old bar), `VOICE_DRIFT_OCTAVES = 0.16` (the tract scale moves up to ±0.16 octave times drift around the chip's voice type, clipped to [0.92, 1.35]), `REGISTER_DEGREES = 3` (the pitch moves by up to `round(3 * drift)` whole scale degrees, held for a syllable). Random-stream tags: consonant choice `0xC05`, drift walks `0xD71F` (three walks: `0xD71F + 0`, `+1`, `+2`), noise `0xB425`. Each stream is seeded `[seed, tag, personality(t)]`.

Syllable rules (the tests fix them):
* Onset triggers are merged, then each starts a syllable. The length is `min(limit, gap to the next onset, duration - start + RELEASE_S)`, and at least `MIN_TRIGGER_GAP_S`.
* Syllable `j` reads GDDR channel `(personality(t) + j) % len(gddr_temps)`. Its vowel is `vowel_sequence(...)` at that channel. Its onset is `pick_consonant(channel_sharpness(...), strength, patch.choir_mix, u_j)`.
* A coda is the first closing trigger strictly inside `(start, start + length)`. It is picked with that trigger's strength and the next channel, and it ends the syllable `CODA_FRICATIVE_S` (fricative), `NASAL_S` (nasal) or `CLOSURE_S + BURST_S` (plosive) after the trigger, or at the next onset, whichever comes first.
* Voiced part: on the existing frame grid (`FRAME_HZ`), each frame belongs to the syllable that holds it. Its formants are the vowel's times the drifting tract scale. Its level is 0 in silence, ramps over `XFADE_S` where a syllable is cut by the next, and falls to 0 over `RELEASE_S` where a syllable ends before the next trigger. A plosive onset adds `CLOSURE_S` of closure before the burst. An unvoiced closure is silent. A voiced closure keeps only the harmonics under `VOICE_BAR_CUTOFF_HZ` at `VOICE_BAR_LEVEL`. After the burst the vowel ramps in over `ONSET_GLIDE_S`. A nasal applies `nasal_gain` to the harmonics for `NASAL_S` with the level at 0.6. Harmonic amplitudes are computed per frame with that frame's `f0`. The phase is the cumulative sum of a per-sample `f0` that glides between syllables over `GLIDE_S`, so a pitch change does not click.
* Noise part: fricatives and bursts are added in the sample domain at `CHOIR_LEVEL * level`, with a fade of `EDGE_FADE_S` at each edge.
* The output stays float32 with `|x| <= 1`.

- [ ] **Step 1: Take the characterization first**

Run `git diff --quiet HEAD -- tensixinet/music/ttchoir.py` and confirm it exits 0. Then compute SHA-256 of `ttchoir(t, d, sr, seed, patch).tobytes()` for these cases: chips `FALLBACK`, `singer()`, `singer(tensix_mask=12473, gddr_temps=(26.0, 99.0) * 4)` and `singer(thm_limit_throttle=61.0)`; `(d, sr)` of (6.0, 16000), (3.0, 8000) and (2.0, 44100); seeds 0 and 7; patches `DEFAULT_PATCH` and `dataclasses.replace(DEFAULT_PATCH, scale=SCALES["dorian"])`. Pin them as literals in:

```python
@pytest.mark.parametrize("case", OLD_CHOIR_HASHES)  # (chip name, duration, sr, seed, patch name, sha256 hex)
def test_with_no_onset_source_the_choir_is_byte_for_byte_the_old_one(case): ...

def test_with_no_onset_source_every_other_choir_field_is_ignored():
    loud = dataclasses.replace(DEFAULT_PATCH, choir_mix=(0.1, 0.1, 5.0), choir_crush=1.0, choir_drift=1.0)
    assert ttchoir(singer(), 4.0, SR, 0, loud).tobytes() == ttchoir(singer(), 4.0, SR, 0).tobytes()
```

Run the test against the unchanged file. It must pass now.

- [ ] **Step 2: Write the failing tests**

Fixtures at the top of `tests/music/test_ttchoir.py`: `ROLLED` and `HARVESTED` as in Task 1; `CHIPS = mock_telemetry(4, seed=0)`; `rms(x) = sqrt(mean(x ** 2))`; `import time`; `draw_styled_patch` from `styles`.

```python
GONG = dataclasses.replace(DEFAULT_PATCH, choir_onset="gong")

def test_merge_drops_triggers_inside_the_gap_and_keeps_the_stronger():
    t, s = merge_triggers(np.array([0.0, 0.03, 0.2, 0.25]), np.array([0.5, 1.0, 0.5, 0.5]))
    assert t.tolist() == [0.0, 0.2] and s.tolist() == [1.0, 0.5]

def test_split_alternate_gives_every_other_trigger_to_the_codas(): ...

def test_each_syllable_starts_on_a_trigger_and_is_clamped_to_the_gap():
    on = (np.array([0.5, 0.9, 3.0]), np.array([1.0, 0.5, 0.5]))
    syl = plan_syllables(singer(), 6.0, 0, GONG, on, None)
    assert [x.start for x in syl] == [0.5, 0.9, 3.0] and syl[0].length == pytest.approx(0.4)
    assert syl[1].length <= syllable_limit(singer(), 0, GONG, np.array([0.9]))[0]

def test_with_no_drift_a_syllable_is_at_most_four_beats(): ...  # limit == 4 * 60 / bpm everywhere

def test_drift_shortens_syllables_and_moves_the_voice_over_bars_and_between_chips():
    drifting = dataclasses.replace(GONG, choir_drift=1.0)
    at = np.linspace(0, 60, 200)
    lim = syllable_limit(singer(), 0, drifting, at)
    beat = 60 / aiclk_to_bpm(singer().aiclk)
    assert lim.min() < 2 * beat and lim.max() <= 4 * beat + 1e-9 and lim.min() >= 0.5 * beat - 1e-9
    assert not np.allclose(lim, syllable_limit(singer(tensix_mask=12473), 0, drifting, at))

def test_a_coda_sounds_only_on_a_closing_trigger_inside_the_syllable():
    on = (np.array([0.0, 2.0]), np.array([0.5, 0.5])); co = (np.array([0.5, 2.5, 9.0]), np.full(3, 0.5))
    syl = plan_syllables(singer(), 4.0, 0, GONG, on, co)
    assert syl[0].coda and syl[0].coda_at == 0.5 and syl[1].coda and syl[1].coda_at == 2.5
    assert plan_syllables(singer(), 4.0, 0, GONG, on, None)[0].coda == ""

def test_the_same_source_for_onsets_and_codas_alternates(): ...  # ttchoir with choir_coda == choir_onset has codas

def test_the_choir_is_silent_between_syllables_and_sings_on_each_trigger():
    t = singer(gddr_temps=(40.0,) * 8)
    strikes = choir_triggers("gong", t, 15.0, SR, 0)[0]
    audio = ttchoir(t, 15.0, SR, 0, GONG)
    # rms in the 50 ms after each strike > 10x the rms in the 50 ms before it, when the gap before is longer than limit + RELEASE_S

def test_a_hot_channel_sings_sharp_onsets_and_a_cool_one_soft_ones(): ...  # via plan_syllables with fixed u

def test_an_unvoiced_closure_is_silent_and_a_voiced_one_hums_low():
    # chip with the onset forced through choir_mix (1, 0, 0) and strength 1: "pa"-like vs a cool chip for "ba"
    # closure window rms of the unvoiced < 0.01 x the vowel rms; voiced closure rms > 0.05 x, >= 80% of it under 600 Hz

def test_no_click_at_a_burst_or_a_retrigger():
    # at 44100 Hz on a dense source: rms of np.diff(audio) in any 2 ms window starting at a syllable start or a burst
    # is at most 3x the rms of np.diff over the middle 50 ms of that syllable's vowel

def test_same_readings_and_seed_give_the_same_line_and_another_seed_another(): ...

def test_a_source_with_no_triggers_gives_silence_of_the_right_length():
    reg = dataclasses.replace(DEFAULT_PATCH, choir_onset="register")
    audio = ttchoir(FALLBACK, 4.0, SR, 0, reg)  # every Tensix column enabled: the register never fires
    assert audio.dtype == np.float32 and audio.shape == (4 * SR,) and not audio.any()

@pytest.mark.parametrize("source", CHOIR_SOURCES)
def test_nothing_is_sung_above_the_nyquist_limit_with_every_source(source):
    for sr in (8000, 22050):
        a = ttchoir(HARVESTED, 6.0, sr, 0, dataclasses.replace(DEFAULT_PATCH, choir_onset=source, choir_mix=(1, 1, 1)))
        assert np.isfinite(a).all() and np.abs(a).max() <= 1.0 and band_share(a, 0.46 * sr, sr / 2, sr) < 1e-3

def test_dense_triggers_merge_and_render_in_time():
    p = dataclasses.replace(DEFAULT_PATCH, choir_onset="rax", rax_bars=14, choir_drift=1.0)
    fast = singer(aiclk=1350.0)
    start = time.perf_counter(); a = ttchoir(fast, 20.0, 16000, 0, p); took = time.perf_counter() - start
    syl = plan_syllables(fast, 20.0, 0, p, choir_triggers("rax", fast, 20.0, 16000, 0, p), None)
    assert min(s.length for s in syl) >= MIN_TRIGGER_GAP_S - 1e-9 and np.isfinite(a).all() and took < 4.0

@pytest.mark.parametrize("duration", [0.0, 0.001, 0.05, 0.2, 0.31])
def test_very_short_renders_with_a_source_are_the_right_length_and_finite(duration): ...  # every source

def test_the_segment_in_the_seed_moves_the_gong_source():
    assert not np.array_equal(ttchoir(ROLLED, 15.0, SR, 9, GONG), ttchoir(ROLLED, 15.0, SR, SEGMENT_SEED_STRIDE + 9, GONG))

def test_in_enya_the_consonants_stay_inside_the_voice():
    p = draw_styled_patch(3, CHIPS, "enya"); t = singer(gddr_temps=(26.0, 99.0) * 4)
    new, old = ttchoir(t, 20.0, 44100, 0, p), ttchoir(t, 20.0, 44100, 0, dataclasses.replace(p, choir_onset="", choir_coda="", choir_howl=""))
    assert band_share(new, 4000, 22050, 44100) <= 1.5 * band_share(old, 4000, 22050, 44100)
    assert rms(new) <= rms(old)
```

Update the existing tests that describe the old choir's bar clock (`test_a_syllable_is_four_beats...`, `test_the_breath_between...`, `test_the_voice_starts_softly...`) only so that they pass `DEFAULT_PATCH` explicitly. Leave their assertions as they are.

- [ ] **Step 3: Run them and see them fail**

Run: `python3 -m pytest tests/music/test_ttchoir.py -q`
Expected: the characterization tests PASS; the new ones FAIL on import (`merge_triggers`, `plan_syllables`).

- [ ] **Step 4: Implement** the interfaces and rules above in `ttchoir.py`. Move the old body to `_vowel_choir` without changing a character. Update the module docstring: consonants now exist, and the tract's zeros are modelled only as the nasal notch.

- [ ] **Step 5: Run them and see them pass, then the whole suite**

Run: `python3 -m pytest tests/music/test_ttchoir.py -q`, then `make test`.
Expected: PASS. A drawn patch's choir now sings syllables, so a test elsewhere that pinned a drawn patch's mix level may move (`test_run.py`'s `--no-normalise` ratio is the likely one). Loosen such a test only when the change follows from the choir, and say so in the task report.

- [ ] **Step 6: Mutation check**

Try: `MIN_TRIGGER_GAP_S` 0.03; merge keeping the first strength; the clamp to the gap removed; `XFADE_S` 0; `RELEASE_S` 0; the coda window inclusive of the end; `split_alternate` giving even triggers to codas; the channel offset dropping `personality(t)`; `SYLLABLE_SHRINK_OCTAVES` 2; the voiced closure silent; the unvoiced closure humming; `ONSET_GLIDE_S` 0 (the click test must go red); the per-frame `f0` replaced by the syllable-start `f0` without a glide; `_vowel_choir` called with the new fields applied (the byte-for-byte test must go red).

- [ ] **Step 7: Leave the changes uncommitted.**

---

### Task 5: Self-oscillation (the howl)

**Files:**
- Modify: `tensixinet/music/ttchoir.py`
- Test: `tests/music/test_ttchoir.py`

**Interfaces:**
- Consumes: `choir_triggers` (Task 1), `rests.rest_flags`, the syllable renderer (Task 4).
- Produces:
  - `HOWL_TAG = 0x4041`, `HOWL_MAX_S = 4.0`, `HOWL_CEILING = 10 ** (-1 / 20)`, `HOWL_START = 0.01`, `HOWL_GROWTH_PER_S = 30.0`, `HOWL_CUT = 0.5`, `HOWL_RELEASE_S = 0.25`, `HOWL_WEIGHTS = (1.0, 0.6, 0.35)`
  - `howl_times(t: Telemetry, duration: float, seed: int, patch: Patch) -> np.ndarray`: the fire times. A trigger of `patch.choir_howl` fires when `rest_flags(seed, HOWL_TAG, n, patch.choir_howl_odds)` marks it. A fire that lands while a howl is sounding (less than `HOWL_MAX_S` after the last fire) is ignored. It returns an empty array when the onset or howl source is "".
  - `howl_voice(formants: Sequence[float], n: int, sample_rate: int) -> np.ndarray`: one howl, `n <= HOWL_MAX_S * sample_rate` samples

The howl's feedback runs at frame rate (`FRAME_HZ`), so there is no sample loop. The tests do not determine the gain law, so it is fixed here. Each frame the level is multiplied by `HOWL_GROWTH_PER_S ** (1 / FRAME_HZ)` while it is under `HOWL_CEILING`. A frame that passes the ceiling multiplies the gain by `HOWL_CUT` for the next frame. The level is interpolated to samples. Over the last `HOWL_RELEASE_S` before `HOWL_MAX_S` the level falls to 0. The tone is three sines at F1, F2 and F3 (clamped under 0.45 of the rate), weighted by `HOWL_WEIGHTS` and normalised so their summed peak is 1. With no buzz source, the result is a whistle or growl at the formants. In `_syllable_choir`, each howl crossfades over `XFADE_S` with the syllables, which are muted while it sounds. It uses the formants of the syllable it replaces (the old choir's frame formants at that time).

- [ ] **Step 1: Write the failing tests**

```python
HOWLING = dataclasses.replace(DEFAULT_PATCH, choir_onset="gong", choir_howl="register", choir_howl_odds=1.0)

def test_the_howl_constants_are_the_spec_numbers():
    assert HOWL_MAX_S == 4.0 and HOWL_CEILING == pytest.approx(0.8913, abs=1e-4)

def test_a_howl_grows_reaches_the_ceiling_and_never_passes_it():
    h = howl_voice(VOWELS["a"] + (3400.0,), int(4.0 * SR), SR)
    env = np.abs(h)
    assert env[: SR // 10].max() < 0.2 and env.max() <= HOWL_CEILING + 1e-6 and env[int(1.5 * SR): int(3 * SR)].max() > 0.5

def test_a_howl_has_no_buzz_only_the_formants():  # >= 90% of its energy within 5% of F1, F2 or F3

def test_howls_fire_on_the_share_of_triggers_the_patch_asks():
    p = dataclasses.replace(HOWLING, choir_howl_odds=0.05)
    fires = sum(len(howl_times(HARVESTED, 120.0, s, p)) for s in range(30))
    hits = sum(len(choir_triggers("register", HARVESTED, 120.0, SR, s // SEGMENT_SEED_STRIDE)[0]) for s in range(30))
    # fires / hits within [0.03, 0.07] (fires inside a running howl are dropped, so allow a little under)

def test_a_howl_on_every_trigger_still_respects_the_cap_and_the_ceiling():
    a = ttchoir(HARVESTED, 20.0, SR, 0, HOWLING)
    fires = howl_times(HARVESTED, 20.0, 0, HOWLING)
    assert len(fires) > 1 and (np.diff(fires) >= HOWL_MAX_S).all()
    assert np.abs(a).max() <= HOWL_CEILING + 1e-6

def test_a_howl_replaces_the_syllable_it_lands_on(): ...  # in the howl's middle, energy at the vowel's harmonics of f0 < 5% of the total

def test_a_howl_that_fires_near_the_end_is_cut_cleanly():
    # duration chosen so a fire lands 0.1 s before the end: right length, finite, no exception

def test_no_howl_source_or_zero_odds_means_no_howl():
    assert len(howl_times(HARVESTED, 20.0, 0, dataclasses.replace(HOWLING, choir_howl_odds=0.0))) == 0
```

- [ ] **Step 2: Run them and see them fail** (`python3 -m pytest tests/music/test_ttchoir.py -q -k howl`). Expected: FAIL on import.

- [ ] **Step 3: Implement** the constants, `howl_times`, `howl_voice` and the replacement in `_syllable_choir`.

- [ ] **Step 4: Run them and see them pass** (`python3 -m pytest tests/music/test_ttchoir.py -q`). Expected: PASS, the characterization tests included.

- [ ] **Step 5: Mutation check**

Try: `HOWL_MAX_S` 5; the ceiling as `10 ** (-1 / 10)`; `HOWL_CUT` 1.0 (no cut: the ceiling test must go red); the overlap rule removed; `rest_flags` given the onset's tag; the howl added to the syllable instead of replacing it; the release removed; the formants unclamped at 8000 Hz.

- [ ] **Step 6: Leave the changes uncommitted.**

---

### Task 6: The speech-chip crush

**Files:**
- Modify: `tensixinet/music/ttchoir.py`
- Test: `tests/music/test_ttchoir.py`

**Interfaces:**
- Consumes: `chip_pressure`, `patch.choir_crush` (Task 2).
- Produces:
  - `CRUSH_ONSET_PRESSURE = 0.3`, `CRUSH_BITS = (12, 4)`, `CRUSH_HOLD = (1, 8)`
  - `crush_depth(pressure: float, crush: float) -> float`: `crush * _unit(pressure, 0.3, 1.0)`. It is 0 at rest (the default chip's pressure is 0.17).
  - `speech_crush(x: np.ndarray, depth: float) -> np.ndarray`: depth 0 returns `x` unchanged. Otherwise it holds each sample for `round(1 + 7 * depth)` samples, then quantises to `round(12 - 8 * depth)` bits by truncating toward zero, so `|out| <= |x|` sample by sample.

It is applied last in `_syllable_choir`, to the whole voice including the howl. The chorus applies cables after the device, so cables see the crushed voice.

- [ ] **Step 1: Write the failing tests**

```python
def test_the_crush_is_off_at_rest_and_grows_with_pressure():
    assert crush_depth(0.17, 1.0) == 0.0 and crush_depth(0.3, 1.0) == 0.0
    assert crush_depth(1.0, 0.6) == pytest.approx(0.6) and 0 < crush_depth(0.65, 1.0) < 1

def test_a_resting_chip_is_not_crushed():
    p = dataclasses.replace(GONG, choir_crush=1.0)
    assert ttchoir(singer(), 6.0, SR, 0, p).tobytes() == ttchoir(singer(), 6.0, SR, 0, dataclasses.replace(p, choir_crush=0.0)).tobytes()

def test_a_strained_chip_is_coarser_and_holds_samples():
    x = np.sin(np.linspace(0, 40, 4000)) * 0.8
    y = speech_crush(x, 1.0)
    assert len(np.unique(np.round(y, 9))) <= 2 ** 4 and (y.reshape(-1, 8) == y.reshape(-1, 8)[:, :1]).all()

def test_the_crush_never_raises_a_peak():
    x = np.random.default_rng(0).uniform(-0.8913, 0.8913, 10000)
    for d in (0.1, 0.5, 1.0):
        assert (np.abs(speech_crush(x, d)) <= np.abs(np.repeat(x[::round(1 + 7 * d)], round(1 + 7 * d))[:len(x)]) + 1e-12).all()

def test_cables_see_the_crushed_voice(): ...  # render_chorus with only=["ttchoir", "gong"], a cable from ttchoir: a strained chip's cable target differs from crush 0
```

- [ ] **Step 2: Run them and see them fail.** Expected: FAIL on import.
- [ ] **Step 3: Implement** the constants and functions above, and call `speech_crush(out, crush_depth(chip_pressure(t), patch.choir_crush))` last.
- [ ] **Step 4: Run them and see them pass** (`python3 -m pytest tests/music/test_ttchoir.py -q`).
- [ ] **Step 5: Mutation check.** Try: `CRUSH_ONSET_PRESSURE` 0; rounding to nearest in place of truncation (the peak test must go red); the hold fixed at 1; the bits fixed at 12; the crush applied before the howl.
- [ ] **Step 6: Leave the changes uncommitted.**

---

### Task 7: DEVICE_INPUTS, the chip explanation and the howl log line

**Files:**
- Modify: `tensixinet/music/devices.py` (`DEVICE_INPUTS`, `CHOIR_SOURCE_INPUTS`), `tensixinet/narrate/explain.py` (`explain_chip`'s ttchoir lines, a new `explain_howls`), `tensixinet/narrate/color.py` (`EVENT_JACKS["choir"] = "pink"`), `tensixinet/run.py`, `tensixinet/live.py`
- Test: `tests/narrate/test_explain.py`, `tests/music/test_choir_triggers.py`, `tests/narrate/test_color.py`, `tests/test_run.py`, `tests/test_live.py`

**Interfaces:**
- Consumes: Tasks 1, 2, 4 and 5.
- Produces:
  - `CHOIR_SOURCE_INPUTS` (the fields each source's trigger times and strengths read, under the patch the test uses for that source):
    `gong`: `{"asic_temperature", "aiclk", "heartbeat", "ddr_status"} | PRESSURE_FIELDS | IDENTITY_FIELDS`;
    `rolls`: `{"asic_temperature", "aiclk", "ddr_status", "eth_mask"} | PRESSURE_FIELDS | IDENTITY_FIELDS`;
    `register`: `{"aiclk", "tensix_mask"}`; `quantus`: `{"aiclk", "axiclk"} | IDENTITY_FIELDS` (identity through the rest gate);
    `lead`: `{"aiclk"}`; `rax`: `{"aiclk"}`.
    These are this plan's reading of the code. The test decides, and a mismatch is fixed in the table after checking which side is wrong.
  - `DEVICE_INPUTS["ttchoir"]` = the old set `| frozenset().union(*CHOIR_SOURCE_INPUTS.values())`, which adds `heartbeat`, `ddr_status` and `axiclk`.
  - `explain_howls(telemetry: Sequence[Telemetry], duration: float, patch: Patch, segment: int = 0) -> List[Tuple[int, Decision]]`: for each slot that plays `ttchoir`, its chip and one decision per fire time, using the seed the chorus gives that slot (`segment * SEGMENT_SEED_STRIDE + slot`).
  - Log kind `"choir"`. `run.py` writes the howls of the whole render once; `live.py` writes each segment's after it renders, with `segment=`. Fast mode (`--interval`) does not log howls in this version.

- [ ] **Step 1: Write the failing tests**

```python
CHOIR_PATCHES = {s: dataclasses.replace(DEFAULT_PATCH, choir_onset=s, rests=0.3) for s in CHOIR_SOURCES}
CHOIR_BASE = {"tensix_mask": 0x3002, "ddr_status": (1, 1, 1, 1)}  # its gong follows a roll and its register fires (checked in the test)

@pytest.mark.parametrize("source", CHOIR_SOURCES)
@pytest.mark.parametrize("field", sorted(CHANGES))
def test_the_choir_reads_its_own_fields_and_its_sources(source, field):
    base = dataclasses.replace(FALLBACK, **CHOIR_BASE); changed = dataclasses.replace(base, **{field: CHANGES[field]})
    p = CHOIR_PATCHES[source]
    differs = not np.allclose(ttchoir(base, LONG, SR, 0, p), ttchoir(changed, LONG, SR, 0, p))
    assert differs == (field in OWN_CHOIR_FIELDS | CHOIR_SOURCE_INPUTS[source])

def test_device_inputs_lists_the_choirs_own_fields_and_every_sources():
    assert DEVICE_INPUTS["ttchoir"] == OWN_CHOIR_FIELDS | frozenset().union(*CHOIR_SOURCE_INPUTS.values())
```

Here `OWN_CHOIR_FIELDS = {"asic_temperature", "gddr_temps", "aiclk"} | PRESSURE_FIELDS | IDENTITY_FIELDS`. The existing `test_device_inputs_match_what_the_device_actually_reads` skips `ttchoir` (`pytest.skip` with a reason that points at the per-source test), because one patch cannot exercise every source. If `CHANGES["tensix_mask"]` (0x1235) leaves the register firing at the same steps as `0x3002`, choose a base mask for which it does not, and say why in a comment.

```python
def test_the_chip_log_says_what_triggers_the_choir():
    p = dataclasses.replace(DEFAULT_PATCH, choir_onset="lead", choir_coda="rax")
    text = subjects(explain_chip(9, FALLBACK, 11, p))  # slot 9 plays ttchoir on 11 chips
    assert "syllables on the lead's notes" in text and "codas on ttrax's gestures" in text
    assert "each bar" in subjects(explain_chip(9, FALLBACK, 11))  # the old choir's line is unchanged

def test_a_source_that_cannot_fire_on_this_chip_is_said():  # register on FALLBACK: "the register never fires on this chip, so the choir is silent"

def test_explain_howls_gives_one_line_per_fire_with_its_chip_and_time():
    p = dataclasses.replace(DEFAULT_PATCH, choir_onset="gong", choir_howl="register", choir_howl_odds=1.0)
    lines = explain_howls([HARVESTED] * 4, 20.0, p)
    assert lines and all(chip == 9 % 4 for chip, _ in lines) and "howls" in lines[0][1].what and "-1 dBFS" in lines[0][1].why

def test_every_kind_of_log_line_the_code_writes_has_a_colour():  # existing test in test_color.py; it must see "choir"

def test_a_live_run_logs_a_howl_in_the_segment_it_fires(): ...  # test_live.py, fake chips and a patch with odds 1.0: a "choir" event with segment=
def test_an_offline_run_logs_its_howls(): ...  # test_run.py
```

- [ ] **Step 2: Run them and see them fail.** Expected: the new parametrized test fails on `CHOIR_SOURCE_INPUTS` being empty, and the log tests fail with no "choir" events.
- [ ] **Step 3: Implement.** Fill `CHOIR_SOURCE_INPUTS` and widen `DEVICE_INPUTS["ttchoir"]`. In `explain_chip`, when `patch.choir_onset` is set, replace the "each bar" vowel line with a syllables line naming the sources and say "the next GDDR channel each syllable". Add `explain_howls` and the log calls. Add `"choir": "pink"` to `EVENT_JACKS`.
- [ ] **Step 4: Run them and see them pass** (`python3 -m pytest tests/narrate tests/music tests/test_run.py tests/test_live.py -q`).
- [ ] **Step 5: Mutation check.** Try: `axiclk` dropped from the quantus inputs; `heartbeat` dropped from the union; `explain_howls` using seed `slot` without the segment stride; `live.py` logging before the render; the colour entry removed.
- [ ] **Step 6: Leave the changes uncommitted.**

---

### Task 8: The Play tab picks the next consonant and vowel

**Files:**
- Modify: `tensixinet/music/patch.py` (the `choir_pick` field), `tensixinet/music/ttchoir.py` (apply the pick), `tensixinet/ui/knobs.py`, `tensixinet/ui/instrument.py`, `app/gradio_app.py`
- Test: `tests/music/test_patch.py`, `tests/music/test_ttchoir.py`, `tests/ui/test_knobs.py`, `tests/ui/test_instrument.py`, `tests/test_app.py`

**Interfaces:**
- Consumes: `CONSONANTS` (Task 3), `VOWEL_ORDER`, `plan_syllables` (Task 4).
- Produces:
  - `Patch.choir_pick: Tuple[str, str] = ("", "")`: (onset consonant, vowel). "" leaves that part to the readings. Validation refuses an onset that is not in `consonants.CONSONANTS` and a vowel that is not in `patch.PICK_VOWELS = "uoaei"`. A test ties `PICK_VOWELS` to `ttchoir.VOWEL_ORDER`. `patch_from_dict` defaults it to `("", "")`.
  - In `plan_syllables`, every syllable takes the pick's onset and vowel where they are set. The source still decides when syllables sound. A pick has no effect while `choir_onset` is "". The pick holds until it is changed; it does not apply to a single syllable.
  - `knobs.CHOIR_ONSETS = ("",) + tuple(CONSONANTS)` and `knobs.CHOIR_VOWELS = ("",) + tuple(VOWEL_ORDER)`
  - `Knobs.set_syllable(onset: str, vowel: str) -> None` (unknown values are ignored, as `Knobs.set` ignores them) and `Knobs.syllable() -> Tuple[str, str]`. Both take the lock.
  - `_Plan.pick: Tuple[str, str]`. `render_block` reads `knobs.syllable()` (`("", "")` with no knobs). A pick that differs from the playing phrase's starts a phrase at once, as an error does. `_render_plan` renders with `dataclasses.replace(self.patch, choir_pick=plan.pick)`.
  - App: two dropdowns in the Play tab, "choir consonant" (shown as "any" for "") and "choir vowel" (labels oo, oh, ah, eh, ee). A handler `syllable_setter(request) -> None` writes to the visitor's knobs, and a change does not restart the session. No fire button.

- [ ] **Step 1: Write the failing tests**

```python
def test_a_pick_sets_every_syllables_onset_and_vowel_and_the_source_still_times_them():
    p = dataclasses.replace(GONG, choir_pick=("m", "a"))
    on = (np.array([0.5, 2.0]), np.array([1.0, 1.0]))
    syl = plan_syllables(singer(gddr_temps=(26.0, 99.0) * 4), 4.0, 0, p, on, None)
    assert [(s.onset, s.vowel) for s in syl] == [("m", "a"), ("m", "a")] and [s.start for s in syl] == [0.5, 2.0]

def test_a_half_pick_leaves_the_other_half_to_the_readings(): ...

def test_a_pick_without_an_onset_source_changes_nothing():
    assert ttchoir(singer(), 3.0, SR, 0, dataclasses.replace(DEFAULT_PATCH, choir_pick=("t", "i"))).tobytes() == ttchoir(singer(), 3.0, SR, 0).tobytes()

def test_a_bad_pick_is_refused(): ...  # ("x", "a"), ("s", "y") -> ValueError; ("", "") and ("sh", "") accepted

def test_knobs_hold_a_syllable_pick_and_ignore_unknown_values():  # tests/ui/test_knobs.py
    k = Knobs(2); k.set_syllable("t", "i"); assert k.syllable() == ("t", "i")
    k.set_syllable("x", "a"); assert k.syllable() == ("t", "i")

def test_a_new_pick_starts_a_phrase_at_once_and_renders_with_it():  # tests/ui/test_instrument.py
    # fake render records the patch it is given; set_syllable between two render_block calls; the second call starts a phrase
    # (phrase count goes up by one) and the recorded patch has choir_pick == ("t", "i"); an unchanged pick starts none

def test_the_play_tab_has_a_consonant_and_a_vowel_picker_and_no_fire_button():  # tests/test_app.py
    # the built page has dropdowns labelled "choir consonant" and "choir vowel" with every CHOIR_ONSETS / CHOIR_VOWELS value
    # and no button whose label contains "fire"; syllable_setter writes the visitor's knobs without restarting the session
```

- [ ] **Step 2: Run them and see them fail.** Run `python3 -m pytest tests/music tests/ui -q`, and run the app tests with `.venv-app/bin/python -m pytest tests/test_app.py -q`. Expected: FAIL (`unexpected keyword 'choir_pick'`, no `set_syllable`).
- [ ] **Step 3: Implement** the interfaces above.
- [ ] **Step 4: Run them and see them pass**, with the same commands. Then check the page by hand in headless Chrome (Playwright, as the earlier Play-tab work did). Change the pick while playing, confirm the sound restarts within a block and that there are no console errors, and confirm the page has no horizontal scroll at 375 px. First check that the server answering on port 7880 is the new one.
- [ ] **Step 5: Mutation check.** Try: the pick applied only to the first syllable; the pick change left out of the phrase-start rule; `_render_plan` using `self.patch` unchanged; `set_syllable` accepting unknown values; the vowel validation reading `"aeiou"` in a different order than `VOWEL_ORDER` (the tie test must go red).
- [ ] **Step 6: Leave the changes uncommitted.**

---

### Task 9: Docs and the full suite

**Files:**
- Modify: `docs/SOUND.md` (the ttchoir section and its row in the device table), `docs/DECISIONS.md`, `docs/ROADMAP.md`, `docs/ARCHITECTURE.md` (the new module), `README.md` (the ttchoir row and the log excerpt only if it shows choir lines), `site/index.html` (the choir card at about line 409, the band table row at about line 687, and the Play tab text if it lists the controls), `CLAUDE.md` (the log entry)
- Test: `tests/test_site.py` (run it; add an assertion only for a number the site states)

**Interfaces:**
- Consumes: everything above. Produces nothing for code.

- [ ] **Step 1: Write the docs.**
  * `SOUND.md`: the syllable rules, the six sources and their strengths (the Task 1 table), the inventory and its pinned numbers, the coda rule (including the same-source alternation), the howl (one in `choir_howl_odds` of the source's triggers, at most 4 s, under -1 dBFS, one at a time), the crush (from pressure 0.3, 12 to 4 bits, hold 1 to 8 samples), the drift (4 to 0.5 beats, ±0.16 octave, ±3 degrees), the draw odds, the style table, and the Play tab pick.
  * `DECISIONS.md`: why the sources live in `devices.py`, why the register source ignores the snare share, why quantus and lead borrow their slot's seed, why gong strength is 1.0, why a coda on the onset's own source alternates, and why the howl runs at frame rate.
  * `ROADMAP.md`: still open are the fire button in the Play tab, MIDI events for the choir, a soft l or r, a kernel, howl lines in fast mode, and listening.
  * `CLAUDE.md`: one log entry under "What happened". It names the prompt ("we're all vowels now. let's get some consonants too.", "triggering a self-oscillation/vocalization mode is a fair feature to add", "not yet. maybe some day"), what was built, the mutation survivors and what each got, the test count from Step 2, and "Measured only, not listened to." Say that the listening clips and videos were not remade.
  * Sweep each changed doc for the phrases in Global Constraints.
- [ ] **Step 2: Run the full suite.** Run `make test`, then `.venv-app/bin/python -m pytest tests/test_app.py tests/ui -q`. Expected: all pass. Put the passed and skipped counts in the `CLAUDE.md` entry, and in any doc that states a test count.
- [ ] **Step 3: Check that replay holds.** Load `demo/chorus.patch.json` and `demo/bio.patch.json` with `load_patch`, render `ttchoir` from each for the slot's chip and seed, and compare the bytes with the same render at `HEAD` (from a `git worktree` of `HEAD`). They must be equal.
- [ ] **Step 4: Leave the changes uncommitted.**
