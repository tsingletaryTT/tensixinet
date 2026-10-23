# Play Tab Implementation Plan

> **Status: done (2026-10-23).** Claude carried this plan out after the first attempt by qwencode (model: Qwen3-Coder-Next on two chips) left code that broke the hygiene tests and
> skipped the plan's test-first order; that attempt was discarded and Tasks 1 to 9 were redone. The commits are on `main`. "What changed while it was built" at the end lists where the
> result differs from the text below, and why. The log entry in `CLAUDE.md` has the measurements.

> **For agentic workers (qwencode):** Execute this plan task by task, in order. Steps use checkbox (`- [x]`) syntax; tick them as you go.
> Read `CLAUDE.md` first. Its rules apply to every task here: test first, mutation-check after, SPDX header on every new `.py`, a test file in the
> matching `tests/` folder for every new module, no `import ttnn/ttl/ttml/torch/sounddevice` in `tensixinet/` or `tests/`.
> When the whole plan is done, add the log entry described in Task 9 and delete nothing from this file (tick the boxes instead).

**Goal:** Add a second tab to the Gradio app where a visitor plays a single device, a group of devices (a module) or the whole system live, by moving
knobs that stand in for chip telemetry and pressing buttons that inject events, and hears the result with about a second of delay.

**Architecture:** The existing page becomes tab 1 ("What is playing", unchanged behaviour). Tab 2 ("Play") runs an `InstrumentSession`: a thread that
every ~1.2 s reads the current knob values, builds `Telemetry` for N synthetic chips, renders one block with the existing `render_chorus`
(`only=[...]` for a device or module, no `only` for the system), stitches blocks with the existing `Stitcher`, and hands audio to the browser the way
`ui/session.py` does. Knobs change the next block and never restart the stream. Changing the scope, device, module or chip count restarts it.
All new logic lives in `tensixinet/ui/` with no Gradio import, so it is tested without a browser. `app/gradio_app.py` only lays the page out.

**Tech Stack:** Python 3, numpy, Gradio (`.venv-app`), pytest. Existing modules reused: `music/chorus.render_chorus`, `music/live.Stitcher` and `Tracker`,
`music/eq`, `music/loudness.LiveGain`, `telemetry.Telemetry` and `mock_telemetry`, `ui/player.Visitors`.

**Spec:** the request, "expand the gradio app to include a tab for playing the instruments, modules, and system live". No other spec exists. The
decisions below are this plan's reading of that request.

## Decisions already made (do not reopen them)

* **Three scopes.** *Device*: one of the eleven devices (`music.chorus.device_names()`). *Module*: a named group of devices (table in Task 2).
  *System*: the whole chorus with cables, the coco bus, stereo and the patch's style, as `render_chorus` plays it with no `only`.
* **What "playing" means here.** The instruments have no keys. They are played by the chip readings, so the performer's hands are the readings:
  sliders for temperature, power, aiclk, GDDR spread and voltage per chip, plus buttons that inject a memory error, a thermal trip, and a load
  burst (power jumps, then the existing eviction logic reacts). A "Load from the chips" button copies one real `tt-smi` reading into the sliders.
* **Source is synthetic chips only** (1 to 4 chips). The real chips are not played live in this tab; tab 1 already does that.
* **CPU rendering only.** No `--kernels` choice in this tab. A kernel worker's start is far longer than the 1.2 s block.
* **Knobs never restart the stream.** Scope, device, module, chip count, style, seed and stereo restart it.
* **One audio stream per visitor.** Pressing Play in either tab ends what that visitor was playing in the other. It uses the same `Visitors`
  registry and the same `TENSIXINET_MAX_PLAYERS` limit.
* **Block timing.** `BLOCK_S = 1.2`, `XFADE_S = 0.2` (the `Stitcher` needs a block longer than two crossfades). Known limit, same as live mode:
  each block restarts the devices' clocks, so a ringing gong fades at a block's end. Say so in the docs.

## Global Constraints

* Python style of the surrounding code: type hints, docstrings that say what and why, comments that match the code.
* New files start with the two SPDX lines used everywhere (`# SPDX-License-Identifier: Apache-2.0` / `# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.`).
* `tests/test_hygiene.py` must stay green: every module has `tests/<same folder>/test_<module>.py`, every folder has `__init__.py`.
* Never open a device and never call a gozer command: nothing here needs a lease. `tt-smi -s` (read-only) is the only hardware contact, in Task 6.
* Tests that touch time take a fake clock and a fake sleep. No test may wait on real time for more than a fraction of a second.
* Docs and log entries follow the "plain English" rules in `~/CLAUDE.md`: state the finding, no "X, not Y" framing, no aphoristic closers. Sweep for
  `, not `, `rather than`, `instead of`, `honest`, `reads as` before the last commit.
* Docs that name numbers (test counts, device counts, ranges) are updated in the same commit as the code.
* Run `make test` before every commit. Commit after each task. Branch is `main`; do not push unless asked.

## Review Focus

Failure modes the tasks' main tests would not catch. Each has a test in the task named in brackets.

1. A slider at an extreme (temperature 0 or 120, power above `tdp_limit`, aiclk 0 or 1500) must still render finite audio, not NaN or a crash. [Task 1, Task 3]
2. A device that is not dealt to any of the chosen chips (a 1-chip box) must not give a silent, endless stream with no explanation; the page says so. [Task 2, Task 3]
3. Ten knob moves in one block interval must all be seen (the last value wins) and none may be lost or crash the render thread. [Task 3]
4. Stop during a render must end the thread within one block interval and leave the visitor's place free. [Task 3, Task 5]
5. Two visitors on the Play tab must not hear or move each other's knobs. [Task 5, Task 7]

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `tensixinet/ui/knobs.py` | create | `Knobs` (thread-safe store of per-chip slider values and pending events) and `to_telemetry` |
| `tensixinet/ui/scopes.py` | create | The module table and `resolve_scope` (scope kind and name to an `only` list) |
| `tensixinet/ui/instrument.py` | create | `InstrumentSession`: the render thread, pacing, snapshot, readout data |
| `tensixinet/ui/readout.py` | create | `readout_html`: per-chip pitch, tempo, pressure and what is playing, as HTML |
| `tensixinet/ui/player.py` | modify | Expose the attached session of a visitor (`Player.session`) so knob handlers can reach it |
| `tensixinet/ui/forms.py` | modify | `new_instrument(...)` and `read_chip_knobs(...)` |
| `app/gradio_app.py` | modify | Wrap the page in `gr.Tabs`; add the Play tab and its handlers |
| `tests/ui/test_knobs.py`, `test_scopes.py`, `test_instrument.py`, `test_readout.py` | create | Tests for the above |
| `tests/ui/test_player.py`, `test_forms.py`, `tests/test_app.py` | modify | Tests for the changes |
| `space/assemble.sh` and `tests/test_space.py` | modify if needed | The Space image must carry the new modules (Task 8) |
| `README.md`, `docs/ARCHITECTURE.md`, `docs/DECISIONS.md`, `docs/ROADMAP.md`, `CLAUDE.md` | modify | Task 9 |

---

### Task 1: Knobs and the telemetry they make

**Files:**
- Create: `tensixinet/ui/knobs.py`
- Test: `tests/ui/test_knobs.py`

**Interfaces:**
- Consumes: `tensixinet.telemetry.Telemetry`, `FALLBACK`, `mock_telemetry(n_chips, seed, ...)` (read their definitions in `tensixinet/telemetry/`).
- Produces:
  - `KNOB_RANGES: dict[str, tuple[float, float, float]]` mapping knob name to `(low, high, default)`. Names: `temperature` (20 to 105 C), `power` (5 to 150 W),
    `aiclk` (200 to 1350 MHz), `gddr_spread` (0 to 30 C: how far apart the eight channel temperatures sit), `voltage` (0.65 to 0.95 V). Check each range against what
    `mock_telemetry` produces and `mapping.py` expects, and adjust the numbers if the mapping clamps differently; write the final numbers in the test.
  - `class Knobs`: `__init__(n_chips: int)`; `set(chip: int, name: str, value: float) -> None` (clamps to the range, ignores NaN); `get(chip: int, name: str) -> float`;
    `trigger(chip: int, event: str) -> None` with `event` in `EVENTS = ("memory_error", "thermal_trip", "load_burst")`; `values() -> list[dict[str, float]]`;
    `take_events() -> list[tuple[int, str]]` (returns and clears the pending events, in the order they were triggered); `resize(n_chips: int) -> None` keeps the
    values of chips that remain.
  - `to_telemetry(knobs: Knobs, tracker: Tracker, now: float) -> list[Telemetry]`: builds one `Telemetry` per chip from the knobs, applies pending events
    (a `memory_error` adds 1 to `corr_errs` for that segment, a `thermal_trip` adds 1 to `thermal_trips`, a `load_burst` adds `LOAD_BURST_W = 70.0` W to that chip's power for
    `LOAD_BURST_S = 4.0` s measured from `now`), then passes the list through `tracker.update(readings, now)` so moving a slider produces `temp_rate` and `power_rate`.
    Events count only for the block in which they are taken, because `Tracker` turns running counters into new events.

- [x] **Step 1: Write the failing tests** in `tests/ui/test_knobs.py`:
  - `test_each_knob_moves_exactly_one_field`: for each knob name, set chip 0 away from its default, build telemetry, and assert that only the mapped field
    (`temperature` to `asic_temperature`, `power` to `power`, `aiclk` to `aiclk`, `voltage` to `voltage`, `gddr_spread` to `gddr_temps`) differs from the defaults,
    and that chip 1 is untouched.
  - `test_gddr_spread_sets_max_minus_min_of_the_eight_channels`: spread 12 gives `max(gddr_temps) - min(gddr_temps) == pytest.approx(12.0)` and 8 channels.
  - `test_values_are_clamped_and_nan_is_ignored`: `set(0, "power", 1e9)` reads back the high bound; `set(0, "power", float("nan"))` leaves the old value.
  - `test_an_event_is_taken_once_and_in_order`: trigger `thermal_trip` on chip 1 then `memory_error` on chip 0; `take_events()` returns both in that order, then `[]`.
  - `test_a_memory_error_shows_in_one_block_only`: with a real `Tracker`, the first `to_telemetry` after the trigger has `corr_errs == 1` on that chip and the next has 0.
  - `test_a_load_burst_adds_power_for_four_seconds_then_stops`: at `now=0` trigger; power is default + 70 at `now=0` and `now=3.9`, default at `now=4.1`.
  - `test_resize_keeps_the_chips_that_remain`: set chip 1, `resize(2)` keeps it, `resize(1)` drops it, `resize(3)` gives the new chip its defaults.
  - `test_moving_a_slider_gives_a_rate`: set temperature 50, call at `now=0`, set 60, call at `now=2.5`; chip 0's `temp_rate` is `4.0` (10 C over 2.5 s).
  - Extremes (Review Focus 1): at every low and high bound for every knob at once, every numeric field of every `Telemetry` is finite.
- [x] **Step 2: Run** `.venv/bin/python -m pytest tests/ui/test_knobs.py -q`. Expected: FAIL with `ModuleNotFoundError: tensixinet.ui.knobs`.
- [x] **Step 3: Implement** `Knobs`, `KNOB_RANGES`, `EVENTS`, `LOAD_BURST_W`, `LOAD_BURST_S` and `to_telemetry` in `tensixinet/ui/knobs.py`. `Knobs` holds a `threading.Lock`;
  every public method takes it. Build each chip from `dataclasses.replace(mock_telemetry(...)[chip], ...)` or `FALLBACK`, whichever gives distinct tensix/eth masks per chip
  (chips need distinct identities, as the mock gives them; check by reading `mock.py`).
- [x] **Step 4: Run** the same command. Expected: PASS.
- [x] **Step 5: Mutation check.** Swap `min`/`max` in the clamp, drop the NaN guard, change `4.0` to `40.0`, drop the `take_events` clear. Each must turn a test red. Add a test for any survivor.
- [x] **Step 6: Commit** `git add tensixinet/ui/knobs.py tests/ui/test_knobs.py && git commit -m "Add the knobs that stand in for chip readings in the Play tab"`.

---

### Task 2: Scopes: device, module, system

**Files:**
- Create: `tensixinet/ui/scopes.py`
- Test: `tests/ui/test_scopes.py`

**Interfaces:**
- Consumes: `tensixinet.music.chorus.device_names() -> list[str]`, `n_slots(n_chips)`, `assign_device(slot)`, `slot_chip`.
- Produces:
  - `MODULES: dict[str, tuple[str, ...]]` with exactly these entries (names are device names from `device_names()`):
    * `"rhythm section"`: `ttrollz`, `gong`, `ttavdog`, `ttsteam`
    * `"drones and air"`: `ttdrone`, `ultrasonic_shimmer`, `ttchoir`
    * `"melody"`: `lead`, `ttquantus`, `ttrax`
    * `"delay and echo"`: `ttquantus`, `ttcoco`
  - `SYSTEM = "the whole system"`
  - `scope_choices() -> list[tuple[str, str]]`: `(label, value)` pairs for a dropdown, system first, then modules, then devices, values like `"system"`, `"module:melody"`, `"device:gong"`.
  - `resolve_scope(value: str) -> Optional[list[str]]`: `None` for the system, the device list for a module, `[name]` for a device; `ValueError` naming the choices for an unknown value.
  - `heard_on(scope_only: Optional[list[str]], n_chips: int) -> list[str]`: the devices of the scope that are dealt to at least one of `n_chips` chips (every device for the system).

- [x] **Step 1: Write the failing tests** in `tests/ui/test_scopes.py`:
  - `test_every_module_member_is_a_real_device` and `test_every_device_is_in_at_least_one_module_or_listed_alone` (devices outside modules are still reachable as device scopes).
  - `test_scope_choices_start_with_the_system_and_have_unique_values_and_one_per_device_and_module`.
  - `test_resolve_scope_round_trips_every_choice`: system gives `None`; a module gives its tuple as a list; a device gives `[name]`.
  - `test_resolve_scope_rejects_an_unknown_value_and_names_the_choices`.
  - `test_heard_on_one_chip_lists_only_the_devices_that_chip_plays` and `test_heard_on_four_chips_lists_every_device` (Review Focus 2). Work out the one-chip expectation from `chip_devices(0, 1)`, and write the expected list out as literals in the test.
- [x] **Step 2: Run** the file. Expected: FAIL (module missing).
- [x] **Step 3: Implement** `MODULES`, `SYSTEM`, `scope_choices`, `resolve_scope`, `heard_on` in `tensixinet/ui/scopes.py`.
- [x] **Step 4: Run** the file. Expected: PASS.
- [x] **Step 5: Mutation check:** drop a module member, change `None` to `[]` for the system, let `heard_on` ignore `n_chips`. Each must fail a test.
- [x] **Step 6: Commit** `git commit -m "Add the device, module and system scopes for the Play tab"`.

---

### Task 3: The instrument engine

**Files:**
- Create: `tensixinet/ui/instrument.py`
- Test: `tests/ui/test_instrument.py`

Read first: `tensixinet/ui/session.py` (the thread, `_UiSink`, `_hand_over`, `next_audio`, `stop`) and `tensixinet/live.py` lines 59 to 215 (how `run_live` applies `Stitcher`, the equaliser and
`LiveGain`). Mirror their use. Do not edit `session.py` or `live.py`.

**Interfaces:**
- Consumes: `Knobs`, `to_telemetry` (Task 1); `resolve_scope`, `heard_on` (Task 2); `render_chorus`, `Stitcher`, `Tracker`, `XFADE_S`, the equaliser and `LiveGain` as `run_live` uses them;
  `make_patch(style, seed, chips)` from `ui/session.py`; `Patch`.
- Produces:
  - `BLOCK_S = 1.2`, `SAMPLE_RATE = 44100`.
  - `@dataclass(frozen=True) class InstrumentConfig`: `scope: str`, `chips: int`, `style: str`, `seed: int`, `stereo: bool`, `cables: bool`, `evict: bool` (the restart key; knobs are not in it).
  - `@dataclass(frozen=True) class InstrumentSnapshot`: `running: bool`, `error: Optional[str]`, `block: int`, `telemetry: tuple` (the readings of the block now playing), `playing: tuple` (device names heard), `notes: tuple` (plain-English lines, for example "ttchoir is not dealt to a 1-chip box").
  - `class InstrumentSession`: `__init__(config, knobs: Knobs, render=render_chorus, clock=time.monotonic, sleep=time.sleep, lead_s: float = 2.0)`;
    `start()`, `stop(timeout: float = 5.0)`, `next_audio(timeout: float = 1.0) -> Optional[np.ndarray]`, `snapshot() -> InstrumentSnapshot`, and `sample_rate` as an attribute.
    The render function is injectable so tests use a stub.

Behaviour to implement: each loop pass takes `knobs.values()` once (so ten moves inside one pass give the last value), builds telemetry with `to_telemetry`, resolves the scope,
calls `render(telemetry, BLOCK_S + XFADE_S, SAMPLE_RATE, patch=patch, segment=block, only=only, stereo=config.stereo, cables=None or [] by config.cables, defrag=<evict state if config.evict, system scope only>)`,
runs the result through the stitcher, EQ and live gain, hands it over, and paces so audio stays at most `lead_s` ahead of the listener. A render error ends the session with `error` set
and the thread stopped (never an unhandled exception in the thread). `evict` follows tab 1: read how `run_live` builds `ChipDefrag` states from `music/evict.py` and do the same with the
synthetic readings; only the system scope uses it (`render_chorus` ignores `defrag` with `only`).

- [x] **Step 1: Write the failing tests** in `tests/ui/test_instrument.py`, using a stub render that records its arguments and returns a deterministic tone, and a fake clock and sleep:
  - `test_each_block_renders_with_the_scope_the_config_names`: system gives `only=None`, a module gives its list, a device gives `[name]`.
  - `test_a_knob_moved_between_blocks_reaches_the_next_render`: set power on chip 0, take the first block, set it again, assert the second render call's telemetry has the new value and the first call's has the old.
  - `test_ten_moves_inside_one_block_give_the_last_value` (Review Focus 3).
  - `test_blocks_use_increasing_segment_numbers` (so blocks are not copies of each other).
  - `test_next_audio_returns_stitched_audio_with_the_crossfade_removed`: total samples over three blocks equals `3 * BLOCK_S * SAMPLE_RATE` within one crossfade.
  - `test_the_session_stays_no_more_than_lead_s_ahead_of_the_listener`: with the fake clock, count blocks handed over before the clock advances.
  - `test_stop_during_a_render_ends_the_thread_within_one_block` (Review Focus 4): the stub render blocks on an event; call `stop()` from the test, release the event, and assert the thread is not alive.
  - `test_a_render_error_stops_the_session_and_is_reported_in_the_snapshot`.
  - `test_snapshot_names_the_devices_heard_and_notes_one_not_dealt_to_the_chips` (Review Focus 2): config `device:ttchoir` with 1 chip, if `heard_on` says it is silent, `notes` says so and the audio is still silence (not an exception).
  - `test_the_snapshot_telemetry_is_the_block_now_playing_not_the_one_just_rendered`: with `lead_s` > 0, the snapshot changes only when the listener reaches the block (copy the `_pending`/`_advance` idea in `Session`; keep it much smaller: a deque of `(start_offset_s, snapshot)`).
  - One real-render smoke test, no stub: `test_every_device_makes_sound_on_four_chips` loops over `device_names()`, renders one real block with 4 chips, 1.0 s, and asserts finite values and an RMS above `1e-4`. Mark with the `slow` marker only if `pyproject.toml` defines one; otherwise keep it, it should take a few seconds.
  - `test_the_real_system_block_renders_faster_than_it_plays`: render one system block with 4 chips, stereo, and assert wall time is below `BLOCK_S * 0.8`. If this is flaky on a loaded machine, change it to log the time into the test output and assert a loose `< BLOCK_S * 2`, and put the measured number in Task 9's docs.
- [x] **Step 2: Run** the file. Expected: FAIL (module missing).
- [x] **Step 3: Implement** `InstrumentConfig`, `InstrumentSnapshot`, `InstrumentSession`, `BLOCK_S`, `SAMPLE_RATE` in `tensixinet/ui/instrument.py`. Use a `queue.Queue` for audio like `Session` does, a `threading.Event` for stop, and slice sleeps of at most 0.25 s so a stop is noticed quickly.
- [x] **Step 4: Run** the file. Expected: PASS.
- [x] **Step 5: Mutation check:** change `BLOCK_S`, drop the stitcher, take `knobs.values()` twice per pass, pass `only=None` for a device scope, drop the stop check in the pacing loop, drop the `try/except` around render. Each must go red. Remember `PYTHONDONTWRITEBYTECODE=1` for mutation runs (see the log in `CLAUDE.md`).
- [x] **Step 6: Commit** `git commit -m "Add the instrument engine: a render loop that reads the knobs every block"`.

---

### Task 4: The readout

**Files:**
- Create: `tensixinet/ui/readout.py`
- Test: `tests/ui/test_readout.py`

**Interfaces:**
- Consumes: `InstrumentSnapshot` (Task 3); `mapping.temp_to_pitch_hz`, `mapping.aiclk_to_bpm`, `mapping.chip_pressure` (confirm the exact names and signatures in `music/mapping.py`; the site sandbox in `site/index.html` ports them as `temp_to_pitch_hz`, `aiclk_to_bpm`, `pressure`); `narrate.color` palette for chip colours (so a chip keeps the colour it has in tab 1).
- Produces: `readout_html(snapshot: InstrumentSnapshot) -> str`: one row per chip with its pitch (Hz and note name), tempo (BPM), a pressure bar, and a line listing `snapshot.playing` and any `notes`. Returns a "press Play" placeholder string constant `READOUT_WAITING` for an empty snapshot.

- [x] **Step 1: Write the failing tests:** `test_a_row_per_chip_with_the_chips_own_colour`; `test_pressure_bar_width_follows_chip_pressure` (a chip at the throttle limit has a wider bar than a cool one; assert on the width number in the markup, not on the colour); `test_notes_and_devices_appear_in_the_output`; `test_html_escapes_notes` (a note containing `<b>` appears escaped); `test_an_empty_snapshot_gives_the_waiting_text`.
- [x] **Step 2: Run** the file. Expected: FAIL.
- [x] **Step 3: Implement** `readout_html` and `READOUT_WAITING`. Inline styles only, same dark-teal look as `ui/chart.py` (read it for the palette; the global brand colours are in `~/CLAUDE.md`). No right-hand borders on any ASCII art (global rule, only relevant if you add any).
- [x] **Step 4: Run** the file. Expected: PASS.
- [x] **Step 5: Mutation check:** swap the pitch and tempo values, drop the escape, invert the bar. Each must go red.
- [x] **Step 6: Commit** `git commit -m "Add the Play tab's readout of pitch, tempo and pressure"`.

---

### Task 5: Reaching a visitor's session from the knob handlers

**Files:**
- Modify: `tensixinet/ui/player.py`
- Test: `tests/ui/test_player.py`

Read `tensixinet/ui/player.py` in full first. Its `Player.attach(token, session)` already stores the session; knob handlers need to reach it without a second registry.

**Interfaces:**
- Consumes: `Player`, `Visitors` as they are.
- Produces: `Player.session` (property: the attached session if the player is active, else `None`); `Visitors.session(key: str)` returning `self.player(key).session`.

- [x] **Step 1: Write the failing tests:** `test_session_is_none_before_attach_and_after_stop`; `test_session_is_the_attached_one_while_active`; `test_two_visitors_have_their_own_sessions` (Review Focus 5); `test_a_session_of_another_kind_counts_towards_the_same_limit` (attach a stub with `.stop()`; the `Visitors(limit=1)` claim for a second visitor raises `Busy`, whatever the config shape: configs are opaque tuples to `Visitors`).
- [x] **Step 2: Run** `tests/ui/test_player.py`. Expected: the new tests FAIL (`AttributeError`), the old ones pass.
- [x] **Step 3: Implement** the property and the method, taking the player's existing lock.
- [x] **Step 4: Run** the file. Expected: PASS, old tests included.
- [x] **Step 5: Mutation check:** return the session after a stop; return the wrong visitor's player. Each must go red.
- [x] **Step 6: Commit** `git commit -m "Let knob handlers reach a visitor's running session"`.

---

### Task 6: Forms: building an instrument and reading the chips into the knobs

**Files:**
- Modify: `tensixinet/ui/forms.py`
- Test: `tests/ui/test_forms.py`

**Interfaces:**
- Consumes: `InstrumentConfig`, `InstrumentSession`, `Knobs`, `KNOB_RANGES`, `scope_choices`, `resolve_scope`; `ui.session.real_sample()`.
- Produces:
  - `new_instrument(scope: str, style: str, seed: int, chips: int, stereo: bool, cables: bool, evict: bool, knobs: Knobs, sleep=time.sleep, lead_s: float = 2.0) -> tuple[InstrumentSession, Patch]`
    raising `ValueError` for an unknown scope or style, and for chips outside 1 to 4.
  - `read_chip_knobs(knobs: Knobs, sample=real_sample()) -> str`: reads one real reading, sets each existing chip's knobs from the matching real chip (extra real chips ignored, missing ones left alone), returns a one-line status (`"loaded 4 chips from tt-smi"` or the error text when `tt-smi` is missing or fails). It never raises.

- [x] **Step 1: Write the failing tests:** `test_new_instrument_builds_a_session_and_a_patch_for_every_scope_choice` (loop `scope_choices()`); `test_new_instrument_rejects_an_unknown_scope_a_bad_style_and_chip_counts_0_and_5`; `test_read_chip_knobs_copies_a_reading_into_the_knobs` (a fake `sample` returning `mock_telemetry(4, 1)`; assert chip 2's temperature knob equals that chip's `asic_temperature` clamped to the range); `test_read_chip_knobs_reports_a_failing_tt_smi_and_leaves_the_knobs_alone` (a `sample` that raises).
- [x] **Step 2: Run** the file. Expected: new tests FAIL.
- [x] **Step 3: Implement** both functions. `gddr_spread` from a real reading is `max(gddr_temps) - min(gddr_temps)`.
- [x] **Step 4: Run** the file. Expected: PASS.
- [x] **Step 5: Mutation check:** let `read_chip_knobs` raise, skip the clamp, accept 5 chips. Each must go red.
- [x] **Step 6: Commit** `git commit -m "Add forms for the Play tab: build an instrument, load the chips into the knobs"`.

---

### Task 7: The Play tab in the Gradio app

**Files:**
- Modify: `app/gradio_app.py`
- Test: `tests/test_app.py`

Read `app/gradio_app.py` in full and the app tests from "Play, Stop and a change of setting" down. The existing handlers, `_stream`, the JS strings and the `Visitors` object stay as they are.
Tab 1's `build()` body moves into `_build_watch_tab()` unchanged; `build()` becomes `with gr.Blocks(...) as demo: with gr.Tabs(): with gr.Tab("What is playing"): _build_watch_tab() ... with gr.Tab("Play"): _build_play_tab()`.
Components created inside a function called within a `with gr.Tab` block belong to that tab; keep `demo.unload(leave)` in `build()`.

**Interfaces:**
- Consumes: everything from Tasks 1 to 6; `visitors` (the one registry), `_key`, `STOPPED`, `STARTING`, `JS_STOP`.
- Produces, in `gradio_app.py`:
  - `play_instrument(scope, style, seed, chips, stereo, cables, evict, request: gr.Request = None)`: a generator; same shape as `_stream` (claim, show starting, stop the old one, build with `new_instrument`, attach, loop on `session.next_audio` and `session.snapshot()`), yielding `(audio, readout_html, status)`. The restart key is `("instrument", scope, style, seed, chips, stereo, cables, evict)`, so it never equals a tab-1 config. Each visitor gets one `Knobs`, kept in a module dict `knob_store: dict[str, Knobs]` and dropped in `leave`.
  - `set_knob(chip, name, value, request)`: sets one knob on the visitor's `Knobs`; returns nothing; works whether or not a session is playing (so values set before Play count).
  - `trigger_event(chip, event, request)`: calls `Knobs.trigger`.
  - `load_chips(request) -> tuple[str, *slider values]`: calls `read_chip_knobs` and returns the status and the new slider values so the page's sliders move.
  - `stop_instrument(request)`: like `stop`.
  - Page: scope dropdown (`scope_choices()`, default system), style, seed, chips (1 to 4), stereo, cables, evict; a grid of sliders (5 knobs × 4 chips; sliders for chips above the chosen count are hidden via `visible`); three buttons per chip (error, trip, load burst); "Load from the chips" (only offered when `tt-smi` is on the PATH, as `source_choices()` decides); Play and Stop; the readout `gr.HTML`; the audio `gr.Audio(streaming=True, autoplay=True)`; a status line.
  - Events: Play and a change of scope/style/seed/chips/stereo/cables/evict use the same streaming wiring as tab 1 (`concurrency_limit=None`, `trigger_mode="multiple"`, `js=` silence strings). Knob sliders use `.input` with `trigger_mode="always_last"`, `concurrency_limit=None`, `show_progress="hidden"`, `queue=False` if Gradio allows, and never touch the audio.
  - Every handler names every parameter (tab 1's comment explains why: Gradio fills `request` only if it is a named parameter, not after `*args`).

- [x] **Step 1: Write the failing tests** in `tests/test_app.py` (use the same `_app()` helper and `importorskip` guard as the neighbouring tests):
  - `test_the_page_has_two_tabs_named_what_is_playing_and_play`.
  - `test_play_instrument_streams_audio_and_a_readout_for_a_device_scope`: drive the generator with a stub `new_instrument` (monkeypatch `gradio_app.new_instrument`) whose session yields two audio chunks, and assert the yielded tuples have audio, the readout string and a status that says playing.
  - `test_set_knob_before_play_is_seen_by_the_session_that_starts_after`.
  - `test_a_knob_change_while_playing_does_not_restart_the_stream`: the stub session counts `start()` calls; set a knob; assert 1.
  - `test_a_change_of_scope_while_playing_restarts_the_stream_with_the_new_scope`.
  - `test_play_in_the_play_tab_ends_what_the_same_visitor_played_in_the_first_tab` and the reverse (use two fake sessions with a `stopped` flag).
  - `test_two_visitors_have_separate_knobs_and_the_third_is_told_the_places_are_taken` (Review Focus 5; uses `TENSIXINET_MAX_PLAYERS=2` through `_limit_from` or a fresh `Visitors`).
  - `test_leave_drops_the_visitors_knobs_and_session`.
  - `test_every_handler_names_request_so_gradio_will_fill_it`: mirror the existing test that asks Gradio's `special_args` (find it in the file and copy its method for the four new handlers).
  - `test_the_slider_ranges_on_the_page_are_the_knob_ranges`: read the components' `minimum`/`maximum`/`value` and compare with `KNOB_RANGES`.
- [x] **Step 2: Run** `.venv-app/bin/python -m pytest tests/test_app.py -q` (the TT venv's gradio does not import, the tests skip there; use `.venv-app`). Expected: new tests FAIL.
- [x] **Step 3: Implement** the handlers and `_build_play_tab`, and restructure `build()`.
- [x] **Step 4: Run** the file in `.venv-app`. Expected: PASS, tab 1's tests included.
- [x] **Step 5: Look at it in a real browser.** Unit tests missed real bugs here twice (see the log). Start `.venv-app/bin/python app/gradio_app.py` (check what answers on port 7880 first; the log has a note on a stale instance) and use headless Chrome with Playwright in a scratch venv outside the repo. Check, and write what you saw in the log entry:
  1. Play on the Play tab starts audio within about 5 s; the readout updates.
  2. Move the temperature slider while playing: the pitch shown in the readout follows within about 2 s and the stream does not restart (status line unchanged, no "starting").
  3. Pressing a button (memory error) while playing causes no restart and no console error.
  4. Switching scope from system to a device restarts with the new scope.
  5. Play in tab 1 then in tab 2 ends the first stream.
  6. Two browser contexts: each plays; the knobs of one do not move the other.
  7. A 375 px wide viewport: no horizontal page scroll.
- [x] **Step 6: Mutation check** (six mutants): make `set_knob` restart the stream; drop the per-visitor `Knobs` key; use the same restart key as tab 1; drop `leave`'s knob drop; clamp the chips slider to 1 to 8; hide the wrong chips' sliders. Each must go red.
- [x] **Step 7: Commit** `git commit -m "Add the Play tab: play a device, a module or the whole system with the knobs"`.

---

### Task 8: The Space and the launcher

**Files:**
- Modify (if needed): `space/assemble.sh`, `space/Dockerfile`, `tests/test_space.py`
- Test: `tests/test_space.py`, `tests/test_launcher.py` (read, change only if they enumerate files)

- [x] **Step 1: Write the failing test** in `tests/test_space.py`: every `tensixinet/ui/*.py` file is copied by `assemble.sh` into `build/space` (read how the existing tests check the file list, extend the same way).
- [x] **Step 2: Run** `make test`. Expected: FAIL only if `assemble.sh` lists files one by one; if it copies the folder, the test already passes, and you then confirm the claim by running `space/assemble.sh` and listing `build/space/tensixinet/ui/`.
- [x] **Step 3: Fix** `assemble.sh` if needed.
- [x] **Step 4: Measure the cost on a small machine.** `taskset -c 0,1 .venv-app/bin/python` a script that renders ten system blocks (4 chips, stereo) through `InstrumentSession` with a fake sleep and reports the longest render time. If the longest is above `BLOCK_S * 0.8`, the Space's two CPUs cannot keep up. Then in `_build_play_tab` cap the chip choice at 2 when `TENSIXINET_MAX_PLAYERS` is below its default or an env `TENSIXINET_PLAY_CHIPS` says so, and record the numbers in `docs/DECISIONS.md`. If it keeps up, record that.
- [x] **Step 5: Run** `make test`. Expected: PASS.
- [x] **Step 6: Commit** `git commit -m "Carry the Play tab's modules into the Space and record what a block costs on two CPUs"`.

---

### Task 9: Documents, the log, and a last check

**Files:**
- Modify: `README.md`, `docs/ARCHITECTURE.md`, `docs/DECISIONS.md`, `docs/ROADMAP.md`, `CLAUDE.md`
- Test: `tests/test_hygiene.py` (run), `tests/test_site.py` (run; do not edit the site)

- [x] **Step 1:** `README.md`: one paragraph and one screenshot-free description of the Play tab under the app section (what the knobs are, the three scopes, the one-second delay, the CPU-only limit).
- [x] **Step 2:** `docs/ARCHITECTURE.md`: add `ui/knobs.py`, `ui/scopes.py`, `ui/instrument.py`, `ui/readout.py` to the data-flow and module list. Update `CLAUDE.md`'s Layout block (`ui/` line) the same way.
- [x] **Step 3:** `docs/DECISIONS.md`: record the decisions in "Decisions already made" above, each with its reason, plus the Task 8 measurement.
- [x] **Step 4:** `docs/ROADMAP.md`: add the open items this leaves: the real chips as a live source in the Play tab (knobs as offsets on a real reading), the kernels in the Play tab, a block that continues the last one (the start-time offset item already in item 3), and a recorded demo of the tab.
- [x] **Step 5:** `CLAUDE.md`: add a log entry dated with the day you finish. Contents: the original prompt ("expand the gradio app to include a tab for playing the instruments, modules, and system 'live'"), the decisions, what the browser check found (Task 7 Step 5, in the order found), the mutation results per task (survivors and the tests they got), and the measurements (block render time on this box and on two CPUs). Say "measured, not listened to" unless a person listened.
- [x] **Step 6: Sweep** the changed docs for `, not `, `rather than`, `instead of`, `honest`, `reads as`, `load-bearing`; rewrite or justify each hit.
- [x] **Step 7: Run** `make test` and `.venv-app/bin/python -m pytest tests/test_app.py -q`. Report the totals in the log entry and make sure any doc that names a test count has the new one.
- [x] **Step 8: Commit** `git commit -m "Document the Play tab and log how it was built"`.

---

## Self-review (done when this plan was written)

* **Coverage of the request:** instruments (device scope, Task 2/3), modules (module scope, Task 2), system (Task 3), "live" (render loop with knob changes heard in the next block, Task 3/7), a tab in the Gradio app (Task 7).
* **Type consistency:** `Knobs`, `to_telemetry`, `InstrumentConfig`, `InstrumentSession`, `InstrumentSnapshot`, `resolve_scope`, `heard_on`, `scope_choices`, `new_instrument`, `read_chip_knobs` keep one signature each across tasks.
* **Things the plan could not check and Qwen must:** the exact names in `music/mapping.py` (Task 4), how `run_live` wires the equaliser and gain (Task 3), whether `Player` stores the session already (Task 5), whether `pyproject.toml` has a `slow` marker (Task 3), and the real ranges `mapping.py` clamps to (Task 1). Each task says where to look. If a name differs, use the real one and say so in the log.
* **Not in this plan (YAGNI):** a keyboard or MIDI input, recording the performance, saving knob presets, real chips as a live source, kernels in the tab.


## What changed while it was built

The text above is the plan as written. The code differs from it in these places, each found by running something.

1. **Phrases instead of blocks (Task 3).** A bare 1.2 s render of the gong is often silent: its first strike is up to 2.2 s in. The real-render test that plays every device found it. `InstrumentSession`
   renders a *phrase* (`SHORT_BLOCKS = 2` after a change, `LONG_BLOCKS = 5` when nothing moved) and hands over 1.2 s slices of it, crossfading only where a phrase is replaced. `Stitcher` is not used.
   `InstrumentSession.__init__` takes the patch: `InstrumentSession(config, knobs, patch, render, clock, sleep, lead_s)`. The sample rate is 22050 and the default lead is 1.5 s; the app asks for 3.0 s.
2. **Review Focus 2 had a false premise (Task 2).** `n_slots(n)` is never fewer than the number of devices, so every device is dealt even on one chip. `heard_on` asks the dealing anyway and
   the "not dealt" note stays, but it cannot fire today. A test reaches it by patching `heard_on`.
3. **Knobs (Task 1).** `Knobs` remembers all four chips' values and plays the first `n_chips` of them, so a hidden chip's slider keeps its value. `current` and `board_power` follow `power`. A burst, an
   error and a trip are applied in one locked step (`Knobs.advance`). The GDDR channels run from 40 C down by the spread.
4. **The readout takes the snapshot (Task 4):** `readout_html(snapshot)`, on its own dark panel, with `note_name(hz)`.
5. **The app (Task 7).** The handlers are named `instrument_play`, `instrument_restart`, `instrument_stop`, `load_chips`, `show_chips`, and `knob_setter(chip, name)` and `event_trigger(chip, event)`
   make the 32 slider and button handlers (their `api_name`s are `knob_<chip>_<name>` and `event_<chip>_<event>`). They are named so the first tab's tests, which count handlers by name, still hold.
   Each tab tells the visitor when the other tab took over, and a setting changed in the tab that is not playing does nothing.
6. **Stop (found in the browser, Task 7 step 5).** Returning `None` for the audio on Stop made Chrome load and play a new source, in the first tab too. Both tabs now return `gr.skip()`.
7. **Task 8:** `assemble.sh` copies the whole `tensixinet/` folder, so nothing needed fixing; a test pins the four new modules. The cost measurement showed no chip cap is needed.
8. **Latency** is 3.6 to 4.2 s from a slider to the readout, not "about a second".
