# Fast sampling and `--with-tt-toplike` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `tensixinet --live --interval MS` reads the chips at that rate and plays so a change is heard within 250 ms, and `--with-tt-toplike` starts tt-toplike and the synth at the same interval in one tmux window.

**Architecture:** A fast reader takes temperature, power, voltage and current from the chips' hwmon files and overlays them on the latest `tt-smi` reading, which a background thread refreshes. The Play tab's phrase engine (`InstrumentSession`) is made configurable (its input, block size, phrase lengths, sample rate) and a new `run_fast` drives it into a player sink. `--with-tt-toplike` builds a tmux command line.

**Tech Stack:** Python 3.10+, numpy, pytest, tmux, tt-toplike, Linux hwmon.

**Spec:** `docs/superpowers/specs/2026-10-23-fast-sampling-design.md` (read its "Settled after the builder check" section; it fixes the calls below).

## Global Constraints

- Test first: write the test, watch it fail for the right reason, write the code, then break the code on purpose and confirm a test goes red. A mutation script that matches nothing proves nothing (single space around `|||` in `scratchpad/mutate.py`'s input).
- SPDX header (`# SPDX-License-Identifier: Apache-2.0` and `# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.`) on every new `.py`; a test file in the matching `tests/` folder for every new module.
- No `ttnn`, `ttl`, `ttml`, `torch` or `sounddevice` imports in `tensixinet/` or `tests/`.
- Run tests with `.venv-app/bin/python -m pytest` (2564 passed, 34 skipped before this work). Live mode without `--interval` must sound and log exactly as it does now; the Play tab must behave as it does now.
- Docs that name flags or numbers change in the same task as the code. Plain English in docs and log text: no "X, not Y" framing, no aphoristic closers; sweep for `, not `, `rather than`, `instead of`, `reads as`, `honest`, `the one`, `which is why`.
- Never `pkill -f` or `pgrep -f`; use PID files. No commits unless the owner asks. No gozer lease is needed (hwmon and tt-toplike's sysfs reads do not open `/dev/tenstorrent/*`).
- Measured facts to rely on: hwmon units are milli-degrees C (`temp1_input`), micro-watts (`power1_input`), millivolts (`in0_input`), milliamps (`curr1_input`); the hwmon `name` file says `blackhole`; `device` links to the PCI address, which has the same text as `board_info.bus_id` in `tt-smi -s` (`0000:01:00.0`). `render_chorus` of four mock chips in stereo at 22.05 kHz takes 45 ms for 0.25 s of audio, 67 ms for 1 s, 186 ms for 4 s.

## Review Focus

- A chip's hwmon directory vanishes or a file read fails mid-run: keep the chip's last fast reading, keep playing, log once.
- hwmon finds a different number of chips than `tt-smi` (or none): fall back to the `tt-smi` rate and say so; never pair chips by position when the bus addresses disagree.
- `tt-smi` hangs for longer than its timeout: audio continues from fast readings and the last slow fields.
- The render takes longer than a block (a 10 ms interval): the loop lengthens its block instead of falling behind, and the log says the real rate.
- `--duration` with `--interval`: the run stops at the duration and the tail rings out; Ctrl-C ends cleanly with the WAV and log closed.
- tmux: a session name that already exists, tmux missing, tt-toplike missing, `--mock-chips` given.

---

### Task 1: The hwmon reader

**Files:**
- Create: `tensixinet/telemetry/hwmon.py`
- Test: `tests/telemetry/test_hwmon.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) class FastReading: asic_temperature: float; power: float; voltage: float; current: float` (C, W, V, A).
  - `@dataclass(frozen=True) class HwmonChip: path: Path; bus_id: str`.
  - `find_chips(root: Path = Path("/sys/class/hwmon")) -> List[HwmonChip]`: directories whose `name` file reads `blackhole`, `wormhole` or `grayskull` (a module constant `CHIP_NAMES`), sorted by `bus_id`, where `bus_id` is the name of the directory `device` points to.
  - `read_chip(chip: HwmonChip) -> Optional[FastReading]`: `None` when any of the four files cannot be read or parsed.

- [ ] **Step 1: Write the failing tests** with a `tmp_path` tree (`hwmon3/name`, `temp1_input`, `power1_input`, `in0_input`, `curr1_input`, and `device` as a symlink to a directory named for the bus id):
  - `test_units_become_celsius_watts_volts_amps`: files `49188`, `41000000`, `818`, `51000` read as 49.188, 41.0, 0.818, 51.0 (tt-smi reports 51.0 A for the same chip).
  - `test_chips_are_ordered_by_bus_address_not_by_hwmon_number`: `hwmon9` at `0000:01:00.0` and `hwmon2` at `0000:02:00.0` come back in bus order.
  - `test_other_hwmon_devices_are_skipped`: a `name` of `coretemp` is not listed.
  - `test_a_missing_or_garbled_file_gives_none`: a missing `power1_input`, and one holding `n/a`, each give `None`.
  - `test_no_hwmon_directory_gives_an_empty_list`.
- [ ] **Step 2: Run** `.venv-app/bin/python -m pytest tests/telemetry/test_hwmon.py -q`; expect failures (`ModuleNotFoundError`).
- [ ] **Step 3: Implement** the three names above in `tensixinet/telemetry/hwmon.py`, plain file reads, no caching.
- [ ] **Step 4: Run the tests**; expect all pass.
- [ ] **Step 5: Mutation check**: swap each divisor (1000, 1e6, 1000, 1000), the sort key, and the name filter; each must turn a test red.

### Task 2: Eviction smoothing in seconds

**Files:**
- Modify: `tensixinet/music/evict.py` (`EMA_ALPHA` at line 36, `_Chip`, `_step` near line 112)
- Test: `tests/music/test_evict.py`

**Interfaces:**
- Produces: the smoothing weight for a reading `dt` seconds after the previous one: `0.5` when `dt >= 1.0` (today's behaviour, since live readings are at least 2 s apart), and `1 - 0.5 ** dt` when `dt < 1.0`. Constant `EMA_FULL_STEP_S = 1.0`; `EMA_ALPHA` stays 0.5 and means the weight at or beyond that spacing. The first reading of a chip sets the average to the reading, as now.

- [ ] **Step 1: Write the failing tests**:
  - `test_the_same_power_curve_evicts_at_the_same_time_at_one_second_and_at_a_tenth`: a chip at rest for 12 s, then 150 W for 12 s, then back to rest; feed it once per 1.0 s and once per 0.1 s (each reading repeats the level). The first `evicting` time differs by less than 1.5 s between the two feeds.
  - `test_readings_two_seconds_apart_smooth_exactly_as_before`: with `dt = 2.0`, the average after readings 10 W then 30 W is 20.0.
  - `test_a_short_gap_moves_the_average_less`: with `dt = 0.1`, 10 W then 30 W gives an average between 10.0 and 11.5.
- [ ] **Step 2: Run**; expect the first and third to fail.
- [ ] **Step 3: Implement**: store each chip's last reading time in `_Chip`; compute alpha as above in `_step`.
- [ ] **Step 4: Run** `tests/music/test_evict.py` and `tests/test_live.py`; expect all pass.
- [ ] **Step 5: Mutation check**: change `1.0` to `2.0`, `0.5 **` to `0.9 **`, and `<` to `<=` at the boundary (the last may be equivalent; say so if it is).

### Task 3: The fast source

**Files:**
- Create: `tensixinet/telemetry/fast.py`
- Modify: `tensixinet/telemetry/snapshot.py` (add `snapshot_bus_ids`)
- Test: `tests/telemetry/test_fast.py`, `tests/telemetry/test_snapshot.py`

**Interfaces:**
- Consumes: `hwmon.find_chips`, `hwmon.read_chip` (Task 1), `snapshot.fetch_snapshot`, `snapshot.parse_snapshot`.
- Produces:
  - `snapshot.snapshot_bus_ids(snapshot: dict) -> List[str]`: each device's `board_info.bus_id` in `parse_snapshot`'s order (empty string when missing).
  - `class FastSource(find=hwmon.find_chips, read=hwmon.read_chip, fetch=fetch_snapshot, clock=time.monotonic, slow_every_s: float = 1.5)` with `start()`, `stop()`, `sample() -> List[Telemetry]`, and attributes `fast: bool` (true when every chip's bus address matched) and `note: str` (a plain sentence for the log saying what is used).
  - `sample()` returns the latest slow reading with `asic_temperature`, `power`, `voltage`, `current` replaced by the fast values for matched chips, and never blocks on `fetch`. Before the first slow reading arrives it blocks up to 3 s for it, then returns `[]`.

- [ ] **Step 1: Write the failing tests** (a fake `fetch` returning a fixture dict from `tests/fixtures/tt_smi_snapshot.json`, a fake `find`/`read`):
  - `test_fast_fields_replace_the_slow_ones_and_slow_fields_stay` (clocks and GDDR temperatures come from `tt-smi`; temperature and power from hwmon).
  - `test_chips_are_matched_by_bus_address_when_the_orders_differ`: hwmon lists chips in the reverse order; each chip still gets its own readings.
  - `test_a_different_number_of_chips_turns_fast_off_and_says_so`: `fast` is false, `note` contains "tt-smi" and the two counts, readings are the slow ones.
  - `test_no_hwmon_turns_fast_off`.
  - `test_a_fetch_that_never_returns_does_not_stop_sample`: `fetch` blocks on an event; after the first reading, `sample()` returns within 50 ms on repeated calls.
  - `test_a_chip_whose_read_fails_keeps_its_last_fast_reading`.
  - In `test_snapshot.py`: `test_bus_ids_follow_the_parse_order` against the fixture (`["0000:01:00.0", ...]`).
- [ ] **Step 2: Run**; expect failures.
- [ ] **Step 3: Implement** `snapshot_bus_ids` and `FastSource`; the slow thread loops `fetch` then waits `slow_every_s` (an `Event.wait`, so `stop()` is prompt); the lock guards the latest slow reading.
- [ ] **Step 4: Run the tests**; expect all pass.
- [ ] **Step 5: Mutation check**: pair by position instead of bus address; drop the count check; make `sample()` call `fetch` itself; each must turn a test red.

### Task 4: The phrase engine takes its input and timing as parameters

**Files:**
- Modify: `tensixinet/ui/instrument.py`
- Test: `tests/ui/test_instrument.py`

**Interfaces:**
- Produces: `InstrumentSession.__init__` gains keyword parameters, all defaulting to today's values so the Play tab is unchanged:
  - `read: Optional[Callable[[float], Sequence[Telemetry]]] = None`: when given, `render_block` calls `read(now)` instead of `to_telemetry(self.knobs, self._tracker, now)`; the readings returned are already tracked (rates filled in by the caller).
  - `block_s: float = BLOCK_S`, `short_s: float = SHORT_BLOCKS * BLOCK_S`, `long_s: float = LONG_BLOCKS * BLOCK_S`, `xfade_s: float = XFADE_S`, `sample_rate: int = SAMPLE_RATE`. Block, phrase and crossfade lengths in samples are worked out from these per instance (replace the module-level `BLOCK_SAMPLES`, `XFADE_SAMPLES`, `phrase_samples` uses). A phrase is `round(short_s / block_s)` or `round(long_s / block_s)` blocks (at least 1) plus the crossfade tail.
  - `knobs` may be `None` when `read` is given.

- [ ] **Step 1: Write the failing tests**:
  - `test_a_session_with_a_read_function_renders_from_what_it_returns` (a fake `render` records its `chips` argument; they are the ones `read` returned).
  - `test_block_and_phrase_lengths_follow_the_parameters`: with `block_s=0.1`, `short_s=1.0`, `long_s=3.0`, a fake render sees phrase durations of `1.0 + xfade_s` and, after the short one runs out, `3.0 + xfade_s`; each block handed over is 0.1 s of samples.
  - `test_the_sample_rate_reaches_the_render_and_the_audio`.
  - The existing Play-tab tests pass unchanged.
- [ ] **Step 2: Run**; expect failures.
- [ ] **Step 3: Implement** the parameters; keep the module constants as the defaults.
- [ ] **Step 4: Run** `tests/ui tests/test_app.py`; expect all pass.
- [ ] **Step 5: Mutation check**: ignore `block_s`, ignore `sample_rate`, use `long_s` for the short phrase; each must turn a test red.

### Task 5: Change detection with a gap, and a held soloist

**Files:**
- Modify: `tensixinet/ui/instrument.py`
- Test: `tests/ui/test_instrument.py`

**Interfaces:**
- Produces: `InstrumentSession.__init__` gains `min_gap_s: float = 0.0`, `changed: Optional[Callable[[tuple, tuple], bool]] = None` and `hold_soloist: Optional[Callable[[Sequence[Telemetry]], float]] = None`:
  - With `changed`, a phrase is replaced when `changed(previous_signature, signature)` is true and at least `min_gap_s` has passed since the last phrase started. A thermal trip, error count change or eviction phase change always counts, and ignores the gap. With `changed=None` the rule is today's exact inequality.
  - With `hold_soloist`, the soloist stays until the number of seconds it returns (the patch's turn length) has passed since it was chosen; a phrase started by a change keeps it. Without it, a new soloist is picked per phrase as now.
- Also in `tensixinet/ui/instrument.py`: `signature_changed(previous, current, temp_c: float = 0.5, power_w: float = 3.0) -> bool` as the default `changed` for fast mode: true when any chip's temperature or power differs by more than the given amounts, or voltage differs by more than 0.01 V, or the phase tuple differs. (Starting values; tune by ear and write the final values in `docs/DECISIONS.md`.)

- [ ] **Step 1: Write the failing tests**:
  - `test_noise_under_the_thresholds_does_not_start_a_phrase` and `test_a_step_in_power_over_the_threshold_starts_a_short_phrase_in_the_next_block`.
  - `test_two_changes_inside_the_gap_make_one_phrase`; `test_a_trip_inside_the_gap_still_starts_a_phrase`.
  - `test_the_soloist_is_held_for_the_turn_length_and_survives_a_change`: with a fake clock and `hold_soloist=lambda chips: 6.0`, three change-started phrases in 2 s render with the same `soloists` argument; after 6 s the next phrase may differ.
  - `test_without_the_new_parameters_nothing_changed`: the existing suite.
- [ ] **Step 2: Run**; expect failures.
- [ ] **Step 3: Implement** inside `render_block` and `_start_phrase`.
- [ ] **Step 4: Run** `tests/ui tests/test_app.py`; expect all pass.
- [ ] **Step 5: Mutation check**: `>` for `>=` at each threshold (one may be equivalent), drop the gap, let a trip obey the gap, re-pick the soloist per phrase.

### Task 6: The measured delay

**Files:**
- Modify: `tensixinet/ui/instrument.py`
- Test: `tests/ui/test_instrument.py`

**Interfaces:**
- Produces: `InstrumentSession` records, for each block, `read_at` (the clock when its readings were taken) and `written_at` (when its audio was queued), and `InstrumentSession.delay_s(buffer_s: float, interval_s: float) -> Optional[float]`: the worst case, `interval_s + (written_at - read_at) + buffer_s`, for the most recent block, `None` before the first.
- A module constant `DELAY_BUDGET_S = 0.25` in `tensixinet/fast.py` (Task 8); this task only exposes the number.

- [ ] **Step 1: Write the failing tests** with a fake clock that advances 0.07 s inside a fake render: `delay_s(buffer_s=0.05, interval_s=0.1)` is 0.22 (within 1e-9); `None` before the first block; it follows the latest block.
- [ ] **Step 2: Run**; expect failures. **Step 3: Implement.** **Step 4: Run** `tests/ui`.
- [ ] **Step 5: Mutation check**: drop the interval term, drop the buffer term, use the first block's time; each must turn a test red.

### Task 7: Players with a small buffer

**Files:**
- Modify: `tensixinet/media/player.py` (`player_command`, `find_player`, `make_sink`)
- Test: `tests/media/test_player.py`

**Interfaces:**
- Produces: `player_command(name, sample_rate, channels=1, latency_ms: Optional[int] = None)`; `find_player` and `make_sink` pass `latency_ms` through; `LATENCY_FLAGS` documents each player's flags. With `latency_ms=None` every command line is exactly what it is today. `paplay` gets `--latency-msec=N`; `aplay` gets `--buffer-time=N*1000` (microseconds); `ffplay` gets `-fflags nobuffer -flags low_delay -probesize 32 -analyzeduration 0` (it has no millisecond setting, so its buffer for the delay report is a stated constant `FFPLAY_BUFFER_MS = 100`). `player_buffer_ms(name, latency_ms) -> int` returns what the delay report adds.

- [ ] **Step 1: Write the failing tests** for the three command lines with `latency_ms=50`, for the unchanged lines with `None`, and for `player_buffer_ms`. Before writing the flags, check each with a one-second silent run (`paplay`, `aplay`, `ffplay` each accepted the flags) and write what was seen in the test's comment.
- [ ] **Step 2: Run**; expect failures. **Step 3: Implement.** **Step 4: Run** `tests/media`.
- [ ] **Step 5: Mutation check**: drop the `--latency-msec` value, multiply `aplay`'s time by 1 instead of 1000.

### Task 8: `run_fast` and `--interval`

**Files:**
- Create: `tensixinet/fast.py`
- Modify: `tensixinet/cli.py` (the parser near line 82, the validation block near line 120, the live branch near line 232)
- Test: `tests/test_fast.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: `FastSource` (Task 3), `InstrumentSession` parameters (Tasks 4 to 6), the time-based `EvictTracker` (Task 2), `Tracker(window_s=0.5)`, `make_sink(..., latency_ms=...)` (Task 7).
- Produces:
  - `run_fast(source, sink, log, sample_rate, interval_s, duration, patch, patch_source, *, normalise, eq, only, cables, stereo, evict, player_name, latency_ms, clock=time.monotonic, sleep=time.sleep, session_factory=InstrumentSession) -> int` (blocks played). Each block: read `source.sample()`, run it through a `Tracker(window_s=0.5)` and (when `evict`) the session's own evict tracker, hand the block to the sink. Block length is `max(interval_s, 0.1)`, lengthened by the loop when a render takes longer than the block (log the real rate once). Short phrase 1.0 s and long phrase 6.0 s, crossfade `min(0.2, block/2)`. Log lines in live mode's style: the readings description once (`describe_readings`), the patch, one `live` line saying fast mode's interval, the fast or fallback `note`, then every 5 s the measured delay (`delay_s`), with a warning when it is over `DELAY_BUDGET_S = 0.25`. The log also says that errors and trips arrive with the 1.5 s `tt-smi` reading and are outside the budget.
  - CLI: `--interval MS` (type int, range 10 to 1000; out of range is a parser error naming the range). Requires `--live` (as `--mock-load` does) and is refused with `--kernels` (message: the kernels hold the chips and the render budget is too small). With `--mock-chips`, the sample is `mock_telemetry` read at the interval (the `FastSource` is not used and `note` says the readings are synthetic). Without `--interval` nothing changes.
  - `tensixinet.py`'s `with_defaults` needs no change (it already adds `--duration 0` for `--live`).

- [ ] **Step 1: Write the failing tests**:
  - `test_run_fast_plays_blocks_and_stops_at_the_duration` with a fake clock, a fake source, a `NullSink`-like recorder: the total audio is the duration plus the crossfade tail; the sink received blocks of `interval` seconds.
  - `test_a_power_step_reaches_the_audio_within_two_blocks`: the recorder's blocks after the step differ from those before it, within two blocks of the step.
  - `test_the_log_names_the_interval_the_fallback_and_the_measured_delay`; `test_a_delay_over_the_budget_is_warned_not_adjusted`.
  - `test_a_render_slower_than_the_block_lengthens_the_block_and_logs_the_real_rate`.
  - `test_ctrl_c_closes_the_sink_and_the_log`.
  - CLI: `--interval 9`, `--interval 1001`, `--interval 100` without `--live`, `--interval 100 --kernels sim`, and `--interval 100 --live --mock-chips 4 --duration 1 --player none` (exit 0, log mentions 100 ms). Without `--interval`, `run_live` is called as before (existing tests).
- [ ] **Step 2: Run**; expect failures. **Step 3: Implement.** **Step 4: Run** `tests/test_fast.py tests/test_cli.py tests/test_live.py`.
- [ ] **Step 5: Mutation check**: ignore the interval, let the loop skip the duration check, report the delay without the interval term, remove the budget warning.
- [ ] **Step 6: Measure on this box** (read-only, no lease; `--player none` first, then `paplay`): `tensixinet --live --interval 100 --duration 30 --player none` against the four real chips. Record the logged delay. Also run it with `tt-smi` unavailable (`PATH` without it) to see the fallback note. Write the numbers and the command in `CLAUDE.md`'s log entry (Task 10).

### Task 9: `--with-tt-toplike`

**Files:**
- Create: `tensixinet/media/toplike.py`
- Modify: `tensixinet/cli.py` (the parser, the validation block, the start of `main` before any chips are read)
- Test: `tests/media/test_toplike.py`, `tests/test_cli.py`

**Interfaces:**
- Produces:
  - `build_session(synth_argv: Sequence[str], interval_ms: int, toplike_args: Sequence[str], session: str = "tensixinet", inside_tmux: bool = False) -> List[List[str]]`: the tmux commands in order. Outside tmux: `new-session -d -s NAME` running the synth command in pane 0, `split-window -h` running `tt-toplike --interval N <toplike_args>`, `set-option -t NAME remain-on-exit off`, then `attach-session -t NAME`. Inside tmux: `split-window -h` in the current window with the same two commands and no attach. The synth command is `sys.executable -m tensixinet` with `synth_argv` minus `--with-tt-toplike` and `--toplike-args`, plus `--interval N`.
  - `run_session(commands, run=subprocess.call, which=shutil.which) -> int`; `check_available(which) -> Optional[str]`: a plain sentence when tmux or tt-toplike is missing, `None` otherwise; `session_exists(name, run) -> bool`.
  - CLI: `--with-tt-toplike` (flag) and `--toplike-args "..."` (a string split with `shlex`). Needs `--live`; interval defaults to 100 when `--interval` is absent; refused with `--mock-chips` (message: the two would show different chips) and with `--kernels`. When the session name exists, the run refuses with a message naming it. The synth pane's Ctrl-C ends it cleanly through the signal handling `run_fast` already has; when tt-toplike ends first, the tmux hook `pane-exited` sends the synth pane `C-c` and the session then closes (commands in `build_session`).

- [ ] **Step 1: Write the failing tests**:
  - `test_outside_tmux_the_session_has_two_panes_with_the_same_interval` (assert `--interval 100` appears in both pane commands, `attach-session` last).
  - `test_inside_tmux_it_splits_the_current_window_and_does_not_attach`.
  - `test_the_toplike_args_are_split_like_a_shell_would` (`"--mode arcade"` gives two arguments).
  - `test_the_flags_for_the_synth_pane_drop_the_launcher_only_flags`.
  - `test_missing_tmux_missing_toplike_mock_chips_kernels_and_an_existing_session_each_refuse_with_a_sentence` (five cases).
  - `test_a_run_returns_the_exit_code_of_the_attach`.
  - No test runs tmux.
- [ ] **Step 2: Run**; expect failures. **Step 3: Implement.** **Step 4: Run** `tests/media tests/test_cli.py tests/test_launcher.py`.
- [ ] **Step 5: Mutation check**: use a different interval for tt-toplike, drop the `--live` requirement, attach when inside tmux.
- [ ] **Step 6: Try it for real** in a throwaway tmux server (`tmux -L scratch`), with `--duration 20`, then close it. Say in the log entry what was seen; it was not heard by a person.

### Task 10: Docs, site and the log

**Files:**
- Modify: `README.md`, `docs/ARCHITECTURE.md` (new modules `telemetry/hwmon.py`, `telemetry/fast.py`, `fast.py`, `media/toplike.py`), `docs/DECISIONS.md` (fast mode's calls: hwmon plus `tt-smi`, phrase lengths, thresholds, the evict smoothing rule, the held soloist, the budget and what is outside it, the rolls' own schedule), `docs/ROADMAP.md` (add part B: tt-toplike publishes fast readings and its view; a defrag song), `CLAUDE.md` (layout block, a dated log entry with the prompt and the measured delay), `tests/test_site.py` and `site/index.html` only if the site names live mode's behaviour in a way that is now wrong.
- Test: `tests/test_hygiene.py` and the full suite.

- [ ] **Step 1:** Update the docs; check any number or flag named in them against the code (the interval range, defaults, the 250 ms budget).
- [ ] **Step 2:** Sweep the new text for the phrases listed in Global Constraints.
- [ ] **Step 3:** Run the full suite: `.venv-app/bin/python -m pytest -q`; expect no failures (the count is 2564 passed, 34 skipped plus the new tests).
- [ ] **Step 4:** Report the measured delay at 100 ms on four real chips, the fallback behaviour, and what has not been heard.
