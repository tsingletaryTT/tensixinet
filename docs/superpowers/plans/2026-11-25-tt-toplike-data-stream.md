# Attach the synth to tt-toplike's data stream Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When `tensixinet --live --with-tt-toplike` starts the two tmux panes, the synth consumes tt-toplike's telemetry stream instead of reading the chips a second time. The stream carries both the four fast hwmon readings and tt-toplike's current view, so a view change in tt-toplike can change the music (the Defrag view gets a defrag song).

**Why now:** Today the synth pane and the tt-toplike pane each read the chips on their own. They tick at the same `--interval`, but they are two readers: the readings can drift, the work is done twice, and the synth cannot know which view tt-toplike is showing. ROADMAP item 7d asks for exactly this: tt-toplike publishes its fast readings and current view on its `--serve` stream, the synth takes both, and a view change can change the music.

**Architecture:** tt-toplike already publishes a `/telemetry` WebSocket frame from its `--serve` mode (`bus::headless` and the TUI's `/serve` block, through `bus::compat`). The frame is an opaque `tt-smi -s` JSON object plus an additive, schema-versioned `tt_toplike` extension. We add a second additive key, `tensixinet`, carrying per-chip fast readings and the current view. On our side, a new `ToplikeSource` implements the same surface as `FastSource` (`start`/`stop`/`sample`/`fast`/`note`/`log`) over a WebSocket client, so `run_fast` needs no change to read it. `--with-tt-toplike` picks a loopback port, starts tt-toplike with `--serve 127.0.0.1:<port>` and the synth with `--feed ws://127.0.0.1:<port>/telemetry`, and keeps the existing end conditions. View changes reach `run_fast` through an `on_view` hook; phase 1 logs them, phase 2 switches the patch at the next phrase boundary.

**Tech Stack:** Python 3.10+, numpy, pytest (tensixinet); Rust, serde/serde_json, tokio, axum/WS (tt-toplike); tmux; Linux hwmon. No new Python dependency: the client is a small stdlib WebSocket reader with an injectable socket for tests.

**Spec / background:** `docs/ROADMAP.md` item 7d, `docs/ARCHITECTURE.md` (`--with-tt-toplike`), and the plan this extends, `docs/superpowers/plans/2026-10-23-fast-sampling.md`.

**Scope split:** Phase A ("attach and consume") is the answer to the question asked here and can land on its own. Phase B ("view changes change the music") depends on A. Each phase is listed with its own tests-first tasks.

---

## Facts established before writing this plan (verify again before coding)

* tt-toplike `--serve [BIND:]PORT` broadcasts through a `tokio::sync::watch` (`TelemetryPublisher`, `src/backend/serve.rs`): **newest frame wins**, no backlog, opaque JSON string. Default port is 8770 (`cli.rs:623`). `--remote HOST:PORT` already consumes `ws://<host:port>/telemetry`, so there is a working WS client to crib framing/reconnect from (`cli.rs:601-606`).
* The frame is built in exactly two publish sites: `src/bus/headless.rs` (headless loop) and `src/bin/tui.rs` (`/serve` block), both through `src/bus/tui_bus.rs::serve_frame`, which calls `bus::compat::telemetry_frame(world, ext)` = newest `tt-smi` raw body + `inject_extension`. `src/backend/remote_ext.rs::inject_extension` adds one key (`EXT_KEY`, `"tt_toplike"`) to the JSON object and is best-effort (bad JSON passes through unchanged).
* The extension pattern is additive and versioned: `TT_TOPLIKE_SCHEMA`, `#[serde(default)]` on fields added later, unknown keys tolerated by serde. Adding a *sibling* key is therefore safe for every existing consumer (tt-station, `--remote` peers) and needs no schema bump on `tt_toplike` itself.
* The bus already has `Event::ViewChanged`, `Bus::note_view()`, and `view_changes()` (`src/bus/tui_bus_tests.rs`, `src/viz/declarative/widgets.rs`). None of that reaches `/telemetry` today; the current view is available in the world/UI state the publish sites already hold.
* Our source boundary is `run_fast(source, ...)` in `tensixinet/fast.py`: it calls `source.sample()` for `List[Telemetry]`, reads `source.fast` and `source.note`, sets `source.log`, and calls `start()`/`stop()`. `FastSource` in `tensixinet/telemetry/fast.py` is the reference implementation. `run_fast`'s `read()` already replays the last good readings with zero rates when `sample()` returns `[]` — the WS source reuses that behaviour for a disconnect.
* `tensixinet/media/toplike.py` builds the two tmux commands from `raw_argv`; the synth pane re-runs without `--with-tt-toplike` and with `TENSIXINET_IN_SESSION=1`. `cli.py::_with_toplike` refuses `--mock-chips`/`--kernels` and requires `--live`. `pyproject.toml` depends only on numpy and soundfile.
* Per the user's standing rule, every new TUI command or explanation goes into the `?` help (`tensixinet/tui/layout.py::HELP`, wired by `tensixinet/tui/app.py::HELP_KEYS`), not into a new README list.

---

## Phase A — attach and consume

### Task A1: Define the `tensixinet` extension on the tt-toplike side (tests first)

**Files:**
* Modify: `tt-toplike/src/backend/remote_ext.rs` (new `TensixinetExt`, `FastChip`, `ViewState`, `inject_tensixinet_extension`, `parse_tensixinet_extension`, `TENSIXINET_SCHEMA`)
* Test: `tt-toplike/src/backend/remote_ext_tests.rs` (new cases alongside the existing `tt_toplike` ones)

Wire shape (JSON added as a sibling of `tt_toplike`):

```json
"tensixinet": {
  "schema": 1,
  "fast": [
    {"bus_id": "0000:00:04.0", "temp_c": 54.2, "power_w": 121.0,
     "voltage_v": 0.86, "current_a": 140.7}
  ],
  "view": {"name": "defrag", "seq": 7}
}
```

* `bus_id` is the PCI address, the same key our hwmon pairing uses (`telemetry/fast.py::_decide`). Never pair fast readings by position.
* `view.seq` is monotonic per process. The synth remembers the last `seq` it handled, so a reconnect cannot replay an old change and a dropped frame cannot hide a newer view.
* All fields `#[serde(default)]` except `schema`; unknown future keys are ignored. Omitted `fast` means "producer does not stream fast readings" (the synth then falls back to the base `tt-smi` frame), not "zero chips".

- [x] **Step 1: Write failing tests** in `remote_ext_tests.rs`: round-trip `TensixinetExt`; `inject_tensixinet_extension` leaves every pre-existing key byte-equivalent in value and preserves an existing `tt_toplike` key; malformed/non-object base JSON passes through unchanged; `parse_tensixinet_extension` returns `None` on a wrong `schema`, absent key, or bad types.
- [x] **Step 2: Run** `cargo test tensixinet_ext` in `~/code/tt-toplike`; expect failures (names not found).
- [x] **Step 3: Implement** the types and functions in `remote_ext.rs`, following `inject_extension`'s best-effort pattern exactly. Do not modify `TtToplikeExt` or bump `TT_TOPLIKE_SCHEMA`.
- [x] **Step 4: Run** the focused tests; they pass. Then `cargo test` for the crate.

### Task A2: Populate and publish the extension at both serve sites (tests first)

**Files:**
* Modify: `tt-toplike/src/bus/compat.rs` (a `telemetry_frame_with` helper), `src/bus/headless.rs`, `src/bin/tui.rs` (serve block), and the backend trait/snapshot used there
* Test: `src/bus/headless_tests.rs` and `src/bus/tui_bus_tests.rs`

* The publisher needs the four fast values at the publish tick. Verify first whether the active backend already exposes them (`backend/luwen.rs` reads temperature/power; `factory.rs` picks hybrid hwmon). If it does, add a small `fast_readings() -> Vec<FastChip>` accessor, keyed by PCI address. If it does not, add the four hwmon reads to the hybrid backend's update path — do **not** shell out to `tt-smi` here.
* The current view is available where the serve block already renders it; thread it into the frame as `{name, seq}` using the bus's `ViewChanged` sequence. Keep the `seq` counter in the publisher so it survives a view that changes twice between frames.
* Both publish sites must call the same helper so headless and TUI cannot drift. The `--remote` relay path must keep relaying the origin frame verbatim (`compat::relay_frame`) and must not add our key.
* Publish only when a frame already exists (`serve_frame` returns `None` when there is no `tt-smi` snapshot); enrichment is additive.

- [x] **Step 1: Write failing tests**: a frame served by the headless loop contains `tensixinet.fast` with one entry per chip and the right `bus_id`; a `ViewChanged` before a publish is reflected in `tensixinet.view` with an incremented `seq`; a relay frame has no `tensixinet` key; no raw frame yields no frame at all.
- [x] **Step 2: Run** the focused tests in tt-toplike; expect red.
- [x] **Step 3: Implement** the accessor, the compat helper, and the two call sites.
- [x] **Step 4: Run** the focused tests, then `cargo test`.

### Task A3: Synth-side WebSocket source (tests first)

**Files:**
* Create: `tensixinet/telemetry/toplike_stream.py`
* Test: `tests/telemetry/test_toplike_stream.py`

`ToplikeSource` implements the `FastSource` surface:

* `start()` connects in a daemon thread; `stop()` closes the socket and joins briefly.
* A minimal stdlib reader: TCP connect, HTTP `Upgrade: websocket` handshake with a random base64 key, then read server→client text frames (handle 7-bit, 126 and 127 lengths; server frames are unmasked). No client frames are needed to subscribe. Inject a socket factory so tests never open a real socket. Keep the reader to one file with a hard cap on frame size (reject absurd lengths).
* `sample()` returns the newest frame's fast readings as `Telemetry` objects: parse the base `tt-smi` JSON with the existing `parse_snapshot`, then lay `tensixinet.fast` over temperature, power, voltage and current by `bus_id`. When no frame has arrived yet, return `[]` (the `run_fast` "no chips" path). When the socket drops, return `[]` so `run_fast` replays the last reading with zero movement; do not invent chips.
* `fast`/`note` describe the source ("Readings come from tt-toplike's stream") for the log; `log` is set by `run_fast` as it is for `FastSource`.
* `view()` returns `(name, seq)` of the newest frame; `drain_views()` returns newly seen `(name, seq)` pairs in order, so no change is lost between blocks.
* Reconnect with **backoff**: first retry after 0.25 s, double each failure up to a 5 s cap, reset on a good frame. Log once on loss and once on recovery (not once per retry). This is the same "back off for longer" rule used elsewhere in this project.

- [x] **Step 1: Write failing tests** with a fake socket handshake + scripted frames: connect/handshake; a frame becomes `Telemetry` values overlaid by `bus_id`; two frames in a row return only the newest; a bad frame is ignored and the last good one kept; disconnect yields `[]`; reconnect timing follows 0.25/0.5/1/…/5 and resets after a good frame; `drain_views()` yields each new `seq` once and replays nothing after reconnect.
- [x] **Step 2: Run** `python3 -m pytest -q tests/telemetry/test_toplike_stream.py`; expect `ModuleNotFoundError`.
- [x] **Step 3: Implement** `toplike_stream.py`.
- [x] **Step 4: Run** the focused tests until green.

### Task A4: `--feed` flag and tmux wiring (tests first)

**Files:**
* Modify: `tensixinet/cli.py`, `tensixinet/media/toplike.py`, `tensixinet/fast.py` (source selection), `README.md`, `docs/ARCHITECTURE.md`
* Test: `tests/test_cli.py` (or the existing CLI test file) and `tests/media/test_toplike.py`

* Add `--feed URL`: live only; refuses `--mock-chips`, `--kernels` and `--with-tt-toplike` used without it (the whole point is that the launcher supplies it). Refuse a URL that is not `ws://`/`wss://` loopback-by-default? No: allow any host but document that a remote feed is the same trust boundary as tt-toplike's `--remote`.
* In the synth pane, `--feed <url>` selects `ToplikeSource` in place of `FastSource`; everything after that is the unchanged `run_fast` path.
* Under `--with-tt-toplike`, choose a free loopback port (bind port 0, read it, close; a tiny race is acceptable on a single-user box, or pass `--serve 127.0.0.1:0` if tt-toplike can report its chosen port — verify). Pass `--serve 127.0.0.1:<port>` to the tt-toplike pane and append `--feed ws://127.0.0.1:<port>/telemetry` to the synth pane's rebuilt `raw_argv`. Keep the existing pane tags and end conditions unchanged.
* `--toplike-args` must not be able to override `--serve` with a conflicting port: detect a `--serve`/`/serve` token there and refuse it with a clear message.
* README: one short sentence that `--with-tt-toplike` now shares tt-toplike's readings through `--feed`; no duplicate TUI instructions (those live under `?`).

- [x] **Step 1: Write failing CLI/media tests**: `--feed` needs `--live`; `--feed` refuses `--mock-chips` and `--kernels`; `--feed` with a plain offline run errors; the rebuilt synth command contains `--feed ws://127.0.0.1:<port>/telemetry` and drops `--with-tt-toplike`; the tt-toplike command contains `--serve 127.0.0.1:<port>`; a `--serve` in `--toplike-args` is refused.
- [x] **Step 2: Run** those tests; expect red.
- [x] **Step 3: Implement** the flag, validation, source selection and command wiring.
- [x] **Step 4: Run** the focused tests, then the whole TUI/CLI test set.

### Task A5: End-to-end attach check and Phase A close-out

- [x] **Step 1: Run** `python3 -m pytest -q` in tensixinet and `cargo test` in tt-toplike. Both must be green before any commit or push (standing rule).
- [x] **Step 2: Manual check** on a box with chips. Telemetry reads are read-only, so no gozer lease is needed for this run. Start `tensixinet --live --interval 100 --with-tt-toplike`; confirm the log says the readings come from tt-toplike's stream, that both panes show the same values, and that killing either pane ends the other as before.
- [x] **Step 3: Confirm backward compatibility** by pointing `--feed` at an older tt-toplike (no `tensixinet` key): the synth must fall back to the base `tt-smi` frame and say so in the log, not crash.
- [ ] **Step 4: Commit Phase A** only if asked. Do not push without an explicit request.

---

## Phase B — view changes change the music

### Task B1: `on_view` hook in `run_fast` (tests first)

**Files:** `tensixinet/fast.py`, `tests/fast/test_view_hook.py` (or the existing fast-mode test file)

* Add `on_view: Optional[Callable[[str, int], None]]` next to `on_segment`. Call it on the loop's thread with `(view_name, seq)` for each new view from `ToplikeSource.drain_views()`, just before a block is queued. Same contract as `on_segment`: store only what it is given, return quickly; if it raises, log one line and stop calling it, never affect the sound.
* A source without `drain_views()` (e.g. `FastSource`) is a no-op.

- [x] **Step 1: Write failing tests**: hook called once per new seq; not called on a repeated seq; a raising hook is logged once and disabled; a source without `drain_views()` is fine.
- [x] **Step 2: Run** focused tests; expect red. **Step 3: implement. Step 4: green.**

### Task B2: View-to-patch map and the defrag song (tests first)

**Files:** `tensixinet/cli.py`, a new `tensixinet/viewmap.py` (or extend the library module), `library/patches/` (a defrag patch), `tests/test_viewmap.py`, TUI `?` help in `tensixinet/tui/layout.py`

* Add `--view-map FILE`: JSON mapping view name to a saved patch name, e.g. `{"defrag": "defrag"}`. Unmapped views change nothing. Only meaningful with `--feed` (or with `--with-tt-toplike`).
* Ship a small defrag patch in the library and document it by name. The patch is ordinary saved-patch JSON; no new patch format.
* On a mapped view change, switch the patch **at the next phrase boundary** using the existing `evict` path so the change does not cut a voice mid-phrase. Defer the switch until the current phrase ends; do not build a second audio graph. If switching proves to fight the phrase engine, fall back to the honest smaller behaviour: log the view change and retune the current patch's tempo/register for Defrag, and say plainly in the help what it does.
* Any new TUI explanation (e.g. "the view shown in the right pane chooses the patch") goes into `HELP` in `tensixinet/tui/layout.py`; the README points to `?` and does not restate it.

- [x] **Step 1: Write failing tests**: map parses and validates unknown patch names; an unmapped view is ignored; a mapped view schedules a switch at a phrase boundary; the TUI help text mentions the behaviour.
- [x] **Step 2: Run** focused tests; expect red. **Step 3: implement. Step 4: green.**

### Task B3: Phase B close-out

- [x] **Step 1: Run** `python3 -m pytest -q` (and `cargo test` if tt-toplike changed). Green before any commit or push.
- [x] **Step 2: Manual check**: with `--with-tt-toplike --view-map`, switch tt-toplike to Defrag and confirm the patch follows on the next phrase, with no audio glitch.
- [x] **Step 3: Update** `docs/ROADMAP.md` item 7d to mark what landed and link this plan. Commit only if asked.

---

## Open questions to settle before Phase A task A2

1. Does the active tt-toplike backend already expose the four fast readings at the publish tick? If not, how much hwmon code must be added to `backend/luwen.rs`? (This is the only substantial unknown in the plan; everything else is wiring.)
2. Can `--serve 127.0.0.1:0` report the port it chose? If not, the launcher's bind-then-close port reservation is fine on a single-user box but should be documented as a small race.
3. Should `view.seq` be process-local (simple) or shared with the bus's own view-change counter (one source of truth)? Process-local is fine as long as the synth keys on `(name, seq)` from the same stream.
4. Is a stdlib WS reader enough, or do ping/pong heartbeats from the tokio server require replies? Test against the real `--serve` endpoint early; if it needs more than text-frame reading, add `websockets` as an optional extra installed by `--with-tt-toplike` rather than a hard dependency.

## Explicit non-goals

* No change to the `tt_toplike` extension schema or to the opaque `tt-smi` frame; existing consumers keep working.
* No second telemetry reader in the synth when `--feed` is active (that is the point), and no `tt-smi` subprocess on the WS thread.
* No TUI commands or explanations outside `?`.
* No commit or push without an explicit request from the user.

## Close-out (2026-11-26)

Manual checks run on the four real chips (read-only, no lease), against a tt-toplike built from the working tree (`target/debug`, not installed):
* A5 step 2: both panes ran; the synth log said "Readings come from tt-toplike's stream (4 chips)"; ending the tt-toplike pane closed the session.
* A5 step 3: the installed 0.19.1 tt-toplike (no `tensixinet` key) served 4 chips, the log said "Readings come from the base tt-smi frame", exit 0, 120 blocks in 12 s.
* B3 step 2: pressing `v` in the arcade view moved tt-toplike to Defrag; the synth logged "switched to defrag, patch is now defrag" about 2 s later. Not listened to, so a glitch-free switch is unconfirmed.

The checks found three defects that the fake-socket tests had passed. Each got a failing test first:
1. The default connector called `socket.create_connection(host, port, timeout=...)`, a `TypeError`; the reader thread died on every real connection.
2. `sample()` returned `[]` while the reader was still connecting, so a fresh run printed "no chips found" and exited.
3. `--serve` is refused by tt-toplike on the auto and hybrid backends, so the launched tt-toplike pane exited at once; the launcher now passes `--backend json` unless the user named a backend.
