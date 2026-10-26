# Fast sampling and `--with-tt-toplike`: design

Date: 2026-10-23. Status: for approval. Part A of a two-part plan; part B is listed under non-goals.

## Intent

When tt-toplike and the synth run side by side, the picture changes first and the sound follows seconds later. You see something
coming and then hear it. The owner wants the beats and the music to track the chips second by second, so that a power spike in
tt-toplike and the sound change land together. The owner's budget for the delay between a chip changing and the change being
heard is **a quarter of a second**.

Two features:

1. `--interval MS` for the live synth, the same idea as tt-toplike's `--interval` (10 to 1000 ms): how often the chips are read.
2. `--with-tt-toplike`: one command that starts tt-toplike and the synth at the same interval so the two stay in step.

## Why it is slow today

- `run_live` (`tensixinet/live.py`) reads the chips once per segment, and a segment is 2 to 10 s (`MIN_SEGMENT_S`, `MAX_SEGMENT_S`).
- It runs `tt-smi -s`, which takes about 0.25 s when healthy and has hung for minutes on this box (`SNAPSHOT_TIMEOUT_S` is 1.5 s
  for that reason).
- It renders up to two segments ahead (`LEAD_IN_SEGMENTS`), and the player buffers on top.
- tt-toplike reads temperature, power, voltage and current from Linux hwmon files on every update (well under 1 ms) and takes
  `tt-smi` records about every 1.5 s. Its `--serve` stream carries only the `tt-smi` records, so it is no faster than 1.5 s.

## What was measured (2026-10-23, this box, four Blackhole chips)

- `/sys/class/hwmon/hwmonN` with `name` = `blackhole` exists for each chip: `temp1_input` (milli-degrees C), `power1_input`
  (micro-watts), `in0_input` (millivolts), `curr1_input` (milliamps). `device` links to the PCI address (`0000:01:00.0` to
  `0000:04:00.0`). All four read instantly while `tt-smi -s` was hanging.
- Rendering `render_chorus` for four mock chips in stereo at 22.05 kHz: 0.25 s of audio takes 45 ms, 1 s takes 67 ms, 4 s takes
  186 ms. Most of the cost is fixed, so short renders are cheap.
- The Play tab already does the hard part of this: a loop that reads its inputs every block (1.2 s), starts a short phrase when an
  input moves, plays long phrases when nothing moves so the slow devices (the gong strikes up to 2.2 s in) get their time, and
  crossfades where a phrase is replaced (`tensixinet/ui/instrument.py`).

## Goals

- With `--live --interval MS`, a change in a chip's temperature or power is audible within 250 ms of the hwmon file changing, at
  the default interval of 100 ms. The run measures and logs its own delay so the claim is checked and not assumed.
- The synth does not stop or stutter when `tt-smi` hangs. Fast readings come from hwmon alone; `tt-smi` fills the slow fields in
  the background.
- `--with-tt-toplike` gives one command, one interval, two panes.
- Everything live mode plays today still plays: patch, style, cables, eviction, stereo, `--only`, the EQ and the loudness gain.

## Non-goals

- **Part B: changes to tt-toplike.** tt-toplike is ours but changes go through a review process, so it ships separately. Part B
  would let tt-toplike publish its fast readings and its current view, so the synth takes both from it. The view signal is what
  would drive a dedicated defrag song when the view changes. Part A builds the fast sampler that part B can reuse. Nothing here
  needs part B.
- A view-driven song of any kind.
- `--kernels` together with `--interval` (the chips are held by the kernels and the render budget is tight). It is refused with a
  message.
- Sound at the same instant as the picture. A block has to be rendered before it plays; the goal is the 250 ms budget.
- macOS or Windows (hwmon is Linux only; without it the synth falls back to `tt-smi` and says so).

## Anti-goals (failures even if it technically works)

- Sound that clicks, drops out or loses the slow devices (gong, rolls) because every step restarted their clocks.
- A fast mode that changes how `--live` sounds when `--interval` is not given.
- Two readings of the same chip that disagree about which chip is which (hwmon and `tt-smi` order must be matched, not assumed).

## Design (calls marked as mine are the builder's to question)

**Reading.** New `tensixinet/telemetry/hwmon.py`: finds the chips' hwmon directories, orders them by PCI address, and returns
temperature, power, voltage and current in the units `Telemetry` already uses (C, W, V, A). A background thread runs `tt-smi -s`
every 1.5 s or so with the existing timeout, never blocking the fast loop; its fields (clocks, GDDR temperatures, errors, masks)
are kept as the latest slow reading and merged under the fast fields. The chip order of the two sources is matched by PCI bus
address (`board_info.bus_id` in the `tt-smi` JSON against the hwmon `device` link). If the two cannot be matched, or hwmon is
missing, the synth reads `tt-smi` only and logs that fast sampling is unavailable. *(My call: sysfs mirrors tt-toplike's own
hybrid backend.)*

**Playing.** A new fast loop, `run_fast`, beside `run_live`. It reuses the Play tab's phrase engine: each block (length: the
interval, at least 100 ms) reads the latest fast reading; a change beyond a threshold starts a short phrase, a quiet stretch
plays long phrases, and a phrase that replaces another crossfades. The phrase logic that `InstrumentSession` holds now is moved
into a shared module so both use it, with the Play tab behaving exactly as it does today. Rates (`temp_rate`, `power_rate`) come
from the fast readings, so power spikes show up as rate and not only as level. *(My call: change thresholds with a minimum gap,
so noise on a 100 ms read does not restart a phrase ten times a second. Values are the builder's to choose and tune by ear, and
to be written down in `docs/DECISIONS.md`.)*

**Latency.** The loop keeps only about one block of audio ahead of the listener and asks the player for a small buffer
(`paplay --latency-msec`, `aplay` buffer arguments, `ffplay` flags). The run logs, every few seconds, the measured delay from a
reading to the audio write plus the player's configured buffer, so the quarter-second claim comes from the run itself.

**`--interval MS`.** Allowed range 10 to 1000, as in tt-toplike. Needs `--live` (as `--mock-load` does). Absent: live mode is
unchanged. Works with `--mock-chips`; the synthetic source is sampled at the same rate. *(My call: not given means today's
behaviour, so no existing run changes.)*

**`--with-tt-toplike`.** Needs `--live`, tmux on the PATH, and real chips. It starts a tmux session with two panes, the synth
(`tensixinet --live --interval N` and the user's other flags) and `tt-toplike --interval N`. Inside tmux it splits the current
window. When either program ends, the session closes. The interval defaults to 100 (tt-toplike's own default) so the two match
without any flag. Refused with a message: `--mock-chips` (the two would show different chips), no tmux, no tt-toplike on the
PATH. Extra arguments for tt-toplike: `--toplike-args "..."` (for example `--mode arcade`). The launcher `tensixinet.py`
passes both flags through. *(My call: tmux was already the plan in the 2026-10-18 log entry.)*

**No gozer lease.** hwmon reads and tt-toplike's sysfs reads do not open `/dev/tenstorrent/*`.

## Settled after the builder check

Each of these is my call unless it says the code decided. They are open to the owner's veto.

1. **What "audible within 250 ms" covers.** A change in a chip's reading must reach the audio of every device that reacts at
   once (the drone, shimmer, lead, choir, the quantus pitch, levels and the mix) within the budget. The slow devices (the gong,
   rolls and steam register) change at their next strike or pulse, as they do in live mode today, because their rhythm needs
   time to exist. The new phrase restarts those devices' clocks as the Play tab's does; its crossfade carries the old phrase's
   continuation so nothing cuts. The owner agreed (2026-10-23): in the Ciat-Lonbarde instruments the rolls keep their own schedule,
   and the rest of the system can only influence them and be influenced by them (the links the code already draws). This is the biggest musical call: it means a power spike is heard at once in the drones and air,
   and in the beats only from the next strike.
2. **Slow fields.** Errors, trips and mask changes arrive on the `tt-smi` cycle (about 1.5 s) and start a short phrase when they
   arrive. They are exempt from the 250 ms budget, and the doc and the log say so.
3. **Eviction and rates (the code decided this).** `music/evict.py` smooths power with `EMA_ALPHA = 0.5` per reading, written for
   readings seconds apart; at 10 readings a second it would smooth over about 0.2 s and react ten times too fast. Its smoothing
   becomes time-based (the same weight per second as today's typical segment), with a test that feeds the same power curve at
   1 s and at 100 ms spacing and gets the same eviction time. `Tracker` already takes a window; fast mode uses a window of about
   0.5 s so rates follow spikes, and live mode keeps its 2 s.
4. **Turn-taking.** `run_fast` is a variant of `run_live`'s rules inside the Play tab's phrase engine. A soloist is held for the
   patch's turn length (`turn_seconds`) and is chosen the way `run_live` chooses it; a short phrase started by a change keeps the
   held soloist. The loudness gain, EQ, `--only`, cables, styles and eviction apply as in live mode. The 0.4 s stitcher is not
   used; the phrase crossfade (0.2 s, or half a block if the block is shorter) replaces it.
5. **No fast source.** If hwmon is missing or cannot be matched to the `tt-smi` order, `--interval` still runs, at the rate
   `tt-smi` gives. The log says so in plain words and makes no 250 ms claim. `--with-tt-toplike` behaves the same way.
6. **The measured delay.** The log reports the worst case: the interval (a change can land just after a read), plus the render,
   plus the audio write, plus the player's configured buffer. It warns when this is over 250 ms and does not change the interval
   by itself.
7. **Shutdown of `--with-tt-toplike`.** Ctrl-C in the synth pane ends the synth cleanly (log and WAV flushed), and tmux then
   closes tt-toplike's pane. If tt-toplike ends first, the session sends the synth a Ctrl-C and waits for it before closing.
8. **Players.** The first target is `paplay` with `--latency-msec`; `aplay` and `ffplay` get their own small-buffer flags and are
   checked to start.

## Constraints from the repo

Test first, then a mutation check; an SPDX header on every `.py`; a test file per module; no `ttnn`, `ttl`, `ttml`, `torch` or
`sounddevice` imports in `tensixinet/` or `tests/`; docs that name numbers and flags updated in the same change (`README.md`,
`docs/ARCHITECTURE.md`, `docs/DECISIONS.md`, the `CLAUDE.md` log, `site/` where it names live mode); plain English in docs, no
"X, not Y" framing; no commits unless asked.

## Testing

- hwmon reading against a fake directory tree (milli-degrees, micro-watts, millivolts, milliamps; ordering by PCI address; a
  missing file; a chip that vanishes mid-run).
- Matching by bus address, with orders that differ between the two sources.
- The loop with an injected clock and a fake source: a step in power starts a short phrase within one block; a quiet source plays
  a long phrase; a `tt-smi` that never returns does not stop the audio.
- The tmux command line (built, not run): the interval in both panes, inside and outside tmux, each refusal.
- Argument checks for both flags.
- Measured on the real chips and not tested: end-to-end delay at 100 ms, reported in the log and the CLAUDE.md entry with the
  command that produced it.

## Left to the builder

Names of modules and functions beyond those above; the change thresholds and minimum gap; block size when the interval is below
100 ms; the player buffer arguments for each player; how the shared phrase module is carved out of `instrument.py`.

## Open for the owner

- Whether the first check of the quarter-second budget is on the speakers or in the log's measured number (I assume the log, with
  the owner listening afterwards).
