# `--tui`: the patch layout in the terminal, with its own log

Status: draft for review. Written from the conversation of 2026-10-30. Calls the author made without being told are marked **(my call)**.

## Why

The Gradio app draws what is playing: the chips, the devices dealt to them, who has the floor, the cables between devices and the stereo field. The owner likes that part best, because "it gives you a feeling of the inputs and outputs and the roles things play." They want the same in a terminal, beside tt-toplike: "brighter and borders for focus works well in ascii/ansi too." They also want the TUI to include "a version of the log that color codes and informs everything."

## What it is

`--tui` (with `--live`) draws a full-screen terminal view of what is playing, redrawn as the segments play. It is built from the same `FlowState` the Gradio chart uses (`ui/flow.py`: each chip's temperature, power, pressure, chaos, stereo position and phase; each device with its chip, whether it has the floor, its turn share and stereo place; the cables with how hard each pushes now). It follows the segment that is playing, as the chart does, not the one just rendered.

## Layout **(my call, except where the owner spoke)**

* **Chips across the top**, one box each: temperature, power, pressure and the phase (running, evicting, idle, filling) with the stereo home. An evicting chip is faint.
* **Devices under their chip**, each as a small box. The device with the floor, and the chip that plays it, get a bright border and bold text. The rest are dim. A device's turn share is a short bar.
* **Cables** between devices as labelled links (kind and depth). They brighten as the push grows.
* **The stereo field** as one line: each voice's position.
* **The log**, built into the same screen: a colour-coded version of the run log that explains each line. It uses the Plumbutter jack colours and the per-chip and per-device colours that `narrate/color.py` already defines, so it matches what the plain terminal log shows. It shows the latest lines for what just happened (the segment, who has the floor and why, cables, evictions, errors) and scrolls.
* No right-hand borders (the global rule): left and bottom bars only. It fits a half-width pane (80 columns) and redraws only the cells that change.

## How it runs

* Beside tt-toplike: `--with-tt-toplike --tui` makes two panes, tt-toplike and the TUI, in the same tmux session and with the same config. The TUI pane replaces the plain log pane, because it has the log inside. The synth itself still runs where it runs now **(my call)**: the TUI is the synth's front end and runs in the synth pane.
* Alone: `--live --tui` takes the whole terminal.
* The plain log, the JSONL and the `.txt` file are still written. `--tui` changes only what the screen shows.
* Quitting: Ctrl-C ends the run as it does now, with the WAV and logs written out.

## Goals

* The TUI and the Gradio chart cannot drift apart: both read `FlowState`. A test builds the TUI's text from a `FlowState` and checks that each device, chip, cable and the soloist appear.
* Focus is visible in ANSI: a test checks that the soloist's box and chip carry the bright attribute and the others do not.
* Replay of colours and layout is deterministic for a given state, so it can be tested as text.
* It works with no colour (`NO_COLOR`, or not a terminal): the focus then shows as brackets or a marker, and nothing breaks.

## Non-goals

* Not a synth panel: nothing changes the sound from the TUI in the first version.
* No stage or scene display, and no Play-tab equivalent.
* Not a replacement for tt-toplike, and no change to tt-toplike.
* No mouse.

## Anti-goals

* A TUI that slows the synth or makes it stutter. The draw runs off the audio path, and a slow terminal drops frames and does not hold up the sound.
* A screen that breaks when the terminal is narrower than expected.

## Constraints

* The rules in `CLAUDE.md`: tests first, a mutation check, an SPDX header, no `ttnn`, `torch` or `sounddevice` imports, a test file per module.
* Standard library only if possible (ANSI escape codes written by hand, as `narrate/color.py` does), so there is no new dependency **(my call)**.
* It must work with `--interval` fast mode, where segments are short, and with `--kernels` runs.

## Decisions from the builder check (all my calls)

* **Fast mode.** The TUI works in both loops. `run_live` already has a per-segment hand-over to the sink. `run_fast` gets a matching hook that builds the same `FlowState` from the readings it just used, and "the segment that is playing" there is the phrase that is playing. This is the largest task in the plan, and it gets its own task and its own tests. The Play tab's `InstrumentSession` is not reused.
* **What is shown follows what is heard, not what is rendered.** Its timing follows the first write on the clock, as `Session._pending` does, so the picture lags the chips by the same delay as the sound. In fast mode that is the delay the run already calculates (the player buffer plus the lead).
* **The TUI owns the synth pane's screen.** It uses the alternate screen buffer, so the terminal is restored on exit. All run-log lines appear in its log, colour-coded: segments, cables, evictions, errors, and also the `delay`, `midi` and backend lines. The plain log goes to the `.txt` file and the JSONL only. If the run ends with an error, the TUI leaves the alternate screen first and prints the error and the log directory, so the existing "Press Enter" message in `media/toplike.py` still shows it.
* **The log comes from structured events** (the kind, what and why that `RunLog` already has), coloured by `narrate/color.py`'s rules. No new explanation text. Lines appear when their segment is heard.
* **Evictions and fills.** The phase label is always shown. A faded chip and its devices use three stepped dim levels that follow the mix's own `PRESENCE` table. Without colour (`NO_COLOR` or not a terminal), the phase label and a marker character say it instead.
* **Size.** 80 columns by 24 rows is the target. Each chip is a box and its devices are one-line rows inside it, not boxes. Only the five most strongly pushing cables are listed. The log gets the rows that are left. More than four chips wrap into further rows of boxes, and the log shrinks. Below 60 columns by 16 rows the TUI shows one line saying the terminal is too small and the run carries on.
* **Keys.** `q` and Ctrl-C both end the run the way Ctrl-C does now (the WAV and logs are written out). A resize redraws.
* **Flags.** `--tui` needs `--live` (otherwise a parser error with a sentence). It works with `--mock-chips`, so a box with no `tt-smi` can run it. The MIDI flags follow whatever the loop being used already allows: they are refused with `--interval`, as now.

## Left to the builder

* The exact box drawing, the widths and the order inside the screen, and how it degrades below 80 columns.
* How it hooks the live loop for the playing segment (the sink's per-segment write that the Gradio session already uses is the likely route).
* How many log lines are kept and how they scroll.
* Where the code lives (`ui/` beside `chart.py`, or a new `tui/` package).
