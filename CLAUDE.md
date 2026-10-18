# tensixinet

Tenstorrent chip telemetry played as a chorus of Ciat-Lonbarde-style devices. Devices are dealt
across the chips and trade solos. Read this file first. It says how to work here. The other docs
say what exists and what is left.

| Read | For |
|---|---|
| `README.md` | What it is and how to run it |
| `docs/ARCHITECTURE.md` | Data flow and where each piece lives |
| `docs/SOUND.md` | Signal-to-sound table, the eight devices, turns, the lead, patches |
| `docs/CIRCUITS.md` | What the Ciat-Lonbarde guides say and which parts the code models |
| `docs/DECISIONS.md` | Why the code is the way it is |
| `docs/CONTRIBUTING.md` | How to add a device or change a mapping, test first |
| `docs/ROADMAP.md` | Open work, in order |
| `legacy/README.md` | The old prototype scripts (unsafe to run) |

## Run it

```bash
make test                                               # about 10 s, no hardware
python3 -m tensixinet --mock-chips 4 --duration 20      # synthetic chips, no hardware
python3 -m tensixinet "your prompt" --duration 60       # real chips + the LLM endpoint
make demo                                               # record the split-pane demo + soundtrack
```

`python3 -m tensixinet --help` lists every option. The run log goes to `logs/` as JSONL (for
programs, every reason kept) and a text file (what you see on screen, each reason once).

## Layout

```
tensixinet/
  telemetry/   snapshot.py (tt-smi JSON to Telemetry), mock.py (synthetic N chips, optional faults)
  music/       mapping, synth, devices, rolzer, sidrax, quantussy, mwtrs, lead, events, turns, patch, chorus
  narrate/     explain (what and why), runlog (JSONL plus the left-bar text view)
  media/       llm (OpenAI-compatible endpoint), mux (WAV under a video)
  run.py cli.py __main__.py     wiring and the command line
tests/         mirrors the package: tests/music/test_devices.py tests the music/devices.py module
demo/          tt-demo manifest, left-pane script, build.sh
docs/          architecture, sound design, decisions, contributing, roadmap
legacy/        first prototype scripts, kept for reference
```

## Rules for working here

* **Test first.** Write the test, watch it fail for the right reason, then write the code.
  After it passes, break the code on purpose (swap a `min` for a `max`, drop a term) and
  confirm a test goes red. A mutant that survives means a test is missing or too weak. Also
  check the test was green before you mutate: a test that is already red "catches" everything.
* **Never `import ttnn`, `ttl`, `ttml`, `torch` or `sounddevice` in `tensixinet/` or `tests/`.**
  Importing `ttnn` opens the device. `tests/test_hygiene.py` fails on any such import.
  `tt-smi -s`, `tt-toplike --backend hybrid` and plain HTTP to the LLM endpoint only read, so
  they need no gozer lease. Anything that opens `/dev/tenstorrent/*` needs one (see the global
  CLAUDE.md).
* **Every module has a test file in the matching `tests/` folder**, and every package folder has an
  `__init__.py`. The hygiene test checks both.
* **The log must match the code.** `DEVICE_INPUTS` lists the fields each device reads, and a test
  fails if that list and the audio disagree.
* **Change the mapping table only with `music/mapping.py` and its tests**, and update `docs/SOUND.md`.
* **Check that a scripted edit changed something.** A search-and-replace that matches nothing
  reports success.
* **Docs that name numbers** (test counts, ranges) are updated in the same change as the code.

## What happened (log)

* **2026-10-16:** Qwen Code wrote the first prototype: telemetry to sine voices, a TT-Lang
  kernel sketch, and a demo-maker stub. No tests.
* **2026-10-17, prompt:** "let's see if we can advance this project a little further, at least
  make it a little more test driven and less experimental, so that qwencode has a better
  chance of finishing it up later."
  * Follow-ups: replace anything hard-coded so anyone with N chips can play it; use telemetry
    across everything so it sounds like a chorus of Ciat-Lonbarde devices; record a
    tt-demo-maker video with tt-toplike on the right, prompting on the left, the music as the
    soundtrack; and make the project log what it does and why.
  * Result: the `tensixinet/` package above, 141 tests, a 50 s demo with soundtrack.
* **2026-10-17, after watching the demo:** "yes several devices per chip. aspects of the system
  should inspire change too ... that lead voice should shift when signals are interpreted like
  shift or invert ... kind of like cocoquantus does." Then: "more turn taking. some more
  gong/gaunge like sounds. a little more of that ultrasonic stuff", and "who is in the lead
  should shift based on the system conditions ... like jazz."
  * Result: seven devices dealt across any chip count, the `lead` device, the gong, the
    ultrasonic shimmer, power-weighted turn-taking, and log lines for each. 226 tests.
  * Mutation checks caught 13 of 16 mutants at first. The three survivors (voice normalisation,
    soloist gains never applied, and the gong's detuned pair) each got a test.
  * `np.convolve` made the turn fades take minutes for a 24 s render. A running-sum average gives
    the same numbers in milliseconds.
* **Mutation checking found three tests that passed against broken code:** `min` swapped for
  `max` in the Sidrax bounds; Cocoquantus ignoring aiclk; and a hard-coded model name coming
  back (the fixture served the same name, so the test could not fail). Three other weak tests
  were caught while writing or running them: an aliasing test that only checked for NaN, a
  scale test that only checked the output was finite, and a CLI width test that was already
  red when its mutant was run, so that "caught" result meant nothing until the test was fixed.
* **Rehearsal can be fooled.** `tt-demo rehearse` reported a "reaction" partly from chip 0,
  which swings between 20 W and 32 W with no prompt at all. A 20 s control sample showed
  chips 2 and 3 (the ones serving the model) stay at 37 to 63 W idle and reach 131 to 137 W
  under the prompt, so that reaction is real and the other is noise. Take a control before
  claiming a prompt moved a chip.
* **Chips 2 and 3 were held by another agent** and carry the
  model on port 8011. Everything here was read-only, so it did not interfere.
* **2026-10-17, repo made public prep:** git repo on `main`; the test fixture was scrubbed of the
  hostname, board and ASIC IDs and process command lines; `archive/` (editor backups and memory
  snapshots) stays ignored. Only `demo/assets/chorus.final.mp4` and its contact sheet are tracked
  from the demo output.
* **2026-10-17, "better structure and organize the whole repo":** the flat package became
  `telemetry/`, `music/`, `narrate/` and `media/`, with tests in matching folders. Prototype
  scripts moved to `legacy/`. Docs split into `docs/`. Added `pyproject.toml` packaging with a
  `tensixinet` command, a Makefile and a CI workflow. A new `tests/test_hygiene.py` reads every
  module and fails on `ttnn`/`ttl`/`ttml`/`torch`/`sounddevice` imports, a package folder without
  `__init__.py`, or a module with no test file in its matching folder. It caught that the RunLog
  tests were hiding inside `test_explain.py`.
  * A mutation run (adding `import ttnn` to a module, removing an `__init__.py`, removing a test
    file) turned the hygiene tests red each time. The first attempt at the import mutant used
    `sed '1a'`, which landed inside a docstring and proved nothing, so it was redone by appending.
  * An earlier edit that was meant to update the "Chip count" decision silently matched nothing,
    and the old text (a 1-chip box plays only Rollz) stayed for a while. Check that a scripted
    text replacement changed something.
* **2026-10-17, "I agree with the above" (replayable one-of-a-kind patches):** each run now draws a
  `Patch` (scale, lead motif, gong tuning and beat, shimmer band, turn length, floor). The system picks
  the scale (GDDR spread) and gong tuning (memory-to-ASIC ratio); a seed from all readings plus the clock
  picks the rest. The patch is saved as JSON, logged with reasons, and replayed with `--patch`. The demo
  replays `demo/chorus.patch.json`, so rebuilding it plays the same music.
  * The mutation battery had three survivors the first time: a seed that hashed only the first fields
    of the readings, the chorus not passing the patch down to the devices (masked by other patch fields
    changing the audio), and the run log not passing the patch to the turn log. Each got a test that
    changes only that one thing.
  * Drawing a patch from no chips crashed before the existing "no chips" message; the CLI now skips the
    draw in that case.
* **2026-10-17, "standard TT license ... make sure our README is amazing":** added the Apache-2.0
  `LICENSE` and `LICENSE_understanding.txt` (copied from tt-gozer) and an SPDX header
  (`Apache-2.0`, 2026 Tenstorrent USA, Inc.) on every `.py` and demo `.sh`. `tests/test_hygiene.py`
  fails on a missing header or license file (removing a header from `run.py` and from
  `left_pane.sh`, and removing `LICENSE`, each turned it red). README rewritten around the demo
  frame, the band, the real log, patches and credits. The README's log excerpt and the hero frame
  (`docs/media/demo-frame.png`) come from a real run and the final video.
* **2026-10-18, "there are still many more signals we could be using ... all of these please":**
  listed what `tt-smi -s` reports that we did not read, checked which fields really differ on the four
  live chips, and wired in all of them except rate of change (that needs live mode, roadmap item 3).
  * New `Telemetry` fields (neutral defaults): board power, AXI and ARC clocks, heartbeat, Tensix and
    ethernet masks, PCIe usage, correctable and uncorrectable memory errors, thermal trips, throttle
    temperature, TDP limit, and `busy` (processes using the chip, `None` when unknown).
  * Sound: voltage detunes the drone and PCIe usage sets its voices; ARC and AXI clocks scale Sidrax and
    Cocoquantus; ethernet links add four Rollz lanes; the heartbeat places the first gong strike; the two
    masks give each chip an identity (lead transpose, gong detune); pressure (closer of temperature to the
    throttle point and power to the TDP limit) brightens the lead and swells the shimmer; board power,
    pressure and processes raise a chip's turn weight; `music/events.py` adds ticks, breaths and sweeps
    for errors and trips; the host description joins the patch seed. `--mock-faults` injects errors.
  * Every new field is in `DEVICE_INPUTS` (or `EVENT_INPUTS`) and the one-field-at-a-time test covers
    them. The old "voltage is unused" test was inverted.
  * Mutation battery: 4 survivors the first time (per-chip event seeds, `tanh` bound on events, the
    mock's distinct-mask rule, the guarantee of at least one faulty mock chip) plus one device-mismatch
    in the explain text. Each got a test. One survivor (`(pid, device)` in the busy set) is equivalent
    because the list is already filtered by device.
  * `ETH_LIVE_STATUS` is read as one bit per port, as tt-toplike does. It changed from 12 links to 2 on one
    chip between two snapshots while another agent used the chips.
  * `demo/chorus.patch.json` still replays, but the devices now read more fields, so rebuilding the demo
    gives different audio from the checked-in video. The video was not rebuilt.
* **2026-10-18, "Did you reference the docs/PDFs on Ciat-Lonbarde instruments we placed in ~/Downloads?"** I had not.
  The three guides (Sidrax, Cocoquantus, Plumbutter & Rolzer, all by Josh Singer) were read in full, including the
  patch photographs, and the devices were rebuilt around what the circuits do. "Yes read them first, and then
  enhance all the current work on the more advanced understandings. These circuits we're imitating are novel."
  * `docs/CIRCUITS.md` lists each point in the guides, what the code does about it and what it leaves out.
  * New modules: `rolzer.py` (E6 barre tempos, steady even rolls, unstable odd rolls), `sidrax.py` (seven barres,
    frequency as bounce over bounds, the figure 1 bounds table, press and release, pressure as chaos),
    `quantussy.py` (five petals and castles, 8-bit buffer), `mwtrs.py` (shift register and snare, a new eighth device).
  * Changed: the gong strikes through a trigger-response blocking window over the chip's own clock, and the shimmer
    snaps and the drone ripples on the same strikes. Cocoquantus lost its random sample-and-hold. The chip identity now
    also sets the gong's window and the Quantussy's leader. Rollz lanes lost the 3, 4, 5, 7 per bar divisions.
  * The `DEVICE_INPUTS` test now renders 15 s, because strike times depend on the clock, heartbeat and identity and
    show up only at the second strike. The snare's base chip needs a harvested column to have anything to play.
  * Mutation checks found survivors again (speed caps, Nyquist clamp, the "3" bounds pair, leader wiring, tanh, the
    snare's low-pass and dust constants). Each got a test. The snare was first white noise with a spectral centre near
    11 kHz, so it got a low-pass.
  * I have measured the audio (levels, spectra, determinism) and have not listened to it.
