# tensixinet

Tenstorrent chip telemetry played as a chorus of Ciat-Lonbarde-style devices. Devices are dealt
across the chips, and they trade solos. The README has the musical background and the signal-to-sound table.

Read this file first. It says what exists, how to check it, and what is left.

## Run it

```bash
python3 -m tensixinet --mock-chips 4 --duration 20      # no hardware needed
python3 -m tensixinet "your prompt" --duration 60       # real chips + the LLM endpoint
python3 -m pytest                                       # 226 tests, about 1 s, no hardware
demo/build.sh                                           # record the split-pane demo + soundtrack
```

`python3 -m tensixinet --help` lists every option. The run log goes to `logs/` as JSONL
(for programs, every reason kept) and a text file (what you see on screen, each reason once).

## Layout

| File | Job |
|---|---|
| `tensixinet/telemetry.py` | `tt-smi -s` JSON to `Telemetry` readings. Pure except `fetch_snapshot`. |
| `tensixinet/mapping.py` | The README table as functions (temperature to pitch, aiclk to BPM, power to volume, DDR to lanes). |
| `tensixinet/synth.py` | Drone voices for any chip count. |
| `tensixinet/devices.py` | Rollz, Sidrax, Cocoquantus, Plumbutter drone, a gong and an ultrasonic shimmer, plus the current-based mixer. `DEVICE_INPUTS` says which telemetry fields each device reads. |
| `tensixinet/lead.py` | The lead voice: a motif that is shifted or inverted by GDDR channel readings. |
| `tensixinet/turns.py` | Who solos each turn (weighted by chip power, never the same twice) and the gain curves. |
| `tensixinet/chorus.py` | Deals device slots to chips, plans the turns, scales by power, mixes by current. |
| `tensixinet/explain.py`, `runlog.py` | The what/why log. |
| `tensixinet/llm.py` | Prompts any OpenAI-compatible endpoint; the model name is read from the endpoint. |
| `tensixinet/mux.py` | Puts the WAV under a video with ffmpeg. |
| `tensixinet/mock.py` | Synthetic N-chip telemetry. |
| `demo/` | `tt-demo` manifest, left-pane script, `build.sh`. |
| `tests/` | One test file per module. `tests/fixtures/tt_smi_snapshot.json` is a real 4-chip capture. |

## Rules for working here

* **Test first.** Write the test, watch it fail for the right reason, then write the code.
  After it passes, break the code on purpose (swap a `min` for a `max`, drop a term) and
  confirm a test goes red. A mutant that survives means a test is missing or too weak. Also
  check the test was green before you mutate: a test that is already red "catches" everything.
* **Never `import ttnn`, `ttl`, `torch` or `sounddevice` in `tensixinet/` or `tests/`.**
  Importing `ttnn` opens the device. `tt-smi -s`, `tt-toplike --backend hybrid` and plain
  HTTP to the LLM endpoint only read, so they need no gozer lease. Anything that opens
  `/dev/tenstorrent/*` needs one (see the global CLAUDE.md).
* **The old top-level scripts are superseded and unsafe to run.** `tensixinet_full_loop.py`
  and `ciat_lonbarde_kernels.py` import `ttnn` and `ttl` at the top, so merely starting them
  opens the device with no lease. `tensixinet_multichip.py`, `phase1_sonification.py` and
  `tensixinet_tt_demo_maker.py` read device 0 only, hard-code 8 voices, and render with
  per-sample Python loops (minutes for 2 minutes of audio; the new code takes about 0.1 s
  for 10 s). They are kept for reference until the TT-Lang kernels are ported.
* The log must match the code. `DEVICE_INPUTS` lists the fields each device reads, and a test
  fails if that list and the audio disagree.
* Bump nothing by hand in the README table without changing `mapping.py` and its tests.

## Decisions

* **Chip count is never assumed.** Voices, devices, GDDR channels and log text all come from
  the telemetry. A 1-chip box plays only Rollz; a 32-chip box plays eight of each device.
  Open question: should a small system play more than one device per chip so it still sounds
  like a chorus? Not decided.
* **A GDDR channel's temperature is the hotter of its two dies.** tt-smi packs each
  `GDDR_x_y_TEMP` word as bottom and top die for two channels (`tt_smi/utils.py`,
  `decode_gddr_pair_temps`). The old code read the bytes as four independent sensors and kept
  8 of 16, so channels 4 to 7 never reached the synth. A test compares our decoder against
  tt-smi's own.
* **Phase is the running sum of frequency.** The old code used `sin(2*pi*f(t)*t)`, which
  smears pitch whenever the frequency changes.
* **Partials above Nyquist are dropped**, so high voices do not alias into the audible band.
* **The chorus applies power and current; the devices never read them.** Power sets each
  chip's volume. Current sets its share of the mix (Plumbutter 2: sources mix by current).
* **`voltage` has no musical use.** A test pins that; adding one is a deliberate change.
* **DDR status:** the code treats the low byte as four 2-bit fields and mutes a lane at 0.
  tt-smi's source says newer firmware puts BIST bits in `[31:16]`. The meaning of values 1 to
  3 is a guess and should be checked against firmware docs.
* The chorus is not normalised to a fixed peak, because that would erase the power-to-volume
  mapping. `tanh` in the mixer keeps it inside full scale.

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

## Open work, in rough order

1. **Demo pacing.** The whole log appears within about 3 s, then the left pane is still for
   the remaining 47 s. Print the chip lines gradually, in time with the music, or loop prompts.
2. **Longer load.** The model answers in about 3 s. Add `--max-tokens` to the CLI (it is fixed
   at 100 in `llm.ask`) or send several prompts so tt-toplike shows load for longer.
3. **Align audio to the LLM reply.** The soundtrack starts at video time 0, but telemetry is
   sampled after the reply.
4. **Live mode.** Today it samples telemetry once and renders. Sampling during playback would
   make the music follow the chips as they change.
5. **Port the TT-Lang kernels** in `ciat_lonbarde_kernels.py` once they can be tested on
   hardware (needs a gozer lease and a hardware-marked test; `pyproject.toml` already
   declares the `hardware` marker, and nothing uses it yet).
6. Move the superseded scripts to `archive/`.
7. This directory is not a git repository. Create one (default branch `main`) before more
   changes land. `audio/`, `logs/`, `demo/assets/` and `*.mp4` are large and should be ignored.
