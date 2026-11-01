# tensixinet

Tenstorrent chip telemetry played as a chorus of Ciat-Lonbarde-style devices. Devices are dealt
across the chips and trade solos. Read this file first. It says how to work here. The other docs
say what exists and what is left.

| Read | For |
|---|---|
| `README.md` | What it is and how to run it |
| `docs/ARCHITECTURE.md` | Data flow and where each piece lives |
| `docs/SOUND.md` | Signal-to-sound table, the ten devices, turns, the lead, patches |
| `docs/CIRCUITS.md` | What the Ciat-Lonbarde guides say and which parts the code models |
| `docs/DECISIONS.md` | Why the code is the way it is |
| `docs/CONTRIBUTING.md` | How to add a device or change a mapping, test first |
| `docs/ROADMAP.md` | Open work, in order |
| `kernels/README.md` | The devices as TT-Lang kernels: how to run them in the simulator and on a chip |
| `site/index.html` | The project site (GitHub Pages, deployed from `site/` by `.github/workflows/pages.yml`): routing diagrams, cable explorer, audio |
| `legacy/README.md` | The old prototype scripts (unsafe to run) |
| `PLAN-PLAY-TAB.md` | **Open plan for qwencode:** a "Play" tab in the Gradio app to play a device, a module or the whole system live. Execute it task by task |

## Run it

```bash
./tensixinet.py --live                                  # the one command: builds .venv on first use, plays until Ctrl-C (also --app, --doctor, --setup)
make test                                               # about 90 s, no hardware
python3 -m tensixinet --mock-chips 4 --duration 20      # synthetic chips, no hardware
python3 -m tensixinet "your prompt" --duration 60       # real chips + the LLM endpoint
python3 -m tensixinet --live --duration 30              # read the chips every turn and play as it goes
python3 -m tensixinet --live --tui --mock-chips 4      # the patch layout and the log, full screen (q or Ctrl-C ends it)
make demo                                               # record the split-pane demo + soundtrack
```

`python3 -m tensixinet --help` lists every option. The run log goes to `logs/` as JSONL (for
programs, every reason kept) and a text file (what you see on screen, each reason once).

## Layout

```
tensixinet/
  telemetry/   snapshot.py (tt-smi JSON to Telemetry), mock.py (synthetic N chips, optional faults), hwmon.py (the chips' hwmon files), fast.py (FastSource: hwmon fast fields plus tt-smi slow fields)
  music/       mapping, synth, devices, ttrolzer, ttrax, ttquantussy, ttsteam, lead, events, loudness, turns, patch, chorus, live
  narrate/     explain (what and why), runlog (JSONL plus the left-bar text view)
  media/       llm (OpenAI-compatible endpoint), mux (WAV under a video), player (live audio out), toplike (tmux session for --with-tt-toplike)
  ui/          flow (what is playing as data), chart (it as animated SVG), logview, session (the live loop on a thread), progress (the steps of a start), player (which session the page shows), forms, knobs / scopes / instrument / readout (the Play tab)
  tui/         layout (FlowState to rows), frame (styled cells, changed-cell ANSI), logpane (the log in the same screen), playhead (shows each segment when it is heard), app (terminal, keys, exit lines); `--tui`; standard library only
  score/       events (Note, Trigger, Control, Score), collect (the devices' events, from the same code as the audio), routing, midi, smf (the file writer), sinks (file and live port), vcv (the VCV Rack voice patch in `vcv/`)
  run.py live.py fast.py cli.py __main__.py     wiring (offline, live, and fast live for --interval) and the command line
tensixinet.py  the launcher: standard library only; finds or builds .venv, falls back to synthetic chips, --live/--app/--doctor/--setup
tests/         mirrors the package: tests/music/test_devices.py tests the music/devices.py module
app/           the Gradio page that draws what is playing (needs .venv-app; see README), with .disco/app.yaml for tt-discolike
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
* **2026-10-18, "build the loudness when you're done":** `music/loudness.py` gives one gain over the finished render
  (-18 dBFS RMS, peak under -1 dBFS, at most +20 dB). `run` applies it and logs it, and `--no-normalise` skips it.
  `render_chorus` still keeps power as volume, so the gain is applied in `run`.
  * The mutation check caught a test that compared the gain cap to the constant it was testing. It now checks 10.0.
* **2026-10-18, demo rebuilt** with the new devices and the loudness gain (`demo/build.sh`, read-only, one prompt to the
  model on port 8011). The soundtrack measures -18.0 dB mean and -3.7 dB max. `demo/chorus.patch.json` is unchanged.
  The README hero frame (`docs/media/demo-frame.png`) is a frame from the new video.
* **2026-10-18, "is this running in real time now or still just at generation time?"** It was generation time. "start on
  live mode" added `--live`: `live.py` reads the chips every turn, renders one turn from that reading and plays it while
  the next is made (a second thread writes to a `paplay`, `aplay` or `ffplay` pipe, or to nothing at the speed of the music).
  * New: `music/live.py` (`advance`, `Tracker` for rates over 2 s, `Stitcher` for 0.4 s equal-power crossfades),
    `media/player.py` (sinks, `make_sink`), `LiveGain` in `loudness.py`, `pick_soloist` in `turns.py`, `turn_seconds` and
    `slot_weights` plus `segment` and `soloists` arguments in `chorus.py`, `Telemetry.temp_rate` and `power_rate`,
    `mapping.chip_chaos` (pressure plus how fast the readings move, read only by Sidrax), `--live` and `--player`.
  * Measured only; nothing live has been heard. 20 s on four mock chips took 20.6 s of wall clock for 10 turns, and 30 s on the four real
    chips took 31.2 s for 7 turns (read-only, no prompt sent: tt-orchard had been told the chips and model were free).
    `paplay`, `aplay` and `ffplay` each accepted a second of silence.
  * The first live run on the real chips showed -1.1 C/s on chips that were holding still: the loop renders its first two turns
    back to back, so the second reading came 0.3 s after the first. Rates are now measured over at least 2 s.
  * Mutation checks found survivors again: the lead-in limit, the early stop after a player failure, the tail's gain,
    who solos after whom (the busy-chip test passed with the "not twice running" rule removed), the exact player
    arguments, the player order, the live gain's immediate peak limit, a one-voice division by zero, and the Tracker's
    reset on losing a chip. Each got a test. An early battery was also unreliable: a restored file kept its size and
    modification second, so Python loaded a stale `.pyc` of the mutant. The mutation scripts now set
    `PYTHONDONTWRITEBYTECODE`.
  * Known limit: each turn restarts the devices' clocks, so a ringing gong fades at the end of its turn. A start-time
    offset in every device would fix it (roadmap item 3).
* **2026-10-18, "let's tt-ize some of our component names that are rips from CL":** the devices and modules now carry a `tt`
  prefix, with credit to Ciat-Lonbarde and Josh Singer's guides and no claim to be a recreation. rollz -> `ttrollz`,
  rolzer -> `ttrolzer`, sidrax -> `ttsidrax`, cocoquantus -> `ttquantus`, quantussy -> `ttquantussy`, plumbutter_drone ->
  `ttdrone`, mwtrs -> `ttsteam`, deerhorn -> `tthorn` (not built). `gong`, `lead` and `ultrasonic_shimmer` are ordinary words
  and keep their names. The old names stay where they describe the originals: the guides' titles, `docs/CIRCUITS.md`'s
  left-hand column and its name table, "after the Gongue" style credit lines, and the history in this log. `legacy/` is untouched.
  * Log lines, `DEVICE_INPUTS` keys and the `device` field in the live log use the new names, so logs from before this change
    name the old devices. Saved patches do not name devices and still replay.
  * Done with a scripted rename and a read through every line that mentions a guide. A scripted rename of "guide" lines
    wrongly renamed the instruments in several docstrings and in `docs/DECISIONS.md`, and those were put back by hand.
* **2026-10-18, "we want this to sound good ... apply some EQ to the final mixed result":** `music/eq.py` and `--no-eq`.
  The curve: cut below 35 Hz, lift 70 to 250 Hz by up to 4 dB, flat midrange, fall from 8 kHz to -33 dB at 20 kHz.
  Offline `run` and live mode both apply it before the loudness gain. Measured: the realised response is within 0.1 dB of
  the curve; a 15 kHz tone falls 17 dB and a 100 Hz tone rises 4 dB. Not listened to.
  * The mutation check caught a test that read its expected curve from the module it was testing (change a point and the
    expectation moved with it). The curve is now written out as literals in the test.
  * Still to do from the same message: a launcher that runs the live synth beside tt-toplike (tmux), and a tighter live turn
    so the sound lags the chips by about a second.
* **2026-10-18, "I still love the idea of running the instruments as kernels ... take turns with the tt-orchard agent":**
  `kernels/ttshock.py` is the first one, with `tensixinet/kernels/packing.py` (host side) and `tests/test_kernels_sim.py`.
  * Found: `gozer status` showed all four chips held by `orchard:supervisor`. tt-orchard-1d agreed to the plan: simulator first,
    no queueing ahead of its 4-chip run, one board at a time later, avoid the board serving port 8011, and it will send a line
    when the boards are free. It also plans its own one-chip kernel smoke test, queued through gozer like ours.
  * The TT-Lang simulator is `tt-lang-sim` (Tenstorrent's package, Apache-2.0). The copy in the TT venv could reach the real
    `ttnn`, so it runs from a fresh venv with CPU torch and no `ttnn` (`kernels/README.md`). The kernel matched the numpy device
    to 3e-7 on the first run. Six deliberate breakages (sin to cos, a dropped mask, a wrong input, a loop one short and others)
    each failed. Two of them crashed the simulator's own error printer, which looks for a file that is not there.
  * Not done: any run on a chip. The right tolerance on hardware is unknown.
* **2026-10-18, "how many kernels can we run per chip ... could all 4 chips self-play a giant self-reactive instrument? ... keep
  on building these in sim from first":** answered in the session (about 120 Tensix cores per chip from the 12 enabled columns,
  one op with different core groups playing different instruments rather than many separate ops; the loop only closes if the
  instrument spends load on purpose, with a governor), then added `ttshimmer` and `ttmix`.
  * Both match their numpy devices in the simulator (1.5e-7 and 6e-8). Seven deliberate breakages of the two kernels and 14 of the
    host helpers each failed, except one: a negative current clamp had no test, and now does.
  * Two simulator facts cost time: block maths refuses Python numbers (`blk * 0.5`), so constants are packed tiles, and `ttl.copy`
    in a list is refused. The real error messages were hidden by a `tt-lang-sim` 1.1.6 bug (it loads a `diagnostics.py` that the
    wheel lacks); `kernels/README.md` has the workaround.
  * `shimmer_parameters` was pulled out of `ultrasonic_shimmer` so the kernel's host side and the device share one set of numbers.
  * Idea not built yet: a software model of a chip (load sets power, power sets temperature through a lag) to test the feedback loop
    and its governor in the simulator before it touches hardware.
* **2026-10-18, "keep building the kernels in the simulator":** `ttgong` (and `gong_parameters` pulled out of `gong`). The strike's
  pitch sag is summed sample by sample in the device, so the host uses the closed form of that sum. A strike rings for up to
  2.5 intervals and strikes are at least one interval apart, so up to three ring at once: the kernel runs three times and the host
  adds them and applies `0.6 * tanh`. It matches the device to 2.2e-7 for all three gong tunings.
  * A first draft expanded `sin a + sin(a + b)` into a form that needed the constant 1, which a block cannot supply. The plain
    form is exact and needs no cosines. The kernel check first used the four-partial bell, so a breakage in the fifth partial slot
    survived; it now uses the five-partial tam-tam.
  * Questions answered in the session: whether we have the AVDogs (the shock wave is the core of one, the rest is not modelled) and
    what a voice engine would take (formant synthesis; see the session).
* **2026-10-18, "avdogs: yes. those purples are important for all these too":** `music/ttavdog.py` is a ninth device (on the end
  of `DEVICES`, so slot numbers did not move). The guide's red out is the second trace (purple in the figures): it answers the
  white voice on the other half of the undulation, same pitch, opposite slope, a little quieter. "Purples" may also mean the purple
  jacks, which the guide says modulate the innards of a module (FM, crunch); those are not modelled. They would be cross-modulation
  between devices, and are open.
  * Mutation checks found five survivors in the tests: the level and red-gain constants compared to themselves, the decay constant,
    a shape test that changed the pitch along with the pressure, and nothing checking that the red voice uses the inverse slope.
  * Adding a device changed three tests that assumed eight slots (soloist test, the pressure-mentions list, and the
    board-power and busy fields, which reach the audio through who solos, so a one-second render may not show them).
* **2026-10-18, `ttavdog` kernel:** matches the device to 1.2e-7 in the simulator; nine kernel breakages and eight host-helper
  breakages failed. One host survivor: nothing exercised a full-pressure chip, where the wave shape is clipped to 0.98 so the
  saw's slope stays finite (without the clip the host divides by zero). The test now includes a chip at full pressure.
  * The full-pressure test also found a latent bug: `pressure()` divided by zero when the throttle limit equalled the 40 C floor.
    `_unit` now treats a degenerate range as a step.
* **2026-10-18, "yes voice choir is needed too. it'll add some pseudo humanity":** `music/ttchoir.py`, a tenth device (on the end of
  `DEVICES`). Additive formant synthesis: harmonics of the chip's pitch (snapped to the patch's scale) shaped by five formant
  resonances, sung on oo/oh/ah/eh/ee that the GDDR channels pick, voice type from the chip's identity, pressure flattening the
  glottal slope, vibrato, a breath dip between syllables. A 20 s voice renders in 0.55 s.
  * Measured only: the harmonics pile up at each vowel's formants (oo has 0.6% of its amplitude above 1500 Hz, ah 3%, ee 9%, eh 16%), the
    fundamental is at the chosen pitch, nothing is above 0.46 of the sample rate. Not heard.
  * Mutation checks found survivors in the pinned numbers (bandwidths, level, vibrato depth and rate), the syllable length and the
    identity wiring (voice type, first channel); each got a test. Frame rate, glide time and drift spacing are tuning constants and
    survive. A short render (0.05 s) crashed `np.interp` because `np.convolve(mode="same")` returns the longer input; found by the
    full suite after registering the device, fixed with a test first.
  * Two tests I wrote first were wrong: one tried to hold a vowel steady through a device that moves on every bar, and one asserted
    on a sine built inside the test. Both were rewritten.
* **2026-10-18, "purple cross-modulation jacks is what I meant ... even our choir should have the equivalent":** the owner of the real
  instruments said the choir and AVDog sound right and that the purples are the cross-modulation jacks. `music/cables.py`: five to eight
  cables drawn from the patch seed (`fm`, `crunch`, `inflate`, `fold`), the choir always one out and one in, pushed by the chaos of the
  chip that plays the source, applied in `render_chorus` before the turn gains. `--no-cables`; one log line per cable.
  * Mutation checks: the first battery left five survivors (the sizes of the FM delay, crunch step, inflate gain, knee and envelope
    window); a test now pins each to a number. The kind offset survives because every kind is still used either way.
  * Three tests had to change: the one that rebuilds the chorus from raw devices now renders unpatched, and the live tests' stand-in
    for `render_chorus` accepts `cables`.
  * Side research for the owner (is a synth driven by a host's own signals new?) is summarised in the session, not in the repo.
* **2026-10-18, "ttsidrax => ttrax":** the device and its module are now `ttrax` (`music/ttrax.py`, `tests/music/test_ttrax.py`, the
  `DEVICE_INPUTS` key and the log lines). The earlier entries in this log keep `ttsidrax`, the name in use when they were written.
  The guide's own name, Sidrax, is unchanged where the docs describe the original.
* **2026-10-18, "create a gh_pages site ... as a site directory ... spotlight the unique instrument" and "show diagrams on how routing works, animate signal paths":**
  `site/index.html` (one page, no build step), `site/assets/` (hero frame, six 30 s MP3s from `--mock-chips 4 --patch-seed 5`, one pair with and
  without `--no-cables`), and `.github/workflows/pages.yml`, which deploys the `site/` folder (not a branch). Sections: the idea, an animated routing
  diagram with five scenarios (quiet box, prompt heats chips 2 and 3, a chip near throttle, a memory error, live mode), a cable explorer drawn from six
  real seeds with a chaos slider per chip, the band, listening, prior art, kernels, credits.
  * `tests/test_site.py` checks that the device list and every drawn patch on the page equal what the code produces, that every local asset exists, and
    that the credit line is present. Changing a depth in the page turned it red (the first attempt matched nothing and proved nothing).
  * Side research (is a synth driven by a host's own signals new?): we found network sonification (Peep, SoNSTAR), early computer loudspeakers and radio
    interference, fan organs and EM-field synths, and no instrument in the Ciat-Lonbarde patch tradition driven by a processor's power, memory and link
    readings. The page says a search is not proof. Not checked: Patch Point's newest releases and Ciat-Lonbarde's current catalogue beyond what the search returned.
* **2026-10-18, "go ahead and cycle through the kernels on the chips and prove things out. record sounds and videos ... if you detect
  success, keep going":** all five kernels ran on all four chips (two `gozer` leases, one per board) and matched their numpy devices.
  `kernels/cycle.py` runs every kernel on every leased chip and writes one JSON line and (with `--save`) the chip's own audio next to
  the numpy reference; `kernels/make_video.py` turns a chip's saved audio into a video (waveform, the numpy reference, a spectrogram,
  the chip's own audio as the soundtrack). Raw results are in `docs/hardware/`; the video and two chip-audio clips are on the site.
  * Two kernels needed changes the simulator had not caught. `ttgong` added two tiles that also fed `sin`, which the chip's compiler
    refuses (one buffer cannot serve both the vector and matrix units); the host now gives each partial's detuned copy its own phase
    tile instead of adding the beat on the chip. `ttgong`'s five-partial sum and `ttavdog`'s one long expression both overflowed the
    chip's four working registers ("insufficient DST registers"); both are now built in stages, each stage's result passed on through
    its own buffer.
  * The chip's `sin` and `exp` matched numpy to 1e-7, same as the simulator. Its multiplies did not: `ttmix`'s `tanh` was off by 2.8e-4
    and `ttavdog`'s chain of multiplies by 2.3e-3, both far below hearing; `--atol` on those two kernels was widened to match, with a
    comment saying why.
  * `gozer run --chips N` worked; `--exact` took a PCI BDF, not a range, and `--chips 0-1` was rejected (`gozer: bad --chips range`).
    A lease release was refused once ("device still open") and left the lease STALE; `gozer reconcile` cleared it, after checking with
    `fuser` that nothing of mine still had the device open.
  * Said yes to running load kernels seldom and for effect (roadmap 7b); none built yet, pending a governor design.
* **2026-10-18, "we should have kernels that are all about STARVATION and DESTRUCTION too ... called into play seldom but with big
  impacts":** logged as roadmap 7b, not yet built. Asked, and was told: a hard wall-clock cap and a live pressure ceiling that aborts
  a burst early (not a cooldown, not a separate tt-orchard sign-off beyond the ordinary one-chip turn); musically, no new device, just
  the burst's real heat and power feeding the existing pressure, chaos and turn-weight signals harder. Built `tensixinet/kernels/governor.py`:
  `MAX_BURST_S = 2.0` (the chosen 1-3 s), `ABORT_PRESSURE = 0.85` read against `mapping.chip_pressure`, `may_start` (a chip already
  near the ceiling may not begin) and `should_stop` (the cap or the ceiling, whichever comes first). Pure host-side logic, no device,
  tested without hardware; a mutation battery found three survivors (both constants unpinned, and `>` for `>=` at the pressure
  boundary) and each got a test. The load kernel itself (the part that actually burns cycles on a chip) is not built.
* **2026-10-18, "make the site behave better on mobile ... the interactive UIs are great explainers ... triple down ... we want people to
  be excited enough that they'd want the hardware":** a third flagship interactive piece, `site/`'s mapping sandbox: four sliders
  (ASIC temperature, power, aiclk, GDDR spread) run the exact ported formulas from `mapping.py` and `patch.py` (`temp_to_pitch_hz`,
  `aiclk_to_bpm`, `pressure`, `scale_for`) and show pitch (with a note name), tempo, a pressure meter and the chosen scale, live. A
  new `#hardware` section makes the case directly: the project only works because real silicon gives up this many live signals for
  free, with links to tenstorrent.com, the Tenstorrent GitHub org and the repo. Mobile: a `@media(max-width:640px)` pass (smaller
  section padding and type, full-width stacked buttons, narrower slider labels) on top of layouts that already had no page-level
  horizontal overflow at 375px (checked with an iframe harness reading `scrollWidth`, not by eye).
  * `tests/test_site.py` now also checks that the sandbox's JS constants equal the Python ones it claims to port (`C3_HZ`,
    `PITCH_SEMITONES`, the temperature/aiclk/bpm ranges, the pressure floors, the scale step and the scale name order) and that the
    page links to tenstorrent.com and github.com/tenstorrent. Changing a constant or the scale order in the page, by hand, turned it red.
* **2026-10-18, "let's also make ttrax configurable in bars ... like the tetrax, sidrax, and quatrax configurations":**
  looked up the real instruments: Sidrax is 7 bars (the guide this module was built from), Tetrax is 4, Quatrax is
  14 (44, 28 and 100 banana jacks respectively). `patch.rax_bars` (4, 7 or 14) picks ttrax's size, drawn uniformly
  from the seed; an old saved patch with no `rax_bars` field defaults to 7, so it still replays as it always did.
  * `glitch_bounds(n_bars, barre)` (next barre, one after) replaces the hardcoded `BOUNDS_TABLE["3"]` lookup in the
    render path; it reproduces that table exactly for seven bars (tested), and for 4 or 14 it is this code's own
    extension of the pattern, since there is no Tetrax or Quatrax guide here to transcribe from. `BOUNDS_TABLE`
    itself is untouched and still documents the guide's figure 1.
  * `ttrolzer.barre_bpm` wraps past its seven-entry tempo table (`barre % len(BARRE_BPM)`) rather than indexing off
    the end, so Quatrax's extra barres play at one of the same seven tempos instead of crashing.
  * Mutation checks: a battery of four deliberate breaks (`glitch_bounds` collapsed to one barre, the bar-count
    tuple changed, the old-patch default changed, the tempo wrap off by one) were each caught by a test.
* **2026-10-18, "let's get the rest of these machines built in tt-lang kernels. look up the delay chips that power the
  cocoquantus 2. let's do our own tensix version of that that really embraces our unique hardware":** researched the
  real chips: a pair of crunchy 8-bit echo ICs, each a buffer of a few tens of kilobits clocked by a VCO, whose length
  trades directly against its resolution (a shorter delay holds more bits). `music/ttcoco.py` is the eleventh device:
  eight parallel delay taps, one per GDDR channel, each independently timed and crunched, echoing `devices.quantussy_tone`
  (factored out of `ttquantus`, a pure refactor, tests unchanged). The real instrument affords two delay lines because
  each is a separate chip; here a tap costs a core's worth of work, so there are eight, and the docs and site say so
  plainly as the concrete answer to "embrace our unique hardware" rather than a slogan.
  * Each tap: a truncated geometric series of delayed, decaying, independently re-crunched copies (`echo_tap`), the
    same "closed-form sum of repeats" shape as `ttgong`'s ring-down, chosen because it is what a kernel can port later.
    Feedback is pressure, scaled to a ceiling of 0.8.
  * Registering an eleventh device touched the same places the ninth and tenth did: `DEVICE_INPUTS`, `explain.py`
    (plus a new pressure-mentions index), the site's embedded device list, cable seeds and band grid, and a handful of
    docs that named "ten" devices. One test needed loosening, not fixing: `--no-normalise`'s raw-vs-normalised ratio
    assumed the old mix's exact raw loudness, which moved when ttcoco joined it.
  * Found and fixed a gap from the ttrax work two entries back: `explain_chip` never saw the patch, so the log always
    said "seven ttrax barres" even when the patch's `rax_bars` was 4 or 14. `explain_chip`/`describe_readings` now
    thread `patch` through (an optional, defaulted parameter, so every existing call site is unaffected).
  * Mutation checks: four survivors, all unpinned constants (`DELAY_RANGE_MS`, `BITS_RANGE`, `FEEDBACK_MAX`,
    `ECHO_FLOOR`); each got a test. Everything else (the delay-vs-bits trade direction, the feedback-pressure
    wiring, the shift helper's padding, the final headroom term) was already caught.
  * Not yet built: the TT-Lang kernel for this device (next), and splitting the eight taps across real Tensix cores
    instead of precomputing them on the host, which would be the first kernel here to use more than one core.
* **2026-10-18, `ttcoco` kernel:** `kernels/ttcoco.py` ports the device's final mix, `tanh(a0 + ... + a7)`, the same shape as
  `ttmix` with eight inputs instead of four; the eight taps themselves (`echo_tap`, sequential and data-dependent) stay on the
  host, the same way the gong's ring-down and the AVDog's undulation do. `tensixinet/kernels/packing.py` gained `coco_inputs`/
  `coco_on_host`, tested against `ttcoco()` directly (not just against the kernel) so the host/chip split is proven correct on
  its own. Matched the simulator to 1.2e-7 on the first run; four deliberate breakages (a subtraction for an addition, a
  dropped term, a missing `tanh`, a swapped input tile) each failed. Ran on all four chips (two leases, one per board) and
  matched to 8.8e-4, inside a 1e-3 tolerance (the sum has seven adds before `tanh`, one more than `ttmix`'s three, so a little
  looser than `ttmix`'s own 2.8e-4). Recorded the same way as the first five kernels: `kernels/make_video.py` now covers all
  six, the site's kernel video and its "On the chip" section are rebuilt, and a `ttcoco`-on-chip audio clip is on the site.
  Splitting the eight taps across real cores, instead of precomputing them on the host, is still open (roadmap 7).
* **2026-10-18, `--only`:** since none of the new sound has been heard, `--only NAME[,NAME]` plays only the named devices at full level with
  no turn-taking (`render_chorus(..., only=[...])`, also in live mode). A log line says what is soloed. Mutation checks found two
  survivors in the command-line parsing (spaces around commas, a list of nothing); both got tests.
* **2026-10-18, "one fresh tt-demo-maker tt-toplike video featuring EVERY DANG FEATURE AT ONCE ... 2 minute video so we can do
  --rotate mode ... integrate the video into the site, and then release to tt-orchard":** a new `features` scene in
  `demo/demos.yaml`: the left pane (`demo/left_pane_features.sh`) runs `tensixinet --live`, reading all eleven devices off
  the four real chips every turn (read-only, no gozer lease); the right pane runs `tt-toplike --rotate`, which cycles
  unattended through every one of its views (table, starfield, memory castle, memory flow, arcade, HivemindSweeper, training
  monitor). 120 s recorded, muxed with the live run's own WAV as the soundtrack (`demo/assets/features.final.mp4`), and
  embedded directly in the site's hero section in place of the old static frame, with a new poster image
  (`site/assets/features.mp4`, `site/assets/features-frame.png`). README gained a line pointing at it alongside the
  original 50 s chorus demo; neither replaces the other.
  * Checked with tt-orchard-1d before recording: no model of theirs was being served to prompt, gozer showed all four chips
    free, and reading them was fine as long as nothing of mine reset a board mid-recording. An unrelated service was listening
    on port 8001 but nobody present could vouch for what it served or whether it touched a chip, so the recording uses idle
    real telemetry rather than prompting it.
  * `tt-demo render features --mp4` took longer than real time to encode (2-minute capture at roughly 0.7 to 0.8x realtime with
    x11grab plus libx264) and outran the foreground command timeout; it was restarted detached and watched for completion
    instead of polled.
  * `.gitignore` gained `!demo/assets/features.final.mp4` / `!demo/assets/features.sheet.png`, matching the chorus demo's
    allow-list entries. `site/assets/` is not under the restrictive `demo/assets/*` rule, so the video and poster there needed
    no gitignore change, only `git add -n` to confirm nothing else came along with them.
* **2026-10-19, "diversify our patch library a bit so there's a lot more variety on possible in every new run":** `music/patch.py`
  now has 19 scales (was 6) and 8 gong tunings (was 3), appended after the originals so the system reaches the first six in the same
  places. The seed also draws a `mode` (the system's scale started from another of its notes, so a quiet box that always lands on one
  scale still gets different shapes), a 1.5% stretch of the gong's partials and a 15% nudge to their levels, a motif of 6 to 12 notes
  in a drawn range, and wider ranges for beat, shimmer band, turn length and floor. A saved patch with no `mode` replays as listed.
  * First mutation run left four survivors, all range bounds: the tests checked values stayed inside the range and never that the draw
    reached past the old one. One tuning (`singing_bowl`) put a partial 0.02 from a whole number at a 3% stretch, so the stretch is 1.5%.
* **2026-10-19, "after running it for awhile, i'm hearing the sound stop and start a bit":** live mode ran 6 minutes, and eight reads of
  `tt-smi -s` in a row (270 to 309 s) hung to their 5 s timeout, each longer than the 4.3 s of music it fed. Rendering (0.26 s) and a
  healthy read (0.23 s) were fine. `SNAPSHOT_TIMEOUT_S` is now 1.5 s. I do not know why `tt-smi` hung; another agent's board reset is a guess.
* **2026-10-19, "the man with the red steam shift register can be used for more than just the noise/snare ... and the noise/snare can be
  used for more than just the sound" (then "used with rollz and gongs and avdogs a pretty good snappy rhythm section can be made"):**
  the register moved to `music/register.py` and four patch-drawn steam routes (`strikes`, `avdog`, `accent`, `pitch`) patch it into the
  gong's clock, the AVDog, ttrollz and the AVDog's pitch. Off in the default patch, so the earlier sound does not move.
  * Needed a chip config where the routes bite: a mask that fires only on odd steps never lines up with the ttrollz clock (its long and
    short steps put pulses on every fourth step and half-way between steps), and most masks give a hit-only gong strike rarely or never.
  * Mutation run: 5 survivors the first time (the hit just before a pulse, the avdog route replacing the strikes instead of adding to
    them, the avdog not following the shared clock, the end-of-render boundary and one equivalent clamp). The four real ones each got a test.
  * Measured only: the routes render finite audio and the log names them. A rhythm-section-only clip is at
    `--only ttrollz,gong,ttavdog,ttsteam`. Not listened to. The kernels do not follow routes.
* **2026-10-19, "are the rollz browns behaving as sandrodes ... are our rollz giving us the kind of odd time signatures associated with the circuit, with a little wander?" then "yeah" to a rebuild:**
  read the guide's text for Sandrodes, Brown Theory, Evens and Geometry (`pdftotext` on the Plumbutter guide works; the first attempt read only the cover
  because it was run on page 1's output) and measured `ttrollz`: lanes were independent, the repeating step pattern was fixed, the wander was
  independent jitter, and the 4-roll and 6-roll had identical step patterns. `roll_ensemble` in `music/ttrolzer.py` now runs the lanes as coupled
  circuits (persistent capacitors, drift that grows with temperature, an odd roll's pattern sliding round its cycle, an accented cycle start, a
  distinct hexagon, and Brown links that pull a target's next step earlier). `ttrollz` uses it; `roll_pulse_times` stays for ttrax.
  * My first sync test had the direction backwards. The guide says a faster pulse syncs an even roll, and the numbers agree (a faster source locks
    the target to 0.08 of its free offset; a slower one barely moves it). The hexagon's triplet feel was swamped by its own swing: I had squared the
    swing amplitude wrongly, so it needed less swing and a stronger triplet.
  * Several first tests passed only on one lucky draw of the capacitors (sync strength, hexagon feel). They now look at 15 seeds and assert a typical
    case, not every case. One test measured step spread, which the square's fixed swing dominates; it now measures the length of a whole cycle.
  * Mutation run: 35 mutants, 7 survivors. Five got tests (the minimum-gap floor, which needs three rolls pulling each other hard; link strengths
    differing; distinct pairs over many seeds; the Ethernet blips' decay). Two are equivalent: the clamp on a step's length (no shape and capacitor
    reach it) and `remaining > 0` against `>= 0` (only differs when two rolls are due at exactly the same instant).
  * Still plain: the chip's own clock that the gong waits on. Measured only; not listened to.
* **2026-10-19, "commit and push this and update the site too":** the site has a new `#rhythm` section (the coupled rolls, the four steam routes and
  the bigger patch library) and a nav link. `tests/test_site.py` checks that every route name, the library sizes (19 scales, 8 gong tunings) and the
  roll model's ranges on the page equal the code's. Not in this commit: the idea of linking the gongs to the rolls and growing the roll network toward
  the Rolzer's 35, which is designed (the gong would take its trigger from one of the chip's rolls, and the gong and `ttrollz` would share one seed so the
  linked pulses are the ones heard) and not started.
* **2026-10-19, "gongs should probably link to the rollz too. it's ok if we have a lot more rolls. like the dedicated rollzer module" (then "yes"):**
  `ttrollz_network` gives each lane that is up a barre of the Rolzer's rolls (3, 4, 5, 6 and 6 sides, the second hexagon half a step out of
  phase): up to 40 rolls per chip, only each lane's own roll heard. `roll_pulses` runs them (cached, one call per render shared by every
  reader). A chip's gong is patched to one of its rolls three times in four (`draw_gong_trigger`, from the chip's identity and the patch) and
  otherwise to its clock; the shimmer's snap, the drone's ripple and the AVDog follow the same strikes. Every reader takes its segment from its
  seed (`seed // SEGMENT_SEED_STRIDE`, now in `ttrolzer.py`), so in live mode the gong and `ttrollz` use the same drift and the linked pulses are
  the ones heard.
  * First version put the links among all 40 rolls and 14 of them never reached a heard roll, so pressure stopped changing what `ttrollz` played
    on a chip. `draw_links` now takes the heard rolls and sends at least half of its links to one. The links also grow with the network (2 + n/10 to
    4 + n/5).
  * The roll-linked gong makes the gong, shimmer, drone and AVDog read the DDR status and pressure, so `DEVICE_INPUTS` gained those and the inputs
    test uses a base chip (`0x3002`) whose gong is on a roll and where every listed field moves its strikes. One drone test compared the tallest
    spectral peak, which jumps between the chord's voices when the ripple changes; it now compares the centre of the spectrum.
  * Mutation run: 31 mutants, 5 survivors (the heard-index filter and the segment in the shimmer, drone and AVDog). Each got a test.
  * Measured only: a 60 s render takes 4.4 s and a live segment 0.4 s. Not listened to.
* **2026-10-19/20, "restart the soundtrack but with TT hardware kernels instead" (then "show me ... all 4 chips ... with kernels on all of them", and a
  gozer lease of an hour or two is trusted):** `--kernels {chip,chips,host}`. A chip worker pool (`kernels/pool.py`, `kernels/worker.py`,
  `tensixinet/media/chip.py`) and a `KernelBackend` (`tensixinet/kernels/backend.py`) run the six kernels' devices on the chips and hand the sound
  back to the CPU with one log line if a worker fails. Details and the hardware findings are in `kernels/README.md`.
  * Found on the hardware, in order: one process with several devices hangs on its first kernel launch; showing a worker one chip of a board makes
    tt-metal call the cluster CUSTOM; four workers that see all chips wait on each other's locks; two processes on one board take turns. The design
    that works is one process per board with the board's two chips visible, switching chips inside the process (0.1 s).
  * `gozer run`'s exit resets the chips for about 40 s. My first `ChipWorker.close()` waited 30 s and killed `gozer run`; the leases were reaped as
    "owning process is dead" and a worker survived holding the chips. Now it waits 180 s and ends the process group on a timeout, and a worker that
    never becomes ready is ended at once. A queued request's ticket is cancelled. Running `gozer status` while a release was under way made two
    releases refuse ("device still open"); I stopped polling during releases.
  * 4 chips, 12 s, `--patch-seed 4`: kernels against CPU max difference 0.0034, correlation 1.00000. Live on four chips for 85 s kept pace (6.7 s segments).
* **2026-10-20, "make the CLI output very colorful ... use the colored inputs and outputs as inspiration ... a little bit of ascii art":** `narrate/color.py`.
  The Plumbutter guide's jack colours (blue modulates the main parameters, green receives a trigger, purple reaches into the innards, white is audio,
  orange is slow control voltages, red is triggering pulses, yellow is modular-level audio, brown is pulses, grey is dust) colour each kind of log line;
  each device and each chip keeps one colour; numbers with units are lifted out; a live segment's bar takes its soloist's colour and has a bar as long as the
  segment. A half-block `TENSIXINET` banner, a patch-cable line and a colour legend open a run. Screen only: the `.txt` log, the JSONL and non-terminals
  are plain, `NO_COLOR` is respected, `--color always` forces it for a recording, `--no-banner` removes the art. No right-hand borders (the global rule).
  * `RunLog.close()` had been closing the terminal's stream through a `_Tee` that refused to; it now never closes stdout or stderr.
* **2026-10-20, "a couple of patches for the 'normies' -- chiptune, enya, blues, aphex":** `music/styles.py` and `--style`. A style narrows the scales, gong
  tunings, motif length and leaps, turn length, floor, steam routes and ttrax size, and sets an `emphasis` (a factor on each device's level and on how often it
  gets the floor; `patch.emphasis`, applied in `render_chorus` and `slot_weights`). The system still picks which of the style's scales and tunings from the
  chips, and the seed draws the rest. A saved patch with no `style` or `emphasis` replays as before. `--only` ignores the balance.
* **2026-10-20, "work on stereo separation ... not just left and right but the full spectrum ... some somewhat fixed, some that travel":** `music/space.py`
  (azimuth, width and depth of a voice; a home for each chip, an offset for each device, an orbit from the chip's chaos and clock), `music/spatial.py`
  (movements from inside the devices: the AVDog's white and red on the two ears, the shimmer's beat, each gong strike's landing, ttquantus's pitch, ttrollz's
  lanes, the snare's alternation, the drone's approach) and `music/stereo.py` (one voice placed). `render_chorus(stereo=True)`; the CLI is stereo by default and
  `--mono` is the old sound. The EQ, loudness gain, stitcher, sinks, players and WAV carry two channels. The mono path is byte-for-byte what it was.
  * First mutation run: 14 survivors. Real gaps: a test that read its own constant (the identity nudge), a depth test that did not isolate the level from the
    lowpass, a dissolve test that passed whichever way the voice drifted, and a smoothing test that a wide voice's own start-up transient dominated. Each got a
    test that measures the one thing; one equivalent mutant (the load flag reset on eviction, which every way back to running sets again).
  * Measured only: left/right correlation 0.27, balance moving -3 to +7 dB, stereo about 0.9 s dearer than mono over 20 s. Not listened to.
* **2026-10-20, "the defrag view has the evict mode -- when that triggers ... should also trigger a certain kind of change in the system":** read tt-toplike's
  `src/animation/defrag.rs`: EVICT fires when a chip that has been running has seen a load (power above 1.5x rest, or 30% of TDP) and falls back to at most 1.2x
  rest (never below 8 W); the "seen a load" gate stops idle chips evicting in a loop (its own phase 29 bug). `music/evict.py` follows each chip's power the same way
  and the same moment moves the band: the chip's voices dissolve right to left across the stereo field at prime-multiplied rates with a falling sweep, the chip goes
  silent and almost never has the floor, and when load returns they fill left to right with a rising sweep. `--mock-load SECONDS` cycles synthetic chips through
  rest, load and rest so it can be heard without a model; `--no-evict` turns it off. Only in live mode; an idle box never leaves "running".
  * My first scenario for a load interrupting an eviction was wrong: four readings back at rest had not yet taken the smoothed power under the threshold, so there was
    nothing to interrupt. The rest baseline is learned only from steady readings, so the edge of a fall is not taken for rest.

* **2026-10-20, "commit and push and update site and recordings":** the stereo and evict code is `ee38da8`. New scene `stereo` (`demo/left_pane_stereo.sh`, `demo/build_stereo.sh`): four synthetic chips on `--mock-load 60` beside `tt-toplike --rotate` on the real chips, 100 s, with the log showing evictions and fills. The site has a `#space` section with the video and a 60 s stereo clip, and `tests/test_site.py` ties its numbers to the code.
  * The first recording used `--mock-load 30` and had no evictions: a loaded stretch of 15 s is too short for the 10 s "ran long enough" rule plus the smoothing. 60 s works (chips evict at 41.6 s, 47.5 s, 70.9 s and 82.6 s in the recording).
* **2026-10-20, "load qwen3-coder-next and give it a go" then "a full re-record of all our audio and video using the HW":** served Qwen3-Coder-Next
  (`tt-model serve raahemnabeel/qwen3-coder-next-blackhole --detach --port 8011 --profile p300 --device-id 0,1`) under a gozer lease held by a `sleep` through
  `--owner-pid`, so the lease outlives a detached container. `demo/load_prompts.py` puts real load on it.
  * Found: long answers (decode) draw only 50 to 57 W against 35 to 45 W at rest, so they never clear the evict rule's 1.5 times rest. Long prompts with 4-token
    answers keep the chips in prefill at 120 to 180 W. The live loop reads the chips about once a turn, so it still misses most short spikes: in the recording chip 1
    evicts once (84.7 s, after a 173 W load), chip 0 never does, and chip 2 (no model) evicts twice on a 27 W blip. The captions say so.
  * Re-recorded: stereo (real load), chorus (prompting Coder-Next), features, kernels (all four chips, no worker failures), the six listening clips (real chips, seed
    5, stereo), three on-chip clips and a new on-chip video. `kernels/cycle.py` passed 12 of 12. Both `gozer run --chips 2` runs were granted board 0, so chips 2 and 3
    were not cycled; the on-chip video is chip 0.
  * Two slips: `pkill -f` matched its own shell and killed the command; `gozer run --exact 0,1` is not a valid form and queued two stray tickets (cancelled).
* **2026-10-20, "we are way overusing the snare still":** measured first, not guessed. On the four real chips the snare is about 1% of the mix's energy and
  fires 0.5 to 0.65 times a second on the one chip that plays it, and no device had the floor too often (ttsteam had 2 to 7 of 40 to 60 segments in the demo logs).
  What spread it was the steam routes: each was on 60% of the time, a patch always had one, so a patch averaged 2.4 routes, 63% had three or four, and `strikes`
  (the whole band follows the snare) was on in 60%. A patch now has no route (25%), one (50%) or two (25%), `strikes` is drawn half as often as the others, no
  style turns on more than two, and the snare's emphasis in the styles is at most 1.0 (it was 1.2 chiptune and 1.5 aphex). Saved patches keep their routes.
  * Not changed: the snare's own level, hit rate and dust. If it still stands out by ear, those are the next dials.
* **2026-10-20, "a tt-discolike enabled gradio app ... an alive illustrated flow chart of what music is happening right now and how it's constructed ... selection of
  the presets/patches ... maybe with the log beside it":** `tensixinet/ui/` and `app/gradio_app.py` (port 7880, `.disco/app.yaml`, its own `.venv-app` because the
  TT venv's gradio does not import). `flow.build_flow` works the state out with the chorus's own functions (device slots, space tables, `draw_cables`, `strength`) so
  the picture cannot drift from the sound; `chart.flow_svg` draws it (CSS-animated wires that run faster with power, cables thicker with push, a glowing soloist, a
  stereo field, evicting chips faded by the mix's own `PRESENCE` table); `session.Session` runs `run_live` on a thread with a paced sink; `forms` turns the form into a
  session. Not a synth panel: nothing changes the sound while it plays.
  * The chart follows the segment that is playing, not the one just rendered. The loop renders up to two segments ahead, so each segment's state waits until the sink
    begins that segment's audio (one write per segment).
  * Found in a browser, not by tests: Gradio's default concurrency limit of 1 queued a new Play behind the running one for ever (now `concurrency_limit=None`, and Stop
    cancels the stream); the log's own text colour was dark under Gradio's theme (each row now sets its colour); `ultrasonic_shimmer` ran into its chip label.
  * Mutation run: 23 mutants; the first run had 6 survivors (the azimuth clamp, dropping played segments, fill events, the sink refusing audio once stopped, the load cycle
    reaching the synthetic source), each now has a test. A second Play takes about 7 s to start because it ends the first session cleanly.
  * Checked by hand in headless Chrome (Playwright, in a scratch venv): the page loads, Play streams audio, chart and log, and the CSS animation runs inside Gradio's HTML.
* **2026-10-20, "can the log stay color coded in the gradio UI":** the app's log had only the coloured left bar. The terminal log also colours each device, each chip and each number with a unit
  inside the line, so the app's log (`ui/logview.decorate_html`) now uses the same token rules (`narrate/color._TOKEN`) and the same colours. Checked by screenshot in Chrome and in Playwright's Firefox
  after 14 s of play, so the colours hold as the log updates. Mutation run: three mutants (number colour, chip colour, no decoration), all caught.
* **2026-10-20, "prepare the HF space with synthetic chips and the tt-lang-sim kernels":** prepared locally, not published. `--kernels sim` (and a Kernels choice in the
  app) starts the chip workers under `tt-lang-sim` with no lease (`media/chip.find_simulator`, `simulator_command`, `kernels/pool.py --sim`); a missing simulator ends the
  session with a `ChipError` message and closes the worker. `space/` holds the Dockerfile, the Space README and `assemble.sh`, which copies only what the Space runs into `build/space`.
  * Built the image (1.8 GB), ran it, and played the simulator mode through `gradio_client`: the log names the simulator and 120 updates streamed. The first run failed on the
    container's log, not in the page: Gradio's streaming audio needs `ffmpeg`, which `python:3.12-slim` lacks. The Dockerfile installs it, with a test.
  * Full suite 2171 passed, 2 skipped, with `TT_LANG_SIM` set (without it the sim tests skip).
* **2026-10-20, "the play/stop and change patch behavior on both gradio and space need some work ... easy to change things and try to start or stop but not have what you changed reflected":**
  `ui/player.py` (which session the page shows) and a rewrite of the app's Play, Stop and change handling. A change of any setting while playing now plays again with it; Stop empties the chart and the
  audio; Stop during a start sticks; the last of several quick changes wins.
  * Found only in a real browser (Playwright, Chrome), in this order: Gradio holds an `input` event back while the previous one is still running, and these streams never finish, so a second
    change waited for ever (`trigger_mode="multiple"` on every stream); a Stop pressed just after a change was undone because the restart checked "is something playing" and then began, with the Stop
    in between (the check and the claim are now one step, `Player.claim`); a frame a stream had sent just before Stop reached the browser after Stop's own result, and `cancels` cannot recall it (each
    stream now ends the page itself, as stopped after a Stop and silently after a newer Play, and Stop no longer cancels); the browser went on playing about ten seconds of audio it had buffered (a `pause()`
    in the page on Stop, Play and a real change; removing the element's `src` made Gradio's HLS player start it again from 0).
  * My first browser run tested the old code: the app on port 7880 was still the instance started before the edits, and the new one could not bind. Check what answers on the port.
  * Mutation run: the first battery left two survivors (the loop's own check and the check after the wait; each needs a test that supersedes the stream at that moment) and one test that hung when its
    mutant removed the restart check (streams never end), so every pull in those tests is bounded (`_take`).
* **2026-10-20, "can we buffer a little more before playing/starting? we get about 2-4s of generated audio before needing to wait for more":** the sink used to wait out each segment before handing
  over the next, so the browser's player was never more than one segment ahead. `Session(lead_s=...)` (the app passes `PREROLL_S` = 10 s) hands audio over until it is that far ahead of the listener, timed from the
  first write on the clock, so render time counts as listening time. The chart and log follow the sound: a segment's state shows when the listener reaches its offset (`_pending`, `_advance`, also run by every snapshot).
  A lead of 0 is the old behaviour, which the command line and most tests use.
  * Measured in headless Chrome over 70 s of play: without the lead the player's buffer fell to 0.1 to 0.5 s each segment and stalled 3 times; with it the buffer stayed at 5 to 9 s and stalled once (the start).
  * Cost: the sound is ten seconds behind the chips. Mutation run: 11 mutants, all caught on the first run.
* **2026-10-20, "the cpu only mode seems to work but when i use tt-lang kernels i don't hear sound or see the connections display or logs starting" (then "it just takes a long time ... maybe we need more progress bars"):**
  it worked, and was slow to start with nothing on the page. Timed on the Space with `gradio_client`: CPU only had the first chart at 4 s; the simulator at 18 s, then a segment about every 5 s. My first guess (a bug) was wrong, and
  the user's own reading ("it just takes a long time") was right. `ui/progress.py` and `Session.progress()` give the steps (start the simulator, render the first segment, fill the buffer, the last with a real bar against the lead);
  the chart area shows them until the first segment plays, the status line names the active step with its seconds, and the log says when the simulator starts and when it is ready.
  * Mutation run: 15 mutants; one survivor (a buffer step listed with no lead), now tested. Looked at in headless Chrome: the checklist and bar show, and the log lines arrive.
  * Not known: how long the Space's own start takes now; the timings above are from before this change, and only the 18 s was measured.

* **2026-10-20, "we need install instructions and helpers. an obvious top level script ... `cd tensixinet && ./tensixinet.py --live` to be all I need":** `tensixinet.py` at the top
  (standard library only; `tests/test_launcher.py` loads it by path because it shares its name with the package, and the package wins an `import`). It uses the running Python if it has numpy and soundfile,
  else `.venv` (built on first use with `pip install -e .`); with no `tt-smi` and no chosen source it plays four synthetic chips and says so; `--live` with no `--duration` plays until Ctrl-C; `--app` builds
  `.venv-app` with Gradio and starts the page; `--doctor` and `--setup`. README's quick start now starts there; `make setup` and `make live`.
  * Mutation run: 15 mutants, all caught. The run itself found a flaw in the tests: one mutant (`pick_python` always None) made a test run the real `setup_env`, which built an 88 MB `.venv` in the repo with a
    network install. The tests now pass a `run` that raises, and use a root that does not exist for `--doctor`. `.venv/` is gitignored.
  * Run for real: on this box (real chips, 8 s, no sound), on the same box with `tt-smi` off the PATH (synthetic fallback), and from a copy of the repo with a system Python that lacks soundfile (it built `.venv`, then played, 10 s in all). Not tried: macOS or Windows (the venv path is `bin/python`).
* **2026-10-20, "it seems like all our patches involve that snare. the shift register can be used on anything and everything. let's use the snare seldom and rely more on gongues for beats":**
  `Patch.snare` (0 to 1) is the share of ttsteam's hits that play, and scales its dust. A drawn patch has 0 (40%), 0.2 (30%), 0.5 (20%) or 1 (10%), from its own random stream; the default patch and
  old saved patches are 1, so they replay as before. The register keeps stepping at any level, so the routes still work with a silent snare. Routes: none 10%, one 35%, two 35%, three 20%, and `strikes`
  (the gong follows the register) is drawn 1.5 times as often as the others, where it was half as often. Styles set the snare (chiptune 0.5, enya 0.1, blues 0.5, aphex 0.6) and raise the gong's balance
  (chiptune 1.0, blues 1.0, aphex 1.3; a test says the gong is never below the snare in any style).
  * The cause, from the code: ttsteam is one of eleven devices dealt to the chips at full level whatever the patch, and its dust ran all the time, so every patch had a snare. The earlier fix only cut the routes.
  * Mutation run: 16 mutants; one survivor was real (the snare draw sharing the routes' random stream, so the two would correlate) and has a test; `snare <= 0` against `< 0` is equivalent.
  * Not changed: how often the gong strikes (its trigger clock and rolls). If the beat is still thin by ear, that is the next dial.
* **2026-10-20, "cocoquantus' is also not just a generator of looping sounds... it takes sound inputs and puts them in a delay influenced by the petals ... maybe even adding some reverb":** `music/cocobus.py`. Each chip's voices
  (after the turn gain, balance and an eviction's fade) are summed at per-device send levels, scaled by `Patch.coco_send`, and run through four 8-bit taps (every second GDDR channel; delay from temperature, bits from delay, feedback
  from pressure, 0.25 to 0.8). A petal's castle, glided over 10 ms, sways each tap's delay by up to 35%. The echoes plus the dry send go through a convolution reverb (`reverb_s`, `coco_reverb`; noise decaying 60 dB over the time, rolled off at 4 kHz, a room per ear).
  Three new patch fields from their own random stream; the default patch and old saved patches send nothing; styles carry `bus`. `--only` skips it. The log has a "coco bus" line, and a "snare" line that was missing.
  * Measured only: a 20 s stereo render on four mock chips went from 2.6 s to 4.1 s, the bus adds about 37% of the dry level (rms), a 3 s live segment takes 0.5 s. Not listened to. The reverb is an addition: the real unit has none. The bus is CPU only (no kernel).
  * Mutation run (31 mutants): 14 survivors the first time, all constants and wiring (the number of taps, sway depth and glide, feedback floor, tap level, reverb cutoff and decay, the reverb's input and share, which channels the taps read,
    each ear's room, the tap bits, the seeds); each got a test, mostly through a spy on `swayed_tap` that checks what the bus hands each tap. One more survivor is equivalent (the early stop in `swayed_tap` that `np.interp` already makes harmless).
  * A test caught a real bug on the first run: the send was taken before an eviction's fade, so an idle chip still echoed.
  * Also fixed: the patch log still said each steam route was on 60% of the time.
* **2026-10-20, "build per-visitor session state":** the app had one global `Player`, so two visitors on the public Space stopped each other's audio. `ui/player.py` gained `Visitors` (a `Player` for each Gradio session hash, `Busy`
  when `limit` visitors are already playing, `drop` and `reset`); the app's handlers take a `gr.Request`, `demo.unload(leave)` ends a visitor's session when they close the page, and `TENSIXINET_MAX_PLAYERS` (default 3) sets the limit. The count and the claim are one step
  under the registry's lock. Only visitors who are playing count towards the limit.
  * Mutation run: 14 mutants; one survivor (the registry lock removed), because the race window was too small to hit. The test now makes the count stale on purpose (a sleep after it), and the lock-less version fails it.
  * Found only with real clients (`gradio_client`, three of them against a running app): every visitor was the same one, because `def play(*settings, request=None)` gets no request from Gradio (it does not look past a *args), so each Play ended the last.
    The unit tests passed `request=` themselves and could not see it. The handlers now name every setting, and a test asks Gradio's own `special_args` whether it would fill `request` (the old form turns it red). With the fix: two play at once, a third is told the places are taken,
    the first's Stop leaves the second going, and the third gets in once a place is free; two new visitors both get in after earlier ones disconnected.
  * Not tried yet: the limit on the Space under real load (cpu-basic, two vCPUs). 3 is a guess from a 3 s segment taking about 0.5 s to render.
* **2026-10-20, "looping cocos need more 'off' notes or rests ... when doing sample and hold to shift registers, or sample and hold to coco style stuff, you add in some silence to get into the rhythms too":**
  `Patch.rests` (0 to below 1) is the share of the coco's held notes that are silent. `devices.ttquantus_gate` takes a second castle of petal 0 that reads the next leader round the castle mod select (`ttquantussy.rest_offset`), held for each note; a note under `rests`
  does not play. `quantussy_tone` multiplies by the gate, so `ttquantus` and every `ttcoco` echo carry the rests. 5 ms glide, true silence between. A drawn patch rests 15 to 45% (own random stream); styles chiptune 25%, enya 30%, blues 35%, aphex 40%;
  the default patch and old saved patches have none. The log has a "rests" line.
  * Measured only (not listened to): the share of resting notes follows `rests` to within 0.1 over 12 seeds; `ttcoco` is quieter with rests, and its echoes fill part of each gap, as a delay does.
  * Mutation run: 14 mutants; 3 survived the first time (the glide's alignment, the petals the gate reads, and `>` against `>=` at the threshold, which is equivalent because a castle value is never exactly `rests`). The first two got tests.
  * Not done: rests for the other sample-and-hold users (ttsteam's register already rests where a column is enabled; the lead's motif and ttrollz are always on). Say if you want them.
* **2026-10-21, "add rests to the lead and ttrollz too then push":** `Patch.lead_rests` and `Patch.roll_rests` (0 to below 1), a shared `music/rests.py` (`rest_flags`: one value per step from the seed, so a bigger share rests every step a smaller one does).
  The lead leaves out notes but always plays the first note of a phrase. ttrollz leaves out pulses of the clock and of each lane separately (a lane's downbeat blip goes with its pulse); `roll_pulses` is untouched, so the gong still follows the whole rolls.
  A drawn patch has 10 to 30% of each from its own stream; styles chiptune (0.1, 0.15), enya (0.2, 0.2), blues (0.25, 0.2), aphex (0.15, 0.3); the default and older patches have none. The log has "lead rests" and "roll rests" lines.
  * The owner listened to the rests (coco, lead and ttrollz) and said "sounds better". Mutation run: 30 mutants; 2 survived the first time (the clock's rests reading only the segment of the seed, and the accent route reading the clock before its rests), each now has a test and goes red.
  * Not done: rests in the other always-on devices (ttdrone, ttchoir, ttavdog, ttrax).
* **2026-10-21, "are the recordings and images in the README and site now up to date" then "yes I want all that, and updated site text too. and samples for the blues, aphex, enya" (and "focus less on rotating tt-toplikes. arcade mode is good enough", "stop start models as needed, but really I like the full on chip version of the instruments best", "for a bonus, please record what a tt-smi -r sounds like at the end"):**
  everything that makes sound was re-made after the snare, coco bus, rests and per-visitor changes. `demo/make_clips.sh` renders the ten listening clips (seed 5, stereo, `--kernels chips`, each run takes and releases its own gozer lease), including the four style samples (`chiptune`, `enya`, `blues`, `aphex`, 40 s each).
  The features and kernels videos run `--kernels chips` and every scene's right pane is tt-toplike's arcade view (`--mode arcade`), not `--rotate`. `demo/build_features.sh` is new. The stereo and chorus videos use Qwen3-Coder-Next on two chips (served under a gozer lease held by a `sleep` through `--owner-pid`, stopped and released after), so their band is on the CPU: the model holds the chips the kernels would need. The site, README and app screenshots are from the current code. The site has a new section on the delay bus and rests, a styles table with each style's snare, bus and rest values and a clip for each style, the 1.5x route odds and snare levels, and the visitor limit.
  * Found by the recording: the first kernels video showed chip 3 evicting every 16 s on a 40 W reading against a 37 W rest. `shows_load` applied the 30%-of-TDP rule even after a rest was known, and this box rests at 36 to 40 W against a 37.5 W line, so wandering across it counted as a load. The line now applies only while no rest is known (tests first, then a mutation check; one survivor, a tiny known rest, got a test). The first stereo recording missed the model's spiky load (one eviction, no fill), so the scene now sends three shorter bursts; the second recording has chip 1 evicting at 1:08 and filling at 1:41.
  * `demo/make_reset_clip.sh` records what a chip reset sounds like: `tensixinet --live` on the CPU reads all four chips while `gozer release` resets them (the reset `tt-smi -r` makes, taken through the lease). For about 40 s `tt-smi` cannot see the chips; the loop logs "no chips in this reading" 19 times and plays the last readings that worked again with no movement, and the chips come back cold at 13 to 15 W. The reset is not a distinct sound: the band holds, then carries on from colder readings. It needs a "chips lost" event in the music to be heard as a reset.
  * Not done: the chip-computed device clips (`ttgong_chip`, `ttavdog_chip`, `ttcoco_chip`) and the hardware kernels video are from the kernel cycle on 2026-10-20 and were not remade, because those kernels did not change.
* **2026-10-21, "the looping cocoquantus almost always sounds the same. nothing is modulating the pitch really. and it really shouldn't be the main thing you hear in an enya mode":**
  measured first (30 s, six seeds, four synthetic chips, each device's energy as it reaches the mix): in `enya` `ttcoco` was 48% and `ttquantus` 8.5%, so the looping coco was 57% of what was heard (`enya` had `ttcoco` at 1.2, to fill the room);
  in a plain draw `ttcoco` was 36% and `ttquantus` 27%. The line itself was a fixed two-octave window of scale steps, every note held and jumping to the next.
  `Patch.quantus_mod` (0 to 1) moves it: a slow castle (the note castle's petals at an eighth of the rate, a third leader, `ttquantussy.pitch_offset`) shifts the register by up to 3 scale degrees every several notes (`devices.ttquantus_register`), each note glides into the next over up to a
  note's length, and a 5 Hz vibrato of up to 40 cents deepens with the chip's pressure. `ttcoco` echoes the same tone, so it carries all three. Drawn patches 0.35 to 0.9 (own stream), styles chiptune 0.15, enya 0.8, blues 0.45, aphex 0.9; the default and older patches have none, so they replay as before.
  In `enya`, `ttcoco` and `ttquantus` are now at 0.3 (blurb: "a long room of delay behind them", since the bus still carries the room); their share of the energy went from 57% to about 7%. The shimmer is now enya's largest device by energy (about 36% before the EQ, which takes it down); not changed.
  * `ttquantus` now lists the pressure fields in `DEVICE_INPUTS`, and the inputs test renders it and `ttcoco` with a patch that moves the pitch (they read pressure only then).
  * Measured only, not listened to. Not changed: a plain draw still has `ttcoco` and `ttquantus` at 36% and 27% of the energy (no emphasis); chiptune keeps `ttquantus` out in front by design.
  * "commit and push update site with new samples": the six full-band clips (`all`, `all_nocables` and the four styles) were re-rendered on the chips with the pitch movement in, and two new cards on the site, `quantus_fixed` and `quantus_moving`, are the looping line from the same patch and the same synthetic readings with `quantus_mod` off and on (`demo/make_clips.sh`; the held one is the moving one's patch with the modulation set to 0). The device clips (`choir`, `avdog`, `gong`, `trax`) do not use the pitch movement and were kept.
  * Not remade: the videos and `stereo.mp3` were recorded before the pitch movement and the enya balance, so their soundtracks still have the fixed line.
* **2026-10-21, "I would want to run tt-bio-demo in a window side by side with the tensixinet CLI output ... 3 chips rendering, 1 chip questioning" then "question mode wasn't working ... the music sounded perfect though, can we get a run recording full screen tt-bio all 4 chips, with the same essential settings and patch" then "let's add the demo to our site":**
  `demo/bio_session.sh` runs tt-bio-demo (`--questions --windowed`, three chips fold, one answers) in a window beside a terminal running `tensixinet --live`, under one gozer lease on all four chips (the booth holds the chips, so the band is on the CPU). Two runs (4 and 6 minutes; the owner cancelled the second at about 80 s): the question chip was reserved
  and ready but no question or score ever appeared, and the second run's UI reconnected 26 times; tt-bio-demo-9a was sent the details, since this was the first real check of tt-bio 0.21.0.
  `demo/bio_record.sh` records the quad (no questions, all four chips folding) full screen with OBS through `demo/bio_capture.sh` (tt-bio-demo's `scripts/record-demo-video.sh` with one change) and muxes `tensixinet --live --patch demo/bio.patch.json` under it (the second run's patch, which the owner heard). 248 s; in the six minutes the booth was up 169 folds started and 165 finished, 3.6 to 18.2 s each, no errors in its log.
  The site has a `#bio` section with a 720p cut (the pointer removed by `delogo`, 42 MB) and a poster.
  * The first recording attempt was refused by the script's own screen check on a clean screen: its dialog probe samples one row where a cell's caption sits. `demo/screen_clean.py` measures how much of the dialog's box is pale instead; it passes the real frame and refuses a fake dialog and a fake taskbar. A tt-bio-demo session was told.
  * The quad run's own `gozer run` refused to reset (`chips [3] still open by [pid]`, the process gone a moment later) and left the lease STALE. Checked with `fuser`, then `gozer reconcile`, then an acquire and release of all four chips to reset them.
  * Not done: the question mode (not working on tt-bio 0.21.0 as of this run), a recording with the kernel devices on the chips (the booth holds them).
* **2026-10-23, "expand the gradio app to include a tab for playing the instruments, modules, and system 'live'" (a plan for qwencode, then "we need to use the 4 chip version qwen code to succeed on this front. for now, fix the mistakes it made and carry out the plan instead"):**
  `PLAN-PLAY-TAB.md` is the plan (it now ends with "What changed while it was built"). The app has a second tab, **Play**: one device, a module (rhythm section, drones and air, melody, delay and echo) or the whole system,
  played with sliders for each chip's temperature, power, aiclk, GDDR spread and voltage and buttons for a memory error, a thermal trip and a load burst. New in `tensixinet/ui/`: `knobs.py`, `scopes.py`, `instrument.py`, `readout.py`;
  changed: `player.py` (`Player.session`, `Visitors.session`), `forms.py` (`new_instrument`, `read_chip_knobs`), `app/gradio_app.py` (two tabs).
  * qwencode's first attempt (Qwen3-Coder-Next on two chips) was discarded. It wrote `knobs.py`, `scopes.py` and `readout.py` out of order with no tests for the last two, replaced the SPDX header of `ui/__init__.py` with imports (hygiene red), had a load burst that forgot
    when it started, a `heard_on` that ignored how devices are dealt, and a gddr test that asserted nothing. All three modules and the tests were redone test-first.
  * **Found by the real-render test:** the gong was silent in a 1.2 s render. It lives on chip 1 and its first strike is up to 2.2 s in. The engine now renders short (2.4 s, after a change) and long (6 s) phrases and plays 1.2 s slices, crossfading only where a phrase is replaced.
    The plan's premise that a 1-chip box leaves devices undealt was also wrong: `n_slots` is at least the device count.
  * **Found only in the browser** (headless Chrome through Playwright, in this order): Gradio never goes network-idle, so a page-load wait on it hangs; a Stop that returns `None` for the streaming audio makes Chrome load a new source and play it for several seconds more. The same happens in the first
    tab and on the commit before this work (run from a worktree), so it is an old bug that the earlier browser check missed. Stop and the stopped frames now return `gr.skip()` in both tabs; after that the element stays paused. Typing a chip count and pressing Tab restarts the stream (blur fires `release`); Enter does not.
  * Browser results: Play starts in 0.6 s; moving a slider changes the readout 3.6 to 4.2 s later with no restart; a scope change restarts in 0.3 s; Play after Stop plays again; two visitors have separate knobs; the fourth player is told the places are taken; 40 s of play had no stall (smallest buffer 1.7 s); no console errors; no horizontal scroll at 375 px.
    The lead is 3 s because the browser's player stalls on less (a shorter lead was not tried). Cost on two CPUs and on one: a short phrase 0.15 s, a long one 0.34 s, so no chip cap for the Space.
  * Mutation runs, per task: knobs 2 survivors (a boundary, an uncorrectable count) plus 1 after hiding chips; scopes 0; instrument 6 the first time (one equivalent: the crossfade tail in the bound that says a phrase has run out) and 1 more (the soloist's phrase number); readout 2; player 1; forms 5; app 1 plus the cross-tab frames. Each got a test.
    The 2546 tests pass in `.venv-app` (34 skipped); with the TT venv's Python the app tests skip because its Gradio does not import.
  * Slips: `pkill -f "app/gradio_app.py"` matched its own shell and killed the command (and a second server); `git add -A tests` swept in an older uncommitted change to `tests/test_site.py` that was not mine, so that commit was redone without it (the file is still uncommitted).
  * The Qwen3-Coder-Next container (port 8011) was stopped on request and its stale lease cleared with `gozer reconcile`; all four chips are free. Measured and watched in a browser; not listened to by a person.
* **2026-10-23, "can the UI during play mode show a little more like in the watch it mode? so as you adjust settings you witness a little how it changes what the system itself looks like":**
  the Play tab now draws the first tab's flow chart above its readout (`readout.play_html`). `InstrumentSession` builds a `FlowState` for each block and the snapshot carries it (`InstrumentSnapshot.state`); `build_flow` takes a `heard` list and
  a soloist of `None` (a scope that plays some devices has nobody on the floor), and `flow_svg` takes a `heading` and draws a device that is not heard faint. The chart column is wider and sticks to the top while the sliders scroll.
  * Found in the browser: the readout's pitch line was dark grey on the dark panel (Gradio's theme colours a div's own text, so the inner row now sets its colour; a test parses the HTML for it). Also, my first screenshot was of the old server: the earlier
    kill had not worked and the new one could not bind port 7880, so check what answers on the port.
  * Mutation run: 2 survivors in the first battery that were real (the stylesheet's faint rule and an empty heading) plus three in the instrument (the segment, the block's time and the phase carried through a phrase); each got a test. One is equivalent (dropping the `!= "running"` filter).
    Two of my first mutation scripts applied nothing, because their separator had two spaces (the script printed BAD for each; "all caught" with BAD lines means nothing ran). The 2564 tests pass in `.venv-app` (34 skipped).
* **2026-10-23, "could we offer --arguments for swifter sampling? kind of like how tt-toplike lets you choose bandwidth. And then a --with-tt-toplike mode where it starts tt-toplike and the synth _at the same refresh rates_ for a synced experience":**
  asked what they wanted to notice: "beats and music tracking second by second. right now you can see the delay a little or know you're seeing something you're about to hear, and then it changes. this could also get to where changing views in tt-toplike could send change signals to the music too. like a dedicated defrag song."
  The delay they would accept is a quarter second. They chose to read hwmon ourselves ("start with A"), said tt-toplike is theirs but changes follow a review process, and on the design agreed that the rolls keep their own schedule ("rollz are on their own schedule. you can only influence and be influenced by them"). Execution was by subagents from a plan (`docs/superpowers/`).
  * Built: `--live --interval MS` (10 to 1000) and `--with-tt-toplike [--interval N] [--toplike-args "..."]`. New modules `telemetry/hwmon.py`, `telemetry/fast.py` (`FastSource`: hwmon for temperature, power, voltage and current at every read, `tt-smi -s` on a thread about every 1.5 s for the rest, matched by PCI bus address, with a fallback note), `fast.py` (`run_fast` and the delay report) and `media/toplike.py` (the tmux session).
    Changed: `ui/instrument.py` (the Play tab's phrase engine takes its read function, block, phrase and crossfade lengths, sample rate, `only`, `eq` and `normalise`, and gained change detection with a minimum gap, a held soloist, the delay it calculates, head and body phrases and render ahead on a worker thread), `music/evict.py` (smoothing in seconds), `music/live.py` (`Tracker(hold=...)`), `media/player.py` (small-buffer flags), `telemetry/snapshot.py`, `narrate/color.py` (a `delay` line colour) and `cli.py`.
  * Calculated by the run on the four real chips (read-only, no lease, `--interval 100`, 22050 Hz, `paplay` at 20 ms): worst-case delay 220 ms, typical 170 ms, 0 gaps in every 5 s report over 30 s. The worst case is the floor, interval 100 + lead 100 + player buffer 20. At `--sample-rate 44100` the budget is missed (worst about 420 ms, typical about 320 ms). The run's own `delay` line produced these numbers from its own timings and the player buffer it asked for; the delay at the speaker has not been measured. Nobody has heard it.
    The 220 ms holds for a change that arrives when the last phrase started more than 1 s earlier. A change inside the one-second gap between phrases is heard when the gap ends, so up to about 1 s plus the delay, and on the real chips a phrase starts about every second, so that is most changes. The runs above were logged before the `delay` line counted that wait (the final review found it; see the next entry).
  * What cost time. The plan assumed short renders, and Task 8 measured the first working loop at a worst delay of 2.10 to 2.45 s: a 6 s phrase took 0.85 to 1.0 s to render at 44.1 kHz, so the loop lengthened its block to over a second. Task 8b fixed it with a 0.6 s head rendered on the loop and the full phrase on a worker thread, a 22050 Hz default in fast mode, and a lead of 0.10 s. A 0.06 s lead gave 2 to 4 gaps every 5 s because a head render (73 to 88 ms) was longer than it; 0.10 s gave none.
    A head's last 50 ms differ from a longer render of the same phrase (correlation 0.47 in that window), so the head is rendered 0.15 s past the join. A reviewer found that an abbreviated `--with-tt` would start tmux sessions without end (the synth pane's command line is rebuilt from full spellings), so the CLI has `allow_abbrev=False` (`--dur 30` is now an error) and a `TENSIXINET_IN_SESSION` guard. Relative tmux targets (`{right-of}`) resolve against the active pane, so the panes tag themselves and each ending acts on the pane tagged the other way.
    `tt-toplike --serve` could not be the source: it carries `tt-smi` frames every 1.5 s and none of the fast readings. The hwmon files read instantly while `tt-smi -s` was hanging, so the synth never waits for `tt-smi`. Errors, thermal trips and mask changes arrive with the `tt-smi` reading, about every 1.5 s, and are outside the quarter-second budget.
  * Mutation checks on every task found survivors and each got a test. A fix to one test file deleted five other tests by replacing a block of text, and they were restored and mapped to mutants. The suite is 2764 passed, 34 skipped.
  * Not done: any listening by a person; 44.1 kHz inside the budget; the lead per player; `ffplay`'s buffer (an estimate of 100 ms, which puts it over the budget) (a slow head render lengthening the block for good was fixed in the next entry); tuning the change thresholds (0.5 C, 3 W, 0.01 V) and the 1 s minimum gap by ear. Part B (tt-toplike publishing its fast readings and its view on `--serve`, and a defrag song driven by the view) is in `docs/ROADMAP.md` 7d and not started.
* **2026-10-23, the final review of fast sampling, fixed in one wave** (two Critical findings, four Important, six Minor; nothing committed, by the owner's rule):
  * **The delay claim left out the one-second gap.** A change that lands less than `MIN_GAP_S` after the last phrase started waits for the gap to end, and on the real chips a phrase starts about every second, so most changes wait. The session now records the wait (`InstrumentSession.held_s`, from the first reading that crossed a threshold to the reading that starts the phrase; a reading that moves back forgets it) and the `delay` line adds it to the worst case and names it. The docs and the site now say the 220 ms holds only for a change that comes more than 1 s after the last phrase started, and that one inside the gap is heard up to about 1 s later. The 1 s gap itself is unchanged: shortening it is the owner's call.
  * **The rolls and the gong do not keep their own schedule.** Every new phrase restarts every device's clock, so while the readings move the gong and the rolls restart about once a second. The docs said otherwise; they now say what the code does. Measured by computation: no gong strike in the first second in 24 of 48 cases with the default patch and 305 of 576 with drawn patches. The reviewer's script passed `draw_patch` its arguments the wrong way round, raised, and measured only the default patch twelve times over. The owner's wish (the rolls on their own schedule) needs a phrase that continues from a start offset, now the open question in `docs/ROADMAP.md` 7d.
  * The delay is calculated by the run from its own timings and the player buffer it asked for. Nothing measured it at the speaker, and the log line, README, DECISIONS, the site and this log now say "calculated".
  * In fast mode (`--interval`) players run in their own session (`start_new_session=True`), so a Ctrl-C reaches only Python (plain `--live` keeps the player in Python's group, so Ctrl-C there stops it at once); a player that still stops after a Ctrl-C is logged and the run exits 0. hwmon reads that keep failing (a chip reset) are logged, searched for again after 10 in a row, and fall back to `tt-smi`'s values with a note. The block is lengthened only after 3 head renders in a row outrun it and comes back after 60 calm blocks. `--only` logs that it turns eviction off in fast mode; the launcher no longer adds `--mock-chips` to `--with-tt-toplike` on a box without `tt-smi`; a detached tmux session gets the terminal's size; the argument checks run before a tmux session is built.
  * Mutation checks: each new behaviour was broken on purpose. Survivors got tests (a held change counted twice when errors start phrases back to back; a chip on `tt-smi`'s values returning to hwmon; renders that still outrun the asked interval keeping the block long). Two are equivalent: the tmux size inside tmux (`build_session` ignores it there) and replacing the whole chip list on a search that misses the failing chip (that branch never runs).
* **2026-10-24, "some of the cv outputs are rather musical. what would a MIDI out option look like?" then the DAW route, a MIDI file route and a live MIDI device route, "someday CV route like in ~/code/tt-cv-agency along with VCV.. let's not block ourselves in", then "go ahead, but self-approve your specs and then execute them with sub agent development ... put this all in a PR":**
  `tensixinet/score/` and `docs/MIDI.md`, built on the branch `midi-out` from a spec and plan in `docs/superpowers/`. `--midi-file`, `--midi-port`, `--midi-offset-ms` and `--midi-map`.
  The lead, `ttquantus`, the gong, `ttrollz` and `ttsteam` send notes and triggers, and each chip sends four controllers.
  * Decisions: the events come from the same functions as the audio (the lead and `ttrollz` were split so the audio reads one event list, with SHA-256 characterization tests taken before the split);
    the file writer is in the repository and `mido` is an optional extra imported only for `--midi-port`; the port sink never blocks the audio and drops everything after its first error. The CV sink is only a seam (the sink protocol) and is not built.
  * What went wrong: two of the implementers wrote code before the tests. Review found that two notes of the same pitch overlapping on a channel let the first off silence the second (fixed with a retrigger),
    and that a dead port was silent (the sink now keeps `error` and the live loop logs one "midi" line).
  * Not done and not measured: nothing has been heard through a real synth. No real port has been connected to a synth (`mido` and `rtmidi` are installed in `~/.tenstorrent-venv`; the port opener is tested against a fake module). `--midi-offset-ms` was never measured against real audio.
    No MIDI clock, no `ttchoir` events, chips share channels on a live port, pitch bend is channel-wide, fast mode (`--interval`) is not joined to the score and the CLI refuses the two together. All are in `docs/ROADMAP.md` item 10.
* **2026-10-29, "bring midi-out into a state where we can share it. proof of concept is midi out leads to vcv rack and we can listen to an instrument playing ON TOP OF tensixinet ... a monophonic voice with lots of LPF and resonance":**
  `vcv/tensixinet-voice.vcv` and `tensixinet/score/vcv.py` (the builder; `python -m tensixinet.score.vcv OUT [--device auto|NAME] [--channel N] [--bend-range ST]`): Core MIDI-CV (mono, channel 1) into a Fundamental VCO saw, a low cutoff (0.30) and high resonance (0.80) VCF that an ADSR opens, and a VCA on the same ADSR.
  * Run for real: Rack Free 2.6.6 headless (`Rack -h`, stdin held open) into a null PulseAudio sink, notes from `mido` into the virtual port. A4 = 440.0 Hz, wheel up 493.8 Hz, wheel down 392.0 Hz, C4 = 261.7 Hz, a channel 2 note silent. A live run on four synthetic chips made sound (RMS 0.04). Nobody has listened to it.
  * Found on the way: the virtual port's ALSA name is `RtMidiOut Client:tensixinet 128:0` (the first matcher looked for a name that started with `tensixinet:` and found nothing); Rack's module slug for MIDI-CV is `MIDIToCVInterface`; with the resonance at 0.8 an autocorrelation pitch tracker reports the filter's ring (249 Hz for A4), so the pitch check ran with the resonance at 0 and the cutoff open;
    a Rack killed with SIGTERM leaves its log without `END` and the next start shows a crash dialog that blocks it (the harness's `vcv-stack` documents this).
  * "similar to what we've done in ~/code/tt-cv-agency, let's record a tt-demo-maker video of tensixinet and vcvrack side by side or stacked to prove this out, with audio": `demo/vcv_record.sh` (a scene the `tt-demo` manifest has no shape for, so a script like `demo/bio_record.sh`). Stacked: tensixinet's live log and an `aseqdump` of its MIDI on top, Rack with the voice and a scope below, the sound of both from one null sink.
    It runs on a private Xvfb display with software GL, so no screen-share dialog and no desktop windows. A two-minute take is 1080p, mean level -17 dB; `build/vcv_rec/vcv.mp4` (41 MB) and a CRF 26 copy `vcv.share.mp4` (22 MB). Frames at 12, 45 and every 20 s were looked at: the scope moves, the envelope and the gate light follow the notes, and the MIDI scrolls.
    Measured on the way: the voice sits about 30 ms ahead of tensixinet's own lead audio (loudness-curve correlation 0.67), so `--midi-offset-ms` stays 0. The scope's time base and gain were found by trying values and looking, since the parameter numbers are not documented (time 1.0 is a solid band, -2.0 shows a whole note, 2.5 shows a few dozen cycles).
    The site has a `#voice` section with the 22 MB recording (`site/assets/voice.mp4`, poster `voice-frame.png`) and `tests/test_site.py` ties its numbers to `score/vcv.py` and the route table.
    Slips: my first Xvfb used display :99, which another agent's `xvfb-run` held, and "Could not initialize GLFW" was the symptom; a Rack stopped with SIGTERM leaves its log without `END` and the next start opens a crash dialog (the recording uses its own user folder so this cannot reach the real log). Nobody has listened to the take, so the balance of the two is untested by ear.
  * Process slip: I wrote `vcv.py` before its tests, then ran a 23-mutant battery. Two survivors (the bend range through the command line, the DC filter flag) got assertions.
* **2026-10-29, "this works! let's ship it! and merge the PR to main":** PR #1 (`midi-out`: MIDI out, the VCV Rack voice and its recording) merged to `main` as a merge commit (`4edc247`); the site has a `#voice` section with the recording. The fast-sampling commit that sat on local `main` was rebased onto it (`de5509e`):
  conflicts in `cli.py` (both sets of flags; `_check_arguments` now returns the `--only` list and the MIDI routes, and `main` takes both `open_midi` and `which`/`tmux_run`), `narrate/color.py`, `tests/test_cli.py` and the docs, all keeping both sides. The merged suite: 3051 passed, 82 skipped.
  * Found by running the merge: `--interval` with `--midi-file` was accepted and wrote no file (fast mode builds no score). It is now refused with a message (test first, mutation checked).
* **2026-10-29, "how hard would it be to support rtpMidi too?" then "try rtpmidid first":** no code. Downloaded the official `rtpmidid` v26.01 Ubuntu `.deb`, unpacked it with `dpkg -x` (nothing installed) and ran two instances on loopback. My first attempt to run the binary was refused by the permission check (code from an external source), so the user started the daemons with `!`.
  The result is a section in `docs/MIDI.md`: tensixinet's `--midi-port virtual` port connected to `rtpA` with `aconnect` becomes a session of its own, and a second instance that connects to it received the lead's notes, bends and the gong's triggers (124 note-ons in 10 s).
  * Slips: `--midi-port` needs `--live` (my first run was an offline run and exited); `aseqdump -p 129` subscribes to port 0 only, and the data was on port 3, so the first dumps showed nothing; the session's UDP port is new each run, so B has to be pointed at the current one.
  * Not tested: Apple's driver, a real network, latency, mDNS discovery. Nobody has listened to it.
  * "commit these including docs and push. Make sure README and website notes the support. If we can offer install help in the tool's scripts, let's do it": `./tensixinet.py --rtpmidi-setup` (in `tensixinet.py`; `find_rtpmidid`, `pick_rtpmidid_asset`, `setup_rtpmidi`) downloads the release `.deb` for this Ubuntu and CPU, asks `dpkg-deb -f` for its package name before unpacking, unpacks with `dpkg -x` into `~/.local/share/tensixinet/rtpmidid` and prints the steps; `--doctor` has an rtpmidid line.
    Tests first (12), then 10 mutants: 2 survivors (a `.deb` suffix check and removing the downloaded package) each got a test. Run for real into a scratch HOME (it downloaded and unpacked; the binary was not started by the helper). README, the site's `#voice` section (a card, tested in `tests/test_site.py`) and `docs/MIDI.md` say it works between two instances here and was not tried with Apple's driver.
* **2026-10-30, "make --with-tt-toplike use the recording config because it makes tmux look better":** outside tmux the `--with-tt-toplike` session now runs on its own tmux socket
  (`tmux -L tensixinet -f tensixinet/media/tmux-toplike.conf`): status bar off and truecolor, the settings tt-demo-maker's new recording setup uses (found uncommitted in its working tree). Inside tmux it still splits the
  owner's window and changes nothing. The owner's `~/.tmux.conf` is not read in the session, so their prefix key and bindings do not apply there; the existing-session message now names the socket (`tmux -L tensixinet attach -t tensixinet`).
  The config is package data. Tests first, then 8 mutants (the CLI wiring, the has-session check, the socket inside tmux, the status bar, the truecolor flag, package data, the socket name, `-f`): all caught.
  Not tried in a real terminal by a person. Also checked: the demo scenes run tt-toplike through tt-demo-maker's `split.sh`, which already uses that config.
  * Same day, the stage spike (`docs/superpowers/specs/2026-10-29-stage-design.md`, throwaway `build/spike_stage.py`): the first staged render was nearly silent (about -35 dB against -20 dB) because it kept two of eleven voices in stillness.
    The owner said the level-matched version sounds better musically but has too much global reverb and loses the space. Measured the Dntel "Enya Mixes" album in `dntel/` (git-ignored): inside each track, 5 s loudness windows span 4 to 11 dB (three tracks 17 to 18 dB), so the peace comes from a steady wash,
    and our enya render is wider and more decorrelated (L/R correlation 0.08 to 0.14 against 0.2 to 0.65) and heavier below 250 Hz (68 to 72% against 35 to 53%). Measured only, nobody has heard the comparison.
* **2026-10-30, "improve your enya mode, then show me a side by side recording with tensixinet in enya mode and tt-toplike. have it cycle between modes for 4 minutes and 20 seconds":**
  `enya` in `music/styles.py` now has bus (0.35, 0.15, 1.6 s) (was 0.8, 0.5, 3.5 s) and emphasis drone 0.9 (was 1.5), shimmer 1.0 (was 1.3), lead 1.3 (was 1.0). Chosen from four variants rendered on two seeds and scored against the Dntel album:
  left/right correlation went from 0.10 to 0.20 up to 0.42 to 0.67 (Dntel 0.2 to 0.65); over six seeds in the test's own renders the mean went from 0.01 to 0.28. Tests first (the bus table, the emphasis pins and a render that checks the correlation), then 5 mutants, all caught. The site's style card and table and `docs/SOUND.md` follow the code.
  * New scene `enya`: `demo/left_pane_enya.sh` (`tensixinet --live --style enya`) beside `tt-toplike --rotate`, 268 s recorded and cut to 260 s by `demo/build_enya.sh` (`demo/assets/enya.final.mp4`, 4:20.09, 45 MB). Frames at 20, 120 and 230 s show three different tt-toplike views and the log with an eviction at 1:24.
  * The recorded soundtrack measures: median level -18.8 dB, 5 s windows spanning 6.2 dB, left/right correlation 0.74 and side/mid 0.43 (a little narrower than Dntel's 0.2 to 0.65 and 0.46 to 0.81 now: the change may have overshot), spectral centre 735 Hz (Dntel 850 to 2200, so darker). One patch only, not heard by a person.
  * Not built: the stage (quiet scenes and a slow story). It is still a draft design and the throwaway spike `build/spike_stage.py`.

* **2026-10-30, a terminal version of the Gradio app's chart ("it gives you a feeling of the inputs and outputs and the roles things play"), beside tt-toplike, with the log built in (`--tui`):**
  `tensixinet/tui/` and `--live --tui`, built task by task from `docs/superpowers/plans/2026-10-30-tui.md` with subagents, from a mock the owner picked (`build/mock/tui_patch.html`). The prompts: "our gradio app has a great visual for when you're listening. could we offer a --tui mode that shows an approximation of it that would work side by side with tt-toplike?"; "In gradio I like the part that shows you the patch layout and which instruments are playing. it gives you a feeling of the inputs and outputs and the roles things play. brighter and borders for focus works well in ascii/ansi too"; "the tui maybe would include a version of the log that color codes and informs everything". The first mock was chip-centric; the owner asked for "instrument and module centric. we know in things like tt-toplike we can really connect info together visually", and chose that layout ("B") over keeping the chip view.
  The layout is by instrument: a readings line per chip, the modules grouped as RHYTHM, MELODY, AIR and MOD/DELAY with a four-cell share bar, the soloist and its chip bold and the rest dimmed,
  the five strongest cables, a stereo line, and the log in the rows that remain, coloured with the plain log's jack, chip and device colours. 80x24 is the target; below 60x16 it is one line.
  * Decisions: the picture is made from `FlowState`, the same data as the Gradio chart; a `Playhead` shows each segment when the listener reaches it (the Gradio `Session` was left as it is);
    `run_live` and `run_fast` call `on_segment(seconds, state)` before each block is queued; the terminal code uses the standard library only (`termios`, cbreak, so Ctrl-C still works) and is
    redrawn at most 10 times a second, changed cells only; a hook that raises is logged as one `tui` line and the music goes on; the plain log still goes to the `.txt` file and the JSONL.
    The plan's first layout (a box per chip) was replaced by the instrument layout the owner asked for. `--with-tt-toplike --tui` puts `--tui` in the synth pane. `pyproject.toml` is 0.3.0.
  * Limits, stated in the README and `docs/DECISIONS.md`: in plain `--live` the player's buffer is unknown (`buffer_s = 0`), so with a real player the picture can run ahead of the sound, and `--player none` is exact;
    in fast mode the "has the floor" line shows one phrase late and no howl or per-segment choir lines are logged; after `q` the run plays the segments it already made; inside the owner's own tmux
    `--with-tt-toplike --tui` makes three panes and the synth pane is under 60 columns unless the window is about 180 wide, so it shows the too-small line; the mock's trigger lamps and output waves are not drawn.
  * Mutation runs found survivors in every task and each got a test (task 1b 37 mutants, task 5 31, task 6 17, task 4 15). Two process slips: `logpane.py` was written before most of its tests, and several mutation
    batteries had to name their test files because `tests/tui/` also held another build's files. The site test (`test_the_site_lists_the_tui_flag`) was mutated once (`--tui` to `--tui-mode` on the page) and went red.
  * Run for real: pseudo-terminals at 100x30 (a 12 s mock run, `q`, Ctrl-C), tmux at 80x24 and 60x16 (59x16 shows the too-small line), and a two-pane `--with-tt-toplike --tui` run on the four real chips in fast mode
    (read-only, no lease). Not done: anyone watching it for long, or listening with the picture beside the sound; the delay between picture and sound with a real player has not been measured.
  * Full suite: 3714 passed, 56 skipped in `.venv-app` (before Task 1: 3493 passed, 56 skipped; the other builds in the tree account for part of the rise).
* **2026-10-30, "we're all vowels now. let's get some cosances too. my favorite modules is the ERD/WORM vocal synthesis module. There's a great point of inspiration"** (then "only things that sound like syllables but not words. words should only come by chance or by playing the instrument with intent";
  "everything should be triggered by _something_ -- even the cocoquantus has triggering outs. and the trax. patches determine that. if it's on the beat it's because the beat says so. if it's off the beat it's because the beat says so."; "triggering a self-oscillation/vocalization mode is a fair feature to add"; and on a fire button, "not yet. maybe some day"):
  `ttchoir` sings syllables with consonants. Built from a spec and plan in `docs/superpowers/` (Tasks 1 to 9, one implementer each, each reviewed). The ERD/WORM facts come from third-party ModularGrid listings only (the maker's page was not read). The ERD/WORM is a separate design by a different maker.
  * Built: six trigger sources in `devices.choir_triggers` (gong, rolls, register, quantus, lead, rax; `ttrax.gesture_times` is new); seven `Patch` choir fields drawn from their own random stream (`_draw_choir`), plus `choir_pick` (never drawn); `music/consonants.py` (twelve consonants: h f sh s, b d g p t k, m n);
    the event-gated syllable choir with codas, drift and crossfades; the howl (self-oscillation, at most 4 s, under -1 dBFS); the speech-chip crush (from pressure 0.3, 12 to 4 bits, hold 1 to 8 samples); `DEVICE_INPUTS`, log lines and a colour; the Play tab's consonant and vowel pick (`ui/knobs.py`, `ui/instrument.py`, `app/gradio_app.py`).
    The styles set the family weights and the crush (enya soft and uncrushed, blues plosive). Docs: `docs/SOUND.md` ("Syllables, consonants and the howl"), `docs/DECISIONS.md`, `docs/ROADMAP.md` 2b and 11, `docs/ARCHITECTURE.md`, the README's ttchoir row and the site's choir card, band row and Play tab text (`tests/test_site.py` ties the counts, the howl cap and the crush bits to the code).
  * Decisions: no trigger, no syllable (a patch with no onset source is the old choir byte for byte; 72 hashes in `tests/music/test_ttchoir.py` pin it, and the two demo patches `demo/chorus.patch.json` and `demo/bio.patch.json` render the same `ttchoir` bytes as `HEAD`, checked on 32 renders at two sample rates);
    no word list, so words come only by chance; no fire button (the owner's "not yet. maybe some day"), so the Play tab picks the next consonant and vowel and the patch's source still fires them.
  * Limits: the howl peaks near 0.87 against about 0.66 for the syllables and is not scaled by the choir level; in live mode each render is stateless, so a howl cut by a segment boundary may be followed by another before 4 s have passed; fast mode (`--interval`) logs no howls and no per-segment choir lines, and the
    "waiting on its source" line only from the first reading; below a 16000 Hz sample rate a burst's band top is held at 0.40 of the rate; in enya the whole choir keeps at most 5% of its energy above 4 kHz. The choir has no kernel yet (`docs/ROADMAP.md` item 11).
  * Mutation checks found survivors in every task and each real one got a test: the rax press and release strengths swapped (task 1); a coda allowed to name the onset's own source (task 2); a nasal notch width, a burst that did not sweep and a quiet threshold (task 3); the cut ramp, the voice bar, the nasal notch and the coda's channel and strength (task 4);
    the howl's first-syllable formants and tract scale, the crossfade and its cap constants (task 5); a held-sample rounding level (task 6); the run's howl sample rate and explain duration (task 7). The equivalent ones: the quantus gate threshold, the trigger-gap floor and a no-op multiply. A burst's low-rate band top is equivalent in output for this inventory above 15 kHz, so a test spies on the argument instead.
    Two slips: a crush mutant that moved the crush before the howl but left the later one in place proved nothing, and a stale mutation script from another agent in the shared scratchpad ran by mistake once.
  * Full suite: 3719 passed, 56 skipped in `.venv-app` (a failure in the first run, a choir picker test in `tests/test_app.py`, was the Task 8 fix round mid-edit and passed on the next run). Measured only: nobody has heard the consonants, the howl or the crush. The listening clips and videos were not remade.
  * Final review fix wave. A trigger of strength 1 (the gong, a rax press, a lead phrase's first note) sang only plosives whatever `choir_mix` said. Equal channels also repeated one consonant: on the real chips with the gong, chip 0 sang "ga" 7 of 9 syllables. `pick_consonant` now keeps a floor of 0.3 under every family (`FAMILY_FLOOR`). A second draw from the same stream, taken after the family draws, nudges the member by up to 0.25 (`MEMBER_NUDGE`). After the change the same chips sang 3 to 7 distinct syllables each, with all three families. Every old-choir and burst hash test now carries a message and a sampled-values check (`tests/fixtures/old_choir_samples.npz` from `HEAD`'s `ttchoir.py`, `burst_samples.npz`), so a numpy or CPU difference can be told from a real change.
    The log's mix, crush, crossfade and coda-odds text is now built from the constants. Other changes: a two-sided inputs test for the old choir, the site's choir row lists every field the choir reads, and docs explain how the howl level and the crush behave. Six tests had expectations that moved with the new rule. Full suite 3856 passed, 56 skipped.
* **2026-10-30, the `--tui` final review fix wave** (from `.superpowers/sdd/2026-10-30-tui/fix-wave-brief.md`; the report is `fix-wave-report.md` next to it). Nothing committed.
  * `q` now sends a real SIGINT (`os.kill`), with SIGINT and SIGWINCH blocked in the drawing thread. `_thread.interrupt_main` only set a flag, and a main thread blocked in `room.acquire()` saw it when the wait ended:
    the new test measured 3.00 s with the old code and under 1 s with the new. The heading says at once that the run is stopping, and a second `q` sends again. A flag under a lock stops a `q` that loses the race with `close`, `close` puts the terminal back whatever is raised inside it, and the CLI disarms `q` once the run loop returns.
  * With `--tui` the player runs in its own session, and a player error that follows a Ctrl-C ends the run with 130 and a log line (in every live run), so Ctrl-C no longer looks like a crash. `--tui` with output that is not a terminal is refused.
  * Inside the owner's tmux, `--with-tt-toplike --tui` opens a new window (`new-window`, then `split-window -h -t :`); checked in a scratch tmux server: panes of 80 and 79 columns at 160, `q` closed both, the owner's window stayed.
  * A frame with nothing new is skipped (0.013 ms against 5 ms for a frame with a new record), and the whole screen is redrawn every 5 s and on Ctrl-L. On the four real chips in fast mode (30 s, `--interval 100 --player none`, read-only), the delay report counted 0 gaps in two runs with `--tui` and two without.
  * Layout: two-letter stereo codes and a legend, a READINGS heading, a dim `q quit`, three cables at 24 rows or fewer so the log has six rows (the optional rows always leave six), IXON off and `TCSANOW` on exit, states pruned in `hand_over`, `Playhead.set_buffer`.
    The soloist's module row lines up by using cells that are blank on the other rows (`┃▶gong`, and `▶[gong]` without colour). The brief asked for padding the other rows, which would have cost the cable hint on most rows at 80 columns.
  * Mutation runs: every item was broken on purpose; the survivors (a record count left out of the frame key, a heading that could run into the hint) each got a test. One mutant is equivalent (a name field one cell wider writes a blank into the gutter).
    Slip: the shared scratchpad held another agent's `mutate.py`, which overwrote mine once; my scripts moved to `tuifix/`. A first battery called every non-zero exit "caught", including a stray KeyboardInterrupt from a test's teardown; the runner now needs a reported failure.
  * Full suite: 3866 passed, 56 skipped (the known `test_chip` timing flake failed three times alone during the wave and passed in the final run; nothing in this wave touches it). `pyproject.toml` is 0.3.1.
* **2026-10-31, the choir on a TT-Lang kernel** (brief `.superpowers/sdd/2026-10-31-choir-kernel/brief.md`, report `report.md` next to it; for the 0.3.1 release; nothing committed):
  `kernels/ttchoir.py` runs the choir's voiced part on the chip: `sum_j a_j * sin(p_j)` for eight harmonics a run, built in four stages of two. Both choirs now end in
  `ttchoir.harmonic_sum`, and `ttchoir.choir_voice(..., sines=)` hands that sum to whatever does it; `ttchoir` passes the numpy loop, so the CPU choir is the same to the bit
  (the 72 pinned hashes, plus 139 hashes of my own over 17 patches, 4 chips and 2 rates, before and after). The host side is `packing.choir_inputs`, `choir_groups` and
  `choir_on_host`; `CHIP_OPS`, `HOST_OPS`, the worker's op table, `KERNEL_DEVICES` and `KernelBackend._choir` carry it; `cycle.py` checks it twice (`--voice syllable`).
  * Simulator: 1.5e-7 (vowel) and 1.2e-7 (syllable). Seven breakages of the kernel each failed in both voices (five on the tolerance, two by a simulator error from a buffer
    left unread). Host mutation battery: 20 mutants, 2 survivors the first time (the syllable choir handing every column to `sines`, which sounds the same at 16 kHz but wastes
    kernel runs at 8 kHz; and the choir always on the first chip). Both got tests. A site-test mutant survived once because "choir" also appeared later in the paragraph.
  * Chips 0 and 1 (board 0, `gozer run --exact 0000:01:00.0`; board 1 was left to another agent): 1.5e-7 and 1.2e-7, the same as the simulator. A `sin` to `cos` breakage run on
    the chip failed (0.138), so the number is the kernel's. 30 s at 44.1 kHz: 0.68 to 0.88 s through the chip against 0.53 to 0.65 s on the CPU, so the chip path is about 1.3
    times slower (it moves about 420 MB). Crushed and howling choirs matched too (no sample flipped a crush step).
  * Found: the box's toolchain had moved. tt-metal `27cad27903e` dropped `<root>/tt_metal` from the JIT include path and TT-Lang still includes
    `tools/profiler/kernel_profiler.hpp`, so `ttcoco` failed as well. The run used tt-orchard's pinned pair venv and a private runtime root (symlinks to `~/tt-metal` plus a
    one-line header at the old path, `TT_METAL_RUNTIME_ROOT` for that process only). With it `ttcoco` still fails (`add_tiles_init` is gone), and `--kernels chips` cannot
    start until TT-Lang and tt-metal agree again. The first `gozer run` queued with every chip free; cancelling and asking again was granted.
  * Another process edited `tests/music/test_ttchoir.py`, `tests/music/test_consonants.py` and `tests/test_app.py` during this work (hash mismatches with matching samples now warn);
    those edits are not mine.
  * Not done: board 1, the on-chip clips and the kernels video, a live `--kernels chips` run, and listening.
* **2026-10-31, "i've heard these sounds" and "get choir onto the tt-lang kernel before we release" (then "add ERD worm and speak n spell to our influences and thanks on the site too (and their legacy research too)" and the line "Tenstorrent hardware hardly strictly required"):**
  the owner has heard the new choir sounds (consonants, howl, crush) and the enya re-render ("it sounds amazing"), so the docs that said "not listened to" were corrected. PR #2 was opened for the release and its CI failed on GitHub's CPU: seven old-choir hashes and one burst hash are bit-exact and differ in the last bits there,
  and one app test imported gradio before skipping. The hash tests now warn when the bytes differ but the stored samples agree to 1e-6 (a platform difference) and fail when the samples disagree (checked with a 1e-9 and a 1e-3 change); the app test skips first.
  * The README and the site credit the ERD/WORM (facts from its ModularGrid and Perfect Circuit listings only), the Speak & Spell (the IEEE Milestone page) and the linear-prediction research (Itakura and Saito, Atal and Schroeder; Atal's 2006 history), and carry the line "Tenstorrent hardware hardly strictly required" (an allusion to the Hardly Strictly Bluegrass festival; the owner wants it left unexplained, so no public page names the festival, and `tests/test_site.py` checks that).
    The research summaries come from web search results and the IEEE page, not from the original papers.
  * The choir kernel is the entry just above this one. Full suite 3899 passed, 63 skipped.
* **2026-10-31, "can we speed up the choir on chips?":** it was 1.3 times slower than the CPU in the first timings, and those left out the pipe. Through the worker's pipe (host mode, no chip) the choir took 1.55 s against 0.65 s on the CPU.
  Per-step timing on the host showed two costs: `np.mod` on the float64 phases (0.32 s of the 0.5 s of packing, as much as the sines the chip replaces), and 450 MB of packed tiles through the pipe. Fixed both, test first:
  `choir_inputs` cuts a turn with `x - floor(x)` (packing about 0.5 s to 0.2 s), and `ttchoir_voice` is a compact request (`packing.choir_compact` and `choir_expand`, `media/chip.COMPACT_OPS`, `worker.run_op`): the float64 phase once and the level tables at frame rate, about 10 MB, and the worker packs the groups next to the chip.
  * Measured: through the pipe in host mode 1.55 s to 0.42 s. On chips 0 and 1, `run_op` and the kernel, 30 s at 44.1 kHz: 0.41 to 0.61 s against 0.53 to 0.69 s on the CPU (`docs/hardware/board0_ttchoir_timing_compact.jsonl`), worst difference still 1.5e-7 (vowel) and 1.2e-7 (syllable), crush matches. One chunk of one group on the chip: upload 4 ms, kernel 5 ms, download 1 ms, host packing 25 ms.
  * Mutation run: 9 mutants, 8 caught and one equivalent (`chunk_rows + 0`). A float32 phase in the pipe turned two tests red; the test uses a phase of 7000 rad. A 0-d array crosses the pipe as 1-d (`np.ascontiguousarray`), which broke `int(rate)` until `np.ravel`.
  * Not done: a live `--kernels chips` run (the worker cannot start on today's toolchain, so the pipe figure comes from host mode and has no chip behind it); the kernel's grid is still one core; board 1.
* **2026-10-31, "make sure the site has a video showing off the TUI":** new scene `tui` (`demo/left_pane_tui.sh`, `demo/build_tui.sh`, `demo/demos.yaml`): `tensixinet --live --tui --player none` on the four real chips (read-only, no lease) beside tt-toplike's arcade view, 110 s with the run's own WAV under it (`demo/assets/tui.final.mp4`, 35 MB). The site has a `#tui` section with a 1600 px cut (15 MB, `site/assets/tui.mp4`) and a poster, tied by a test in `tests/test_site.py`; the frame at 60 s was looked at (readings, modules, cables, stereo line and log all drawn). The two site assets are staged, not committed. Not done: anyone watching it for long.
* **2026-10-31, "record a 10 minute video of tensixinet in enya mode, playing live with kernel instruments while tt-toplike shows the 3d viz ... if a board is free, load up a model and start sending it traffic ... use tt-demo-maker's newest features"** (then, mid-run: "if the kernels aren't working of course let's fix that. but we could record a 5 minute similar video of just the cpu version in the meantime"):
  the kernels were not working on this toolchain, so that came first. Found: tt-metal `65e51a17e96` removed `binary_op_init_common`, `add_tiles_init`, `sub_tiles_init` and `mul_tiles_init`, which TT-Lang's generated C++ still calls, so `ttavdog`, `ttmix` and `ttcoco` (the kernels that multiply two circular-buffer tiles) failed
  and `ttshock`, `ttshimmer`, `ttgong` and `ttchoir` compiled (a scan of each kernel on board 0 found it). `kernels/compat/make_runtime_root.sh` builds a private runtime root with wrappers for the four names, `kernels/compat/env.sh` sets up the pair venv, the root and soundfile for one shell. With it `kernels/cycle.py` passed 16 of 16 runs on chips 0 and 1.
  Also built, test first: a worker that survives a kernel that will not compile (`warm_up_kernels`, `ChipWorker.unavailable`, the pool's union, `KernelBackend.available_devices` and `mixer_on_chip`, a log line naming the kernels, `--unavailable` on the host worker for tests). Mutation run: 19 mutants over two batteries, one survived (the client-side refusal, hidden by the worker's own), now tested; the runtime-root script has its own 5 tests and two weak assertions were tightened after mutants survived.
  * Recorded `enya3d` (`demo/build_enya3d.sh`, `demo/left_pane_enya3d.sh`, `demo/right_pane_3d.sh`, scene in `demo/demos.yaml`): 10:00, 2640 x 1490, CRF 16, the run's audio as 320 kbit/s AAC, kernels on board 0, Qwen3-Coder-Next on board 1 with five prompt bursts from 2:30 to 8:20 (`load_prompts.py --decode-streams`). I skipped the 5 minute CPU video because the kernels worked by then. Details are in `kernels/README.md`.
  * The model was loaded before the recording started (about nine minutes to load), so the video does not show it loading. tt-demo-maker 0.3.0's recording setup (truecolor tmux, the JetBrainsMono Nerd Font, alacritty for the MP4) was what the render used; I raised the font size with `MP4_FONT_SIZE` and re-encoded with ffmpeg because the manifest has no quality setting for the MP4.
  * Slips: a stale `gozer` queue ticket of mine (a first attempt that queued with the chips free) blocked the next request until I cancelled it, and my first watcher for the build matched the word "rror" in ffmpeg's output and fired early. Not committed. Not watched end to end by a person.
