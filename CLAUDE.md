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
| `legacy/README.md` | The old prototype scripts (unsafe to run) |

## Run it

```bash
make test                                               # about 14 s, no hardware
python3 -m tensixinet --mock-chips 4 --duration 20      # synthetic chips, no hardware
python3 -m tensixinet "your prompt" --duration 60       # real chips + the LLM endpoint
python3 -m tensixinet --live --duration 30              # read the chips every turn and play as it goes
make demo                                               # record the split-pane demo + soundtrack
```

`python3 -m tensixinet --help` lists every option. The run log goes to `logs/` as JSONL (for
programs, every reason kept) and a text file (what you see on screen, each reason once).

## Layout

```
tensixinet/
  telemetry/   snapshot.py (tt-smi JSON to Telemetry), mock.py (synthetic N chips, optional faults)
  music/       mapping, synth, devices, ttrolzer, ttrax, ttquantussy, ttsteam, lead, events, loudness, turns, patch, chorus, live
  narrate/     explain (what and why), runlog (JSONL plus the left-bar text view)
  media/       llm (OpenAI-compatible endpoint), mux (WAV under a video), player (live audio out)
  run.py live.py cli.py __main__.py     wiring (offline and live) and the command line
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
* **2026-10-18, `--only`:** since none of the new sound has been heard, `--only NAME[,NAME]` plays only the named devices at full level with
  no turn-taking (`render_chorus(..., only=[...])`, also in live mode). A log line says what is soloed. Mutation checks found two
  survivors in the command-line parsing (spaces around commas, a list of nothing); both got tests.
