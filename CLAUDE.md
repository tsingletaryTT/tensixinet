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
