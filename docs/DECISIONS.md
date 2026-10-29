# Decisions

Why the code is the way it is. Each item says what was chosen and what the evidence was. Change one only with a test that shows the new behaviour.

* **Chip count is never assumed.** Voices, devices, GDDR channels and log text all come from
  the telemetry. Devices are slots, and there are always at least as many slots as devices
  (`max(chips, 11)`). Slot `s` goes to chip `s % chips`. A 1-chip box plays all eleven devices,
  and a box with 10 or more chips plays one device per chip. New devices go on the end of the list so slot numbers do not move.
* **Players trade solos.** Each turn lasts two bars at the average tempo. One slot is featured and
  the rest drop to 35% volume, with 0.25 s fades. The soloist is drawn in proportion to the
  power its chip draws, and never repeats back to back (`music/turns.py`). A busy chip solos more.
* **The lead changes with the system.** Each bar reads the next GDDR channel and compares it
  with the average channel. Hotter than average shifts the motif up (1 scale degree per degree C,
  at most 4). Cooler inverts it around its first note. Within 1 C it repeats. Flat readings loop.
  Only the snapshot varies, so the variety comes from the channels. Live sampling (see the
  roadmap) would let the lead respond to change over time.
* **The devices follow the instrument guides.** The first versions were impressions from memory. After reading
  the Sidrax, Cocoquantus and Plumbutter & Rolzer guides they were rebuilt around what the circuits do
  (`docs/CIRCUITS.md` has the full list). The reasons, one by one, follow.
* **ttrollz lanes are rolls on barres.** The guide says the Rolzer's barres run at about 2:3 of each
  other, capacitors are good to 20% and the rolls are never locked, so the old lanes of 3, 4, 5 and 7 pulses per bar
  were the opposite of the circuit. Barre 2 (84 BPM) plays at the chip's tempo so the lanes keep about the old density.
* **One clock drives the gong, the shimmer and the drone.** On a Plumbutter one trigger can ring the Gongue, the
  Ultrasound and the AVDog together. `strike_times` gives the strike moments and the three devices read the same
  ones, so each of them lists `aiclk`, `heartbeat` and the identity masks in `DEVICE_INPUTS`. The test that checks
  `DEVICE_INPUTS` renders 15 s so it can see a second strike.
* **The gong's trigger response is a blocking window.** A strike needs a clock pulse after the window. The window
  is two bars times 1 to 1.375 by chip identity, which makes the strikes sit on the chip's rhythm and makes chips
  divide their clock differently. This changes the strike times of any chip whose identity picks a longer window.
* **ttrax is seven barres by default, and 4 or 14 by the patch.** The real instrument comes in three sizes (Tetrax,
  Sidrax, Quatrax); `patch.rax_bars` picks one, drawn uniformly from the seed, with the Sidrax default so an old
  saved patch with no `rax_bars` field still plays as it always did. Frequency is bounce over bounds (clamped to a
  quarter to eight times, so a closing window cannot alias), and the bounds of a released barre are the next barre
  and the one after (`glitch_bounds`), a rule chosen because it reproduces the Sidrax guide's own table exactly for
  seven bars; Tetrax and Quatrax have no guide here, so it is this code's own extension, not a transcription. The
  press and release gestures come from the ttrolzer barres, whose guide-given tempo table has seven entries; past
  that (Quatrax's fourteenth barre) the table wraps rather than indexing off the end. Pressure is the chaos knob.
  The output is mono, so the guide's stereo press and release channels sum. `sidrax_bounds` (coolest channel sets
  the lower bound) is gone because the walls are now per barre.
* **ttquantus notes come from the ttquantussy.** The guide says the castle is deterministic,
  that low prime ratios between petals make a pattern that repeats, and that the buffer is 8-bit with the clock as
  its sample rate. The seed only nudges the ratios by up to 3%, so patterns drift. `sample_hold` is gone.
* **The Tensix holes are a rhythm.** The shift register has one step per Tensix column and fires where a column is
  harvested. A chip with every column enabled is silent, which is correct for it. Every live chip has two holes. The
  snare noise is rolled off above 4 kHz because raw white noise put the device's spectral centre near 11 kHz.
* **The Ultrasound's aliasing is not modelled.** It needs ultrasonic signals, and a 44.1 kHz render cannot hold them.
* **The gong is inharmonic.** Partial ratios are 1, 2.76, 5.40 and 8.45, each doubled and detuned
  0.7 Hz. Its ring is about a third of the strike interval, so a faster clock rings shorter.
* **The ultrasonic shimmer is two tones in 12 to 20 kHz**, shrunk to fit below 0.45 of the sample
  rate. Temperature sets the pitch and GDDR spread sets the beat rate (0.2 Hz per degree C). At
  44.1 kHz the top is 19.8 kHz. Use `--sample-rate 96000` to go higher.
* **A patch holds every drawn choice, and the default patch is the old constants.** That makes
  "nothing changed" a test (`render_chorus(..., patch=DEFAULT_PATCH)` equals the render without a patch).
  The system decides the scale (GDDR spread) and the gong tuning (memory-to-ASIC temperature ratio); a
  seed decides the motif, gong beat, shimmer band, turn length and floor. The seed is a SHA-256 of all
  readings plus entropy, so any one reading changing gives a new patch. The bucket rules (3 C per scale,
  ratio times 10 for the tuning) are arbitrary but deterministic; their job is to move with the machine.
* **The patch file is one line per field,** so it diffs well and can be edited by hand. `load_patch` refuses
  a file whose gong partials do not start at 1.0, whose amplitudes do not match the partials, whose band is
  not low-to-high, or whose turn length or floor is out of range.
* **With no chips the CLI draws no patch,** so `run` still reports the clear "no chips" error.
* **Turn fades use a running-sum average.** `np.convolve` over an 11,025-sample window made a 24 s
  render take minutes; the running sum gives the same numbers in milliseconds.
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
* **Every `tt-smi` reading that varies now reaches the sound.** Voltage detunes the drone, PCIe usage sets its voices,
  ARC and AXI clocks scale ttrax and ttquantus, ethernet links add ttrollz lanes, the heartbeat places the first gong
  strike, and the harvesting and link masks give each chip an identity. New `Telemetry` fields have neutral defaults, so
  a source that does not report them plays as before.
* **Ethernet links:** the code reads the low 16 bits of `ETH_LIVE_STATUS` as one bit per port, bit N set when port N
  has a live link. tt-toplike reads it the same way. On a four-chip box the value changes while an agent is using
  the chips (one chip went from 12 links to 2 between two snapshots), so it is a live signal.
* **Pressure is the closer of two ramps** (temperature up to the throttle point, power up to the TDP limit). The clock
  is left out because chips lower it under pressure, so a high clock does not mean a chip is near a limit.
* **Chip identity is a hash of two masks, reduced to small numbers.** Three lead offsets and 21 gong detunes mean
  two chips sometimes match. A wider range would sound more different but less like one ensemble.
* **Errors are an event layer.** A new device would have changed the slot count and every existing render at the time.
  Events are added after the turn gains, so they sound whoever has the floor, and all three counters are 0 on a healthy
  chip, so the layer is silent.
* **`busy` is None when the snapshot has no process list,** and adds nothing to the turn weight. `tt-smi` lists itself
  on every chip, so it is not counted. An idle chip is not turned down below its plain power, because a quiet box
  would otherwise be all drone.
* **The host description feeds the seed** (hash of host name, kernel, driver and memory) in the real path only, so the
  same chips on another machine draw another patch. `--patch` replay does not depend on it.
* **Rate of change comes from live mode.** One snapshot has no rate, so `Telemetry.temp_rate` and `power_rate` default
  to 0 and an offline run behaves as it did. In live mode `music/live.Tracker` fills them in, over a baseline at least
  2 s old (a 0.3 s gap between two readings turned a 0.3 degree wobble into a "heating" of 1 C/s). Only ttrax reads
  them: its chaos knob is pressure plus the faster of the two rates, each scaled so 1 C/s or 25 W/s is the full amount,
  capped at 1. The other devices keep reading pressure.
* **DDR status:** the code treats the low byte as four 2-bit fields and mutes a lane at 0.
  tt-smi's source says newer firmware puts BIST bits in `[31:16]`. The meaning of values 1 to
  3 is a guess and should be checked against firmware docs.
* **The chorus keeps its absolute level, and `run` normalises the finished file.** `render_chorus` leaves power as
  volume, and `tanh` in the mixer keeps it inside full scale. `run` then applies one gain over the whole render to
  reach -18 dBFS RMS, limited to a peak of -1 dBFS and at most +20 dB (`music/loudness.py`). One gain keeps every
  chip's level against the others and the swells and gaps, so the power-to-volume mapping survives. A quiet box
  peaked near 0.2 and a loaded one near 0.4 before this. `--no-normalise` writes the render as it came.
* **Live mode renders one turn at a time, from a fresh reading.** `render_chorus(..., segment=k, soloists=[slot])` gives
  each turn its own seeds (a fixed stride per segment) and a named soloist, and `live.py` crossfades the turns together
  (`music/live.Stitcher`, 0.4 s, equal power). A turn is the patch's bars at the mean tempo, held between 2 s and 10 s.
  The last turn takes whatever would otherwise be a sliver, so it can run a little long. The sound lasts the duration
  plus the 0.4 s tail.
* **Each turn restarts the devices' clocks.** That was the cheapest way to render in turns without touching all eight
  devices. The costs: a gong that is still ringing at the end of its turn fades over the 0.4 s crossfade (offline it rings
  for up to ten bars), rolls and strikes start from the top of each turn, and oscillator phases do not carry across. The
  clean fix is a start-time offset in every device so a turn continues the last (roadmap item 3).
* **The loop reads the chips only when a turn has finished playing.** A second thread writes audio to the sink. The sink's
  writes block at the speed of the music, and a semaphore lets the loop get two turns ahead, so each turn is rendered from a
  reading at most one turn old when it starts playing. Without the semaphore the loop would render the whole run at once.
* **The player is a command line program on a pipe.** `paplay`, `aplay` or `ffplay` takes raw 16-bit mono on standard
  input. `sounddevice` stays out of the package (the hygiene test forbids it), and a player process needs no
  Python audio library. `--player none` sleeps for the length of each turn so a run still goes at the speed of the music.
  `--out` records what was played.
* **New errors play once.** Memory error and thermal trip counters count up for the life of a chip. In live mode a turn
  plays only the increase since the last reading, so an old error does not tick in every turn.
* **A failed reading plays the last good one again,** with no rates and no new errors, and the log says so. A reading with a
  different number of chips starts the rate baseline over. A first reading that fails stops the run.
* **Cables act on rendered voices, drawn from the patch seed.** The devices have fixed signatures and render independently, so a
  cable cannot reach their internals without rewriting them all. Acting on the finished voice gives the guides' degrade, mutate and
  inflate, and keeps the devices testable alone. The wiring comes from the patch seed (not the segment seed), so live mode does
  not rewire between turns, and no `Patch` field was added, so saved patches still replay (an old patch gets cables from its seed).
  `inflate` and `fold` pass a level unchanged below 0.9 so that a silent source or a depth of zero leaves the target exactly alone.
* **Live loudness is a running gain.** `LiveGain` moves the gain half way (in the log domain) toward each turn's own gain,
  ramps in a straight line across the turn, applies the peak limit at once and holds the gain through silence. A whole
  render still gets one gain.
* **One EQ on the final mix, as a minimum-phase FIR.** The lows are quiet on small speakers and the shimmer is harsh or
  inaudible, so `music/eq.py` shapes the mix after mixing and before the loudness gain. An FIR built from a curve with the
  cepstral method needs only numpy (scipy would be a new dependency), has no pre-ringing and almost no delay, which live
  mode needs, and runs on chunks with carried state. A linear-phase design would have delayed every live turn by about
  0.1 s or more. The end of a live run drops the filter's own ring-out because the last turn's tail has faded to nothing.
  Without the loudness gain, a render whose lows were lifted past full scale is scaled down to fit. The curve is a
  starting point chosen by reasoning, not by listening.

## The Play tab

* **Three scopes, one engine.** A device, a module and the whole system all go through `render_chorus`: `only=[...]` for a device or module, no `only` for the
  system. A module is a fixed group of devices (`ui/scopes.MODULES`), so the page does not need a patch editor. The system keeps cables, the delay bus, turn-taking and
  eviction, which `only` switches off.
* **The performer's hands are the readings.** The devices read chip telemetry, so the Play tab has knobs for the readings (`ui/knobs.py`) and does not add a
  separate control path into the devices. Current and board power follow the power knob, as `mock_telemetry` has them, because the mix is weighted by current. Each
  chip keeps its masks (its identity in the music) however many chips are played, because the masks come from one `mock_telemetry(4)` call.
* **The sound is rendered in phrases of 2 or 5 blocks.** The first version rendered every 1.2 s block on its own. The gong lives on chip 1 and its first strike is up to half a two-bar
  interval in (2.2 s at 110 BPM), so a bare 1.2 s render of the gong was silent, and the test that played every device caught it. A block is now a slice of a phrase. A change (a knob, an error,
  a trip, an eviction phase) starts a 2-block phrase, which renders in 0.15 s and is heard within a block. A phrase that runs out with nothing moving is followed by a 5-block one.
  The slowest devices are still silent for the first block or two of a short phrase, and a gong whose first strike is later than 2.4 s is silent while a slider is being moved.
* **An error, a trip or a new knob value starts a phrase; the error counter going back to zero does not.** The counter falls back the block after an error, and a phrase started by that would cut
  the phrase that carries the tick before the tick was heard.
* **Latency is about four seconds.** The page's player is kept `INSTRUMENT_LEAD_S = 3.0` s ahead of the listener. Over a 40 s run in headless Chrome that gave a smallest
  buffer of 1.7 s and no stall; a move showed in the readout 3.6 to 4.2 s after it was made. A shorter lead was not tried.
* **The Play tab draws the first tab's flow chart.** The session builds a `FlowState` for each block (`build_flow`, from the playing phrase's readings, its soloist and its eviction phases) and the snapshot carries the one the
  listener has reached, so the chart follows the sound as the readout does. A scope that plays some devices has nobody on the floor (it plays at full level, with no turn-taking), and the devices it leaves out are drawn faint
  (`DeviceView.heard`). The chart's heading names who has the floor and has no segment number or time, so the page is sent again only when something in it changes (a new phrase can change the floor).
  The chart and readout are one HTML output, so the handlers still yield three values. The readout's inner rows set their own colour: Gradio's theme colours a div's text itself, and the pitch line was dark grey on the dark panel.
* **Stop returns `gr.skip()` for the audio.** Returning `None` for a streaming audio component made Chrome load a new source and play it, so the sound went on for several seconds after Stop (found in
  the browser, on Gradio 6.30, in the first tab too). Stop and the stopped frames return `gr.skip()`, and the page's own `pause()` does the silencing. The first frame of a new Play still returns
  `None`, which starts the new stream from 0.
* **One audio stream for a visitor.** Both tabs share `Visitors`, so Play in one ends the other, and a visitor holds one of the `TENSIXINET_MAX_PLAYERS` places whichever tab it is. A setting
  changed in the tab that is not playing does nothing.
* **Cost on a small machine.** One short phrase (4 chips, stereo, system) took at most 0.15 s and a long one 0.34 s with the process pinned to two CPUs, and the same pinned to one. A visitor who
  keeps a slider moving starts a short phrase every block, so about 12% of a core. The Play tab therefore has no chip cap for the Space.

## MIDI out

* **The events come from the code that makes the audio.** The lead and `ttrollz` used to compute their notes and pulses inside the render. They now build an event list
  (`lead_notes`, `ttrollz_pulse_sets`) and render from it, and `score/collect.py` reads the same lists, so a MIDI note and a heard note cannot disagree. A second copy of the
  lead's or the rolls' logic would drift. SHA-256 characterization tests, written against the code before the split, pin the lead's and `ttrollz`'s audio to the bytes it had.
  The other devices' extractors (`ttquantus`, the gong, `ttsteam`) call the helpers their devices use.
* **The file writer is in this repository and `mido` is optional.** A format 1 file is a short header and a few chunks, so `score/smf.py` writes it with `struct` and the
  tests need no library. `mido` and `python-rtmidi` are in the `midi` extra and imported only when `--midi-port` opens a port, so the suite and every other run work without them.
  The cost is that the writer is ours to keep correct. Its tests read the bytes back.
* **The port sink never blocks the audio and stops after an error.** `MidiPortSink.add` pushes to a heap and returns. A thread sends what is due. A failing port or an unroutable score
  sets `error` once and drops everything after it. The thread does not log, because the run log belongs to the loop's thread, so the loop reads `error` after each `add` and logs one
  "the MIDI port stopped" line. A review found that the first version of the sink kept the failure silent, which left a dead port looking like a quiet one. `close` waits a bounded time, so a
  port whose send hangs cannot hang the exit.
* **A note that starts on a sounding number retriggers it.** MIDI has one note-on state per channel and number. Review found two overlapping notes of the same pitch whose first note-off
  silenced the second, so the second note-on now ends the first at that moment and the first's own off is dropped.
* **The first synth is a VCV Rack patch the code builds.** The proof of concept is a monophonic voice with a dark, resonant low-pass filter on the lead's channel, played on top of tensixinet's own sound.
  `score/vcv.py` builds the patch as data and `vcv/tensixinet-voice.vcv` is its output, with a test that the two agree, so the bend knob, the channel and the numbers cannot drift from `Routes`.
  The patch names no MIDI device and no audio output, because the ALSA name of the virtual port carries a client number that changes each run; `--device auto` writes a copy for the running port.
  The wheel's 5 V swing and the module and port numbers were measured by playing the patch in Rack (440.0 Hz for A4, 493.8 and 392.0 Hz at the wheel's ends). They were not read from Rack's source.
  The filter and envelope values are a first guess. The alternative, MIDI-CAT as in `~/code/vcv-cv-harness`, is for CCs onto knobs and is not needed to play notes.
