# Decisions

Why the code is the way it is. Each item says what was chosen and what the evidence was. Change one only with a test that shows the new behaviour.

* **Chip count is never assumed.** Voices, devices, GDDR channels and log text all come from
  the telemetry. Devices are slots, and there are always at least as many slots as devices
  (`max(chips, 10)`). Slot `s` goes to chip `s % chips`. A 1-chip box plays all ten devices,
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
* **ttrax is seven barres.** Frequency is bounce over bounds (clamped to a quarter to eight times, so a closing window
  cannot alias), and the bounds of a released barre are two other barres from the guide's table. The press and release
  gestures come from the ttrolzer barres, so the seven barres keep different tempos. Pressure is the chaos knob. The
  output is mono, so the guide's stereo press and release channels sum. `sidrax_bounds` (coolest channel sets the lower
  bound) is gone because the walls are now per barre.
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

