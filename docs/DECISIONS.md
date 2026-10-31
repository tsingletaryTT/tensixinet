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

## Fast sampling (`--interval`)

* **hwmon plus `tt-smi`, as tt-toplike's own hybrid backend does it.** Temperature, power, voltage and current come from `/sys/class/hwmon` (milli-degrees, micro-watts, millivolts
  and milliamps; a read takes well under a millisecond and worked while `tt-smi -s` was hanging). Everything else (clocks, GDDR temperatures, errors, trips, masks) comes from
  `tt-smi -s` on a background thread about every 1.5 s. The two orders are matched by PCI bus address, never by position. With no hwmon, a different chip count or different
  addresses, the run uses `tt-smi` alone, says so, and states no delay claim. A chip whose hwmon reads fail keeps its last reading for a while, and the first failure of a stretch
  is logged with the chip's index and bus address. After ten failures in a row (`REFIND_AFTER`) the chips are searched for again, because a reset can move a chip's hwmon files,
  and only that chip's entry is replaced, so the pairing of the other chips stands. A chip that still fails then takes `tt-smi`'s values, `fast` turns off and the note names it,
  so the delay report counts `tt-smi`'s period. A chip that reads again is logged once. Before this a reset could freeze a chip's readings for the rest of the run with no log line. `--interval` is refused with `--kernels` (the kernels hold the chips and the render budget is small).
* **Budget: a quarter second.** The owner's number. The log reports the worst case (interval, plus read-to-write time, plus queued audio, plus the player's buffer, plus the time a change
  waited for the minimum gap between phrases) and the typical case, and says when the worst case is over 250 ms. Its floor is interval + lead + player buffer. These are calculated by the run
  from its own timings and the player buffer it asked for. The delay at the speaker has not been measured: the run cannot see the sound server's own latency, the player's start-up offset or
  audio waiting in the pipe (a pipe holds up to 64 KB, about 0.74 s at 22050 Hz in stereo). Errors, thermal trips and mask changes arrive with the `tt-smi` reading, about every 1.5 s,
  and are outside the budget. The run does not change the interval to fit.
* **Change thresholds and a minimum gap.** A phrase starts when a chip's temperature moves more than 0.5 C, power more than 3 W, voltage more than 0.01 V, aiclk more than 0.5 MHz, or a
  GDDR temperature more than 0.5 C, or when the chip count, the GDDR channel count or an eviction phase changes. The clock and GDDR figures are whole units in `tt-smi`, so any step of one
  counts. A change starts a phrase at most once a second (`MIN_GAP_S = 1.0`), and a change held back by the gap is heard when it passes. All of these are starting values to tune by ear.
  They are not yet tuned: nobody has listened. On the four real chips the readings cross them often enough that a phrase starts about every second, so the gap applies most of the time: a
  change is heard up to about 1 s plus the delay below after it happens. The session records how long the change it plays waited (`held_s`, from the first reading that crossed a threshold
  to the reading that starts the phrase), and the `delay` line adds that to its worst case. Shortening the gap would hear a change sooner and restart the gong and the rolls more often
  (next entry); that is the owner's choice and is not made here.
* **Head and body.** The plan assumed short renders. Task 8 measured a 6 s phrase at 44.1 kHz at 0.85 to 1.0 s and a 1 s phrase at 0.2 s on the real chips, and the worst delay was 2.10 to
  2.45 s. A change now plays from a 0.6 s head (`HEAD_S`) rendered on the loop, while the full phrase (short 1.0 s, long 6.0 s, `SHORT_PHRASE_S` and `LONG_PHRASE_S`) renders on a worker thread
  and takes over at the same sample offset. The head is rendered 0.15 s past its crossfade (`HEAD_MARGIN_S`), because the last 50 ms of a short render differ from a longer render of the
  same phrase (correlation 0.47 to 0.85 in that window for four of eight cases, 0.95 to 1.000 with the margin). The head and body join with an equal-gain crossfade: they correlate 0.98 to 1.0,
  and an equal-power fade of correlated signals is up to 3 dB loud in the middle. Joins between phrases keep the equal-power fade, because the devices restart there. A body that is late is
  replaced by the head's own continuation fading out, then silence, and it comes in when ready. The loop never waits. In a quiet stretch the next long phrase renders ahead once the playing one is half played.
* **Rate and lead.** The default sample rate with `--interval` is 22050 Hz (44100 Hz without it), because a head renders in 73 to 88 ms there and 138 to 150 ms at 44100 Hz, longer than a 100 ms block.
  The lead is 0.10 s at 22050 Hz and 0.18 s at 44100 Hz, with a straight line between. At 0.06 s paplay had 2 to 4 gaps of 20 to 30 ms every 5 s, because a head render was longer than the lead.
  At 0.10 s there were none. A 0.10 s margin measured the same as 0.15 s. Players are asked for small buffers: `paplay --latency-msec=20`, `aplay --buffer-time`, `ffplay` low-delay flags.
  `ffplay`'s buffer is an estimate of 100 ms and was not measured, which puts it over the budget. `--player auto` picks `paplay` first.
* **Calculated by the run** (four real chips, read-only, `--interval 100`, 22050 Hz, `paplay` at 20 ms, 30 s): worst case 220 ms (100 interval + 100 lead + 20 buffer), typical 170 ms,
  0 gaps in every 5 s report. These hold for a change that arrives when the last phrase started more than 1 s earlier; the runs were logged before the `delay` line counted the wait for the
  gap. A change inside the gap is heard when the gap ends, so up to about 1 s plus that delay. The numbers come from the run's own timings and the player buffer it asked for, and the delay
  at the speaker has not been measured. At 44100 Hz the head render outruns the block, the loop lengthens it to 200 ms, and the worst case is about 420 ms and the typical about 320 ms.
  44100 Hz is over budget.
* **When the block is lengthened, and when it comes back.** A head render at 22050 Hz takes 73 to 88 ms against a 100 ms block, so one stall from another process used to lengthen
  the block to at least 0.15 s for the rest of the run. The loop now lengthens it only after three head renders in a row outrun it (`OVERRUNS_TO_LENGTHEN`; a render on the loop that
  fits starts the count again), and returns to the interval asked for after 60 blocks in a row whose render fits that interval (`CALM_BLOCKS_TO_SHORTEN`). Each change is logged and
  carries the eviction tracker, the held soloist, the gain and the EQ over. Two overruns still make two gaps. Both numbers are starting values.
* **Eviction smoothing in seconds.** `music/evict.py` weights a power reading 0.5 when it comes at least 1 s after the chip's last one, and `1 - 0.5 ** dt` when closer. Live mode's readings
  are seconds apart, so it is unchanged. A test feeds one power curve at 1 s and at 0.1 s spacing and gets the same eviction time.
* **Rates over 0.5 s.** `Tracker(window_s=0.5, hold=True)` in fast mode (live mode keeps 2 s). `hold` carries the last measured rate on the readings between two measurements, because
  a rate exists only on the reading where a window closes, and four in five readings would have said the chip was still.
* **A new phrase restarts the rolls and the gong.** A new phrase is rendered from its own start, with a new segment number, as the Play tab's is, and a crossfade carries the old phrase's
  continuation. So every device's clock restarts with it: the drones, shimmer, lead, choir, the quantus pitch, the levels and the mix follow the change, and the gong and the rolls start again
  from their beginning. While the readings keep moving a phrase starts about once a second, so the gong and the rolls restart about once a second, and a 6 s phrase in a quiet stretch lets them
  play out. The owner's wish is the opposite: in the Ciat-Lonbarde instruments the rolls are on their own schedule, and the rest of the system can influence them and be influenced by them only
  through the links. A phrase that continues from a start offset (`ROADMAP.md`, items 3 and 7d) would deliver that, and it is the open design question of this feature.
  * Measured (a computation, no hardware): `devices.strike_times(chip, 1.0, patch, segment)` for each chip of three sets (the parsed fixture `tests/fixtures/tt_smi_snapshot.json`,
    `mock_telemetry(4, 0)` and `mock_telemetry(4, 3)`), segments 0 to 3, gives no gong strike in the first 1.0 s in 24 of 48 cases (50%) with the default patch, and in 305 of 576 (53%) with
    the patches `draw_patch(seed, chips)` draws for seeds 0 to 11. The first strike over 6 s lands 0.27 to 2.29 s in with the default patch and 0.11 to 3.67 s in with the drawn ones. The
    reviewer's first count (288 of 576) called `draw_patch` with its arguments the wrong way round, which raised and fell back to the default patch twelve times over.
* **In fast mode the player runs in its own session.** `PlayerSink(own_session=True)`, which `make_sink` asks for whenever a small buffer is requested (`--interval`), starts `paplay`, `aplay` or `ffplay` with `start_new_session=True`. Plain `--live` leaves the player in Python's group, so a Ctrl-C stops it with Python and queued segments do not play out. In Python's process group a terminal Ctrl-C (or tt-toplike's
  `send-keys C-c`) reached the player too; `paplay` exits at once on SIGINT, the tail written after it failed with a broken pipe, and the run ended with exit code 3, which kept the
  synth pane open behind "Press Enter" and tt-toplike's pane with it. Now only Python gets the signal, and `close()` still waits for the player to finish. If a player does stop
  after a Ctrl-C, fast mode logs it and still ends as stopped (exit 0).
* **The soloist is held for the patch's turn length** (`turn_seconds`) and chosen the way live mode chooses it. A short phrase started by a change keeps the held soloist.
* **`--with-tt-toplike` runs on a tmux socket of its own, with a config of its own.** `tensixinet/media/tmux-toplike.conf` turns the status bar off and asks for truecolor
  (`terminal-features ",*:RGB"`, `COLORTERM`), which is what tt-demo-maker's recordings use; without it tmux flattens tt-toplike's 24-bit gradients to 256 colours. tmux reads `-f` only when it starts a new server,
  so outside tmux the session is `tmux -L tensixinet -f <conf>`, and your own tmux and `~/.tmux.conf` are not involved (and your prefix key and bindings do not exist in it). Inside tmux we split your
  window on your server and change nothing about it. The config ships as package data (`pyproject.toml`).
* **`--with-tt-toplike` uses tmux, and a command line without abbreviations.** Each pane tags itself (`@tensixinet_role`), and when its program ends it closes or interrupts the pane tagged the other way
  (tmux resolves `{right-of}` against the window's active pane, whichever pane ran the command). A reviewer found that an abbreviated `--with-tt` would not be stripped from the synth pane's command line and the
  pane would start another session without end. The parser therefore has `allow_abbrev=False`, and a `TENSIXINET_IN_SESSION` variable in the synth pane refuses a nested start. Abbreviated flags
  (`--dur 30`) now fail with "unrecognized arguments". Undoing it means restoring abbreviations and stripping the abbreviated forms from the synth command.

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
  The demo recording runs on a private Xvfb display, a null sink and a private Rack user folder (`-u`), which `ffmpeg -f x11grab` can read, where the desktop is Wayland and a screen grab there is black or needs a share dialog. Rack Free renders there with Mesa's software GL.
  The first attempts failed because display :99 belonged to another agent's `xvfb-run` and Rack died with "Could not initialize GLFW"; `Xvfb -displayfd` now picks a free number.
  The filter and envelope values are a first guess. The alternative, MIDI-CAT as in `~/code/vcv-cv-harness`, is for CCs onto knobs and is not needed to play notes.

## Terminal view (`--tui`)

* **A `Playhead` in the terminal package, next to `Session._pending` and not in place of it.** The Gradio `Session` already holds each segment's state until the listener
  reaches its offset. The terminal view needs the same rule in a process with no Gradio, and the fast loop has no `Session`. `tui/playhead.py` holds the rule for both loops
  (`hand_over(seconds, state)` before a block is queued, `note(record)` for each log record, `wrap(sink)` to learn when audio is written). The Gradio session was left as it is,
  so the page's behaviour did not change. The cost is two copies of one idea. Merging them is open.
* **`buffer_s` is 0 in plain live mode.** The player's own buffer is unknown there: `paplay`, `aplay` and `ffplay` each hold some audio the loop cannot see. With a real player the
  picture can therefore run ahead of the sound by that buffer. `--player none` has no player buffer and keeps exact time. Fast mode asks the player for a small buffer, so it
  passes the buffer it asked for (`player_buffer_s`), and the picture follows the sound to within that estimate.
* **In fast mode the "has the floor" line is logged after the block is queued, so the screen shows it one phrase late.** `InstrumentSession` logs it when a phrase has been
  made, and the hook runs before the put. Fast mode also logs no howls and no per-segment choir lines, because the Play tab's engine does not produce them. The screen is drawn from the
  `FlowState` and is complete without them.
* **The screen is made from `FlowState`, the data behind the Gradio chart.** A test compares them. The layout is built around instruments: a readings line per chip, then the modules
  grouped by role, then the five strongest cables, then the stereo line, then the log in whatever height remains (at least 3 rows at 80x24). An early plan drew a box per chip with its
  devices under it. That was replaced after the owner asked for a layout organised by instrument, since the cables and the soloist cross chips.
* **Limits of the size.** Below 60x16 the screen is a single line. At 100 columns MOD/DELAY becomes a fourth column. Each chip has one readings line; when rows run short the
  strip collapses to four lines (the soloist's chip among them) and a `+N more chips` line. The terminal is redrawn at most 10 times a second and only the cells that changed are sent.
* **24 rows: three cables, so the log has six rows.** At 80x24 with four chips the log had four rows after the five cables and the two labels, so most reasons never showed. At 24 rows
  or fewer three cables are listed; taller, five. The optional rows below take only rows above six rows of log at any height, so a taller screen does not give the log fewer
  rows for them. When the plan cuts cables, folds and collapses, it still keeps three rows of log, as before.
* **Optional rows go first, in a fixed order.** The CABLES and STEREO labels, a legend of the stereo codes and a READINGS heading that names the readings columns take only rows left over
  above the log's floor. When rows run short the READINGS heading goes first, then the legend, then the STEREO label, then the CABLES label. A dim `q quit` sits at the right end of the
  heading when the heading fits beside it (only the patch's parts are given up for it).
* **Two-letter stereo codes.** With one letter, the lead (`L`) and ttrollz (`R`) could not be told from the `L` and `R` that label the ends of the line. Each device now has two lower-case
  letters (`ld`, `rz`, `tx`, `sh` and so on, the owner's mock used several of them), and a code stops short of the end labels.
* **The soloist's module row lines up without widening the others.** The soloist's row had an arrow and room for brackets, so its chip mark sat one or two cells right of the others.
  Padding every row by those cells would have cost the cable hint on most rows at 80 columns. Instead the arrow takes the blank cell after the bar (`┃▶gong`), and without colour the row is
  `▶[gong]`: the arrow takes the bar's cell and the brackets take cells that are blank on every other row. The readings line keeps the heavy bar for the focused chip.
* **A frame with nothing new is not drawn again.** Re-rendering every frame cost 5 to 6 ms (about 5% of a core at 10 frames a second, under the GIL), and fast mode has little headroom.
  The log rows are kept for the same records and width, and a frame is skipped when the state, the records, the size, the colour flag and the heading's notice are unchanged
  (measured: 0.013 ms for such a frame against 5 ms for one with a new record). The whole screen is still redrawn every 5 s and on Ctrl-L, because diff-only drawing never repairs
  a screen that something else wrote to. On the four real chips in fast mode (`--interval 100 --player none`, 30 s, two runs with `--tui` and two without) the run's own delay report
  counted 0 gaps in every run; the worst block renders were 57 to 82 ms without `--tui` and 74 to 117 ms with it, while another agent held two of the chips.
* **Standard library only, and cbreak, so Ctrl-C keeps working.** `tui/app.py` sets cbreak with `termios` (as `tty.setcbreak` does), which leaves signal characters on, so Ctrl-C reaches
  Python as it does in the plain log. It also turns off IXON: with it on, Ctrl-S stops the terminal's output and the exit would wait on a frame stuck in `write`. The mode is put back with
  `TCSANOW`, which does not wait for the output to drain. `q` and Ctrl-C both end the run through the loop's existing stop path, which plays the segments already made to the end.
* **`q` sends a real SIGINT.** `_thread.interrupt_main()` only sets a flag, which a main thread blocked in the live loop's `room.acquire()` sees when the wait ends (measured: an
  interrupt sent 0.3 s into a 3 s wait raised after 3.0 s; a real SIGINT raised after 0.3 s). So `q` calls `os.kill(os.getpid(), SIGINT)`, and the drawing thread blocks SIGINT and
  SIGWINCH for itself so the main thread takes them. The heading says at once that the run is stopping, and a second `q` sends again, as a second Ctrl-C would. The check that `q` may
  still send and the send are one step under a lock that `close` and `disarm` take, and the CLI disarms `q` once the run loop has returned, so it cannot interrupt the 40 s chip release.
* **With `--tui` the player has a session of its own.** Plain live mode leaves the player in Python's process group, so a terminal Ctrl-C killed it, the drain write failed and the run
  ended with exit code 3 and an error printout, while `q` drained cleanly. With `--tui` the player starts with `own_session=True`, as fast mode's does. And in every live run, a player
  error that follows a Ctrl-C ends the run as Ctrl-C does (exit code 130), with a log line saying the player stopped after Ctrl-C.
* **`--tui` needs a terminal.** With its output going to a pipe or a file, the alternate-screen codes would be written into it, so it is refused at the argument check. A run that ends with an error leaves the
  alternate screen first, prints the last five log records and the message on the normal screen, and a `--with-tt-toplike` pane then waits for Enter.
* **The screen is off the audio path.** A hook that raises is logged as one `tui` line, switched off, and the music goes on. The plain log goes to the `.txt` file and the JSONL exactly as before.
* **`--with-tt-toplike --tui` runs `--tui` in the synth pane.** Inside the owner's own tmux, splitting the current window twice gave the synth a third of it, under 60 columns unless
  the window was about 180 wide. With `--tui` it now opens a new window named `tensixinet` (`tmux new-window`, then `split-window -h -t :`, where `:` is the session's current window,
  the new one), so the owner's pane is left as it was and each pane gets half the width. The panes still tag themselves and end each other: `close` lists the panes of its own window.
  Checked in a scratch tmux server at 160 columns: the synth pane was 80 wide and tt-toplike's 79, `q` in the synth closed both, and the owner's window stayed. Without `--tui` the
  current window is split as before. Outside tmux there are two panes.
* **Known limit: a cable hint names a device and leaves out its chip.** With 12 or more chips a device is dealt to two chips, and the hint does not say which.
* **Not done.** Nobody has watched the screen for long; it was measured on pseudo-terminals and in tmux. The design mock's trigger lamps and output waves are not drawn, because `FlowState`
  has no data for them.

## The choir's syllables (`ttchoir`)

* **The owner's words.** "we're all vowels now. let's get some cosances too. my favorite modules is the ERD/WORM vocal synthesis module. There's a great point of inspiration";
  "only things that sound like syllables but not words. words should only come by chance or by playing the instrument with intent"; "everything should be triggered by _something_ --
  even the cocoquantus has triggering outs. and the trax. patches determine that. if it's on the beat it's because the beat says so. if it's off the beat it's because the beat says so.";
  "triggering a self-oscillation/vocalization mode is a fair feature to add". What we know of the ERD/WORM comes from third-party ModularGrid listings only (the maker's page was not read).
  The ERD/WORM is a separate design by a different maker. It is a point of inspiration, and nothing is copied from it.
* **No trigger, no syllable.** With an onset source in the patch the choir sings on each trigger and is silent between them. A quiet chip, or a source with rests, gives a quiet choir.
  With no source (`choir_onset == ""`) the choir is the old one on its bar clock, kept byte for byte as `_vowel_choir`, so a patch saved before syllables replays as it did. 72 hashes in
  `tests/music/test_ttchoir.py` pin that, and every other choir field is ignored in that case. `CHOIR_ONSET_ODDS` has no "none" entry, so every drawn patch has an onset source,
  and `--patch-seed N` no longer gives that seed's old vowel choir. A saved JSON patch still replays as it was.
* **No word list.** The inventory is twelve consonants and five vowels. It can spell a few words by chance, and nothing aims at any.
* **The sources live in `devices.py`, beside `strike_times`.** `choir_triggers` calls the functions the source devices already use (`strike_times`, `roll_pulses`, `step_times`, `lead_notes`,
  `ttquantus_onsets`, and `ttrax.gesture_times`, which is new), so the choir and the device it follows cannot disagree about when something happens. `ttchoir` imports `devices` and `devices`
  does not import `ttchoir`. A source is recomputed from the chip's own readings. The choir does not follow another chip's device.
* **The register source ignores the snare share.** The shift register keeps stepping whatever share of its hits the snare plays (`Patch.snare`), so a patch with a silent snare still has a
  register to follow. The choir fires on every step that fires, which keeps "use the register for more than the snare" true for the voice as well.
* **`quantus` and `lead` borrow the seed of their own chorus slot** (`SOURCE_SLOTS`: slot 2 and slot 4, tied by a test to `chorus.DEVICES`). Their rests and petals come from the device's seed,
  so the choir's triggers are the notes that device plays. The choir draws nothing of its own for them.
* **A gong strike has strength 1.0.** A strike is an event with no velocity. The trigger's strength leans the consonant: a stronger trigger leans to plosives, a weaker one to fricatives
  and nasals, and the patch's `choir_mix` is the base lean. So the gong, the beat's own voice, sings more plosives than its mix alone would give. The register and the roll pulses have no
  strength of their own and give 0.5 (`NO_STRENGTH`).
* **Every family keeps a floor, and a second draw nudges the consonant** (`consonants.pick_consonant`). The first rule weighted the families `mix[0] * s`, `mix[1] * (1 - s)` and
  `mix[2] * (1 - s)` for a trigger of strength s, and took the member nearest the channel's sharpness. The final review measured what that did on the real chips with the gong as the source:
  every syllable was a plosive, chip 0 sang "ga" 8 of 10 syllables and chip 1 "bo" 9 of 11, and an enya patch on the gong was all plosives against a plosive weight of 0.1. The owner had asked
  for range and variance. Now the weights are `mix[0] * (0.3 + 0.7 s)`, `mix[1] * (1 - 0.7 s)` and `mix[2] * (1 - 0.7 s)` (`FAMILY_FLOOR` 0.3), so a strength-1 source still leans to plosives and
  sings some fricatives and nasals. A second uniform from the same per-syllable stream moves the sharpness by up to 0.25 either way (`MEMBER_NUDGE` 0.5), held inside 0 to 1, before the member
  is chosen. The nudges are drawn after the family draws, so each family draw keeps its place in the stream. The old choir (`choir_onset == ""`) never picks a consonant and is unchanged.
* **A coda on the onset's own source alternates.** With one source for both roles the even triggers start syllables and the odd ones close them. Without that rule a closing trigger would
  always be the next onset and would never land inside a syllable, so the coda would never sound.
* **The howl runs at frame rate.** Its level follows a gain law at `FRAME_HZ` (200 a second), the rate the tract is updated at. The level changes slowly against that rate, and a loop per sample
  in numpy would be slow for no audible gain. The cut that keeps it under -1 dBFS is applied to the frame whose growth would pass the ceiling, so the level never passes it and no clamp is needed.
  A howl is capped at 4 s and one runs at a time. The cost is the pulse: once at the ceiling the level is halved and regrows about every 0.2 s, which is audible as a 5 Hz flutter.
* **The howl is louder than the syllables and is not scaled by `CHOIR_LEVEL`.** It peaks near 0.87 where the syllables peak near 0.6 to 0.7, about 4 dB louder in RMS over its body. It was left
  at the ceiling the spec asks for. The owner has heard it in the enya recording and made no complaint about it. The difference matters most for `--only ttchoir` and the Play tab's single device, where no other voice covers it. If it is too
  loud by ear, scaling the howl by `CHOIR_LEVEL` is the knob.
* **In live mode every render is stateless.** A howl cut by a segment boundary has no memory of its start, so the next segment may fire another before 4 s have passed. The stitcher's crossfade
  hides the cut. Carrying a howl across segments needs the start-time offset that is open for the other devices (`docs/ROADMAP.md`, item 3).
* **The crush is inside the choir's own render,** after the howl and before the final clip, so the cables see the crushed voice. It rounds toward zero, so it cannot raise a peak above the howl's ceiling.
* **Low sample rates.** Below 16000 Hz a plosive's burst has its band top lowered to 0.40 of the rate, because a short burst's envelope spreads its spectrum past Nyquist (1.5% of its power on
  average at 8000 Hz, up to 8% on some draws). At 22050 Hz and above the bursts are unchanged.
* **Enya.** The whole choir keeps at most 5% of its energy above 4 kHz (a test), so the consonants sit inside the voice and the style stays dark.
* **The Play tab picks a consonant and a vowel and has no fire button.** The owner, asked about a button to fire a syllable at will: "not yet. maybe some day". The pick only fills in the next
  syllables. The patch's source still decides when they sound. A fire button needs a new control type in `ui/knobs.py` and `ui/instrument.py`.
* **Fast mode (`--interval`) logs no howls and no per-segment choir lines,** because the Play tab's engine does not produce them. The line saying a chip's source is silent is logged from the first reading only.
* **The choir's kernel does only the sum of sines.** Only the harmonics are a `sin` per sample; the syllable plan, the consonants' noise, the howl and the crush are sequential or need a table, so they stay on the host. `ttchoir.choir_voice` takes a `sines` callable, and the CPU choir passes `harmonic_sum`, the loop it always had, so the CPU output is unchanged to the bit. The kernel takes eight harmonics a run (sixteen input buffers and two of its own, the gong's count), and the host adds the runs.
* **Not done.** MIDI events for the choir, a soft l or r, and a longer listen. The owner has heard the consonants, the howl and the crush (2026-10-31).
