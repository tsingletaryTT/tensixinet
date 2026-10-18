# Decisions

Why the code is the way it is. Each item says what was chosen and what the evidence was. Change one only with a test that shows the new behaviour.

* **Chip count is never assumed.** Voices, devices, GDDR channels and log text all come from
  the telemetry. Devices are slots, and there are always at least as many slots as devices
  (`max(chips, 7)`). Slot `s` goes to chip `s % chips`. A 1-chip box plays all seven devices,
  and a box with 7 or more chips plays one device per chip.
* **Players trade solos.** Each turn lasts two bars at the average tempo. One slot is featured and
  the rest drop to 35% volume, with 0.25 s fades. The soloist is drawn in proportion to the
  power its chip draws, and never repeats back to back (`music/turns.py`). A busy chip solos more.
* **The lead changes with the system.** Each bar reads the next GDDR channel and compares it
  with the average channel. Hotter than average shifts the motif up (1 scale degree per degree C,
  at most 4). Cooler inverts it around its first note. Within 1 C it repeats. Flat readings loop.
  Only the snapshot varies, so the variety comes from the channels. Live sampling (see the
  roadmap) would let the lead respond to change over time.
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
* **`voltage` has no musical use.** A test pins that; adding one is a deliberate change.
* **DDR status:** the code treats the low byte as four 2-bit fields and mutes a lane at 0.
  tt-smi's source says newer firmware puts BIST bits in `[31:16]`. The meaning of values 1 to
  3 is a guess and should be checked against firmware docs.
* The chorus is not normalised to a fixed peak, because that would erase the power-to-volume
  mapping. `tanh` in the mixer keeps it inside full scale.

