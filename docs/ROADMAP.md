# Roadmap

Open work in rough order. Each item starts with a failing test (see `CONTRIBUTING.md`).

1. **More of the patch.** Which devices run at all is not in the patch yet. More fields could join it: the
   motif length, the transform rules (shift and invert thresholds), the number of lanes in ttrollz, the ttrax
   wall spread, the ttquantussy's ratios and the snare's cutoff. Each new field needs the "only this field changes the chorus" test.
2. **Let more of the system pick.** Today GDDR spread picks the scale and the memory-to-ASIC ratio picks the gong
   tuning. Power spread across chips could pick the turn length, and aiclk spread could pick ttrollz's asymmetry.
2b. **The voice, next steps.** `ttchoir` sings vowels. Still to build, each a step on its own: noise and fricatives (shaped noise: /s/ at
   4 to 8 kHz, /sh/ at 2.5 to 4 kHz), plosives (a closure, a burst and aspiration, then formants gliding in), nasals (a murmur with an
   antiresonance), liquids and glides (moving formants), coarticulation, a text-to-phoneme layer with morphemes setting stress and
   duration, and having the chips sing the language model's reply. A kernel version (a sum of harmonics, each a `sin` of a multiple of
   one phase; noise and filtering as a convolution, with noise supplied by the host) comes after the numpy voice sounds right.
2c. **Cables, next steps.** Cables act on finished voices. Open: cables into a device's parameters (the ttrax chaos, a clock,
   the choir's vowel), cables that feed back, cables chosen by the readings as well as the seed, and a kernel version of the four kinds.
3. **Live mode, the rest.** `--live` works: it reads the chips every turn, draws the soloist from the current
   weights, plays new errors once, moves ttrax's chaos with how fast temperature and power change, and
   crossfades the turns. What is open:
   * A start-time offset in every device, so a turn continues the last one. Today each turn restarts its clocks,
     and a ringing gong fades at the end of its turn.
   * Re-drawing the patch as the readings drift.
   * A tthorn that follows the chips' movement (see `docs/CIRCUITS.md`).
   * Recording a live demo (the demo still uses the offline render) and testing the players on a machine
     with real speakers. Only silence has been sent to `paplay`, `aplay` and `ffplay` so far.
   * Rates for more readings than temperature and power (clocks, link counts), and for the other devices.
4. **Demo pacing.** The whole log appears within about 3 s, then the left pane is still for the
   rest of the clip, and the longer log scrolls the first lines off the top. Print the chip lines
   gradually, in time with the music, or loop prompts.
5. **Longer load.** The model answers in about 3 s. Add `--max-tokens` to the CLI (it is fixed at
   100 in `media/llm.py`) or send several prompts so tt-toplike shows load for longer.
6. **Align audio to the LLM reply.** The soundtrack starts at video time 0, but telemetry is
   sampled after the reply.
7. **Run the devices as TT-Lang kernels.** Six kernels (`ttshock` the drone's ripple, `ttshimmer`, `ttgong` up to five partials and
   three overlapping strikes, `ttavdog`, `ttmix` the mixer up to four chips, `ttcoco` the eight-tap delay mix) match their numpy devices
   in the TT-Lang simulator and have now run on all four chips of both boards (`kernels/README.md`, `kernels/cycle.py`, `docs/hardware/`).
   Next, in order: splitting
   `ttcoco`'s eight taps across real Tensix cores instead of pre-computing them on the host, which would make it the first kernel here
   to use more than one core, then the ttrax bounce, ttrollz's blips, the lead, the EQ (a convolution, which is a matmul), a mixer
   for more than four chips, and a
   `hardware`-marked test that runs in CI's absence of a chip as a skip, which `pyproject.toml` declares and nothing uses yet. The
   old sketches are in `legacy/ciat_lonbarde_kernels.py`. The kernels' host side uses the default patch, so they do not follow the
   steam routes (`docs/SOUND.md`); the `pitch` route's cumulative phase is the part that would need a new kernel.
7b. **Load kernels, used seldom and with a hard governor.** Kernels whose purpose is to spend as much power or as many cycles or
   cores as the box allows, briefly, so a burst of real load becomes a real event in the music and in the readings that drive the
   other devices. Open design questions: how the governor caps temperature, power and wall time before such a kernel is allowed to
   run at all; how rarely "seldom" is (a cooldown, a cap per run, both); whether it needs its own gozer conversation with tt-orchard
   beyond the ordinary one-chip turn, since it is deliberately not polite load; and what in the music marks the moment (a device of
   its own, or an event like the error ticks). Nothing here is built. This only runs with an explicit lease and an explicit opt-in;
   it is not something a live run turns on by itself.
8. **Loudness.** Done in `music/loudness.py`: one gain to -18 dBFS RMS. A perceptual measure (LUFS, with
   K-weighting) would match the level of runs with very different spectra more closely.
9. **More of the guides.** The Gongue's Trad and Gonz filter modes, the shift register's feedback taps and buttons,
   the coco's looping, flip, skip, Dolby switches and iron cross (the delay and feedback itself is now `ttcoco`), and
   the Ultrasound's folding of ultrasound into the audible band (needs a higher sample rate) are not modelled.
   `docs/CIRCUITS.md` lists each one.
