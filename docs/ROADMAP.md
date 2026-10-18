# Roadmap

Open work in rough order. Each item starts with a failing test (see `CONTRIBUTING.md`).

1. **More of the patch.** Which devices run at all is not in the patch yet. More fields could join it: the
   motif length, the transform rules (shift and invert thresholds), the number of lanes in ttrollz, the ttsidrax
   wall spread, the ttquantussy's ratios and the snare's cutoff. Each new field needs the "only this field changes the chorus" test.
2. **Let more of the system pick.** Today GDDR spread picks the scale and the memory-to-ASIC ratio picks the gong
   tuning. Power spread across chips could pick the turn length, and aiclk spread could pick ttrollz's asymmetry.
3. **Live mode, the rest.** `--live` works: it reads the chips every turn, draws the soloist from the current
   weights, plays new errors once, moves ttsidrax's chaos with how fast temperature and power change, and
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
7. **Port the TT-Lang kernels** in `legacy/ciat_lonbarde_kernels.py` once they can be tested on
   hardware. That needs a gozer lease and a test with the `hardware` marker, which `pyproject.toml`
   declares and nothing uses yet.
8. **Loudness.** Done in `music/loudness.py`: one gain to -18 dBFS RMS. A perceptual measure (LUFS, with
   K-weighting) would match the level of runs with very different spectra more closely.
9. **More of the guides.** The Gongue's Trad and Gonz filter modes, the shift register's feedback taps and buttons,
   the coco's delay and loops, and the Ultrasound's folding of ultrasound into the audible band (needs a higher
   sample rate) are not modelled. `docs/CIRCUITS.md` lists each one.
