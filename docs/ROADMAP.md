# Roadmap

Open work in rough order. Each item starts with a failing test (see `CONTRIBUTING.md`).

1. **More of the patch.** Which devices run at all is not in the patch yet. More fields could join it: the
   motif length, the transform rules (shift and invert thresholds), the number of lanes in Rollz, the Sidrax
   wall spread, the Quantussy's ratios and the snare's cutoff. Each new field needs the "only this field changes the chorus" test.
2. **Let more of the system pick.** Today GDDR spread picks the scale and the memory-to-ASIC ratio picks the gong
   tuning. Power spread across chips could pick the turn length, and aiclk spread could pick Rollz's asymmetry.
3. **Live mode.** Today it samples telemetry once, draws one patch and renders. Sampling during playback
   would make the lead, the turns and the gong follow the chips as they heat up under load, and could
   re-draw the patch as the readings drift. It would also allow signals that need two readings: rate of
   change of temperature and power as a gesture, and a link that drops or an error that appears mid-run.
   Live mode would also let a Deerhorn follow the chips' change over time, and let rate of change move the
   Sidrax chaos knob (see `docs/CIRCUITS.md`).
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
8. **Loudness.** The chorus peaks around 0.4 on purpose, because power sets the volume.
   Decide whether the soundtrack gets loudness normalisation when it is muxed.
9. **More of the guides.** The Gongue's Trad and Gonz filter modes, the shift register's feedback taps and buttons,
   the coco's delay and loops, and the Ultrasound's folding of ultrasound into the audible band (needs a higher
   sample rate) are not modelled. `docs/CIRCUITS.md` lists each one.
