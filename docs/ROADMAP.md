# Roadmap

Open work in rough order. Each item starts with a failing test (see `CONTRIBUTING.md`).

1. **Replayable one-of-a-kind patches.** Each run draws a "patch" (lead motif, scale or mode, gong
   partial ratios, shimmer band, turn length, which devices run) from a seed taken from the chips'
   readings. The patch is saved as JSON, printed in the run log with the reason for each choice,
   and replayed with `--patch FILE`. Tests: same seed repeats exactly, different seeds differ, and
   the patch file round-trips.
2. **Let the system pick the mode.** For example the GDDR spread chooses the scale and the ratio of
   memory to ASIC temperature chooses the gong tuning, so the diversity follows the machine.
3. **Live mode.** Today it samples telemetry once and renders. Sampling during playback would make
   the lead, the turns and the gong follow the chips as they heat up under load.
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
8. **Loudness.** The chorus peaks around 0.23 to 0.31 on purpose, because power sets the volume.
   Decide whether the soundtrack gets loudness normalisation when it is muxed.
9. **License.** The repo has none yet.
