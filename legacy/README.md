# Legacy prototype scripts

These are the first prototype, written before the `tensixinet/` package and its tests. They are
kept for reference, mainly `ciat_lonbarde_kernels.py`, whose TT-Lang kernel sketches are not ported
yet (see `docs/ROADMAP.md`). Nothing in the package imports them.

**Do not run them without a gozer lease.** `tensixinet_full_loop.py` and `ciat_lonbarde_kernels.py`
import `ttnn` and `ttl` at the top, and importing those opens the device.

The other three read device 0 only, hard-code 8 voices and render with per-sample Python loops
(minutes for 2 minutes of audio; the package takes about 0.5 s for 24 s).

| Script | What it was |
|---|---|
| `phase1_sonification.py` | First telemetry-to-sine-voices sketch |
| `tensixinet_multichip.py` | Two-chip, 8-voice synth |
| `tensixinet_full_loop.py` | LLM to hardware to music loop, opens the device |
| `tensixinet_tt_demo_maker.py` | Early demo-maker stub |
| `ciat_lonbarde_kernels.py` | TT-Lang kernel sketches (VCO, bounds-modulated VCO, granular, polyrhythm) |
