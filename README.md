# Tensixinet: Hardware Telemetry → Generative Music

A generative music instrument that sonifies Tenstorrent hardware telemetry using Ciat Lonbarde modular synthesizer principles.

## Overview

**Tensixinet** creates music from hardware telemetry with a unique feedback loop:

1. **LLM thought process** (Qwen3-Coder-Next) generates text/ideas
2. **Hardware telemetry** changes based on the LLM's activity
3. **TT-Lang synthesizer kernels** run on Tenstorrent chips
4. **Telemetry from those kernels** shapes the music in real-time

This creates a profound integration: **AI thought → hardware synthesis → music output**

## Core Principles

### Ciat Lonbarde Inspiration
- **Plumbutter 2**: Current-based (not voltage-based) control with banana plug patching
- **Rollz**: Organic pulse generation with asymmetric timing
- **Cocoquantus**: Sample-and-hold with multiple waveforms
- **Sidrax**: Bounds modulation creating complex oscillator behavior

### Key Design Philosophy
1. **Current-based mixing**: Patching multiple banana plugs mixes by current flow, not voltage addition
2. **Organic exponential curves**: Natural charging/discharging behavior creates musical timing
3. **Controlled instability**: Systems that appear random but have underlying structure
4. **Polyrhythmic patterns**: Multi-channel telemetry creates complex rhythmic relationships

## Hardware → Music Mapping

| Hardware Signal | Musical Parameter | Range |
|----------------|-------------------|-------|
| `asic_temperature` | Pitch, linear in semitones | 25-100°C → C3-C5 |
| `aiclk` | Tempo/BPM | 400-1350 MHz → 60-180 BPM |
| `power` | Volume with exponential curve | 0-200W → 0.3-1.0 |
| `GDDR temperatures` | Sidrax swing bounds (coolest channel sets the lower bound, hottest the upper) | every channel the board reports |
| `DDR status` | Rollz polyrhythm lanes of 3, 4, 5, 7 pulses per bar; status 0 mutes a lane | 4 fields |

## What it does now

Seven devices are dealt across your chips. There are always at least seven slots, and slot `s`
goes to chip `s % chips`, so a 1-chip box plays the whole ensemble and a box with 7 or more chips
plays one device each.

| Device | Character | Reads |
|---|---|---|
| Rollz | Lopsided pulse clock plus polyrhythm lanes | temperature, aiclk, DDR status |
| Sidrax | Oscillator bouncing between two bounds | temperature, GDDR temperatures |
| Cocoquantus | Sample-and-hold melody on a pentatonic scale | temperature, aiclk |
| Plumbutter drone | Three-voice C major drone | temperature |
| Lead | Eight-note motif that is shifted or inverted bar by bar | temperature, aiclk, GDDR temperatures |
| Gong | Inharmonic, long-ringing strikes with beating pairs | temperature, aiclk |
| Ultrasonic shimmer | Two close tones at 12-20 kHz that swell and fade | temperature, GDDR temperatures |

**The lead changes with the system.** Each bar reads the next GDDR channel and compares it with the
average channel. Hotter shifts the motif up, cooler inverts it, and a reading near the average
repeats it, so uneven memory temperatures make the melody wander and even ones make it loop.

**Players trade solos.** Every two bars one device takes the floor and the others drop to 35%
volume. Chips drawing more power get more solos, and nobody solos twice in a row.

Power sets each chip's volume. Current sets its share of the mix (sources mix by current, as
patched banana plugs do). Voltage is not used.

## Use it

```bash
pip install -r requirements.txt

python3 -m tensixinet --mock-chips 8 --duration 20        # no hardware: 8 synthetic chips
python3 -m tensixinet --duration 60                       # your real chips, via tt-smi -s
python3 -m tensixinet "describe a hardware synth" --duration 60
```

With a prompt, it is sent to an OpenAI-compatible endpoint first (`--endpoint`, default
`http://127.0.0.1:8011`; the model name is read from the endpoint, or give `--model`). The chips
are sampled after the reply, so the model's load is in the telemetry. If the endpoint is down
the music still plays and the log says so.

Output is a 16-bit WAV (`--out`, default `audio/`) and a run log (`--log-dir`, default `logs/`).
The log says what each telemetry value became and why:

```
║ [00:02.9] chip 2 plays cocoquantus
║          why: chip index 2 % 4 = 2; devices cycle so any chip count plays all 4 characters
║ [00:02.9] aiclk 1350 MHz -> 180.0 BPM
║          why: 400-1350 MHz maps onto 60-180 BPM; a faster clock sets a faster pulse
```

Reading telemetry (`tt-smi -s`) does not open the chips, so none of this needs a gozer lease.

## Test it

```bash
python3 -m pytest        # 226 tests, about 1 s, no hardware, no Tenstorrent packages
```

`tests/fixtures/tt_smi_snapshot.json` is a real 4-chip capture. See `CLAUDE.md` for how the
tests are written and checked.

## The demo video

`demo/build.sh` records `demo/demos.yaml` with `tt-demo`: the prompt and the run log on the
left, `tt-toplike` on the right, then lays the chorus under it as the soundtrack. It reads
telemetry only. Output: `demo/assets/chorus.final.mp4`.

## Superseded and not yet ported

The scripts in the top directory (`phase1_sonification.py`, `tensixinet_full_loop.py`,
`tensixinet_multichip.py`, `tensixinet_tt_demo_maker.py`) are the first prototype. They read
only device 0 and render slowly. `tensixinet_full_loop.py` and `ciat_lonbarde_kernels.py`
import `ttnn` and `ttl`, which opens the device, so do not run them without a lease.

`ciat_lonbarde_kernels.py` holds the TT-Lang kernel sketches (VCO clock generator,
bounds-modulated VCO, granular engine, polyrhythmic clock). They have no tests and have not
been run on hardware. Porting them is open work, listed in `CLAUDE.md`.
