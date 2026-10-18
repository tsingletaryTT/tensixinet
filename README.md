# tensixinet

Tenstorrent chip telemetry, played as a chorus of Ciat-Lonbarde-style synthesizers.

Each chip's temperature, clock, power, current, memory temperatures and DDR status shape the
sound. Seven devices are dealt across however many chips you have, and they trade solos. A
local language model can be prompted first, so its load shows up in the chips' readings.
Every choice the program makes is written to a log with the reason for it.

**Demo:** [`demo/assets/chorus.final.mp4`](demo/assets/chorus.final.mp4) is 50 seconds with the
prompt and run log on the left, tt-toplike on the right, and the
chorus as the soundtrack.

## Try it

```bash
pip install .                                  # numpy and soundfile
tensixinet --mock-chips 8 --duration 20        # no hardware: 8 synthetic chips
tensixinet --duration 60                       # your real chips, read with `tt-smi -s`
tensixinet "describe a hardware synth" --duration 60   # prompt a local model first
tensixinet --patch logs/patch_20261017_120000.json     # replay an earlier run's choices exactly
```

Output is a 16-bit WAV (`--out`, default `audio/`) and a run log (`--log-dir`, default `logs/`).
The log looks like this:

```
║ [00:00.0] chip 0 plays rollz, sidrax, cocoquantus, plumbutter_drone, lead, gong, ultrasonic_shimmer
║          why: slots 0-6 of 7 go to chip 0: slots are dealt to 1 chip in turn (slot % chips)...
║ [00:00.0] aiclk 800 MHz -> 110.5 BPM
║          why: 400-1350 MHz maps onto 60-180 BPM; a faster clock sets a faster pulse
║ [00:00.0] turn 1 (0s): lead on chip 0 solos
║          why: the soloist is drawn in proportion to the power each chip draws...
```

Reading telemetry does not open the chips. `tt-smi -s` is read-only, so none of this needs a
lease on a shared machine. The language model is any OpenAI-compatible endpoint (`--endpoint`,
default `http://127.0.0.1:8011`); the model name is read from it. If it is down, the music still
plays and the log says so.

## What you hear

| Device | Character | Reads |
|---|---|---|
| Rollz | Lopsided pulse clock plus polyrhythm lanes | temperature, aiclk, DDR status |
| Sidrax | Oscillator bouncing between two bounds | temperature, GDDR temperatures |
| Cocoquantus | Sample-and-hold melody on a pentatonic scale | temperature, aiclk |
| Plumbutter drone | Three-voice C major drone | temperature |
| Lead | Eight-note motif, shifted or inverted bar by bar | temperature, aiclk, GDDR temperatures |
| Gong | Inharmonic, long-ringing strikes with beating pairs | temperature, aiclk |
| Ultrasonic shimmer | Two close tones at 12-20 kHz that swell and fade | temperature, GDDR temperatures |

Power sets each chip's volume and current sets its share of the mix. A 1-chip box plays all seven
devices; a box with 7 or more chips plays one each. The full signal-to-sound table, the turn-taking
rule and the reasons behind them are in [docs/SOUND.md](docs/SOUND.md).

## A different patch every run

Each run draws a patch from the chips' readings plus the clock, so no two runs sound alike. The scale
follows how unevenly the memory channels are heated, the gong tuning follows the memory-to-ASIC temperature
ratio, and a seed picks the lead's motif, the shimmer band, how long solos last and how quiet the others get.
The patch is saved as JSON next to the log, and every choice is logged with its reason. Replay one with
`--patch FILE`. Details are in [docs/SOUND.md](docs/SOUND.md).

## Test it

```bash
python3 -m pytest        # no hardware, no Tenstorrent packages, about 1 s
```

## Where things are

| Path | What |
|---|---|
| `tensixinet/` | The package: `telemetry/`, `music/`, `narrate/`, `media/` |
| `tests/` | One test file per module, in a folder that matches the module's |
| `demo/` | The `tt-demo` manifest and scripts that record the video |
| `docs/` | [Architecture](docs/ARCHITECTURE.md), [sound design](docs/SOUND.md), [decisions](docs/DECISIONS.md), [contributing](docs/CONTRIBUTING.md), [roadmap](docs/ROADMAP.md) |
| `legacy/` | The first prototype, kept for reference. Some scripts open the device; see its README |

Contributors and coding agents: start with [CLAUDE.md](CLAUDE.md).
