# tensixinet

**Your Tenstorrent chips, playing in a band.**

[![tests](https://github.com/tsingletaryTT/tensixinet/actions/workflows/test.yml/badge.svg)](https://github.com/tsingletaryTT/tensixinet/actions/workflows/test.yml)
[![license](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

[![The demo: the run log on the left, tt-toplike on the right](docs/media/demo-frame.png)](demo/assets/chorus.final.mp4)

*Click the picture for the 50-second demo with sound (`demo/assets/chorus.final.mp4`). Left: the prompt
and the run log. Right: [tt-toplike](https://github.com/tenstorrent/tt-toplike) watching the same chips.
The soundtrack is what the chips played.*

Every chip reports its temperature, clock, power, current, memory temperatures and DDR status. tensixinet
turns those readings into music. Each chip joins a small band of synthesizers inspired by
[Ciat-Lonbarde](https://www.ciat-lonbarde.net/) instruments, and the band trades solos. A cool chip and a hot chip
sound different, a busy chip gets more solos, and uneven memory temperatures make the melody wander.

You can prompt a local language model first. Its work heats the chips, so you can hear the model
thinking. Every choice the program makes is written to a log with the reason for it, and each run
draws a different patch, so no two runs sound alike unless you replay one.

You do not need Tenstorrent hardware to try it. It can play any number of synthetic chips.

## Try it in a minute

```bash
git clone https://github.com/tsingletaryTT/tensixinet && cd tensixinet
pip install .                                  # numpy and soundfile
tensixinet --mock-chips 4 --duration 20        # four synthetic chips, no hardware
```

It writes a WAV file to `audio/` and the run log to `logs/`. Play the WAV with anything.

On a machine with Tenstorrent chips:

```bash
tensixinet --duration 60                                  # reads your chips with `tt-smi -s`
tensixinet "describe a hardware synth" --duration 60      # prompt a local model first
tensixinet --patch logs/patch_20261017_120000.json        # replay an earlier run's choices exactly
```

Reading telemetry does not open the chips. `tt-smi -s` is read-only, so none of this needs a lease on a shared
machine, and nothing in the package imports `ttnn` (a test checks that for every file).

## Meet the band

Seven devices, dealt out across however many chips you have. One chip plays all seven. Seven or more
chips play one each. Anything in between shares them.

| Device | What you hear | What drives it |
|---|---|---|
| **Rollz** | A lopsided pulse clock with polyrhythm lanes of 3, 4, 5 and 7 | temperature, clock, DDR status |
| **Sidrax** | An oscillator bouncing between two bounds | temperature, GDDR temperatures |
| **Cocoquantus** | A sample-and-hold melody, quantised to a scale | temperature, clock |
| **Plumbutter drone** | A three-voice C major drone | temperature |
| **Lead** | An eight-note motif that gets shifted up or flipped upside down, bar by bar | temperature, clock, GDDR temperatures |
| **Gong** | Inharmonic strikes with slowly beating partials, a gamelan or tam-tam ring | temperature, clock |
| **Ultrasonic shimmer** | Two close tones at 12 to 20 kHz that swell and fade | temperature, GDDR temperatures |

**They trade solos.** Every one to four bars, one device steps forward and the others drop back. The soloist is drawn
in proportion to the power each chip draws, so the busiest chip solos most. Nobody solos twice in a row, and
quiet chips still get turns.

**The lead listens to the memory.** Each bar reads the next GDDR channel and compares it with the average.
A hotter channel shifts the motif up. A cooler one inverts it. A channel near the average repeats it. Even
memory makes a loop, and uneven memory makes the melody wander.

**Power and current mix the band.** Power sets each chip's volume. Current sets its share of the mix, the
way patched banana plugs mix by current on a Plumbutter.

## It tells you why

Every line of the log says what a reading became and the reason. This is a real run on four chips, trimmed:

```
║ [00:00.2] patch seed 22
║          why: replayed from demo/chorus.patch.json, so this run sounds like
║               the one that saved it; the seed fixes every drawn choice below
║ [00:00.2] scale minor_pentatonic (GDDR spread 20 C)
║          why: each 3 C between the hottest and coolest GDDR channel moves one
║               scale along, so even memory plays the first and uneven memory
║               wanders the list
║ [00:00.2] chip 0 plays rollz, lead
║ [00:00.2] asic 47.5 C -> pitch 198.3 Hz
║          why: 25-100 C maps onto C3-C5, linear in semitones, so a hotter chip
║               plays higher
║ [00:00.2] lead phrases: 1 shift +2, 2 repeat, 3 repeat, 4 repeat, 5 repeat, 6
║           repeat, 7 shift +2, 8 repeat
║ [00:00.2] chip 3 current 70 A -> 39% of the mix
║          why: sources mix by current, as patched banana plugs do, so the chip
║               drawing more current is louder
║ [00:00.2] turn 1 (0s): plumbutter_drone on chip 3 solos
║          why: the soloist is drawn in proportion to the power each chip draws,
║               nobody solos twice in a row, and everyone else drops to a lower
║               level while it plays
```

The screen shows each reason once. A JSONL file in `logs/` keeps every line for programs. A test changes one
reading at a time and requires the audio to change exactly where the log says it does, so the log cannot
describe a mapping that is not in the sound.

## No two runs alike

Each run draws a **patch**: the scale, the lead's motif, the gong's tuning and beat, the shimmer band, how long
solos last and how quiet everyone else gets.

- **The machine decides two things.** The scale follows how unevenly the memory channels are heated. The gong
  tuning follows the ratio of memory to ASIC temperature.
- **A seed decides the rest.** The seed hashes every reading on every chip plus the clock, so changing any one
  reading gives a new patch.
- **A patch is a small JSON file** saved next to the log. `--patch FILE` replays it and gives a byte-identical
  WAV. `--patch-seed N` fixes the seed instead.

The demo replays [`demo/chorus.patch.json`](demo/chorus.patch.json), so rebuilding it plays the same music.

## What the readings do

| Reading | Becomes | Range |
|---|---|---|
| ASIC temperature | Pitch, linear in semitones | 25-100 °C is C3 to C5 |
| AI clock | Tempo | 400-1350 MHz is 60-180 BPM |
| Power | Chip volume, exponential curve | 0-200 W is 0.3 to 1.0 |
| Current | Share of the mix | each chip's fraction of the total |
| GDDR temperatures | Sidrax bounds, the lead's shifts and flips, the shimmer's beat | every channel the board reports |
| DDR status | Rollz polyrhythm lanes; status 0 mutes a lane | 4 fields |
| Voltage | Nothing. A test pins that | |

The full reasoning is in [docs/SOUND.md](docs/SOUND.md). One detail matters for correctness: a GDDR channel's
temperature is the hotter of its two dies, because `tt-smi` packs two channels with two dies each into every
register. The first prototype read those bytes as independent sensors and lost half the channels.

## Options

| Option | What it does |
|---|---|
| `PROMPT` | Send this to the language model first, so its load is on the chips when they are read |
| `--duration SECONDS` | Length of the audio (default 60) |
| `--sample-rate HZ` | Default 44100. Use 96000 for more ultrasonic headroom |
| `--mock-chips N` | Play N synthetic chips and read no hardware. `--seed` varies them |
| `--patch FILE` / `--patch-seed N` | Replay a saved patch, or draw from a fixed seed |
| `--out`, `--log-dir` | Where the WAV and the log go (`audio/`, `logs/`) |
| `--endpoint`, `--model` | Any OpenAI-compatible endpoint (default `http://127.0.0.1:8011`). The model name is read from it |

If the endpoint is down, the music still plays and the log says so.

## How it is built

```
 prompt ─► media/llm            tt-smi -s ─► telemetry/snapshot    --mock-chips ─► telemetry/mock
                                                      │                                  │
                                                      └──────────── one reading per chip ┘
                                                                          │
                                         ┌────────────────────────────────┴───────────┐
                                         ▼                                            ▼
                                 music/  patch ► devices ► turns ► chorus     narrate/  explain ► runlog
                                         │                                            │
                                         ▼                                            ▼
                                       WAV  ──► media/mux ──► video soundtrack      JSONL + text log
```

| Folder | Job |
|---|---|
| `tensixinet/telemetry/` | `tt-smi -s` parsing for any chip count, and synthetic chips |
| `tensixinet/music/` | The mapping table, the seven devices, patches, turn-taking, the mixer |
| `tensixinet/narrate/` | The what-and-why log |
| `tensixinet/media/` | The language model client and the ffmpeg soundtrack step |
| `tests/` | One test file per module, in a folder that matches the module's |
| `demo/` | The [tt-demo-maker](https://github.com/tsingletaryTT/tt-demo-maker) manifest and scripts that record the video |
| `docs/` | [Architecture](docs/ARCHITECTURE.md), [sound design](docs/SOUND.md), [decisions](docs/DECISIONS.md), [contributing](docs/CONTRIBUTING.md), [roadmap](docs/ROADMAP.md) |
| `legacy/` | The first prototype. Some scripts open the device; see its README |

## Tested without hardware

```bash
make test        # about 4 s: 580+ tests, no Tenstorrent packages, no chips
```

The tests are written first, then the code is broken on purpose (swap a `min` for a `max`, drop a term, stop
passing a value down) to check that a test goes red. Every mutant that survived got a new test. A hygiene test
reads every source file and fails on any import that would open a device, a folder without `__init__.py`,
a module without a test file, or a missing license header.

To rebuild the demo you need [tt-demo-maker](https://github.com/tsingletaryTT/tt-demo-maker), tt-toplike and
ffmpeg: `make demo`.

## Status

The instrument works and is still growing.

- **Done:** reading any number of chips, seven devices, turn-taking, patches, the log, the demo.
- **Not done:** playing live while the chips change (today it reads once and renders), and the TT-Lang kernels
  in [`legacy/`](legacy/README.md), which are sketches that have never run on hardware.
- **Next:** [docs/ROADMAP.md](docs/ROADMAP.md) lists what is open, in order. Contributions are welcome. Start
  with [CLAUDE.md](CLAUDE.md) and [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md), which apply to people and to
  coding agents alike.

## Credits

The devices are inspired by instruments from [Ciat-Lonbarde](https://www.ciat-lonbarde.net/), Peter Blasser's
workshop: Rollz, Sidrax, Cocoquantus and Plumbutter. They are impressions of how those instruments behave.
They do not emulate the circuits. Telemetry comes from [tt-smi](https://github.com/tenstorrent/tt-smi), and the
demo uses [tt-toplike](https://github.com/tenstorrent/tt-toplike).

## License

Apache License 2.0. See [LICENSE](LICENSE) and [LICENSE_understanding.txt](LICENSE_understanding.txt).
Source files carry an SPDX header (`Apache-2.0`, copyright Tenstorrent USA, Inc.).
