# Architecture

## Data flow

```
 prompt ──► media/llm ──► (the chips work harder)
                                   │
 tt-smi -s ──► telemetry/snapshot ─┤        --mock-chips ──► telemetry/mock
                  parse_snapshot   │                               │
                                   ▼                               ▼
                         list of Telemetry, one per chip
                                   │
              ┌────────────────────┴─────────────────────┐
              ▼                                          ▼
   music/chorus.render_chorus                  narrate/explain
   plan_turns ► turn_gains                     what and why for every choice
   devices ► ×gain ► per chip ► ×power ►              │
   current_mix                                        ▼
              │                                 narrate/runlog
              ▼                                 JSONL + left-bar text
        WAV (run.py) ──► media/mux ──► video soundtrack
```

`run.py` wires the pieces and `cli.py` parses arguments. Nothing in `telemetry/`, `music/` or
`narrate/` touches the network, the filesystem or the device, apart from `fetch_snapshot`, which
runs `tt-smi -s`.

## Packages

| Package | Module | Job |
|---|---|---|
| `telemetry/` | `snapshot.py` | `Telemetry` type, `tt-smi -s` JSON parsing (any chip count, any number of GDDR channels), `fetch_snapshot` |
| | `mock.py` | Deterministic synthetic readings for N chips |
| `music/` | `mapping.py` | The signal-to-sound table as functions |
| | `synth.py` | Drone voices for any chip count |
| | `devices.py` | Rollz, Sidrax, Cocoquantus, Plumbutter drone, gong, ultrasonic shimmer, and the current mixer. `DEVICE_INPUTS` says which fields each device reads |
| | `lead.py` | The lead voice and its shift/invert transforms |
| | `turns.py` | Who solos each turn, and the gain curves |
| | `patch.py` | The `Patch` (every drawn choice of a run), drawing it from a seed and the chips, JSON save and load |
| | `chorus.py` | Deals device slots to chips, plans turns, mixes |
| `narrate/` | `explain.py` | Plain-language what/why for every choice, built from the same functions the audio uses |
| | `runlog.py` | JSONL for programs, wrapped left-bar text for people |
| `media/` | `llm.py` | Prompts any OpenAI-compatible endpoint |
| | `mux.py` | Puts the WAV under a video with ffmpeg |

## Dependencies run one way

`telemetry` depends on nothing in the package. `music` depends on `telemetry`. `narrate` depends on
`music` and `telemetry`. `media` depends on nothing. `run.py` and `cli.py` depend on all of them.
Keep it that way: `music` must never import `narrate`.

## Why the tests are shaped like this

* A device is `f(telemetry, duration, sample_rate, seed, patch=DEFAULT_PATCH) -> float32 array`, so every device is
  tested the same way, with no hardware.
* `DEVICE_INPUTS` is a claim about what each device reads. `tests/narrate/test_explain.py` changes
  one field at a time and requires the audio to move exactly when the claim says it should, so the
  log cannot describe a mapping that is not in the sound.
* `tests/test_hygiene.py` reads the source: no module may import anything that opens a device,
  every package folder needs an `__init__.py`, and every module needs a test file in the matching
  folder.
