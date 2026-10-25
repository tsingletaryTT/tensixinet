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
   current_mix ► one loudness gain (run.py)           ▼
              │                                 narrate/runlog
              ▼                                 JSONL + left-bar text
        WAV (run.py) ──► media/mux ──► video soundtrack
```

`run.py` wires the pieces and `cli.py` parses arguments. `live.py` is the live twin of `run.py`: where `run.py` reads the chips
once, it reads them every turn, renders one turn at a time with `render_chorus(..., segment=k, soloists=[slot])`,
cross-fades the turns together (`music/live.py`), and hands the audio to a sink (`media/player.py`) from a second
thread. The sink's writes block at the speed of the music, which paces the loop. Nothing in `telemetry/`, `music/` or
`narrate/` touches the network, the filesystem or the device, apart from `fetch_snapshot`, which
runs `tt-smi -s`.

With `--midi-file` or `--midi-port`, `run.py` and `live.py` also build a `Score` for each render (`score/collect.chorus_score`), which deals and seeds the
devices the way `render_chorus` does and asks the lead, `ttquantus`, the gong, `ttrollz` and `ttsteam` for their events from the functions the audio reads.
`score/midi.to_messages` routes the events to timed MIDI bytes, and a sink takes them: `MidiFileSink` keeps them for `score/smf.py` to write when the run ends, and `MidiPortSink` sends
them from its own thread at their time. Neither sink touches the audio path.

```
 devices (same functions as the audio) ──► score/collect ──► Score (Note, Trigger, Control)
                                                                 │ score/midi + score/routing
                                                                 ▼
                                              timed MIDI bytes ──► MidiFileSink ──► smf.py ──► .mid
                                                                └─► MidiPortSink ──► mido port (thread)
```

## Packages

| Package | Module | Job |
|---|---|---|
| `telemetry/` | `snapshot.py` | `Telemetry` type (temperatures, clocks, power, limits, harvesting and link masks, error counters, process count), `tt-smi -s` JSON parsing (any chip count, any number of GDDR channels), `fetch_snapshot`, `host_fingerprint` |
| | `mock.py` | Deterministic synthetic readings for N chips |
| `music/` | `mapping.py` | The signal-to-sound table as functions, including pressure, chip identity and turn weight |
| | `synth.py` | Drone voices for any chip count |
| | `devices.py` | ttrollz, ttquantus, ttdrone, gong, ultrasonic shimmer, the strike times they share, and the current mixer. Re-exports ttrax. `DEVICE_INPUTS` says which fields each device reads |
| | `ttrolzer.py` | The ttrolzer's rolls: barre tempos from E6 capacitors, steady even rolls, unstable odd ones |
| | `ttrax.py` | 4, 7 or 14 barres (Tetrax, Sidrax or Quatrax, from `patch.rax_bars`) bouncing between each other's oscillators, with press and release gestures |
| | `ttquantussy.py` | The ttquantussy's five petals and castles, and the coco's 8-bit buffer |
| | `ttsteam.py` | A shift register that steps a noise snare, with the harvested Tensix columns as its pattern |
| | `ttchoir.py` | A singer: a glottal buzz shaped by formant resonances, sung on vowels that the GDDR channels pick |
| | `ttavdog.py` | The AVDog's dogtone under the two halves of an undulation: a white voice and a red voice that answer each other |
| | `ttcoco.py` | The Cocoquantus's own delay chips, as eight parallel echo taps instead of the real instrument's two |
| | `lead.py` | The lead voice and its shift/invert transforms |
| | `events.py` | Tick, breath and sweep sounds for memory errors and thermal trips; silent on a healthy chip |
| | `turns.py` | Who solos each turn, and the gain curves |
| | `eq.py` | The final-mix EQ: a minimum-phase FIR cut from a curve, with no scipy, that runs on a whole render or on a stream of chunks |
| | `loudness.py` | One gain for a finished render: -18 dBFS RMS, peak under -1 dBFS, at most +20 dB. `LiveGain` follows the music as it plays |
| | `patch.py` | The `Patch` (every drawn choice of a run), drawing it from a seed and the chips, JSON save and load |
| | `chorus.py` | Deals device slots to chips, plans turns (`turn_seconds`, `slot_weights`), mixes. `render_chorus` takes a segment number and a named soloist for live mode |
| | `live.py` | Live mode's pure pieces: `advance` (rates and new errors from two readings) and `Stitcher` (cross-fades turns together) |
| `narrate/` | `explain.py` | Plain-language what/why for every choice, built from the same functions the audio uses |
| | `runlog.py` | JSONL for programs, wrapped left-bar text for people |
| `media/` | `llm.py` | Prompts any OpenAI-compatible endpoint |
| | `mux.py` | Puts the WAV under a video with ffmpeg |
| | `player.py` | Where live audio goes: a `paplay`, `aplay` or `ffplay` process, a growing WAV, or no sound at the speed of the music |
| `score/` | `events.py` | `Note`, `Trigger`, `Control` and `Score`, with the Hz-to-MIDI helpers |
| | `collect.py` | One extractor per device and `chorus_score`, which builds a render's `Score` the way `render_chorus` deals the devices |
| | `routing.py` | The route table (`Routes`, `DEFAULT_ROUTES`) and `load_routes` for `--midi-map` |
| | `midi.py` | `to_messages`: a `Score` to timed MIDI bytes |
| | `smf.py` | The Standard MIDI File writer |
| | `sinks.py` | `MidiFileSink`, `MidiPortSink` (a thread and a heap, never blocks the audio), `TeeScoreSink`, and the lazy `mido` opener |

## The app

`app/gradio_app.py` lays out two tabs and moves data. Everything else is in `tensixinet/ui/`, which imports no UI library.

| Module | Job |
|---|---|
| `flow.py`, `chart.py`, `logview.py`, `progress.py` | The first tab: what is playing as data, as an animated SVG, as a coloured log, and the steps of a start |
| `session.py` | The first tab's session: `run_live` on a thread, with a sink that hands each segment to the page |
| `player.py` | `Player` (which session a page shows) and `Visitors` (one `Player` for each visitor, with a limit on how many play). Both tabs use it, so a visitor plays in one tab at a time |
| `forms.py` | The form's choices and how they become a session (`new_session`) or an instrument (`new_instrument`, `read_chip_knobs`) |
| `knobs.py` | The Play tab's performer: `Knobs` (five values for each of four chips, and pending events) and `to_telemetry`, which turns them into `Telemetry` through a `Tracker` |
| `scopes.py` | What the Play tab plays: `resolve_scope` turns `system`, `module:<name>` or `device:<name>` into the `only` list `render_chorus` takes |
| `instrument.py` | `InstrumentSession`: a thread that reads the knobs every block (1.2 s), renders a phrase when something changed or the last one ran out, and hands the audio on through the EQ and the live gain |
| `readout.py` | The Play tab's view: `play_html` is the flow chart of the block being heard (`chart.flow_svg`, from the snapshot's `FlowState`) above `readout_html`, pitch, tempo and pressure for each chip from the same `mapping.py` functions the devices use |

The Play tab's data flow: sliders and buttons write `Knobs` (per visitor, `knob_store` in the app) → the render thread reads them once a block →
`to_telemetry` → `render_chorus(only=...)` for a phrase → blocks cut from the phrase, crossfaded where a phrase changes → EQ → loudness → the page.

## Dependencies run one way

`telemetry` depends on nothing in the package. `music` depends on `telemetry`. Inside `music`, `devices.py` builds on
`ttrolzer`, `ttrax` and `ttquantussy`, and `ttsteam.py` builds on `devices.py` for the strike helpers. `narrate` depends on
`music` and `telemetry`. `media` depends on nothing. `run.py`, `live.py` and `cli.py` depend on all of them.
Keep it that way: `music` must never import `narrate`. `score` depends on `music` and `telemetry`, and `music` never imports `score`; `mido` is imported only inside `open_mido_output`.

## Why the tests are shaped like this

* A device is `f(telemetry, duration, sample_rate, seed, patch=DEFAULT_PATCH) -> float32 array`, so every device is
  tested the same way, with no hardware.
* `DEVICE_INPUTS` is a claim about what each device reads. `tests/narrate/test_explain.py` changes
  one field at a time and requires the audio to move exactly when the claim says it should, so the
  log cannot describe a mapping that is not in the sound.
* `tests/test_hygiene.py` reads the source: no module may import anything that opens a device,
  every package folder needs an `__init__.py`, and every module needs a test file in the matching
  folder.
