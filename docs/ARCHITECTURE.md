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
| | `hwmon.py` | The chips' Linux hwmon files (`/sys/class/hwmon`): `find_chips` orders them by PCI address, `read_chip` gives temperature, power, voltage and current in C, W, V and A |
| | `fast.py` | `FastSource`: the fast fields from hwmon at every read, the slow fields (clocks, GDDR, errors, masks) from `tt-smi -s` on a background thread about every 1.5 s, matched by bus address. With no match it hands back the `tt-smi` readings and a note that says so |
| `music/` | `mapping.py` | The signal-to-sound table as functions, including pressure, chip identity and turn weight |
| | `synth.py` | Drone voices for any chip count |
| | `devices.py` | ttrollz, ttquantus, ttdrone, gong, ultrasonic shimmer, the strike times they share, and the current mixer. Re-exports ttrax. `choir_triggers` gives the six sources the choir can sing on (gong, rolls, register, quantus, lead, rax). `DEVICE_INPUTS` says which fields each device reads |
| | `ttrolzer.py` | The ttrolzer's rolls: barre tempos from E6 capacitors, steady even rolls, unstable odd ones |
| | `ttrax.py` | 4, 7 or 14 barres (Tetrax, Sidrax or Quatrax, from `patch.rax_bars`) bouncing between each other's oscillators, with press and release gestures |
| | `ttquantussy.py` | The ttquantussy's five petals and castles, and the coco's 8-bit buffer |
| | `ttsteam.py` | A shift register that steps a noise snare, with the harvested Tensix columns as its pattern |
| | `ttchoir.py` | A singer: a glottal buzz shaped by formant resonances, sung on vowels that the GDDR channels pick. With an onset source in the patch it sings syllables that start on that source's triggers, with an optional coda, a self-oscillating howl and a speech-chip crush. Both choirs end in one sum of sines (`harmonic_sum`); `choir_voice(..., sines=)` hands that sum to the kernel backend and keeps everything else on the host |
| | `consonants.py` | The choir's twelve consonants (fricatives, plosives, nasals) as pure signal functions, and how a chip's readings and a trigger's strength choose one |
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
| | `player.py` | Where live audio goes: a `paplay`, `aplay` or `ffplay` process, a growing WAV, or no sound at the speed of the music. `player_command` and `player_buffer_ms` give the small-buffer flags fast mode asks for |
| | `toplike.py` | `--with-tt-toplike`: builds the tmux commands for a synth pane and a tt-toplike pane at one interval, and runs them (outside tmux on its own socket, `tmux -L tensixinet -f media/tmux-toplike.conf`: status bar off, truecolor). The panes tag themselves so that each ending closes the other |
| | `chip.py` | `ChipWorker`, the client of `kernels/pool.py` and `kernels/worker.py` (a subprocess under `gozer run`), and `CHIP_OPS`, the seven kernels and how many packed inputs each takes |
| `kernels/` | `packing.py` | Samples to 32 x 32 tiles and back, and each kernel's host side: the arguments it is given (phases reduced to one turn in float64) and its formula in numpy (`*_on_host`). For the choir, `choir_groups` packs eight harmonics a run |
| | `backend.py` | `KernelBackend`: the six devices with kernels (gong, shimmer, drone ripple, AVDog, ttcoco, choir) and the mixer, run through a chip worker, falling back to the CPU on a `ChipError` |
| | `governor.py` | The limits for a load burst on a chip (`MAX_BURST_S`, `ABORT_PRESSURE`); no load kernel uses it yet |
| `score/` | `events.py` | `Note`, `Trigger`, `Control` and `Score`, with the Hz-to-MIDI helpers |
| | `collect.py` | One extractor per device and `chorus_score`, which builds a render's `Score` the way `render_chorus` deals the devices |
| | `vcv.py` | Builds the VCV Rack voice patch (`vcv/tensixinet-voice.vcv`): one mono voice on the lead's channel; `python -m tensixinet.score.vcv` writes a copy for another channel or port |
| | `routing.py` | The route table (`Routes`, `DEFAULT_ROUTES`) and `load_routes` for `--midi-map` |
| | `midi.py` | `to_messages`: a `Score` to timed MIDI bytes |
| | `smf.py` | The Standard MIDI File writer |
| | `sinks.py` | `MidiFileSink`, `MidiPortSink` (a thread and a heap, never blocks the audio), `TeeScoreSink`, and the lazy `mido` opener |
| `tui/` | `layout.py` | `--tui`: turns a `FlowState` into rows of styled text for a width and a height: a readings line per chip, the devices grouped by role (RHYTHM, MELODY, AIR, MOD/DELAY), the five strongest cables, the stereo line, and what is left of the height for the log. Below 60x16 it is one line |
| | `frame.py` | The grid of styled cells, the dim levels, and the ANSI that redraws only the cells that changed |
| | `logpane.py` | Wraps log records to the pane's width and colours them with `narrate/color`'s jack, chip and device colours |
| | `playhead.py` | `Playhead`: holds each segment's state until the listener reaches it, keeps the newest 400 log records, and wraps the sink to learn when audio is written |
| | `app.py` | `TuiApp` and the terminal: the alternate screen, cbreak input (so Ctrl-C still works), `q`, resizes, at most 10 frames a second, and the exit lines |
| (top) | `fast.py` | `run_fast`, the loop behind `--interval`, and the delay report |

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

## Fast mode

`--live --interval MS` does not use `live.py`. `fast.py` reads every block and borrows the Play tab's phrase engine:

```
 /sys/class/hwmon ─► telemetry/hwmon ─┐
                                      ├─► telemetry/fast.FastSource ─► Tracker(0.5 s window, hold) ─► read(now)
 tt-smi -s (thread, ~1.5 s) ──────────┘                                                                  │
                                                                                                        ▼
  ui/instrument.InstrumentSession: change detection (thresholds, 1 s minimum gap) ► head (0.6 s) on the loop,
  body (1 s or 6 s phrase) on a worker thread ► crossfade ► EQ ► loudness ► blocks ► media/player (paplay, 20 ms)
```

`run_fast` drives `InstrumentSession.render_block` from its own loop and paces itself to stay `lead_for(rate)` ahead of the listener (0.10 s at 22050 Hz,
0.18 s at 44100 Hz). Each block's delay is the interval, plus the time from reading to writing the block, plus the audio queued in front of it, plus the player's
buffer, plus, for a change that arrived less than 1 s after the last phrase started, the time it waited for that gap (`InstrumentSession.held_s`). `run_fast`
logs the worst and typical values every 5 s. They are calculated from the loop's own timings and the player buffer it asked for; the delay at the speaker is
not measured. If three head renders in a row outrun the block, the loop lengthens the block (and the lead with it) and logs it; after 60 blocks in a row
whose render fits the interval asked for, it returns to that interval and logs that too. The session keeps the soloist for the patch's turn length (`turn_seconds`). `music/evict.py` smooths power by elapsed seconds, so ten readings a
second evict at the same time as one reading a second. `InstrumentSession` takes its read function, block, phrase and crossfade lengths, sample rate, `only`, `eq` and
`normalise` as parameters; the Play tab passes none of them and behaves as before.

With `--tui`, `run_fast` calls `on_segment(seconds, state)` just before it queues each block. The state is `InstrumentSession.handed_state`, the `FlowState` of the newest block
made, and the `Playhead` shows it when the listener reaches that block (the player's buffer is counted). The plain `live.py` loop calls the same hook once per segment, with
the state it built for the segment (`build_flow`, the function the Gradio chart uses).

`--with-tt-toplike` runs no chips itself. `cli.py` builds a tmux session through `media/toplike.py`, and each pane runs its own program. The synth pane runs
this same command line without `--with-tt-toplike` and with `TENSIXINET_IN_SESSION=1`.

## Dependencies run one way

`telemetry` depends on nothing in the package. `music` depends on `telemetry`. Inside `music`, `devices.py` builds on
`ttrolzer`, `ttrax` and `ttquantussy`, `ttchoir.py` builds on `devices.py` and `consonants.py` (which imports nothing from the package), and `ttsteam.py` builds on `devices.py` for the strike helpers. `narrate` depends on
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
