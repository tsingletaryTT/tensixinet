# MIDI and CV out: design

Date: 2026-10-24. Status: self-approved by the builder at the owner's request ("self-approve your specs and then execute them").
The owner's own words: "we could go the DAW route, a midi file route, a live midi device route ... someday CV route like in
`~/code/vcv-cv-harness` (tt-cv-agency) along with VCV ... let's not block ourselves in."

## Intent

Several devices make events that are musical in themselves: the lead's notes, the quantus line, the gong strikes, the roll
pulses, the register hits, and slow controls such as pressure and chaos. Today they exist only inside the audio render. This
work hands the same events out as data, and turns them into MIDI (a file for a DAW, a live port for a synth) with a seam
that a CV sink can use later.

## Goals

- A `Score`: timed, typed events in physical units, produced by the same code paths that make the audio.
- `--midi-file PATH`: write a Standard MIDI File for an offline run and for a live run (the file grows as segments play).
- `--midi-port NAME`: send the same events to a live MIDI output port, with `--midi-offset-ms` to line MIDI up with the audio.
- `--midi-map FILE`: a JSON file overriding the default routing (channels, trigger note numbers, CC numbers).
- Controls leave as MIDI CCs on named lanes, so a VCV Rack patch mapped through MIDI-CAT (as `vcv-cv-harness` does) can use them.
- The audio does not change: every existing render is byte-identical with and without these flags.

## Non-goals

- A DC-coupled CV sink (Expert Sleepers or similar). The event types and the route table are built so one can be added.
- MIDI in (a controller driving the Play tab's knobs).
- MIDI clock messages on a live port (the file carries the tempo). Listed in the roadmap.
- `--midi-*` together with fast mode (`--interval`): that code is on another branch, and its phrases restart every device's clock.
  The roadmap records the join.
- Events for `ttchoir`, `ttrax`, `ttcoco`, `ttdrone`, shimmer, AVDog: they are audio-rate or share another device's notes.
  `ttcoco` echoes `ttquantus`'s notes, so those notes stand for both. `ttchoir` pitches are a later addition.

## Anti-goals

- Events that disagree with the audio. For the lead and `ttrollz`, whose events and audio are made in one loop, the loop is
  split so the audio reads the event list. One source for both, with a characterisation test that the audio is unchanged.
- A new required dependency. Writing a `.mid` file uses a small writer in this repo. `mido` and `python-rtmidi` are an
  optional extra (`pip install -e .[midi]`), imported only when `--midi-port` is used.
- A live port that blocks the audio loop.

## Design

**Package `tensixinet/score/`.**

- `events.py`: `Note(t, dur, pitch_hz, velocity, source, chip, bend)`, `Trigger(t, strength, source, chip, lane)`,
  `Control(t, name, value, chip)`, `Score(events, duration, bpm)`. Times are seconds from the start of the segment.
  `velocity`, `strength` and `value` are 0 to 1. `bend` is a tuple of `(seconds into the note, semitones from pitch_hz)` points.
  Helpers `hz_to_midi(hz) -> float` and `note_and_bend(hz) -> (int note, float semitone remainder)`.
- `collect.py`: per-device extractors and `chorus_score(telemetry, duration, patch, segment, soloists, only, defrag) -> Score`,
  which deals devices to chips, seeds and turn gains exactly as `render_chorus` does and scales velocity by the turn gain and
  the patch's balance for that device. A chip that is idle under `defrag` emits nothing.
- `routing.py`: `Routes` (a frozen dataclass): MIDI channel per source, note number per trigger source (and per roll lane),
  CC number per control name, the pulse length of a trigger, the bend range in semitones. `DEFAULT_ROUTES`, and
  `load_routes(path)` (JSON, unknown keys refused, partial files merge over the defaults).
- `midi.py`: `to_messages(score, routes) -> list[(seconds, bytes)]`, a pure function: notes become pitch bend (set before the
  note) + note on + bend points + note off; triggers become a note on and an off after the pulse length; controls become CC
  values 0 to 127, sent when they change.
- `smf.py`: `write_midi_file(path, messages, bpm, ppq=480)`, a Standard MIDI File (format 1, one track per chip, a tempo track).
- `sinks.py`: `MidiFileSink` and `MidiPortSink` and `TeeScoreSink`, each with `add(score, start_s)` and `close()`.
  `MidiPortSink` keeps its own scheduler thread on a monotonic clock, takes its output port and clock from arguments (so a test
  gives it a fake), sends all-notes-off on `close()`, and never raises into the caller (a port error is logged once and the
  sink goes quiet).

**Events from the devices.** The functions that already return times are used directly: `strike_times` (gong), `roll_pulses`
(each heard roll lane; a pulse that begins a cycle is stronger), `step_times` with the register's pattern (ttsteam, scaled by
the patch's `snare` share). The lead and `ttquantus` notes are made inline today. `lead` is split into `lead_notes(...)`,
which returns `(start_s, dur_s, pitch_hz)` for every note that plays (rests removed) and `lead` renders from that list.
`ttquantus`'s notes are read from `ttquantus_steps` and `ttquantus_gate`: a note is one eighth, its pitch is the held scale
step plus the register shift, glide and vibrato appear in `bend`, and a note whose gate is shut is a rest and is dropped.
`ttrollz`'s pulse selection (including the roll rests) is moved into one function that both the audio and the extractor call.
Controls come once per segment per chip: temperature, power, pressure and chaos (0 to 1, from `mapping.py`'s own scales).

**Live and offline wiring.** `run` builds a `Score` for the whole render and hands it to the sinks. `run_live` builds one for
each segment (same `segment`, `soloists` and `defrag` as the render) and hands it over with the segment's start in seconds;
events that start after the segment's own length (the crossfade tail) are dropped, since the next segment plays them. The port
sink's time base is the wall clock when the first segment's audio is handed to the player, plus `--midi-offset-ms`.

**Routing defaults.** One channel per source (lead 1, ttquantus 2, ttrollz 10 with a note per lane, gong 10, ttsteam 10,
controls 16 for the CCs), chosen once in `DEFAULT_ROUTES` and listed in `docs/MIDI.md`. In a file each chip is its own track.
On a live port the chips share the channel.

**CV later.** A CV sink takes the same `Score` and `Routes`: controls map to lanes the way `vcv-cv-harness`'s channel file
names them, and a DC-coupled sink would write the same controls as samples at a fixed volts-per-unit. Nothing in `collect.py`
or `events.py` mentions MIDI.

## Constraints from the repo

Test first, then a mutation check; SPDX header on every `.py`; a test file for every module in the matching `tests/` folder;
no `ttnn`, `ttl`, `ttml`, `torch` or `sounddevice` imports; docs that name flags or numbers are updated in the same change
(`README.md`, `docs/ARCHITECTURE.md`, `docs/SOUND.md` where it lists inputs, `docs/ROADMAP.md`, the `CLAUDE.md` layout and log);
plain English with no "X, not Y" framing; sweep the new prose for `, not `, `rather than`, `instead of`, `reads as`, `honest`.
The pytest run needs no hardware and no MIDI library.

## Left to the builder

Exact GM note numbers for the triggers (listed in the docs), the strength of a cycle-start pulse against an ordinary one, the
sampling step of a bend curve, the control smoothing, and the module split inside `score/` beyond the names above.

## Open for the owner

- Whether a chip should get its own channel on a live port (16 channels cap it at a few chips).
- Whether `ttchoir` should join with its pitches and vowels as CCs.
- MIDI clock on a live port.
