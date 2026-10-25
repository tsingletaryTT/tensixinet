# MIDI out

`--midi-file` and `--midi-port` hand out what the devices play as MIDI: notes, drum triggers and controllers. The audio does not change.
The events come from the same functions that make the sound, so a note in the MIDI is a note you hear (see `docs/DECISIONS.md`).

Status: the tests pass with no MIDI library and no synth. Nothing has been heard through a real synth or DAW. The live port has been tested against a
fake `mido` module only. The library is not installed on the box this was written on, so no real `mido` or `rtmidi` run has happened.

## What each device sends

| Device | Sends | Notes |
|---|---|---|
| `lead` | notes | Pitch and timing are the lead's own, rests removed |
| `ttquantus` | notes, with pitch bend | One note per held note that plays (the rest gate decides). With `quantus_mod` above 0 the glide and vibrato are a bend curve sampled every 20 ms |
| `gong` | triggers | One per strike |
| `ttrollz` | triggers | The roll clock and each lane that is up. A pulse that starts a roll's cycle has velocity 127 before scaling, any other 0.6 of that |
| `ttsteam` | triggers | One per register hit the snare plays |
| every chip | controllers | Temperature, power, pressure and chaos, each 0 to 1, read once at the start of each render and sent only when the 0 to 127 value changes |

The other devices (`ttdrone`, `ttrax`, `ttavdog`, `ttchoir`, `ttcoco`, the shimmer, the cables and the delay bus) send nothing. Velocity is the event's strength times the device's
balance and the turn gain at that moment (the same gain the audio gets), kept between 0.05 and 1. A chip that is idle under eviction has no notes or triggers. A chip that is
evicting or filling is scaled by the same fade as its audio. A note still sounding at the end of a render is ended there. In live mode each segment's score is cut where the
crossfade starts, so the crossfade tail is not sent twice.

## Default routes

Channels are shown 1 to 16.

| Source | Where | Number |
|---|---|---|
| `lead` | channel 1, note | pitch, with a pitch-bend remainder |
| `ttquantus` | channel 2, note | pitch, with a pitch-bend remainder and curve |
| `gong` | channel 10, note | 49 |
| `ttsteam` | channel 10, note | 38 |
| `ttrollz` clock | channel 10, note | 36 |
| `ttrollz` lanes | channel 10, note | 42, 44, 46, 54, 56, 60, 61, 62, 63, 64, cycling by lane index |
| temperature | channel 16, CC | 20 |
| power | channel 16, CC | 21 |
| pressure | channel 16, CC | 22 |
| chaos | channel 16, CC | 23 |

A trigger is a note-on followed 50 ms later (`pulse_s`) by a note-off. Pitch bend spans 2 semitones for the full wheel (`bend_range`). A pitch outside 0 to 127 is moved by
octaves into range, and a bend larger than the range is clamped. A note that starts on a number already sounding on its channel ends the earlier note at that moment.
Messages at the same time go out in this order: note-off, pitch bend, note-on, controller.

## The file

`--midi-file PATH` writes a format 1 Standard MIDI File at 480 ticks per quarter note. Track 0 holds the name `tensixinet` and one tempo, the first render's (`aiclk_to_bpm` of chip 0).
Chip `k` has a track named `tensixinet chip k`, and a chip with no messages has no track. The writer is in this repository (`score/smf.py`) and needs no library.
Offline, the file holds the single render. With `--live` it holds every segment and is written when the run ends. A tempo change between segments is not written.

## Live port

`--midi-port NAME` needs `--live` and the extra: `pip install -e .[midi]` (`mido` and `python-rtmidi`). The name is an existing MIDI output, or `virtual`, which makes an output called `tensixinet`.
A missing library or an unknown port name ends the run with one message and exit code 2, before any chip is read.

A scheduler thread sends each message at `start of run + segment start + time in segment + offset`. Adding a score never blocks the audio loop. If the port fails, or a score cannot be
routed, the sink keeps the first error, drops everything after it, and the live loop logs one `midi` line, "the MIDI port stopped: ...", and goes on playing. On exit the sink ends every
sounding note, sends all-notes-off (CC 123) on each channel it used and closes the port.

`--midi-offset-ms MS` shifts every message later (positive) or earlier (negative). It exists to line the MIDI up with the audio, which the player delays by an amount that depends on the player.
It has never been measured against real audio, so the default is 0 and the right value for a setup is found by ear.

### How to hear it

None of this has been tried with this code. The setup is the usual one for a virtual port.

1. Install the extra: `pip install -e .[midi]`.
2. Start a software synth or a DAW that listens for MIDI input. For FluidSynth with a General MIDI soundfont, `~/code/tt-midi-maker` has a worked setup (see its README and `INSTALL.md`).
3. Run `./tensixinet.py --live --midi-port virtual --player none` and connect the `tensixinet` output to the synth (a DAW lists it as an input; with FluidSynth use `aconnect` or the DAW's patchbay).
   Add `--midi-offset-ms` if the MIDI and the audio from tensixinet itself are both audible and drift.
4. Channel 10 is the General MIDI drum channel, so the gong, snare and rolls play as drums on a GM synth. Channel 16 carries controllers, which a synth ignores unless it maps CC 20 to 23.

To use other channels, notes or controllers, pass a map.

## `--midi-map`

A JSON file with any of these keys, merged over the defaults (the three mappings merge key by key). It needs `--midi-file` or `--midi-port`. Channels in this file are zero-based, 0 to 15,
so channel 10 is 9. Unknown keys and out-of-range numbers are refused with a message.

```json
{
  "note_channels": {"lead": 4, "ttquantus": 5},
  "trigger_notes": {"gong": 51, "ttsteam": 40},
  "roll_lane_notes": [42, 44, 46],
  "control_ccs": {"temperature": 1, "power": 74},
  "control_channel": 0,
  "trigger_channel": 9,
  "pulse_s": 0.1,
  "bend_range": 12
}
```

## The CV sink

Not built. The events are plain data (`Note`, `Trigger`, `Control` in `score/events.py`) and a sink takes a `Score` and a start time (`ScoreSink` in `score/sinks.py`), so a CV sink would be a third
sink beside the file and the port. A route from the control lanes into MIDI CCs for `~/code/vcv-cv-harness` (which takes MIDI CCs through MIDI-CAT) has not been tried. A direct CV output as in `~/code/tt-cv-agency` is open work.

## Limits

* Nothing has been heard through a real synth, and no real `mido` or `rtmidi` has run. The offset was never measured against real audio.
* No fast mode: `--interval` lives on another unmerged branch, and joining it to the score is on the roadmap.
* No MIDI clock, and no events from `ttchoir`.
* On a live port the chips share channels (every chip's lead plays on channel 1), so two chips' notes on one number retrigger each other. The file keeps chips apart with a track each.
* Pitch bend is channel-wide in MIDI, so overlapping notes on one channel share one bend.
* Controllers are sampled once at the start of each render. They are not continuous.
* A nearly dissolved chip still sends its quiet events until it is idle.
