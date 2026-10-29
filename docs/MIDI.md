# MIDI out

`--midi-file` and `--midi-port` hand out what the devices play as MIDI: notes, drum triggers and controllers. The audio does not change.
The events come from the same functions that make the sound, so a note in the MIDI is a note you hear (see `docs/DECISIONS.md`).

Status: the tests pass with no MIDI library and no synth. Nothing has been heard through a real synth or DAW. The live port has been tested against a
fake `mido` module only. `mido` and `python-rtmidi` are installed in `~/.tenstorrent-venv` (the Python that `python3` uses on this box), so the library is present. Nobody has opened a real port and connected a synth to it.

## What each device sends

| Device | Sends | Notes |
|---|---|---|
| `lead` | notes | Pitch and timing are the lead's own, rests removed |
| `ttquantus` | notes, with pitch bend | One note per held note that plays (the rest gate decides). The note number is the held scale step (the castle's degree plus the register shift) at the note's middle. With `quantus_mod` above 0 the glide and vibrato are a bend curve sampled every 20 ms. A glide wider than the bend range (2 semitones by default) is clamped, so a big jump between two notes is heard in MIDI as a bend that stops at the range |
| `gong` | triggers | One per strike |
| `ttrollz` | triggers | The roll clock and each lane that is up. A pulse that starts a roll's cycle has velocity 127 before scaling, any other 0.6 of that |
| `ttsteam` | triggers | One per register hit the snare plays |
| every chip | controllers | Temperature, power, pressure and chaos, each 0 to 1, read once at the start of each render. Within one score a controller is sent only when its 0 to 127 value changes. In live mode each segment is its own score, so each segment starts by sending all four values again |

The other devices (`ttdrone`, `ttrax`, `ttavdog`, `ttchoir`, `ttcoco`, the shimmer, the cables and the delay bus) send nothing. Velocity is the event's strength times the device's
balance and the turn gain at that moment (the same gain the audio gets), kept between 0.05 and 1. A chip that is idle under eviction has no notes or triggers. A chip that is
evicting or filling is scaled by the same fade as its audio, and a voice whose fade factor is at or below 0.02 sends nothing (it has dissolved in the audio). A note still sounding at the end of a render is ended there. In live mode each segment's score is cut where the
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

The VCV Rack route below has been run end to end. The synth and DAW routes in this list have not been tried with this code; the setup is the usual one for a virtual port.

1. Install the extra: `pip install -e .[midi]`.
2. Start a software synth or a DAW that listens for MIDI input. For FluidSynth with a General MIDI soundfont, `~/code/tt-midi-maker` has a worked setup (see its README and `INSTALL.md`).
3. Run `./tensixinet.py --live --midi-port virtual --player none` and connect the `tensixinet` output to the synth (a DAW lists it as an input; with FluidSynth use `aconnect` or the DAW's patchbay).
   Add `--midi-offset-ms` if the MIDI and the audio from tensixinet itself are both audible and drift.
4. Channel 10 is the General MIDI drum channel, so the gong, snare and rolls play as drums on a GM synth. Channel 16 carries controllers, which a synth ignores unless it maps CC 20 to 23.

To use other channels, notes or controllers, pass a map.

## A voice on top, in VCV Rack

`vcv/tensixinet-voice.vcv` is a Rack patch with one monophonic voice. It listens on MIDI channel 1, where the lead plays. A note makes a saw wave. The wave passes through a low-pass filter with the cutoff
low (0.30) and the resonance high (0.80). An envelope opens the cutoff on each note and also sets the level. The pitch wheel bends the voice by the score's two semitones. It uses only VCV's free modules
(Core and Fundamental).

1. Install the extra (`pip install -e .[midi]`) and start tensixinet with a virtual port: `./tensixinet.py --live --midi-port virtual`.
2. Open `vcv/tensixinet-voice.vcv` in VCV Rack (Free is enough). Click the MIDI-CV module's device line and pick `RtMidiOut Client:tensixinet` (the number after it changes). Pick the audio output on the Audio module.
   To skip the first click, write a copy that already names the port: `python -m tensixinet.score.vcv my-voice.vcv --device auto` while tensixinet is running.
3. The voice plays over tensixinet's own sound. If the two drift apart, move the voice with `--midi-offset-ms` (positive is later).

The patch also has a scope on the voice's output. `--audio-device NAME` on the same command names the PulseAudio or PipeWire output, so the copy needs no clicks at all.

Other channels: `--channel N` and `--bend-range SEMITONES` on `python -m tensixinet.score.vcv` write a patch for another channel or a `--midi-map` with a different `bend_range`. Point it at `ttquantus` with `--channel 2`.
The patch is made by `score/vcv.py` and a test checks that the file in the repository is what the builder writes.

**Timing.** `--midi-offset-ms` was measured once for this route. With tensixinet playing only the lead (`--only lead`) into one null sink and the Rack voice into another, both recorded at once for 38 s, the correlation of the two sounds' loudness
curves peaked at 0.67 with the voice 30 ms ahead of the lead (0.65 at 20 ms, 0.64 at 40 ms; the two sounds have different attacks, so the real figure is within a few tens of milliseconds). The default of 0 stays. A player that buffers more, or a Rack audio buffer other than 512 samples at 48 kHz, moves this.

**A recording.** `demo/vcv_record.sh` records the pair in one 1080p frame with the sound of both: tensixinet's log top left, the MIDI it sends (`aseqdump`) top right, Rack with the voice and its scope along the bottom. It runs on a private X server (Xvfb, software GL),
a private sound sink and a private Rack user folder, so it does not touch your desktop or your Rack settings, and needs no screen-share dialog. `VCV_SECONDS=45 demo/vcv_record.sh` for a shorter take; the file is `build/vcv_rec/vcv.mp4`. The sound is what the two programs played together
(mean level -17 dB, peak -0.5 dB over a two-minute take).

What was measured, on this box with Rack Free 2.6.6 running without a window (`Rack -h`) into a null audio sink, and notes sent with `mido` into the virtual port: A4 plays at 440.0 Hz, a full wheel up plays 493.8 Hz
and down 392.0 Hz (two semitones either side), C4 plays at 261.7 Hz, and a note on channel 2 makes no sound. A live run of four synthetic chips into the voice produced sound (RMS 0.04, peak 0.20); the notes
in that run were not compared with the audio one by one. Nobody has listened to it. The cutoff, resonance and envelope times are a first guess and are best set by ear in Rack. The chips share the lead's channel, so on a live port two chips'
lead notes can overlap on a single voice. What the voice does then was not checked.

## Over the network (rtpMIDI), tried with rtpmidid

tensixinet needs no code for this. The virtual port is an ordinary ALSA port, so `rtpmidid` (David Moreno's daemon, which speaks Apple's network MIDI session protocol, RTP-MIDI) can carry it to a Mac or iPad. A Mac shows the session in Audio MIDI Setup, Network.
`rtpmidid` is not in apt. Upstream publishes `.deb` files (v26.01 was used). `./tensixinet.py --rtpmidi-setup` fetches the `.deb` built for this Ubuntu and CPU, checks that the package is called `rtpmidid`, and unpacks it with `dpkg -x` into `~/.local/share/tensixinet/rtpmidid`, so nothing is installed and no sudo is used. It then prints the commands. `--doctor` says whether `rtpmidid` is found (on the PATH or there). The helper covers Ubuntu 24.04 and 25.10 on x86_64 and arm64, the Ubuntu builds upstream publishes (upstream also has Debian trixie and Fedora 43 packages, which the helper does not fetch); on another system it says so.

What was run, on one machine, with two `rtpmidid` instances standing in for the two ends (ports 15004 and 15104, discovery off):

```bash
rtpmidid --port 15004 --name rtpA --control /tmp/rtp/A.sock --rtpmidi-discover false &
rtpmidid --port 15104 --name rtpB --control /tmp/rtp/B.sock --rtpmidi-discover false &
tensixinet --live --midi-port virtual
aconnect 130:0 128:0            # tensixinet's port to rtpA's port (use the numbers aconnect -l shows)
rtpmidid-cli --control /tmp/rtp/B.sock connect hostname=127.0.0.1 port=PORT name=tensixinet
aseqdump -p 129:3               # what arrived at the far end
```

The `aconnect` line makes `rtpA` open a session named `RtMidiOut Client-tensixinet` on a UDP port of its own. `rtpmidid-cli ... status` on A shows that port. `PORT` is that number, and it changes each run.
For a real Mac, skip instance B: leave `rtpmidid` running, connect the port to it with `aconnect`, and pick the session in Audio MIDI Setup (or use `connect` from the daemon to the Mac's session).

Measured: `aseqdump` on B's port received the lead's notes, pitch bends and the gong's triggers, with the channels and note numbers the score sends (124 note-on events in 10 s from four synthetic chips). The test did not check them one by one against what was sent.
Not tested: Apple's own driver, a Mac or iPad, a network that is not loopback, latency and jitter, `rtpmidid`'s mDNS discovery (it was off), or the `rtpmidid` package from a distribution. Nobody has listened to it.

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

* Nothing has been heard through a real synth, and no one has connected a real port to a synth (the library is installed here). The offset was never measured against real audio.
* No fast mode: `--interval` with `--midi-file` or `--midi-port` is refused with a message, because fast mode builds no score. Joining them is on the roadmap.
* No MIDI clock, and no events from `ttchoir`.
* On a live port the chips share channels (every chip's lead plays on channel 1), so two chips' notes on one number retrigger each other. The file keeps chips apart with a track each.
* Pitch bend is channel-wide in MIDI, so overlapping notes on one channel share one bend.
* Controllers are sampled once at the start of each render. They are not continuous.
* A voice that is dissolving sends quiet events until its fade factor reaches 0.02, then none. Other quiet voices keep a velocity floor of 0.05.
