# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""A VCV Rack patch that plays tensixinet's MIDI: one monophonic voice, a low-pass filter and a lot of resonance.

The voice listens on MIDI channel 1, where the lead plays. The notes make a saw wave. The wave goes through a resonant low-pass filter
whose cutoff an envelope opens on every note, then through an amplifier on the same envelope. Pitch bend moves the pitch by the same
two semitones the score uses (`Routes.bend_range`).

A `.vcv` file is a zstd-compressed tar holding one `patch.json`. The module numbers below are Fundamental's and Core's own
(checked against `sequencer_test.vcv` in `~/code/vcv-cv-harness` and by playing the result in Rack).
"""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Dict, List, Optional

from tensixinet.score.routing import DEFAULT_ROUTES

RACK_VERSION = "2.6.4"
ALSA = 2  # RtMidi's ALSA backend, the driver number Rack saves

# Module ids in the patch.
MIDI, VCO, VCF, ADSR, VCA, OUT, SCOPE = 1, 2, 3, 4, 5, 6, 7

# Core MIDI-CV outputs, Fundamental VCO / VCF / ADSR / VCA ports and parameters.
MIDI_PITCH, MIDI_GATE, MIDI_BEND = 0, 1, 4
VCO_PITCH_IN, VCO_FM_IN, VCO_SAW_OUT = 0, 1, 2
VCO_FM_PARAM = 4
VCF_CUTOFF_IN, VCF_AUDIO_IN, VCF_LOW_OUT = 0, 3, 0
VCF_CUTOFF_PARAM, VCF_RES_PARAM, VCF_CUTOFF_CV_PARAM = 0, 2, 3
ADSR_GATE_IN, ADSR_OUT = 4, 0
VCA_AUDIO_IN, VCA_OUT = 2, 0
VCA_LEVEL_IN = 1
SCOPE_X_IN = 0
SCOPE_GAIN_PARAM, SCOPE_TIME_PARAM = 0, 4  # gain 1 and the time base; the rest are Rack's defaults

# The wheel's full swing is 5 V on MIDI-CV (measured in Rack: with the VCO's FM knob at 2 / 12 / 5, a full bend up from A4 plays B4).
BEND_VOLTS = 5.0

CUTOFF = 0.30  # knob position, 0 to 1
RESONANCE = 0.80
ZOOM = 1.25  # the modules fill a 540 pixel high window
ENVELOPE_TO_CUTOFF = 0.40


def _param(values: Dict[int, float], n: int) -> List[dict]:
    return [{"id": i, "value": values.get(i, 0.0)} for i in range(n)]


def _module(mid: int, plugin: str, model: str, x: int, params: List[dict], data: Optional[dict] = None) -> dict:
    m = {"id": mid, "plugin": plugin, "model": model, "version": RACK_VERSION, "params": params, "pos": [x, 0]}
    if data is not None:
        m["data"] = data
    return m


def bend_knob(bend_range: float) -> float:
    """The VCO's FM knob that turns MIDI-CV's bend output (5 V for the full wheel) into `bend_range` semitones (1 V is 12)."""
    return bend_range / 12.0 / BEND_VOLTS


def voice_patch(device: Optional[str] = None, audio: Optional[dict] = None, channel: int = DEFAULT_ROUTES.note_channels["lead"],
                bend_range: float = DEFAULT_ROUTES.bend_range) -> dict:
    """The patch as a dict. `device` is the ALSA name of the MIDI input, or None to pick it in Rack's menu. `audio` is an
    AudioInterface2 `audio` block, or None to pick the output in Rack's menu. `channel` is the zero-based MIDI channel the voice
    plays and `bend_range` the semitones the full wheel covers; both default to the score's own routes."""
    midi_in = {"driver": ALSA, "channel": channel}
    if device is not None:
        midi_in["deviceName"] = device
    out_data = {"dcFilter": True}
    if audio is not None:
        out_data["audio"] = audio
    modules = [
        _module(MIDI, "Core", "MIDIToCVInterface", 0, [], {"channels": 1, "midi": midi_in}),
        _module(VCO, "Fundamental", "VCO", 14, _param({1: 1.0, VCO_FM_PARAM: bend_knob(bend_range), 5: 0.5}, 8)),
        _module(VCF, "Fundamental", "VCF", 24, _param({VCF_CUTOFF_PARAM: CUTOFF, VCF_RES_PARAM: RESONANCE, VCF_CUTOFF_CV_PARAM: ENVELOPE_TO_CUTOFF}, 7)),
        _module(ADSR, "Fundamental", "ADSR", 32, _param({0: 0.10, 1: 0.40, 2: 0.50, 3: 0.35}, 9)),
        _module(VCA, "Fundamental", "VCA", 38, _param({0: 1.0, 1: 1.0}, 2)),
        _module(OUT, "Core", "AudioInterface2", 44, [], out_data),
        _module(SCOPE, "Fundamental", "Scope", 53, _param({SCOPE_GAIN_PARAM: 1.0, SCOPE_TIME_PARAM: 2.5, 7: 1.0}, 8)),
    ]
    wires = [
        (MIDI, MIDI_PITCH, VCO, VCO_PITCH_IN),
        (MIDI, MIDI_BEND, VCO, VCO_FM_IN),
        (MIDI, MIDI_GATE, ADSR, ADSR_GATE_IN),
        (VCO, VCO_SAW_OUT, VCF, VCF_AUDIO_IN),
        (ADSR, ADSR_OUT, VCF, VCF_CUTOFF_IN),
        (VCF, VCF_LOW_OUT, VCA, VCA_AUDIO_IN),
        (ADSR, ADSR_OUT, VCA, VCA_LEVEL_IN),
        (VCA, VCA_OUT, OUT, 0),
        (VCA, VCA_OUT, OUT, 1),
        (VCA, VCA_OUT, SCOPE, SCOPE_X_IN),
    ]
    cables = [
        {"id": 100 + i, "outputModuleId": a, "outputId": b, "inputModuleId": c, "inputId": d, "color": "#1b8eb1"}
        for i, (a, b, c, d) in enumerate(wires)
    ]
    return {"version": RACK_VERSION, "zoom": ZOOM, "gridOffset": [0.0, 0.0], "modules": modules, "cables": cables}


def write_vcv(path: Path, patch: dict, run: Callable = subprocess.run) -> None:
    """Pack `patch` as `path`, a zstd-compressed tar with one `patch.json`. Needs `tar` and `zstd` on the PATH."""
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "patch.json").write_text(json.dumps(patch, indent=1))
        run(["tar", "--zstd", "-cf", str(path), "-C", tmp, "patch.json"], check=True)


PULSE = 2  # RtAudio's PulseAudio backend, the driver number Rack saves


def pulse_audio(device: str, sample_rate: float = 48000.0, block_size: int = 512) -> dict:
    """An AudioInterface2 `audio` block for the PulseAudio (or PipeWire) output called `device`."""
    return {"driver": PULSE, "deviceName": device, "sampleRate": sample_rate, "blockSize": block_size, "inputOffset": 0, "outputOffset": 0}


def alsa_input_name(prefix: str = "tensixinet", names: Optional[List[str]] = None) -> Optional[str]:
    """The name Rack lists for tensixinet's virtual port (`--midi-port virtual`), or None when it is not running.
    ALSA names it `RtMidiOut Client:tensixinet 128:0`, and the client number changes from run to run."""
    if names is None:
        import mido  # lazy: the `midi` extra

        names = mido.get_input_names()
    for n in names:
        if f":{prefix} " in n:
            return n
    return None


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="python -m tensixinet.score.vcv", description="Write the VCV Rack voice patch.")
    ap.add_argument("out", help="the .vcv file to write")
    ap.add_argument("--channel", type=int, default=DEFAULT_ROUTES.note_channels["lead"] + 1, help="MIDI channel to play, 1 to 16 (default: the lead's)")
    ap.add_argument("--bend-range", type=float, default=DEFAULT_ROUTES.bend_range, help="semitones the full pitch wheel covers; match --midi-map's bend_range")
    ap.add_argument("--audio-device", help="name of the PulseAudio or PipeWire output for the Audio module; default: pick it in Rack")
    ap.add_argument("--device", help="ALSA name of the MIDI input; 'auto' finds the running tensixinet port; default: pick it in Rack")
    args = ap.parse_args(argv)
    device = args.device
    if device == "auto":
        device = alsa_input_name()
        if device is None:
            print("no tensixinet MIDI port is running; start `tensixinet --live --midi-port virtual` first")
            return 2
    if not 1 <= args.channel <= 16:
        print("--channel must be 1 to 16")
        return 2
    audio = pulse_audio(args.audio_device) if args.audio_device else None
    write_vcv(Path(args.out), voice_patch(device, audio, channel=args.channel - 1, bend_range=args.bend_range))
    print(f"wrote {args.out}" + (f" listening to {device}" if device else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
