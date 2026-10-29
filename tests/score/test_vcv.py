# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Tests for the VCV Rack voice patch: what is wired to what, the numbers that match the score, and the packing."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from tensixinet.score import vcv
from tensixinet.score.routing import DEFAULT_ROUTES


def _mods(patch):
    return {m["model"]: m for m in patch["modules"]}


def _wires(patch):
    names = {m["id"]: m["model"] for m in patch["modules"]}
    return {(names[c["outputModuleId"]], c["outputId"], names[c["inputModuleId"]], c["inputId"]) for c in patch["cables"]}


def _param(module, pid):
    return {p["id"]: p["value"] for p in module["params"]}[pid]


def test_the_signal_path_is_wired_as_a_mono_voice():
    # The ids are Fundamental's and Core's own; each line below was checked by playing the patch in Rack.
    assert _wires(vcv.voice_patch()) == {
        ("MIDIToCVInterface", 0, "VCO", 0),  # note -> pitch
        ("MIDIToCVInterface", 4, "VCO", 1),  # wheel -> FM input
        ("MIDIToCVInterface", 1, "ADSR", 4),  # gate -> envelope
        ("VCO", 2, "VCF", 3),  # saw -> filter audio in
        ("ADSR", 0, "VCF", 0),  # envelope -> cutoff
        ("VCF", 0, "VCA", 2),  # low-pass out -> amplifier
        ("ADSR", 0, "VCA", 1),  # envelope -> level
        ("VCA", 0, "AudioInterface2", 0),
        ("VCA", 0, "AudioInterface2", 1),
    }


def test_ids_are_unique_and_every_cable_ends_on_a_module():
    p = vcv.voice_patch()
    ids = [m["id"] for m in p["modules"]]
    assert len(ids) == len(set(ids))
    cable_ids = [c["id"] for c in p["cables"]]
    assert len(cable_ids) == len(set(cable_ids))
    for c in p["cables"]:
        assert c["outputModuleId"] in ids and c["inputModuleId"] in ids
    for m in p["modules"]:
        assert len({q["id"] for q in m["params"]}) == len(m["params"])


def test_the_voice_listens_to_the_leads_channel_and_the_score_bend():
    p = vcv.voice_patch()
    m = _mods(p)
    assert m["MIDIToCVInterface"]["data"]["midi"]["channel"] == 0
    assert m["MIDIToCVInterface"]["data"]["channels"] == 1  # one voice
    assert _param(m["VCO"], 4) == pytest.approx(2.0 / 12.0 / 5.0)


def test_channel_and_bend_range_follow_the_arguments():
    m = _mods(vcv.voice_patch(channel=4, bend_range=12.0))
    assert m["MIDIToCVInterface"]["data"]["midi"]["channel"] == 4
    assert _param(m["VCO"], 4) == pytest.approx(12.0 / 12.0 / 5.0)


def test_bend_knob_turns_five_volts_into_the_range():
    assert vcv.bend_knob(2.0) * 5.0 * 12.0 == pytest.approx(2.0)
    assert vcv.bend_knob(7.0) * 5.0 * 12.0 == pytest.approx(7.0)


def test_the_voice_has_a_dark_filter_with_a_lot_of_resonance():
    vcf = _mods(vcv.voice_patch())["VCF"]
    assert _param(vcf, 0) == pytest.approx(0.30)  # cutoff
    assert _param(vcf, 2) == pytest.approx(0.80)  # resonance
    assert _param(vcf, 3) == pytest.approx(0.40)  # how far the envelope opens the cutoff


def test_the_envelope_is_a_plucked_one_that_sustains():
    adsr = _mods(vcv.voice_patch())["ADSR"]
    assert [_param(adsr, i) for i in range(4)] == [0.10, 0.40, 0.50, 0.35]


def test_device_and_audio_are_left_out_unless_given():
    m = _mods(vcv.voice_patch())
    assert "deviceName" not in m["MIDIToCVInterface"]["data"]["midi"]
    assert "audio" not in m["AudioInterface2"]["data"]
    assert m["AudioInterface2"]["data"]["dcFilter"] is True  # keeps a drifting offset out of the speakers
    m = _mods(vcv.voice_patch("RtMidiOut Client:tensixinet 128:0", {"driver": 2}))
    assert m["MIDIToCVInterface"]["data"]["midi"]["deviceName"] == "RtMidiOut Client:tensixinet 128:0"
    assert m["MIDIToCVInterface"]["data"]["midi"]["driver"] == 2  # ALSA
    assert m["AudioInterface2"]["data"]["audio"] == {"driver": 2}


def test_write_vcv_packs_one_patch_json_with_zstd_tar(tmp_path):
    calls = []

    def fake(cmd, check):
        calls.append((cmd, check))
        assert (Path(cmd[cmd.index("-C") + 1]) / "patch.json").exists()

    vcv.write_vcv(tmp_path / "v.vcv", {"version": "x"}, run=fake)
    cmd, check = calls[0]
    assert check is True
    assert cmd[:3] == ["tar", "--zstd", "-cf"] and cmd[3] == str(tmp_path / "v.vcv") and cmd[-1] == "patch.json"


@pytest.mark.skipif(not (shutil.which("tar") and shutil.which("zstd")), reason="needs tar and zstd")
def test_a_written_file_unpacks_to_the_same_patch(tmp_path):
    patch = vcv.voice_patch("x")
    vcv.write_vcv(tmp_path / "v.vcv", patch)
    out = subprocess.run(["tar", "--zstd", "-xOf", str(tmp_path / "v.vcv"), "patch.json"], capture_output=True, text=True, check=True).stdout
    assert json.loads(out) == patch


def test_alsa_input_name_finds_the_virtual_port():
    names = ["Midi Through:Midi Through Port-0 14:0", "RtMidiOut Client:tensixinet 128:0"]
    assert vcv.alsa_input_name(names=names) == "RtMidiOut Client:tensixinet 128:0"
    assert vcv.alsa_input_name(names=names[:1]) is None
    assert vcv.alsa_input_name(names=["Other:tensixinet-old 9:0"]) is None


def test_main_writes_the_file_and_says_so(tmp_path, monkeypatch, capsys):
    seen = {}
    monkeypatch.setattr(vcv, "write_vcv", lambda path, patch, run=None: seen.update(path=path, patch=patch))
    assert vcv.main([str(tmp_path / "a.vcv"), "--device", "dev", "--channel", "5", "--bend-range", "12"]) == 0
    m = _mods(seen["patch"])
    assert m["MIDIToCVInterface"]["data"]["midi"]["deviceName"] == "dev"
    assert m["MIDIToCVInterface"]["data"]["midi"]["channel"] == 4  # the flag is one-based
    assert _param(m["VCO"], 4) == pytest.approx(12.0 / 12.0 / 5.0)
    assert "listening to dev" in capsys.readouterr().out


def test_main_default_channel_is_the_leads(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(vcv, "write_vcv", lambda path, patch, run=None: seen.update(patch=patch))
    vcv.main([str(tmp_path / "a.vcv")])
    assert _mods(seen["patch"])["MIDIToCVInterface"]["data"]["midi"]["channel"] == DEFAULT_ROUTES.note_channels["lead"]


def test_main_auto_finds_the_port_or_refuses(tmp_path, monkeypatch, capsys):
    seen = {}
    monkeypatch.setattr(vcv, "write_vcv", lambda path, patch, run=None: seen.update(patch=patch))
    monkeypatch.setattr(vcv, "alsa_input_name", lambda *a, **k: "found:tensixinet 1:0")
    assert vcv.main([str(tmp_path / "a.vcv"), "--device", "auto"]) == 0
    assert _mods(seen["patch"])["MIDIToCVInterface"]["data"]["midi"]["deviceName"] == "found:tensixinet 1:0"
    seen.clear()
    monkeypatch.setattr(vcv, "alsa_input_name", lambda *a, **k: None)
    assert vcv.main([str(tmp_path / "a.vcv"), "--device", "auto"]) == 2
    assert not seen and "no tensixinet MIDI port" in capsys.readouterr().out


def test_main_refuses_a_channel_out_of_range(tmp_path, capsys):
    for bad in ("0", "17"):
        assert vcv.main([str(tmp_path / "a.vcv"), "--channel", bad]) == 2
    assert not (tmp_path / "a.vcv").exists()


@pytest.mark.skipif(not (shutil.which("tar") and shutil.which("zstd")), reason="needs tar and zstd")
def test_the_shipped_patch_is_what_the_builder_makes():
    shipped = Path(__file__).resolve().parents[2] / "vcv" / "tensixinet-voice.vcv"
    out = subprocess.run(["tar", "--zstd", "-xOf", str(shipped), "patch.json"], capture_output=True, text=True, check=True).stdout
    assert json.loads(out) == vcv.voice_patch()
