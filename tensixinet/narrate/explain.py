# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Plain-language what/why for every choice the chorus makes.

Numbers come from the same mapping functions the audio uses, and only inputs a
device really reads (devices.DEVICE_INPUTS) are mentioned, so the log cannot
describe a mapping that is not in the sound.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import math

from ..music.cables import Cable, strength
from ..music.chorus import DEVICES, assign_device, chip_devices, n_slots, plan_turns, slot_chip
from ..music.devices import (
    DEVICE_INPUTS,
    REFERENCE_AXICLK,
    SHIMMER_BEAT_HZ_PER_DEG,
    SHOCK_DECAY_BEATS,
    SHOCK_RATE_BEATS,
    TRIGGER_RESPONSE_SPREAD,
    TRIGGER_RESPONSE_STEPS,
    gong_detune_cents,
    gong_trigger_roll,
    strike_times,
    ttrollz_network,
)
from ..music.space import ORBIT_CALM, ORBIT_CHAOS, ORBIT_HZ_PER_BPM, chip_azimuth
from ..music.ttrolzer import draw_links, roll_drift
from ..music.ttsteam import STEPS, dust_per_second, snare_decay, step_pattern
from ..music.ttchoir import voice_scale, vowel_sequence
from ..music.ttquantussy import castle_offset
from ..music.ttcoco import N_TAPS, tap_bits, tap_delay_ms, tap_feedback
from ..music.ttrax import FALLBACK_ARCCLK, N_BARRES, barre_bounds
from ..music.events import MAX_BREATHS, MAX_SWEEPS, MAX_TICKS
from ..music.lead import BRIGHTNESS, identity_offset, lead_transforms
from ..music.patch import (
    DEFAULT_PATCH,
    DEGREES_PER_SCALE_STEP_C,
    GONG_TUNING_NAMES,
    SCALE_NAMES,
    Patch,
    gddr_spread,
    memory_to_asic_ratio,
)
from ..music.mapping import (
    BUSY_WEIGHT_MAX_PROCESSES,
    BUSY_WEIGHT_PER_PROCESS,
    PRESSURE_WEIGHT,
    aiclk_to_bpm,
    chip_chaos,
    chip_pressure,
    ddr_lane_sides,
    rate_activity,
    drone_voices,
    eth_lane_levels,
    heartbeat_phase,
    personality,
    power_to_volume,
    temp_to_pitch_hz,
    voltage_to_cents,
)
from ..telemetry import Telemetry


@dataclass(frozen=True)
class Decision:
    what: str
    why: str


def explain_chip(index: int, t: Telemetry, n_chips: Optional[int] = None, patch: Patch = DEFAULT_PATCH) -> List[Decision]:
    """What this chip plays and what each reading it feeds into those devices becomes."""
    n_chips = len(DEVICES) if n_chips is None else n_chips
    devices = [d.__name__ for d in chip_devices(index, n_chips)]
    uses = frozenset().union(*(DEVICE_INPUTS[d] for d in devices))
    slots = [s for s in range(n_slots(n_chips)) if slot_chip(s, n_chips) == index]
    if len(slots) == 1:
        slot_text = f"slot {slots[0]}"
    elif slots == list(range(slots[0], slots[-1] + 1)) and len(slots) > 2:
        slot_text = f"slots {slots[0]}-{slots[-1]}"
    else:
        slot_text = "slots " + ", ".join(map(str, slots))
    chips_text = f"{n_chips} chip{'' if n_chips == 1 else 's'}"
    out = [
        Decision(
            f"chip {index} plays {', '.join(devices)}",
            f"{slot_text} of {n_slots(n_chips)} go to chip {index}: slots are dealt to {chips_text} in turn "
            f"(slot % chips) and cycle through {len(DEVICES)} devices. There are always at least "
            f"{len(DEVICES)} slots, so a small system still plays the whole ensemble",
        )
    ]
    if "asic_temperature" in uses:
        out.append(Decision(
            f"asic {t.asic_temperature:.1f} C -> pitch {temp_to_pitch_hz(t.asic_temperature):.1f} Hz",
            "25-100 C maps onto C3-C5, linear in semitones, so a hotter chip plays higher",
        ))
    if "aiclk" in uses:
        out.append(Decision(
            f"aiclk {t.aiclk:.0f} MHz -> {aiclk_to_bpm(t.aiclk):.1f} BPM",
            "400-1350 MHz maps onto 60-180 BPM; a faster clock sets a faster pulse",
        ))
    if "ddr_status" in uses:
        out.append(Decision(
            f"DDR status {t.ddr_status} -> rolls with {ddr_lane_sides(t.ddr_status)} sides",
            "each DDR field adds a ttrolzer roll of 3, 4, 5 or 6 sides on its own barre, so the lanes drift against "
            "each other; an odd roll is unsteady and an even one swings; status 0 grounds the roll and mutes it",
        ))
    out.extend(_explain_wider_readings(t, devices, patch))
    out.append(Decision(
        f"power {t.power:.0f} W -> volume {power_to_volume(t.power):.2f}",
        "0-200 W maps onto volume 0.3-1.0 on an exponential curve",
    ))
    return out


def _percent_list(levels) -> str:
    return ", ".join(f"{lv:.0%}" for lv in levels)


def _explain_wider_readings(t: Telemetry, devices: List[str], patch: Patch = DEFAULT_PATCH) -> List[Decision]:
    """Decisions for the readings beyond temperature, clock and power, only for devices on this chip."""
    out: List[Decision] = []
    pressure = chip_pressure(t)
    ident = personality(t)
    pressure_why = (
        f"how close the chip is to a limit: ASIC temperature from 40 C up to the {t.thm_limit_throttle:.0f} C throttle "
        f"point, or power from a quarter of the {t.tdp_limit:.0f} W limit up to it, whichever is closer"
    )
    if "ttrollz" in devices:
        net = ttrollz_network(t)
        browns = draw_links(len(net.rolls), patch.seed, personality(t), chip_pressure(t), [lane.index for lane in net.lanes])
        if browns:
            def name(i: int) -> str:
                return f"{net.rolls[i].sides}-roll ({net.rolls[i].bpm:.0f} BPM)"
            wires = ", ".join(f"{name(l.source)} to {name(l.target)}" for l in browns[:6])
            if len(browns) > 6:
                wires += f", and {len(browns) - 6} more"
            out.append(Decision(
                f"{len(browns)} Browns patched between its {len(net.rolls)} rolls, {len(net.lanes)} of them heard: {wires}",
                "each lane that is up brings a barre of the Rolzer's rolls (a triangle, square, pentagon and two hexagons) and only one "
                "roll of each barre is sounded; at least half the links end at a heard roll, so the silent rolls shape what you hear. "
                "A Brown is a Sandrode, input and output at once, so each patch weaves two rolls together: every pulse of the "
                "first pulls the next step of the second earlier, which syncs an even roll to a faster one and knocks it off its "
                f"pattern when the source is odd. The chip's identity and the patch pick the wiring; pressure {chip_pressure(t):.2f} "
                f"sets the pull (up to {max(l.pull for l in browns):.0%} of a step); ASIC temperature {t.asic_temperature:.0f} C sets how far "
                f"each capacitor drifts a step ({roll_drift(t.asic_temperature):.1%}). The first step of every roll's cycle gets an "
                "octave-up accent, so an odd roll's 3 or 5 can be heard",
            ))
        links = bin(t.eth_mask).count("1")
        out.append(Decision(
            f"{links} of 16 ethernet links up -> lanes at {_percent_list(eth_lane_levels(t.eth_mask))}",
            "each group of four links drives one more slow roll on a grave barre, as loud as "
            "the share of its group that is up, so a link going down quietens its lane",
        ))
    if "ttrax" in devices:
        n_bars = patch.rax_bars
        walls = [barre_bounds(t.gddr_temps, b) for b in range(n_bars)]
        out.append(Decision(
            f"GDDR {min(t.gddr_temps):.0f}-{max(t.gddr_temps):.0f} C -> barre walls {min(w[0] for w in walls):.2f} to {max(w[1] for w in walls):.2f}",
            f"each of the {n_bars} ttrax barres bounces between two walls: its own GDDR channel sets the upper one and the "
            "next channel the lower, so a hotter channel pushes a wall further out",
        ))
        out.append(Decision(
            f"ARC clock {t.arcclk:.0f} MHz -> bounce x{t.arcclk / FALLBACK_ARCCLK:.2f}",
            f"ttrax bounces at the chip's pitch when the ARC clock is {FALLBACK_ARCCLK:.0f} MHz and scales with it",
        ))
        chaos = chip_chaos(t)
        moving = rate_activity(t.temp_rate, t.power_rate)
        movement = (
            f"; the readings are moving ({t.temp_rate:+.1f} C/s, {t.power_rate:+.0f} W/s), which adds {moving:.2f} "
            "(the faster of the two, up to 1) to the knob"
        ) if moving else ""
        out.append(Decision(
            f"pressure {pressure:.2f} -> ttrax chaos {chaos:.0%}",
            f"{pressure_why}{movement}; the ttrax chaos knob cross-modulates the barres, and at the top it is noise. "
            f"Each barre is pressed every four steps of its own ttrolzer barre, at {aiclk_to_bpm(t.aiclk):.0f} BPM here",
        ))
    if "ttquantus" in devices:
        out.append(Decision(
            f"AXI clock {t.axiclk:.0f} MHz -> notes change x{t.axiclk / REFERENCE_AXICLK:.2f} as often",
            f"ttquantus holds each note for an eighth when the AXI clock is {REFERENCE_AXICLK:.0f} MHz, "
            "and a faster AXI clock holds it for less",
        ))
    if "ttquantus" in devices:
        out.append(Decision(
            f"chip identity {ident:08x} -> castle samples the petal {castle_offset(ident)} places round",
            "the ttquantussy's five petals each have a castle that samples another petal's triangle; the chip's "
            "Tensix and ethernet masks pick how far round the ring it looks (2, 3, 4 or 1 places), which changes the pattern of notes",
        ))
    if "ttdrone" in devices:
        out.append(Decision(
            f"voltage {t.voltage:.2f} V -> {voltage_to_cents(t.voltage):+.0f} cents",
            "0.75 V is in tune; the drone goes sharp above it and flat below, 4 cents per 10 mV, up to 40 cents",
        ))
        out.append(Decision(
            f"PCIe usage {t.pcie_usage} -> {drone_voices(t.pcie_usage)} drone voices",
            "the drone has one voice per PCIe usage step, from 1 to 6",
        ))
        beat = 60.0 / aiclk_to_bpm(t.aiclk)
        out.append(Decision(
            f"each gong strike ripples the drone: {SHOCK_RATE_BEATS / beat:.1f} swings a second, fading over {SHOCK_DECAY_BEATS * beat:.1f} s",
            "the drone rings in sympathy with the gong, after the shock wave an AVDog gives a Gongue: every gong strike on this chip sets off a damped swing "
            "that changes direction every half cycle, so the drone's level ripples and settles back",
        ))
    if "ttchoir" in devices:
        scale_name = {1.0: "bass", 1.08: "baritone", 1.17: "alto", 1.25: "soprano"}[voice_scale(ident)]
        vowels = "".join(vowel_sequence(t.gddr_temps, min(len(t.gddr_temps), 8), ident % len(t.gddr_temps)))
        out.append(Decision(
            f"ASIC {t.asic_temperature:.0f} C -> ttchoir pitch {temp_to_pitch_hz(t.asic_temperature):.0f} Hz, snapped to the patch's scale",
            "the chip's pitch is moved to the nearest note of the patch's scale so a choir of chips sings in tune with itself, with a gentle vibrato",
        ))
        out.append(Decision(
            f"GDDR channels -> vowels {' '.join(vowels)}",
            "each bar the singer reads the next GDDR channel: a channel hotter than the average sings toward ee, a cooler one toward oo, "
            "and equal temperatures sing ah; the formants glide from one vowel to the next",
        ))
        out.append(Decision(
            f"chip identity {ident:08x} -> a {scale_name} voice (vocal tract x{voice_scale(ident):.2f}); pressure {pressure:.2f} -> glottal slope",
            f"{pressure_why}; the Tensix columns and ethernet links a chip has pick its voice type, and pressure flattens the glottal "
            "slope so a chip near a limit sings with more effort and a brighter tone",
        ))
    if "ttavdog" in devices:
        out.append(Decision(
            f"pressure {pressure:.2f} -> ttavdog wave shape {pressure:.2f} (0 is a triangle, 1 a saw) at {temp_to_pitch_hz(t.asic_temperature):.0f} Hz",
            f"{pressure_why}; the dogtone drones at the chip's pitch and is a triangle with no pressure, skewed toward a saw "
            "as pressure rises, and the red voice has the opposite slope",
        ))
        out.append(Decision(
            "each gong strike pings the ringing filter: the white voice sounds on one half of the swing, the red voice on the other",
            "after the AVDog, whose undulation opens a silent drone oscillator as it swings; the voices answer each other, at the same pitch "
            "with the opposite slope, and the red is a little quieter. The swing is the one that ripples the ttdrone",
        ))
    if "ttcoco" in devices:
        delays = [tap_delay_ms(g) for g in t.gddr_temps[:N_TAPS]]
        bits = [tap_bits(d) for d in delays]
        feedback = tap_feedback(t)
        out.append(Decision(
            f"GDDR channels -> {N_TAPS} delay taps, {min(delays):.0f} to {max(delays):.0f} ms, {min(bits):.0f} to {max(bits):.0f} bits",
            "each GDDR channel drives one tap of the coco's own delay chips: a hotter channel echoes faster, and a "
            "shorter delay holds more bits, the real chips' own trade of buffer length against resolution. The real "
            f"instrument affords two such chips; a chip here is {N_TAPS} cores' worth of work, not {N_TAPS} chips' worth of board space",
        ))
        out.append(Decision(
            f"pressure {pressure:.2f} -> tap feedback {feedback:.2f}",
            f"{pressure_why}; every tap feeds back on itself, so a chip under pressure rings with more, longer-lived echoes",
        ))
    if "lead" in devices:
        out.append(Decision(
            f"pressure {pressure:.2f} (ASIC {t.asic_temperature:.0f} C, {t.power:.0f} W) -> lead overtone {BRIGHTNESS * pressure:.0%}",
            f"{pressure_why}; more pressure adds an octave overtone to each lead note",
        ))
        out.append(Decision(
            f"chip identity {ident:08x} -> lead starts +{identity_offset(ident)} degrees up",
            "the Tensix columns and ethernet links a chip has enabled differ from chip to chip, so they pick "
            "0, 1 or 2 scale degrees to transpose its phrases",
        ))
    if "gong" in devices:
        bpm = aiclk_to_bpm(t.aiclk)
        interval = 2 * 4 * 60.0 / bpm
        out.append(Decision(
            f"chip identity {ident:08x} -> gong tuned {gong_detune_cents(ident):+d} cents",
            "the same chip fingerprint detunes the gong by up to 10 cents either way, so gongs on different chips beat against each other",
        ))
        net = ttrollz_network(t)
        which = gong_trigger_roll(t, patch)
        if which is None:
            out.append(Decision(
                "gong is patched to the chip's own ttrollz clock",
                "the chip has no rolls to patch it to, or its identity and the patch picked the clock; the strike lands on the next "
                "clock pulse after the blocking window",
            ))
        else:
            roll = net.rolls[which]
            out.append(Decision(
                f"gong is patched to roll {which} of {len(net.rolls)}, a {roll.sides}-roll ({roll.bpm:.0f} BPM)"
                + (", one of the rolls that is heard" if which in [lane.index for lane in net.lanes] else ", a roll that is not heard"),
                "the Gongue takes its trigger from a roll's orange output, so the strike lands on the roll's own pulses, which wander "
                "off any grid. The chip's identity and the patch pick the roll, and the same pulses are the ones ttrollz plays",
            ))
        strikes = strike_times(t, 60.0, patch)[:4]
        out.append(Decision(
            f"trigger response: gong strikes at {', '.join(f'{x:.1f}' for x in strikes)} s",
            f"the gong ignores its trigger for a blocking window after each strike, as a Gongue does with its trigger response: two bars, times "
            f"{1 + TRIGGER_RESPONSE_SPREAD * (ident % TRIGGER_RESPONSE_STEPS):.2f} for this chip's identity. "
            "The strike lands on the next pulse of its trigger after the window, so a chip divides its trigger its own way",
        ))
        out.append(Decision(
            f"heartbeat {t.heartbeat} -> first gong strike {heartbeat_phase(t.heartbeat) * interval / 2:.1f} s in",
            "the firmware counter modulo 16, in sixteenths of half a strike interval, sets where the first strike lands",
        ))
    if "ttsteam" in devices:
        hits = [i for i, hit in enumerate(step_pattern(t.tensix_mask)) if hit]
        if hits:
            out.append(Decision(
                f"Tensix mask {t.tensix_mask:#06x} -> snare hits on steps {', '.join(map(str, hits))} of {STEPS}",
                "the shift register has one step per Tensix column and fires a step where its column is harvested "
                "(the mask bit is 0), so each chip has its own rhythm of holes",
            ))
        else:
            out.append(Decision(
                f"Tensix mask {t.tensix_mask:#06x} has no harvested columns -> the snare stays silent",
                "the shift register fires a step where a Tensix column is harvested (the mask bit is 0), "
                "and this chip has every column enabled",
            ))
        out.append(Decision(
            f"pressure {pressure:.2f} -> snare decay {snare_decay(pressure):.2f} s",
            f"{pressure_why}; the snare decay runs from 0.75 s with no pressure down to 75 ms at full pressure",
        ))
        out.append(Decision(
            f"PCIe usage {t.pcie_usage} -> {dust_per_second(t.pcie_usage):.0f} dust pops a second",
            "PCIe traffic is data moving, so each step of usage (up to 6) adds 15 pops a second to the dust crackle under the snare",
        ))
    if "ultrasonic_shimmer" in devices:
        spread = max(t.gddr_temps) - min(t.gddr_temps)
        out.append(Decision(
            f"GDDR spread {spread:.0f} C -> shimmer beat {SHIMMER_BEAT_HZ_PER_DEG * spread:.1f} Hz",
            "the two shimmer tones sit further apart the more unevenly the memory channels are heated, "
            "and the Ultrasound gives a short snap on every gong strike, as the guide's Deconstructed Drum does",
        ))
        out.append(Decision(
            f"pressure {pressure:.2f} (ASIC {t.asic_temperature:.0f} C, {t.power:.0f} W) -> shimmer level {0.5 * (1 + pressure):.2f}",
            f"{pressure_why}; more pressure swells the shimmer, up to twice its resting level",
        ))
    return out


def explain_turn_weights(telemetry: Sequence[Telemetry]) -> List[Decision]:
    """Each chip's turn weight and the factors that made it."""
    mean_board = sum(t.board_power for t in telemetry) / len(telemetry)
    out = []
    for i, t in enumerate(telemetry):
        load = t.board_power / mean_board if mean_board > 0 else 1.0
        pressure = chip_pressure(t)
        parts = [f"power {t.power:.0f} W", f"board {t.board_power:.0f} W (x{load:.2f})", f"pressure {pressure:.2f} (x{1 + PRESSURE_WEIGHT * pressure:.2f})"]
        weight = t.power * load * (1 + PRESSURE_WEIGHT * pressure)
        if t.busy is not None:
            counted = min(t.busy, BUSY_WEIGHT_MAX_PROCESSES)
            factor = 1 + BUSY_WEIGHT_PER_PROCESS * counted
            weight *= factor
            parts.append(f"{t.busy} process{'' if t.busy == 1 else 'es'} (x{factor:.2f})")
        out.append(Decision(
            f"chip {i} turn weight {weight:.0f}: " + ", ".join(parts),
            "the weight is the chip's power times its board load against the average board, times 1 plus "
            f"{PRESSURE_WEIGHT:g} per unit of pressure, times 1 plus {BUSY_WEIGHT_PER_PROCESS:g} per process using "
            f"it (up to {BUSY_WEIGHT_MAX_PROCESSES}); a chip that does not list its processes counts none",
        ))
    return out


CABLE_WORDS = {
    "fm": "mutates it: the target is read through a delay that follows the source waveform",
    "crunch": "degrades it: the target is rounded to coarse steps, coarser while the source is loud",
    "inflate": "inflates it: the target is amplified while the source sounds, with a soft ceiling",
    "fold": "folds it: the target is driven and reflected back on itself, which adds overtones",
}


def explain_cables(cables: Sequence[Cable], telemetry: Sequence[Telemetry]) -> List[Decision]:
    """One decision per patch cable: what it connects, and how hard the source's chip pushes it."""
    n = len(telemetry)
    names = [assign_device(s).__name__ for s in range(n_slots(n))]
    out = []
    for c in cables:
        slot = names.index(c.source) if c.source in names else None
        if slot is None:
            continue
        chip = slot_chip(slot, n)
        chaos = chip_chaos(telemetry[chip])
        out.append(Decision(
            f"{c.source} -> {c.target} ({c.kind}), depth {c.depth:.2f}, pushed {strength(c, chaos):.2f} by chip {chip}",
            f"the voice of {c.source} {CABLE_WORDS[c.kind]}. The push is the depth, a quarter of it from a calm chip and all of it "
            f"from a chaotic one; chip {chip} plays {c.source} and its chaos (pressure plus how fast its readings move) is {chaos:.2f}",
        ))
    return out


def explain_events(index: int, t: Telemetry) -> Decision:
    """What the error counters of one chip add to the sound."""
    parts, sounds = [], []
    if t.corr_errs:
        parts.append(f"{t.corr_errs} correctable memory error{'' if t.corr_errs == 1 else 's'}")
        sounds.append(f"each correctable error is a short high tick (up to {MAX_TICKS})")
    if t.uncorr_errs:
        parts.append(f"{t.uncorr_errs} uncorrectable memory error{'' if t.uncorr_errs == 1 else 's'}")
        sounds.append(f"each uncorrectable error is a low breath of noise (up to {MAX_BREATHS})")
    if t.thermal_trips:
        parts.append(f"{t.thermal_trips} thermal trip{'' if t.thermal_trips == 1 else 's'}")
        sounds.append(f"each thermal trip is a falling sweep (up to {MAX_SWEEPS})")
    if not parts:
        return Decision(
            f"chip {index} reports no memory errors or thermal trips",
            "a healthy chip adds no event sounds; errors would play as ticks, a breath and a falling sweep",
        )
    return Decision(f"chip {index} reports " + ", ".join(parts), "; ".join(sounds) + ", and they sound whoever has the floor")


def explain_mix(telemetry: Sequence[Telemetry]) -> List[Decision]:
    total = sum(max(t.current, 0.0) for t in telemetry)
    out = []
    for i, t in enumerate(telemetry):
        share = max(t.current, 0.0) / total if total else 1 / len(telemetry)
        out.append(Decision(
            f"chip {i} current {t.current:.0f} A -> {share:.0%} of the mix",
            "sources mix by current, as patched banana plugs do, so the chip drawing more current is louder",
        ))
    return out


def explain_lead(t: Telemetry, duration: float) -> Decision:
    """How the lead's phrases change over the run, from this chip's GDDR channels."""
    bars = max(1, math.ceil(duration * aiclk_to_bpm(t.aiclk) / 240.0))
    shown = lead_transforms(t.gddr_temps, min(bars, 8))
    if all(tr.kind == "repeat" for tr in shown):
        what = "lead loops its motif unchanged"
    else:
        names = [f"shift +{tr.amount}" if tr.kind == "shift" else tr.kind for tr in shown]
        what = "lead phrases: " + ", ".join(f"{i + 1} {n}" for i, n in enumerate(names))
    return Decision(
        what,
        "each bar reads the next GDDR channel and compares it with the average channel: "
        "hotter than average shifts the motif up, cooler inverts it, within 1 C repeats it",
    )


def explain_turns(
    telemetry: Sequence[Telemetry], duration: float, limit: int = 12, patch: Patch = DEFAULT_PATCH
) -> List[Decision]:
    """Who solos in each turn, and why."""
    n = len(telemetry)
    solo_slots, turn_s = plan_turns(telemetry, duration, patch)
    out = []
    for k, slot in enumerate(solo_slots[:limit]):
        chip = slot_chip(slot, n)
        out.append(Decision(
            f"turn {k + 1} ({k * turn_s:.0f}s): {assign_device(slot).__name__} on chip {chip} solos",
            "the soloist is drawn in proportion to each chip's turn weight (its power, raised by a heavy board "
            "load, by pressure near a limit and by processes using it), nobody solos twice "
            "in a row, and everyone else drops to a lower level while it plays",
        ))
    if len(solo_slots) > limit:
        more = len(solo_slots) - limit
        out.append(Decision(
            f"{more} more turns follow the same way",
            f"the run has {len(solo_slots)} turns of {turn_s:.1f}s; only the first {limit} are listed to keep the log short",
        ))
    return out


def _side(azimuth: float) -> str:
    return "in the centre" if abs(azimuth) < 0.15 else ("on the left" if azimuth < 0 else "on the right")


def explain_space(telemetry: Sequence[Telemetry], patch: Patch) -> List[Decision]:
    """Where each chip sits in the stereo field and how far it travels: one line on how the field works, then one for each chip."""
    out = [Decision(
        "stereo field: azimuth, width and depth",
        "a voice has an azimuth from -1 (hard left) to +1 (hard right), a width from a point to wide, and a depth from near to far (farther is "
        "quieter, darker and carries a reflection). Each chip has a home, spread left to right by its index and nudged by its identity, and each "
        "device has its own offset, width and depth, so the air of the shimmer is wide and far and the lead is near. A voice also moves: it orbits "
        "its place by an amount that comes from its chip's chaos at a speed that comes from the clock, the voice with the floor comes forward, "
        "the AVDog swings with its undulation, the shimmer pans at its beat, each gong strike lands somewhere new, ttquantus pans with its pitch "
        "and ttrollz hops from lane to lane",
    )]
    n = len(telemetry)
    for i, t in enumerate(telemetry):
        az = chip_azimuth(i, n, t)
        chaos = chip_chaos(t)
        reach = ORBIT_CALM + ORBIT_CHAOS * min(1.0, chaos)
        period = 1.0 / (aiclk_to_bpm(t.aiclk) * ORBIT_HZ_PER_BPM)
        out.append(Decision(
            f"chip {i} sits {_side(az)} of the field ({az:+.2f}) and orbits {reach:.2f} around it",
            f"its chaos is {chaos:.2f} (pressure plus how fast its readings move), which sets how far it travels: {ORBIT_CALM:.2f} when calm and up "
            f"to {ORBIT_CALM + ORBIT_CHAOS:.2f}; its {t.aiclk:.0f} MHz clock sets how fast, one orbit in about {period:.0f} s",
        ))
    return out


STEAM_ROUTE_WHY = {
    "strikes": "the register's hits join the clock the gong waits on, so the gong, the shimmer's snap, the drone's ripple and the AVDog follow the snare",
    "avdog": "every snare hit pings the AVDog's filter as well, a call and response on the snare rhythm",
    "accent": "ttrollz clock pulses that land on a snare hit sound louder",
    "pitch": "the register holds one note of the scale per step, from the seed, and steers the AVDog's dogtone with them",
}


def _snare_decision(patch: Patch) -> Decision:
    """How much of ttsteam's snare plays in this patch."""
    if patch.snare <= 0.0:
        what = "snare: silent"
    elif patch.snare >= 1.0:
        what = "snare: every hit plays"
    else:
        what = f"snare: {patch.snare:.0%} of the register's hits play"
    where = f"set by the {patch.style} style" if patch.style else (
        "drawn from the seed: silent 40% of the time, one hit in five 30%, one in two 20%, every hit 10%")
    return Decision(what, f"{where}; the register keeps stepping whatever the level, so its steam routes still work")


def _rests_decision(patch: Patch) -> Decision:
    """How many of the coco's held notes are silent."""
    if patch.rests <= 0.0:
        return Decision("rests: none", "this patch plays every held note of the Quantussy's castle")
    where = f"set by the {patch.style} style" if patch.style else "drawn from the seed: 15 to 45% of the notes"
    return Decision(
        f"rests: {patch.rests:.0%} of the coco's held notes are silent",
        f"{where}; a second castle of the first petal, reading another petal than the note castle does, is held for each note and a note under that value does not play, "
        "so the silences repeat and drift with the notes (ttquantus and every ttcoco echo of it)",
    )


def _pulse_rests_decisions(patch: Patch) -> list:
    """How many of the lead's notes and of ttrollz's pulses are left out."""
    where = f"set by the {patch.style} style" if patch.style else "drawn from the seed: 10 to 30%"
    out = []
    for name, share, plays, how in (
        ("lead rests", patch.lead_rests, "every note of the motif",
         "each note after the first of a phrase has its own value from the seed, and one under that share is not played, so the downbeat holds the phrase together"),
        ("roll rests", patch.roll_rests, "every pulse of the clock and the rolls",
         "each pulse of the clock and of every roll lane has its own value from the seed, and one under that share is not sounded (the gong still follows the whole rolls)"),
    ):
        if share <= 0.0:
            out.append(Decision(f"{name}: none", f"this patch plays {plays}"))
        else:
            out.append(Decision(f"{name}: {share:.0%} of the {'notes' if name.startswith('lead') else 'pulses'} are silent", f"{where}; {how}"))
    return out


def _bus_decision(patch: Patch) -> Decision:
    """How much of the band goes through the Cocoquantus's delay, and the room."""
    if patch.coco_send <= 0.0:
        return Decision("coco bus: off", "this patch sends nothing through the Cocoquantus's delay, so its devices are heard dry")
    where = f"set by the {patch.style} style" if patch.style else "drawn from the seed: a send of 25 to 80%, a room of 15 to 50% for 0.8 to 3.5 s"
    return Decision(
        f"coco bus: {patch.coco_send:.0%} of the band through the delay, then a {patch.reverb_s:.1f} s room at {patch.coco_reverb:.0%}",
        f"{where}; each chip's voices, as heard, go into four crunchy 8-bit taps whose echo times the petals' castles sway (a hotter GDDR channel echoes sooner, "
        "a longer delay holds fewer bits, pressure feeds the repeats back), and the echoes and the dry send go through the room",
    )


def _steam_routes_decision(patch: Patch) -> Decision:
    """Which of ttsteam's register routes are on: the snare's hits and held notes used as triggers, accents and pitch."""
    if not patch.steam_routes:
        return Decision("steam routes: none", "this patch does not patch ttsteam's register into any other device, so the snare is only a sound")
    return Decision(
        f"steam routes: {', '.join(patch.steam_routes)}",
        "drawn from the seed (none 10%, one 35%, two 35%, three 20%; the strikes route a half more often than the others); "
        + "; ".join(f"{r}: {STEAM_ROUTE_WHY[r]}" for r in patch.steam_routes),
    )


def _style_decision(patch: Patch) -> List[Decision]:
    """A line for a patch drawn in a style: what the style is and the balance it sets for the band."""
    if not patch.style:
        return []
    from ..music.styles import STYLES
    blurb = STYLES[patch.style].blurb if patch.style in STYLES else "a style this version does not know"
    balance = ", ".join(f"{name} x{factor:g}" for name, factor in sorted(patch.emphasis, key=lambda p: -p[1]))
    return [Decision(f"style {patch.style}",
                     f"{blurb} Its balance, as a factor on each device's level and on how often it gets the floor: {balance or 'none'}")]


def explain_patch(patch: Patch, telemetry: Sequence[Telemetry], source: str) -> List[Decision]:
    """Every choice a patch makes, with the reading or the seed that decided it."""
    bars = f"{patch.bars_per_turn} bar{'' if patch.bars_per_turn == 1 else 's'}"
    return [
        Decision(
            f"patch seed {patch.seed}",
            f"{source}; the seed fixes every drawn choice below, so the same seed replays the same sound",
        ),
        Decision(
            f"scale {patch.scale_name}, mode {patch.mode} (GDDR spread {gddr_spread(telemetry):.0f} C)",
            f"each {DEGREES_PER_SCALE_STEP_C:g} C between the hottest and coolest GDDR channel moves one scale along "
            f"({', '.join(SCALE_NAMES)}), so even memory plays the first and uneven memory wanders the list; "
            f"the seed starts the scale from its note number {patch.mode} (0 plays it as listed)",
        ),
        Decision(
            f"lead motif {patch.motif}",
            f"{len(patch.motif)} scale degrees drawn from the seed, starting at 0 and moving at most 3 degrees a step",
        ),
        Decision(
            f"gong tuning {patch.gong_tuning} (memory/ASIC temperature {memory_to_asic_ratio(telemetry):.2f}), "
            f"beat {patch.gong_beat_hz:g} Hz",
            f"the average GDDR temperature divided by the average ASIC temperature, times 10 and rounded, "
            f"picks one of {len(GONG_TUNING_NAMES)} tunings ({', '.join(GONG_TUNING_NAMES)}); the seed picks the beat, "
            f"stretches the partials by up to 1.5% and nudges their levels",
        ),
        Decision(
            f"shimmer band {patch.shimmer_band[0]:.0f}-{patch.shimmer_band[1]:.0f} Hz",
            "drawn from the seed: a low edge of 8 to 15 kHz and a width of 3 to 9 kHz, "
            "squeezed under 0.45 of the sample rate when needed",
        ),
        Decision(
            f"turns last {bars}; everyone else drops to {patch.floor:.0%}",
            "drawn from the seed: 1 to 6 bars per turn, and a floor between 20% and 50%",
        ),
        _steam_routes_decision(patch),
        _snare_decision(patch),
        _bus_decision(patch),
        _rests_decision(patch),
        *_pulse_rests_decisions(patch),
    ] + _style_decision(patch)
