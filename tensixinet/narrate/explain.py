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
    strike_times,
)
from ..music.ttsteam import STEPS, dust_per_second, snare_decay, step_pattern
from ..music.ttquantussy import castle_offset
from ..music.ttsidrax import FALLBACK_ARCCLK, N_BARRES, barre_bounds
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


def explain_chip(index: int, t: Telemetry, n_chips: Optional[int] = None) -> List[Decision]:
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
    out.extend(_explain_wider_readings(t, devices))
    out.append(Decision(
        f"power {t.power:.0f} W -> volume {power_to_volume(t.power):.2f}",
        "0-200 W maps onto volume 0.3-1.0 on an exponential curve",
    ))
    return out


def _percent_list(levels) -> str:
    return ", ".join(f"{lv:.0%}" for lv in levels)


def _explain_wider_readings(t: Telemetry, devices: List[str]) -> List[Decision]:
    """Decisions for the readings beyond temperature, clock and power, only for devices on this chip."""
    out: List[Decision] = []
    pressure = chip_pressure(t)
    ident = personality(t)
    pressure_why = (
        f"how close the chip is to a limit: ASIC temperature from 40 C up to the {t.thm_limit_throttle:.0f} C throttle "
        f"point, or power from a quarter of the {t.tdp_limit:.0f} W limit up to it, whichever is closer"
    )
    if "ttrollz" in devices:
        links = bin(t.eth_mask).count("1")
        out.append(Decision(
            f"{links} of 16 ethernet links up -> lanes at {_percent_list(eth_lane_levels(t.eth_mask))}",
            "each group of four links drives one more slow roll on a grave barre, as loud as "
            "the share of its group that is up, so a link going down quietens its lane",
        ))
    if "ttsidrax" in devices:
        walls = [barre_bounds(t.gddr_temps, b) for b in range(N_BARRES)]
        out.append(Decision(
            f"GDDR {min(t.gddr_temps):.0f}-{max(t.gddr_temps):.0f} C -> barre walls {min(w[0] for w in walls):.2f} to {max(w[1] for w in walls):.2f}",
            "each of the seven ttsidrax barres bounces between two walls: its own GDDR channel sets the upper one and the "
            "next channel the lower, so a hotter channel pushes a wall further out",
        ))
        out.append(Decision(
            f"ARC clock {t.arcclk:.0f} MHz -> bounce x{t.arcclk / FALLBACK_ARCCLK:.2f}",
            f"ttsidrax bounces at the chip's pitch when the ARC clock is {FALLBACK_ARCCLK:.0f} MHz and scales with it",
        ))
        chaos = chip_chaos(t)
        moving = rate_activity(t.temp_rate, t.power_rate)
        movement = (
            f"; the readings are moving ({t.temp_rate:+.1f} C/s, {t.power_rate:+.0f} W/s), which adds {moving:.2f} "
            "(the faster of the two, up to 1) to the knob"
        ) if moving else ""
        out.append(Decision(
            f"pressure {pressure:.2f} -> ttsidrax chaos {chaos:.0%}",
            f"{pressure_why}{movement}; the ttsidrax chaos knob cross-modulates the barres, and at the top it is noise. "
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
        strikes = strike_times(t, 60.0)[:4]
        out.append(Decision(
            f"trigger response: gong strikes at {', '.join(f'{x:.1f}' for x in strikes)} s",
            f"the gong ignores its trigger for a blocking window after each strike, as a Gongue does with its trigger response: two bars, times "
            f"{1 + TRIGGER_RESPONSE_SPREAD * (ident % TRIGGER_RESPONSE_STEPS):.2f} for this chip's identity. "
            "The strike lands on the next pulse of the chip's own ttrollz clock after the window, so a chip divides its clock its own way",
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


def explain_patch(patch: Patch, telemetry: Sequence[Telemetry], source: str) -> List[Decision]:
    """Every choice a patch makes, with the reading or the seed that decided it."""
    bars = f"{patch.bars_per_turn} bar{'' if patch.bars_per_turn == 1 else 's'}"
    return [
        Decision(
            f"patch seed {patch.seed}",
            f"{source}; the seed fixes every drawn choice below, so the same seed replays the same sound",
        ),
        Decision(
            f"scale {patch.scale_name} (GDDR spread {gddr_spread(telemetry):.0f} C)",
            f"each {DEGREES_PER_SCALE_STEP_C:g} C between the hottest and coolest GDDR channel moves one scale along "
            f"({', '.join(SCALE_NAMES)}), so even memory plays the first and uneven memory wanders the list",
        ),
        Decision(
            f"lead motif {patch.motif}",
            "eight scale degrees drawn from the seed, starting at 0 and moving at most 2 degrees a step",
        ),
        Decision(
            f"gong tuning {patch.gong_tuning} (memory/ASIC temperature {memory_to_asic_ratio(telemetry):.2f}), "
            f"beat {patch.gong_beat_hz:g} Hz",
            f"the average GDDR temperature divided by the average ASIC temperature, times 10 and rounded, "
            f"picks one of {len(GONG_TUNING_NAMES)} tunings ({', '.join(GONG_TUNING_NAMES)}); the seed picks the beat",
        ),
        Decision(
            f"shimmer band {patch.shimmer_band[0]:.0f}-{patch.shimmer_band[1]:.0f} Hz",
            "drawn from the seed: a low edge of 10 to 14 kHz and a width of 4 to 7 kHz, "
            "squeezed under 0.45 of the sample rate when needed",
        ),
        Decision(
            f"turns last {bars}; everyone else drops to {patch.floor:.0%}",
            "drawn from the seed: 1 to 4 bars per turn, and a floor between 25% and 45%",
        ),
    ]
