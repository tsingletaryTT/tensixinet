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
from ..music.devices import DEVICE_INPUTS, sidrax_bounds
from ..music.lead import lead_transforms
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
    aiclk_to_bpm,
    ddr_lane_divisions,
    power_to_volume,
    temp_to_pitch_hz,
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
            f"DDR status {t.ddr_status} -> lanes {ddr_lane_divisions(t.ddr_status)}",
            "each DDR field adds a polyrhythm lane of 3, 4, 5 or 7 pulses per bar; status 0 mutes it",
        ))
    if "gddr_temps" in uses:
        lo, hi = sidrax_bounds(t.gddr_temps)
        out.append(Decision(
            f"GDDR {min(t.gddr_temps):.0f}-{max(t.gddr_temps):.0f} C -> swing bounds {lo:.2f} to {hi:.2f}",
            "the coolest memory channel sets the lower bound and the hottest the upper",
        ))
    out.append(Decision(
        f"power {t.power:.0f} W -> volume {power_to_volume(t.power):.2f}",
        "0-200 W maps onto volume 0.3-1.0 on an exponential curve",
    ))
    return out


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
            "the soloist is drawn in proportion to the power each chip draws, nobody solos twice "
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
