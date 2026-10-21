# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Evict: tt-toplike's Defrag view shows a chip's memory dissolving when its model unloads. Here the same trigger moves the band.

tt-toplike's rule (src/animation/defrag.rs): a chip that has been running, has seen a load (power above 1.5 times its rest, or above 30%
of its TDP) and has fallen back to about rest (at most 1.2 times, and never below 8 W) has unloaded its model. Its blocks dissolve right to
left at staggered prime-multiplied rates, and when load returns they fill left to right. Here that chip's voices do the same through the stereo
field, a sweep travels with them, and the chip leaves the rotation until it is loaded again."""
import dataclasses

import numpy as np
import pytest

from tensixinet.music.chorus import DEVICES, assign_device, render_chorus, slot_chip, slot_weights
from tensixinet.music.evict import (
    DRAIN_PRIMES, EVICT_SECONDS, FILL_SECONDS, ChipDefrag, EvictTracker, drain_progress, evict_eligible, fill_sweep, evict_sweep,
    presence_weight, shows_load,
)
from tensixinet.music.patch import DEFAULT_PATCH
from tensixinet.telemetry import FALLBACK
from tensixinet.telemetry.mock import mock_telemetry

SR = 8000


def chip(power, tdp=125.0):
    return dataclasses.replace(FALLBACK, power=power, tdp_limit=tdp)


# --- the trigger, as tt-toplike has it ------------------------------------------------------------------------------------------------

def test_a_chip_shows_a_load_above_one_and_a_half_times_its_rest_or_above_a_third_of_its_tdp_when_rest_is_unknown():
    assert shows_load(25.0, 15.0, 125.0) and not shows_load(22.0, 15.0, 125.0)       # 22 W is 1.47 times a 15 W rest
    assert shows_load(40.0, 0.0, 125.0) and not shows_load(30.0, 0.0, 125.0)          # no rest seen: 30% of 125 W is 37.5 W
    assert not shows_load(60.0, 0.0, None) and not shows_load(60.0, 0.0, 0.0)          # no limit to compare with, and no rest


def test_once_a_rest_is_known_only_one_and_a_half_times_it_counts_as_a_load_whatever_the_tdp_says():
    assert not shows_load(40.0, 36.0, 125.0)      # 40 W is over 30% of 125 W, but it is 1.1 times this chip's own rest
    assert not shows_load(54.0, 36.0, 125.0) and shows_load(55.0, 36.0, 125.0)   # 1.5 times 36 W is 54 W
    assert shows_load(30.0, 0.5, 125.0)            # even a tiny known rest is a rest: 30 W is far over it, though under 30% of the limit


def test_a_chip_that_rests_near_a_third_of_its_tdp_does_not_evict_as_its_power_wanders_across_that_line():
    """This box's chips rest at 36 to 40 W, and 30% of a 125 W limit is 37.5 W. Wandering across the line is not a load."""
    t = EvictTracker()
    phases = feed(t, [36.0, 36.5, 36.0, 36.2] + [36.0, 39.0, 37.0, 40.0, 36.5, 39.5] * 20)
    assert {p for p, _ in phases} == {"running"} and not t.take_events()


def test_a_chip_evicts_only_after_a_load_when_it_has_fallen_back_near_rest_and_has_run_a_while():
    assert evict_eligible(60.0, 17.0, 15.0, saw_load=True)                              # at most 1.2 times 15 W is 18 W
    assert not evict_eligible(60.0, 19.0, 15.0, saw_load=True)
    assert evict_eligible(60.0, 7.0, 4.0, saw_load=True) and not evict_eligible(60.0, 9.0, 4.0, saw_load=True)   # never below 8 W
    assert not evict_eligible(5.0, 17.0, 15.0, saw_load=True)                           # it has not run long enough
    assert not evict_eligible(60.0, 14.0, 15.0, saw_load=False)                         # it never saw a load: an idle chip does not evict


# --- the tracker over a chip's life -----------------------------------------------------------------------------------------------------

def feed(tracker, powers, start=0.0, step=3.0, chips=1):
    """Feed one power reading per `step` seconds; returns the phases the first chip was in."""
    phases = []
    for i, p in enumerate(powers):
        out = tracker.update([chip(p) for _ in range(chips)], start + i * step)
        phases.append((out[0].phase, round(out[0].elapsed, 3)))
    return phases


def test_an_idle_chip_that_never_loaded_never_evicts_however_long_it_sits():
    t = EvictTracker()
    phases = feed(t, [14.0 + (i % 3) * 0.5 for i in range(200)])
    assert {p for p, _ in phases} == {"running"} and not t.take_events()


def test_a_chip_that_loads_then_unloads_evicts_sits_out_and_fills_when_the_load_comes_back():
    t = EvictTracker()
    idle, load = [14.0] * 6, [62.0] * 6
    phases = feed(t, idle + load + [14.0] * 8)
    names = [p for p, _ in phases]
    assert names[:12] == ["running"] * 12                                              # the load itself is not an eviction
    first_evict = names.index("evicting")
    assert first_evict >= 12 and phases[first_evict][1] == 0.0                          # it starts at once, with no elapsed time
    assert names[first_evict:first_evict + 3] == ["evicting"] * 3                       # the dissolve lasts a few readings
    assert "idle" in names[first_evict:] and names.index("idle") - first_evict >= 2
    events = t.take_events()
    assert [e.kind for e in events] == ["evict"] and events[0].chip == 0
    after = feed(t, [62.0] * 8, start=100.0)
    assert [p for p, _ in after][0] == "filling" and after[0][1] == 0.0
    assert [p for p, _ in after][-1] == "running"
    assert [e.kind for e in t.take_events()] == ["fill"]


def test_the_evict_event_says_which_chip_and_the_numbers_that_decided_it():
    t = EvictTracker()
    feed(t, [14.0] * 6 + [62.0] * 6 + [14.0] * 6)
    (event,) = t.take_events()
    assert event.what.startswith("chip 0 evicted") and "62 W" in event.what and "14 W" in event.what
    assert "right to left" in event.why and "1.2" in event.why and "1.5" in event.why
    assert t.take_events() == []                                                         # events are handed over once


def test_each_chip_is_judged_on_its_own_readings():
    t = EvictTracker()
    for i in range(30):
        loaded = 62.0 if 5 <= i < 12 else 14.0
        out = t.update([chip(loaded), chip(14.0), chip(14.0)], i * 3.0)
    phases = [t.update([chip(14.0)] * 3, 100.0 + i)[j].phase for i in range(1) for j in range(3)]
    assert phases[0] in ("evicting", "idle", "running") and phases[1] == phases[2] == "running"
    assert {e.chip for e in t.take_events()} == {0}


def test_a_chip_that_starts_under_load_has_no_known_rest_and_so_waits_for_a_real_drop():
    t = EvictTracker()
    phases = feed(t, [70.0] * 6 + [50.0] * 6 + [6.0] * 6)
    names = [p for p, _ in phases]
    assert "evicting" not in names[:12] and "evicting" in names[12:]                      # 50 W is not rest, 6 W is below the 8 W floor


def test_the_load_returning_during_an_eviction_turns_it_round_into_a_fill():
    t = EvictTracker()
    before = feed(t, [14.0] * 6 + [62.0] * 6 + [14.0] * 5)
    assert before[-1][0] == "evicting" and before[-1][1] == 0.0                         # it began at 48 s
    phases = feed(t, [62.0] * 4, start=51.0)                                             # and 3 s in, the load is back
    assert phases[0] == ("filling", 5.0)                                                 # it turns round from where it had got to: 3 of 8 s gone, 5 to fill
    assert "evicting" not in [p for p, _ in phases] and [e.kind for e in t.take_events()] == ["evict", "fill"]


def test_a_different_number_of_chips_restarts_the_tracking():
    t = EvictTracker()
    feed(t, [14.0] * 6 + [62.0] * 6, chips=2)
    out = t.update([chip(14.0)] * 3, 100.0)
    assert [d.phase for d in out] == ["running"] * 3


# --- the dissolve and the fill, per voice -----------------------------------------------------------------------------------------------------

def drain(phase, elapsed, n=SR * 10, rank=0, count=4, prime=0):
    return drain_progress(phase, elapsed, n, SR, rank=rank, count=count, prime_index=prime)


def test_a_running_chip_has_not_dissolved_and_an_idle_one_has_completely():
    assert np.allclose(drain("running", 0.0), 0.0) and np.allclose(drain("idle", 0.0), 1.0)


def test_an_evicting_voice_dissolves_from_nothing_to_nothing_left_and_never_comes_back():
    p = drain("evicting", 0.0)
    assert p[0] == 0.0 and p[-1] == 1.0 and (np.diff(p) >= -1e-12).all() and p.max() <= 1.0
    late = drain("evicting", EVICT_SECONDS - 1.0, n=SR * 2)
    assert 0.0 <= late[0] < 1.0 and late[-1] == 1.0                                       # picking up part-way through


def test_a_filling_voice_comes_back_from_nothing_to_whole():
    p = drain("filling", 0.0)
    assert p[0] == 1.0 and p[-1] == 0.0 and (np.diff(p) <= 1e-12).all()


def test_voices_go_right_to_left_on_the_way_out_and_left_to_right_on_the_way_in():
    right, left = drain("evicting", 0.0, rank=0), drain("evicting", 0.0, rank=3)           # rank 0 is the rightmost voice
    assert right[SR] > left[SR] and np.argmax(right > 0.01) < np.argmax(left > 0.01)
    fill_left, fill_right = drain("filling", 0.0, rank=3), drain("filling", 0.0, rank=0)
    assert fill_left[SR * 2] < fill_right[SR * 2]                                           # the leftmost fills first, so it is further on


def test_the_drain_rates_are_prime_multiples_so_voices_of_one_chip_finish_at_different_times():
    assert DRAIN_PRIMES[:4] == (2, 3, 5, 7) and all(DRAIN_PRIMES[i] < DRAIN_PRIMES[i + 1] for i in range(len(DRAIN_PRIMES) - 1))
    done = [int(np.argmax(drain("evicting", 0.0, prime=k) >= 1.0)) for k in range(5)]
    assert done == sorted(done, reverse=True) and len(set(done)) == 5                        # a higher prime drains faster


# --- the sweeps that travel with it -------------------------------------------------------------------------------------------------------------

def centroid(x):
    spec = np.abs(np.fft.rfft(x))
    return float((spec * np.arange(len(spec))).sum() / spec.sum())


def test_the_evict_sweep_falls_in_pitch_and_the_fill_sweep_rises_and_both_are_finite_and_quiet_outside_their_length():
    out, back = evict_sweep(5.0, 44100, 3), fill_sweep(5.0, 44100, 3)
    assert len(out) == len(back) == 5 * 44100 and np.isfinite(out).all() and np.abs(out).max() <= 0.5
    half = int(1.2 * 44100)
    assert centroid(out[:half]) > 1.6 * centroid(out[half:2 * half]) and centroid(back[half:2 * half]) > 1.6 * centroid(back[:half])
    assert np.abs(out[-1000:]).max() < 1e-6 and not np.array_equal(out, evict_sweep(5.0, 44100, 4))
    assert np.array_equal(out, evict_sweep(5.0, 44100, 3))


def test_a_chips_presence_is_how_much_of_the_band_it_gets_to_be():
    assert presence_weight("running") == 1.0 and presence_weight("idle") < presence_weight("evicting") < presence_weight("filling") < 1.0


# --- through the chorus ----------------------------------------------------------------------------------------------------------------------------

CHIPS = mock_telemetry(4, 3)


def states(*phases):
    return [ChipDefrag(p, 100.0) if p in ("idle",) else ChipDefrag(p, 0.0) for p in phases]


def test_an_idle_chip_is_silent_and_the_others_play_as_they_did():
    with_idle = render_chorus(CHIPS, 4.0, SR, DEFAULT_PATCH, only=None, defrag=states("running", "running", "running", "idle"), cables=[])
    base = render_chorus(CHIPS, 4.0, SR, DEFAULT_PATCH, cables=[])
    assert not np.allclose(with_idle, base)
    assert np.array_equal(render_chorus(CHIPS, 4.0, SR, DEFAULT_PATCH, defrag=states("running", "running", "running", "running")),
                          render_chorus(CHIPS, 4.0, SR, DEFAULT_PATCH))                       # nothing running is nothing changed
    assert np.array_equal(render_chorus(CHIPS, 4.0, SR, DEFAULT_PATCH, defrag=None), render_chorus(CHIPS, 4.0, SR, DEFAULT_PATCH))


def test_auditioning_a_device_ignores_the_defrag_so_it_is_heard_as_it_is():
    a = render_chorus(CHIPS, 4.0, SR, DEFAULT_PATCH, only=["ttsteam"], defrag=states("running", "running", "running", "idle"))
    assert np.array_equal(a, render_chorus(CHIPS, 4.0, SR, DEFAULT_PATCH, only=["ttsteam"]))


def test_an_evicting_chips_voices_leave_the_stereo_field_toward_the_left_edge():
    """ttsteam is on chip 3, at the right of the field. While that chip evicts its voice drifts left and fades."""
    kw = dict(stereo=True, duration=8.0, cables=[], soloists=[7])
    base = render_chorus(CHIPS, 8.0, SR, DEFAULT_PATCH, **{k: v for k, v in kw.items() if k != "duration"})
    gone = render_chorus(CHIPS, 8.0, SR, DEFAULT_PATCH, defrag=states("running", "running", "running", "evicting"), **{k: v for k, v in kw.items() if k != "duration"})
    def share(x):
        return float(np.sum(x[:, 1] ** 2) / (np.sum(x ** 2) + 1e-12))
    assert not np.allclose(base, gone) and share(gone) < share(base) - 0.02
    quiet = render_chorus(CHIPS, 8.0, SR, DEFAULT_PATCH, defrag=states("running", "running", "running", "idle"), **{k: v for k, v in kw.items() if k != "duration"})
    assert np.sum(quiet ** 2) < np.sum(base ** 2)


def test_the_defrag_list_must_have_one_state_per_chip():
    with pytest.raises(ValueError, match="4 chips"):
        render_chorus(CHIPS, 2.0, SR, DEFAULT_PATCH, defrag=[ChipDefrag("running", 0.0)])


def test_a_chip_that_has_left_the_band_hardly_gets_the_floor_and_one_that_is_coming_back_gets_it_less_often():
    base = slot_weights(CHIPS, DEFAULT_PATCH)
    weights = slot_weights(CHIPS, DEFAULT_PATCH, states("running", "running", "evicting", "filling"))
    for slot in range(len(DEVICES)):
        factor = {0: 1.0, 1: 1.0, 2: presence_weight("evicting"), 3: presence_weight("filling")}[slot_chip(slot, 4)]
        assert weights[slot] == pytest.approx(base[slot] * factor)


# --- gaps a mutation run found ---------------------------------------------------------------------------------------------------------

def test_rest_is_not_learned_from_the_edge_of_a_fall_so_a_chip_that_started_under_load_is_not_evicted_the_moment_it_drops():
    t = EvictTracker()
    phases = [p for p, _ in feed(t, [70.0] * 6 + [28.0] * 12)]
    first_low = 6
    assert "evicting" not in phases[first_low:first_low + 3]                                            # 28 W has not settled yet: 3 readings in, still falling
    assert "evicting" in phases[first_low:]                                                              # once it has, it is the chip's rest


def test_a_sweep_is_cut_to_the_part_that_falls_inside_the_render_and_there_is_none_once_it_has_finished():
    from tensixinet.music.evict import SWEEP_S, phase_sweep
    full = evict_sweep(SWEEP_S, SR, 5)
    part = phase_sweep("evicting", 1.0, 1.5, SR, 5)
    assert len(part) == int(1.5 * SR) and np.allclose(part, full[SR:SR + len(part)])
    assert phase_sweep("evicting", SWEEP_S + 0.1, 2.0, SR, 5) is None and phase_sweep("running", 0.0, 2.0, SR, 5) is None
    assert phase_sweep("idle", 0.0, 2.0, SR, 5) is None and np.allclose(phase_sweep("filling", 0.5, 1.0, SR, 5), fill_sweep(SWEEP_S, SR, 5)[SR // 2:SR // 2 + SR])


def test_an_idle_chip_is_exactly_a_chip_whose_voices_have_all_been_turned_down():
    """With the turns fixed, a chip that is idle is the same as one whose devices all have a factor of nothing: its voices are gone and the rest are untouched."""
    from tensixinet.music.chorus import assign_device
    names = {assign_device(s).__name__ for s in range(len(DEVICES)) if slot_chip(s, 4) == 3}
    muted = dataclasses.replace(DEFAULT_PATCH, emphasis=tuple((n, 1e-12) for n in names))
    a = render_chorus(CHIPS, 4.0, SR, DEFAULT_PATCH, soloists=[0], defrag=states("running", "running", "running", "idle"), cables=[])
    b = render_chorus(CHIPS, 4.0, SR, muted, soloists=[0], cables=[])
    assert np.allclose(a, b, atol=1e-6)


def test_the_sweep_is_really_added_and_it_travels_right_to_left_on_the_way_out_and_left_to_right_on_the_way_in(monkeypatch):
    import tensixinet.music.chorus as chorus_module
    one = mock_telemetry(1, 3)
    monkeypatch.setattr(chorus_module, "drain_progress", lambda phase, elapsed, n, sr, rank, count, prime: np.ones(n))   # the voices are already gone
    noise = np.random.default_rng(1).uniform(-0.3, 0.3, int(3.0 * SR))       # noise and not a constant, which a wide voice would favour one ear with
    monkeypatch.setattr(chorus_module, "phase_sweep", lambda phase, elapsed, duration, sr, seed: noise if phase in ("evicting", "filling") else None)
    def halves(phase):
        out = chorus_module.render_chorus(one, 3.0, SR, DEFAULT_PATCH, stereo=True, cables=[], defrag=[ChipDefrag(phase, 0.0)])
        h = len(out) // 2
        return [float(np.sum(out[a:b, c] ** 2)) for a, b in ((0, h), (h, len(out))) for c in (0, 1)]
    ll, lr, rl, rr = halves("evicting")[0], halves("evicting")[1], halves("evicting")[2], halves("evicting")[3]
    assert lr > ll and rr < rl                                           # evicting: right ear first, left ear later
    fl, fr, gl, gr = halves("filling")
    assert fl > fr and gr > gl                                           # filling: the other way round
    monkeypatch.setattr(chorus_module, "phase_sweep", lambda *a, **k: None)
    silent = chorus_module.render_chorus(one, 3.0, SR, DEFAULT_PATCH, stereo=True, cables=[], defrag=[ChipDefrag("evicting", 0.0)])
    assert float(np.sum(silent ** 2)) < 1e-9                             # with no sweep and no voices there is nothing


def test_the_voices_of_a_chip_are_ranked_from_the_right_by_where_they_sit_and_given_their_primes_in_chip_order():
    from tensixinet.music.chorus import assign_device, n_slots, voice_order
    from tensixinet.music.space import device_space
    for chip in range(4):
        slots = [s for s in range(n_slots(4)) if slot_chip(s, 4) == chip]
        ranks = {s: voice_order(s, 4) for s in slots}
        assert sorted(r[0] for r in ranks.values()) == list(range(len(slots))) and {r[1] for r in ranks.values()} == {len(slots)}
        by_rank = sorted(slots, key=lambda s: ranks[s][0])
        offsets = [device_space(assign_device(s).__name__)[0] for s in by_rank]
        assert offsets == sorted(offsets, reverse=True)                                             # the rightmost first
        assert [ranks[s][2] for s in slots] == list(range(len(slots)))
