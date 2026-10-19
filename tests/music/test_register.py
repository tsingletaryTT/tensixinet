# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The shift register of Man With The Red Steam as a clock and a source of held notes, apart from the snare it plays."""
import dataclasses

import numpy as np
import pytest

from tensixinet.music.mapping import aiclk_to_bpm
from tensixinet.music.patch import DEFAULT_PATCH, SCALES
from tensixinet.music.register import STEPS, held_semitones, step_pattern, step_times, step_values
from tensixinet.telemetry import FALLBACK


def tel(**kw):
    return dataclasses.replace(FALLBACK, **kw)


def test_the_register_has_one_step_per_tensix_column_and_fires_on_harvested_ones():
    assert STEPS == 14
    assert [i for i, hit in enumerate(step_pattern(0x37F7)) if hit] == [3, 11]


def test_step_times_are_the_firing_steps_on_an_eighth_note_clock():
    t = tel(tensix_mask=0x37F7, aiclk=800.0)
    step_s = 30.0 / aiclk_to_bpm(800.0)
    got = step_times(t, 30 * step_s)
    expected = [k * step_s for k in range(100) if k % STEPS in (3, 11) and k * step_s < 30 * step_s]
    assert got.tolist() == pytest.approx(expected)


def test_step_times_stop_at_the_duration_and_are_empty_for_a_chip_with_every_column():
    t = tel(tensix_mask=0x37F7)
    assert step_times(t, 20.0).max() < 20.0 and (np.diff(step_times(t, 20.0)) > 0).all()
    assert len(step_times(tel(tensix_mask=0x3FFF), 20.0)) == 0


def test_a_faster_clock_fires_more_often():
    slow, fast = tel(tensix_mask=0x37F7, aiclk=500.0), tel(tensix_mask=0x37F7, aiclk=1300.0)
    assert len(step_times(fast, 30.0)) > len(step_times(slow, 30.0))


def test_the_held_values_are_one_per_step_in_zero_to_one_and_come_from_the_seed():
    v = step_values(7)
    assert v.shape == (STEPS,) and (v >= 0).all() and (v < 1).all()
    assert np.array_equal(v, step_values(7)) and not np.array_equal(v, step_values(8))
    assert len(set(v.tolist())) > 8  # not a constant


def test_held_semitones_are_notes_of_the_patch_scale_held_for_a_step_and_looping_with_the_register():
    t = tel(aiclk=800.0)
    sr, step_s = 8000, 30.0 / aiclk_to_bpm(800.0)
    patch = dataclasses.replace(DEFAULT_PATCH, seed=3)
    n = int(2.2 * STEPS * step_s * sr)
    semis = held_semitones(t, n, sr, patch)
    assert semis.shape == (n,)
    allowed = set(SCALES["major_pentatonic"])
    assert set(np.unique(semis).tolist()) <= allowed
    per_step = int(step_s * sr)
    assert len(np.unique(semis[: per_step - 2])) == 1                      # held through a step
    lap = round(STEPS * step_s * sr)  # a step is not a whole number of samples, so a lap lines up to within a sample at each boundary
    assert np.mean(semis[:lap] == semis[lap: 2 * lap]) > 0.999  # one lap repeats the last
    assert len(np.unique(semis)) >= 3                                        # and the register does move


def test_held_semitones_follow_the_patch_seed_and_scale():
    t, sr = tel(aiclk=800.0), 8000
    a = held_semitones(t, 8000, sr, dataclasses.replace(DEFAULT_PATCH, seed=1))
    b = held_semitones(t, 8000, sr, dataclasses.replace(DEFAULT_PATCH, seed=2))
    whole = dataclasses.replace(DEFAULT_PATCH, seed=1, scale_name="whole_tone", scale=SCALES["whole_tone"])
    assert not np.array_equal(a, b) and not np.array_equal(a, held_semitones(t, 8000, sr, whole))
