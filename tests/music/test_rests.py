# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Which steps of a note or pulse sequence rest. The lead and ttrollz use it; the coco's rests come from a castle (devices.ttquantus_gate)."""
import numpy as np

from tensixinet.music.rests import rest_flags


def test_no_share_means_no_rest_and_a_share_near_one_rests_nearly_every_step():
    assert not rest_flags(3, 1, 500, 0.0).any() and not rest_flags(3, 1, 500, -0.2).any()
    assert rest_flags(3, 1, 500, 0.95).mean() > 0.9


def test_the_share_of_resting_steps_follows_the_asked_share():
    for share in (0.1, 0.3, 0.6):
        got = np.mean([rest_flags(s, 7, 400, share).mean() for s in range(10)])
        assert abs(got - share) < 0.03


def test_the_flags_are_booleans_of_the_asked_length_and_repeat_for_the_same_seed_and_tag():
    a = rest_flags(5, 2, 40, 0.4)
    assert a.dtype == bool and a.shape == (40,) and np.array_equal(a, rest_flags(5, 2, 40, 0.4))
    assert rest_flags(5, 2, 0, 0.4).shape == (0,)


def test_a_different_seed_or_tag_gives_a_different_pattern():
    a = rest_flags(5, 2, 200, 0.4)
    assert not np.array_equal(a, rest_flags(6, 2, 200, 0.4)) and not np.array_equal(a, rest_flags(5, 3, 200, 0.4))


def test_asking_for_more_steps_does_not_change_the_earlier_ones():
    assert np.array_equal(rest_flags(5, 2, 30, 0.4), rest_flags(5, 2, 90, 0.4)[:30])


def test_a_bigger_share_rests_every_step_a_smaller_share_does():
    small, big = rest_flags(5, 2, 300, 0.2), rest_flags(5, 2, 300, 0.5)
    assert (big | ~small).all() and big.sum() > small.sum()
