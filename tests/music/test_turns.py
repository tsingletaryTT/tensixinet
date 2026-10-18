# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Turn-taking: who solos is decided by system conditions, like players trading solos."""
import numpy as np
import pytest

from tensixinet.music.turns import soloists, turn_gains


def test_one_voice_always_solos():
    assert soloists([5.0], 6) == [0] * 6


def test_no_turns_means_no_soloists():
    assert soloists([1.0, 2.0], 0) == []


def test_nobody_solos_twice_in_a_row_when_there_is_someone_else():
    for weights in ([1, 1], [1, 1, 10], [3, 1, 1, 1], [100, 1]):
        s = soloists(weights, 60)
        assert all(a != b for a, b in zip(s, s[1:])), weights


def test_a_busier_voice_takes_more_solos():
    s = soloists([1.0, 1.0, 10.0], 60)
    assert s.count(2) > s.count(0) and s.count(2) > s.count(1)


def test_a_quiet_voice_still_gets_turns():
    s = soloists([1.0, 1.0, 1.0, 40.0], 80)
    assert all(s.count(i) >= 1 for i in range(4))


def test_a_voice_with_no_weight_never_solos():
    assert 0 not in soloists([0.0, 5.0, 5.0], 40)


def test_equal_weights_share_turns_evenly():
    s = soloists([1.0] * 4, 40)
    assert sorted(s.count(i) for i in range(4))[0] >= 8


def test_all_zero_weights_are_treated_as_equal():
    s = soloists([0.0, 0.0, 0.0], 30)
    assert set(s) == {0, 1, 2}


def test_soloists_are_deterministic():
    assert soloists([1, 2, 3], 20) == soloists([1, 2, 3], 20)


# --- gains ---------------------------------------------------------------------

SR = 1000


def test_gains_have_one_row_per_voice_and_one_column_per_sample():
    g = turn_gains([0, 1], 3, 2000, SR, turn_s=1.0, floor=0.3, fade_s=0.1)
    assert g.shape == (3, 2000)


def test_the_soloist_is_full_volume_mid_turn_and_the_others_sit_at_the_floor():
    g = turn_gains([0, 1], 3, 2000, SR, turn_s=1.0, floor=0.3, fade_s=0.1)
    assert g[0, 500] == pytest.approx(1.0) and g[1, 500] == pytest.approx(0.3) and g[2, 500] == pytest.approx(0.3)
    assert g[1, 1500] == pytest.approx(1.0) and g[0, 1500] == pytest.approx(0.3)


def test_hand_overs_fade_instead_of_clicking():
    g = turn_gains([0, 1], 2, 2000, SR, turn_s=1.0, floor=0.1, fade_s=0.2)
    assert np.abs(np.diff(g, axis=1)).max() < 0.1  # a step of 0.9 spread over 200 samples
    assert 0.1 < g[0, 1000] < 1.0  # mid-fade at the boundary


def test_the_last_soloist_holds_to_the_end_of_the_audio():
    g = turn_gains([0, 1], 2, 3000, SR, turn_s=1.0, floor=0.3, fade_s=0.1)
    assert g[1, -1] == pytest.approx(1.0)  # more samples than turns: hold the last soloist


# --- live mode picks one soloist at a time -----------------------------------------------------------------

from tensixinet.music.turns import pick_soloist  # noqa: E402


def test_picking_one_turn_at_a_time_matches_planning_them_all_at_once():
    weights = [1.0, 2.0, 5.0, 3.0]
    planned = soloists(weights, 30)
    previous = None
    for k, expected in enumerate(planned):
        previous = pick_soloist(weights, k, previous)
        assert previous == expected


def test_a_live_pick_never_repeats_the_player_who_just_soloed_even_when_the_weights_changed():
    for k in range(40):
        for previous in range(3):
            assert pick_soloist([1.0, 1.0, 10.0], k, previous) != previous


def test_a_live_pick_skips_a_voice_with_no_weight():
    assert all(pick_soloist([0.0, 5.0, 5.0], k, None) != 0 for k in range(40))


@pytest.mark.filterwarnings("error")  # drawing from nobody divides by zero
def test_a_lone_voice_solos_every_turn_even_though_it_just_did():
    assert [pick_soloist([4.0], k, 0) for k in range(5)] == [0] * 5
    assert [pick_soloist([0.0, 4.0], k, 1) for k in range(5)] == [1] * 5


def test_the_sequence_of_soloists_is_pinned_so_a_saved_patch_replays_the_same_run():
    assert soloists([1.0, 2.0, 5.0, 3.0], 12) == [2, 1, 3, 2, 0, 2, 1, 3, 2, 1, 3, 2]  # what the code gave before pick_soloist was split out of it
