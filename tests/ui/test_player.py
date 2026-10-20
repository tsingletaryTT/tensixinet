# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Player: which session the page is showing, so that Play, Stop and a change of setting each do what the page says.

The page streams from a generator per Play. Three things went wrong without this: a change of setting did nothing until Play was pressed
again, a Stop while a session was still starting was ignored (the session then started), and the generator of an ended session could still
write to the page after a newer Play had begun."""
from tensixinet.ui.player import Player


class FakeSession:
    def __init__(self):
        self.stopped = 0

    def stop(self):
        self.stopped += 1


def test_a_new_player_is_idle():
    p = Player()
    assert not p.active and p.config is None


def test_begin_makes_the_player_active_with_the_config_and_the_token_current():
    p = Player()
    t = p.begin(("synthetic", 4))
    assert p.active and p.config == ("synthetic", 4) and p.is_current(t)


def test_begin_ends_the_session_that_was_playing_and_the_old_token_is_no_longer_current():
    p, old = Player(), FakeSession()
    t1 = p.begin("a")
    assert p.attach(t1, old)
    t2 = p.begin("b")
    assert old.stopped == 1 and not p.is_current(t1) and p.is_current(t2) and p.config == "b"


def test_a_session_that_finishes_starting_after_a_newer_begin_is_refused():
    p = Player()
    t1 = p.begin("a")
    p.begin("b")
    assert not p.attach(t1, FakeSession())


def test_stop_ends_the_session_and_leaves_the_player_idle():
    p, s = Player(), FakeSession()
    t = p.begin("a")
    p.attach(t, s)
    p.stop()
    assert s.stopped == 1 and not p.active and p.config is None and not p.is_current(t)


def test_stop_while_the_session_is_still_starting_refuses_it_when_it_arrives():
    p, s = Player(), FakeSession()
    t = p.begin("a")
    p.stop()
    assert not p.attach(t, s) and not p.active


def test_finish_marks_a_session_that_ended_on_its_own_as_idle_so_a_change_does_not_restart_it():
    p = Player()
    t = p.begin("a")
    p.finish(t)
    assert not p.active


def test_a_stale_generator_finishing_does_not_idle_the_newer_session():
    p = Player()
    t1 = p.begin("a")
    p.begin("b")
    p.finish(t1)
    assert p.active and p.config == "b"


def test_claim_hands_back_the_old_session_without_stopping_it_so_the_caller_can_show_progress_first():
    p, old = Player(), FakeSession()
    p.attach(p.begin("a"), old)
    token, handed = p.claim("b")
    assert token is not None and handed is old and old.stopped == 0 and p.config == "b" and p.is_current(token)


def test_a_restart_claim_is_refused_when_idle_or_unchanged_and_that_is_decided_with_the_claim():
    p = Player()
    assert p.claim("a", only_if_changed=True) == (None, None)       # nothing is playing
    p.begin("a")
    assert p.claim("a", only_if_changed=True) == (None, None)       # nothing changed
    token, _ = p.claim("b", only_if_changed=True)
    assert token is not None and p.config == "b"


def test_a_stop_that_comes_after_a_claim_is_not_undone_by_the_work_that_follows_it():
    p = Player()
    p.begin("a")
    token, old = p.claim("b", only_if_changed=True)
    p.stop()                                                        # the user presses Stop while the restart is yet to build its session
    assert not p.is_current(token) and not p.attach(token, FakeSession()) and not p.active


def test_a_change_after_the_session_ended_or_was_stopped_is_refused():
    p = Player()
    t = p.begin("a")
    p.finish(t)
    assert p.claim("b", only_if_changed=True) == (None, None)
    p.begin("a")
    p.stop()
    assert p.claim("b", only_if_changed=True) == (None, None)
