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


# --- Visitors: one Player for each person on the page ---------------------------------------------------------

import threading

import pytest

from tensixinet.ui.player import Busy, Visitors


def test_each_visitor_has_their_own_player_and_the_same_visitor_always_gets_the_same_one():
    v = Visitors(limit=5)
    assert v.player("a") is v.player("a") and v.player("a") is not v.player("b")


def test_one_visitors_stop_does_not_touch_anothers_session():
    v, sa, sb = Visitors(5), FakeSession(), FakeSession()
    pa, ta, _ = v.claim("a", "x")
    pb, tb, _ = v.claim("b", "y")
    pa.attach(ta, sa)
    pb.attach(tb, sb)
    v.player("a").stop()
    assert sa.stopped == 1 and sb.stopped == 0 and pb.is_current(tb) and pb.active and pb.config == "y"


def test_a_new_play_by_one_visitor_leaves_the_others_playing():
    v, sb = Visitors(5), FakeSession()
    pb, tb, _ = v.claim("b", "y")
    pb.attach(tb, sb)
    v.claim("a", "x")
    v.claim("a", "x2")
    assert sb.stopped == 0 and pb.is_current(tb)


def test_a_play_that_would_pass_the_limit_is_refused_with_the_numbers():
    v = Visitors(limit=2)
    v.claim("a", 1)
    v.claim("b", 1)
    with pytest.raises(Busy) as e:
        v.claim("c", 1)
    assert e.value.playing == 2 and e.value.limit == 2 and "2" in str(e.value)
    assert not v.player("c").active                       # the refused visitor was not made to play


def test_a_visitor_who_is_already_playing_can_play_again_at_the_limit_and_a_change_is_not_refused():
    v = Visitors(limit=1)
    v.claim("a", 1)
    assert v.claim("a", 2)[1] is not None                 # Play again
    assert v.claim("a", 3, only_if_changed=True)[1] is not None   # a change of setting


def test_a_place_opens_when_someone_stops_or_leaves_or_their_session_ends():
    v = Visitors(limit=1)
    pa, ta, _ = v.claim("a", 1)
    pa.stop()
    assert v.claim("b", 1)[1] is not None
    v.drop("b")
    assert v.claim("c", 1)[1] is not None
    v.player("c").finish(v.player("c")._epoch)
    assert v.claim("d", 1)[1] is not None


def test_only_players_that_are_playing_count_towards_the_limit_not_everyone_who_has_the_page_open():
    v = Visitors(limit=1)
    for key in "abcde":
        v.player(key)                                     # five people looking at the page
    assert v.playing() == 0 and v.claim("a", 1)[1] is not None


def test_dropping_a_visitor_stops_their_session_and_forgets_them_and_dropping_a_stranger_is_fine():
    v, s = Visitors(5), FakeSession()
    p, t, _ = v.claim("a", 1)
    p.attach(t, s)
    v.drop("a")
    assert s.stopped == 1 and len(v) == 0
    v.drop("nobody")
    assert v.player("a") is not p                         # a visitor who comes back starts fresh


def test_claiming_hands_back_the_session_that_was_playing_for_that_visitor_only():
    v, sa, sb = Visitors(5), FakeSession(), FakeSession()
    pa, ta, _ = v.claim("a", 1)
    pa.attach(ta, sa)
    pb, tb, _ = v.claim("b", 1)
    pb.attach(tb, sb)
    _, _, old = v.claim("a", 2)
    assert old is sa and sa.stopped == 0                  # handed over to be stopped by the caller, as Player.claim does


def test_the_limit_holds_when_many_visitors_press_play_at_once(monkeypatch):
    import time
    v, ok, busy = Visitors(limit=3), [], []
    count = Visitors.playing
    monkeypatch.setattr(Visitors, "playing", lambda self: (lambda n: (time.sleep(0.002), n)[1])(count(self)))   # the count goes stale before the claim, unless the registry's lock holds the two together
    def press(i):
        try:
            v.claim(f"v{i}", i)
            ok.append(i)
        except Busy:
            busy.append(i)
    threads = [threading.Thread(target=press, args=(i,)) for i in range(40)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(ok) == 3 and len(busy) == 37 and v.playing() == 3


def test_reset_stops_everyone_and_forgets_them_and_a_limit_of_nothing_is_refused():
    v, s = Visitors(5), FakeSession()
    p, t, _ = v.claim("a", 1)
    p.attach(t, s)
    v.reset()
    assert s.stopped == 1 and len(v) == 0
    with pytest.raises(ValueError):
        Visitors(0)


# --- reaching the running session (the Play tab's knob handlers need it) -------------------------------------------------------------

def test_session_is_none_before_attach_after_stop_and_after_finish():
    player = Player()
    assert player.session is None
    token = player.begin("a")
    assert player.session is None  # claimed, still being built
    stub = FakeSession()
    assert player.attach(token, stub) and player.session is stub
    player.stop()
    assert player.session is None
    token = player.begin("b")
    player.attach(token, FakeSession())
    player.finish(token)
    assert player.session is None


def test_a_newer_claim_takes_the_old_session_away():
    player = Player()
    first = FakeSession()
    player.attach(player.begin("a"), first)
    player.claim("b")
    assert player.session is None


def test_two_visitors_have_their_own_sessions():
    visitors = Visitors(3)
    a, b = FakeSession(), FakeSession()
    for key, stub in (("a", a), ("b", b)):
        player, token, _ = visitors.claim(key, ("cfg", key))
        player.attach(token, stub)
    assert visitors.session("a") is a and visitors.session("b") is b and visitors.session("nobody") is None


def test_the_limit_counts_a_session_of_any_kind():
    visitors = Visitors(1)
    player, token, _ = visitors.claim("a", ("instrument", "system"))
    player.attach(token, FakeSession())
    with pytest.raises(Busy):
        visitors.claim("b", ("watch", "synthetic"))
    visitors.drop("a")
    visitors.claim("b", ("watch", "synthetic"))


def test_asking_for_a_visitors_session_does_not_add_the_visitor():
    visitors = Visitors(2)
    assert visitors.session("stranger") is None and len(visitors) == 0
