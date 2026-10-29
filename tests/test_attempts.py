from __future__ import annotations

import numpy as np
import pytest
from conftest import square_hold

from beta.attempts import Attempt, estimate_floor, feet_on_floor, find_attempts
from beta.contacts import Contact

# Ids 1-20, a higher id higher on the wall; 20 is the top. Every bbox bottom is <= 850.
HOLDS = [square_hold(i, 300, 800 - 30 * i) for i in range(1, 21)]
F, FPS, TORSO = 240, 30.0, 100.0


def C(limb: str, hold: int, a: int, b: int) -> Contact:
    return Contact(limb, hold, a, b)


def contacts(*cs: Contact) -> dict[str, list[Contact]]:
    return {limb: [c for c in cs if c.limb == limb] for limb in ("LH", "RH", "LF", "RF")}


def floor(on_from=()) -> np.ndarray:
    """(F, 2) on-floor mask: both feet on the floor in the given [a, b) spans."""
    m = np.zeros((F, 2), bool)
    for a, b in on_from:
        m[a:b] = True
    return m


def hips(y0: float = 400.0, drops=()) -> np.ndarray:
    """(F, 2) mid-hip; drops are (start, end, dy) linear changes in y."""
    h = np.column_stack([np.full(F, 300.0), np.full(F, y0)])
    for a, b, dy in drops:
        h[a:b, 1] += np.linspace(0, dy, b - a)
        h[b:, 1] += dy
    return h


def test_sequence_merges_regrips():
    a = Attempt(0, 10, "clip_ended", 7, contacts(C("LH", 7, 0, 3), C("LH", 7, 5, 8), C("LH", 9, 9, 10)))
    assert a.sequence()["LH"] == [7, 9]


def test_floor_is_where_feet_rest_with_hands_free():
    # A short rest at the wall (900), climbing (600), then a much longer stand near the camera (1050).
    feet = np.full((F, 2, 2), 900.0)
    feet[30:90, :, 1] = 600.0
    feet[90:, :, 1] = 1050.0
    busy = np.zeros(F, bool)
    busy[30:90] = True
    assert estimate_floor(feet, np.ones((F, 2), bool), busy, HOLDS, TORSO, FPS) == pytest.approx(900.0, abs=1)


def test_no_floor_without_resting_feet():
    feet = np.full((F, 2, 2), 600.0)
    assert estimate_floor(feet, np.ones((F, 2), bool), np.zeros(F, bool), HOLDS, TORSO, FPS) is None


def test_feet_on_floor_carry_through_unseen_frames():
    feet = np.full((F, 2, 2), 900.0)
    seen = np.ones((F, 2), bool)
    seen[10:20] = False
    feet[10:20] = 0.0
    on = feet_on_floor(feet, seen, np.zeros((F, 2), bool), 900.0, TORSO)
    assert on[5:25].all()


def test_attempt_starts_when_both_feet_leave_the_floor():
    cs = contacts(C("LF", 1, 2, 8), C("LH", 3, 10, 50), C("RH", 4, 30, 60))
    a = find_attempts(cs, floor([(0, 20)]), hips(), HOLDS, TORSO, FPS)[0]
    assert a.start_frame == 20 and a.contacts["LH"] == [C("LH", 3, 20, 50)] and a.contacts["LF"] == []


def test_attempt_from_the_first_frame_when_the_clip_starts_on_the_wall():
    assert find_attempts(contacts(C("LH", 3, 0, 50)), floor(), hips(), HOLDS, TORSO, FPS)[0].start_frame == 0


def test_topped():
    cs = contacts(C("LH", 18, 0, 99), C("LH", 20, 100, 200), C("RH", 20, 110, 200))
    a = find_attempts(cs, floor(), hips(), HOLDS, TORSO, FPS)[0]
    assert (a.outcome, a.end_frame, a.high_point) == ("topped", 139, 20)


def test_top_hold_override():
    cs = contacts(C("LH", 18, 100, 200), C("RH", 18, 100, 200))
    assert find_attempts(cs, floor(), hips(), HOLDS, TORSO, FPS, top_hold=18)[0].outcome == "topped"


def test_fall_with_a_pose_gap():
    cs = contacts(C("RF", 2, 0, 55), C("LF", 9, 0, 57), C("LH", 14, 0, 58), C("RH", 15, 0, 59))
    h = hips(drops=[(59, 69, 130.0)])
    h[60:69] = np.nan  # 0.3 s without a pose mid-fall
    a = find_attempts(cs, floor([(75, F)]), h, HOLDS, TORSO, FPS)[0]
    assert a.outcome == "fell" and a.end_frame == 59 and a.fall.frame == 59
    assert a.fall.last_holds == {"LH": 14, "RH": 15, "LF": 9, "RF": 2}
    assert [r.limb for r in a.fall.release_order] == ["RF", "LF", "LH", "RH"]


def test_cutting_feet_is_not_a_fall():
    cs = contacts(C("LH", 14, 0, 200), C("RH", 15, 0, 200))
    assert find_attempts(cs, floor(), hips(drops=[(50, 59, 120.0)]), HOLDS, TORSO, FPS)[0].outcome == "clip_ended"


def test_a_fall_after_cutting_feet_is_still_a_fall():
    cs = contacts(C("LH", 14, 0, 150), C("RH", 15, 0, 150))  # feet cut at 50 while the hands hold on
    a = find_attempts(cs, floor([(170, F)]), hips(drops=[(50, 59, 120.0), (150, 160, 130.0)]), HOLDS, TORSO, FPS)[0]
    assert (a.outcome, a.end_frame) == ("fell", 150)


def test_came_down():
    cs = contacts(C("LH", 5, 0, 85), C("RH", 6, 0, 85))
    a = find_attempts(cs, floor([(90, F)]), hips(drops=[(60, 120, 150.0)]), HOLDS, TORSO, FPS)[0]
    assert (a.outcome, a.end_frame) == ("came_down", 90)


def test_two_attempts_and_no_contacts_from_walking_back():
    cs = contacts(
        C("LH", 14, 0, 59), C("RH", 15, 0, 59), C("RF", 1, 90, 100),  # RF brushes hold 1 walking back
        C("LH", 3, 160, 230), C("RH", 4, 160, 230),
    )  # fmt: skip
    on = floor([(75, 150)])
    on[90:101, 1] = False  # the busy foot isn't "on the floor", the other one is
    attempts = find_attempts(cs, on, hips(drops=[(59, 69, 130.0)]), HOLDS, TORSO, FPS)
    assert [a.outcome for a in attempts] == ["fell", "clip_ended"] and attempts[1].start_frame == 150
    assert attempts[0].contacts["RF"] == []


def test_single_attempt_without_a_floor():
    a = find_attempts(contacts(C("LH", 3, 40, 100)), None, hips(), HOLDS, TORSO, FPS)
    assert len(a) == 1 and a[0].start_frame == 40 and a[0].outcome == "clip_ended"
