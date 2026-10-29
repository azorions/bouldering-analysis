from __future__ import annotations

import numpy as np
import pytest
from conftest import body, track_of

from beta.align import CameraTrack, FrameTransform
from beta.limbs import effector_points, fix_left_right, hip_centers, torso_length, wall_landmarks

HANDS, FEET = ((200.0, 100.0), (300.0, 100.0)), ((220.0, 400.0), (280.0, 400.0))


def test_hand_point_reaches_past_the_knuckles():
    lm = body(HANDS, FEET)
    lm[15, :2], lm[19, :2], lm[17, :2] = (100, 100), (110, 100), (110, 110)  # wrist, index, pinky
    pts, seen = effector_points(lm[None])
    assert np.allclose(pts[0, 0], (112, 106)) and seen[0, 0]


def test_hand_falls_back_to_the_forearm_when_fingers_are_hidden():
    lm = body(HANDS, FEET)
    lm[13, :2], lm[15, :2] = (80, 100), (100, 100)  # elbow, wrist
    lm[[17, 19], 3] = 0.2
    assert np.allclose(effector_points(lm[None])[0][0, 0], (108, 100))


def test_foot_point_is_the_front_of_the_shoe():
    lm = body(HANDS, FEET)
    lm[31, :2], lm[29, :2] = (200, 300), (180, 300)  # left toe, left heel
    assert np.allclose(effector_points(lm[None])[0][0, 2], (197, 300))


def test_seen_follows_wrist_and_toe_visibility():
    lm = body(HANDS, FEET)
    lm[15, 3], lm[32, 3] = 0.4, 0.6
    assert effector_points(lm[None])[1][0].tolist() == [False, True, True, True]


def test_wall_landmarks_apply_the_camera():
    w = np.array([[1.0, 0, 10], [0, 1.0, -5], [0, 0, 1.0]])
    transforms = (FrameTransform(0, w, "anchor", 99), FrameTransform(1, None, "lost", 0), FrameTransform(2, w, "track", 99))
    cam = CameraTrack("moving", 30.0, 3, 640, 480, (700, 500), 0, transforms, {})
    lm = wall_landmarks(track_of([body(HANDS, FEET), body(HANDS, FEET), None]), cam)
    assert np.allclose(lm[0, :, :2], body(HANDS, FEET)[:, :2] + (10, -5)) and np.allclose(lm[0, :, 3:], body(HANDS, FEET)[:, 3:])
    assert np.isnan(lm[1]).all() and np.isnan(lm[2]).all()


def test_torso_length_is_the_median():
    frames = [body(HANDS, FEET), body(HANDS, FEET), body(HANDS, FEET, torso=300.0)]
    assert torso_length(np.stack(frames)) == pytest.approx(100.0)


def test_torso_length_needs_shoulders_and_hips():  # Review Focus 5
    with pytest.raises(ValueError, match="shoulders and hips"):
        torso_length(np.full((3, 33, 5), np.nan))


PAIRS = [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16), (17, 18), (19, 20), (21, 22),
         (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)]  # fmt: skip


def mirrored(lm: np.ndarray) -> np.ndarray:
    """Every left landmark labelled right and vice versa, as MediaPipe does when it gets the facing wrong."""
    out = lm.copy()
    for left, right in PAIRS:
        out[[left, right]] = lm[[right, left]]
    return out


def test_global_swap_when_most_frames_face_the_camera():
    lm = np.stack([mirrored(body(HANDS, FEET))] * 10)
    fixed, report = fix_left_right(lm, 100.0, 30.0)
    assert report["global_swap"] and np.allclose(fixed[0, 15, :2], HANDS[0])
    assert report["frames_swapped"] == {"arms": 0, "legs": 0}


def test_no_global_swap_when_facing_the_wall():
    assert not fix_left_right(np.stack([body(HANDS, FEET)] * 10), 100.0, 30.0)[1]["global_swap"]


def test_continuity_swaps_a_flipped_arm_pair():
    lm = np.stack([body(HANDS, FEET)] * 10)
    left_arm, right_arm = [13, 15, 17, 19, 21], [14, 16, 18, 20, 22]
    lm[5, left_arm], lm[5, right_arm] = lm[5, right_arm].copy(), lm[5, left_arm].copy()
    fixed, report = fix_left_right(lm, 100.0, 30.0)
    assert report["frames_swapped"] == {"arms": 1, "legs": 0} and np.allclose(fixed[5, 15, :2], HANDS[0])


def test_continuity_leaves_close_hands_alone():
    close = ((240.0, 100.0), (260.0, 100.0))  # 0.2 torso apart: a match, not a swap
    lm = np.stack([body(close, FEET)] * 10)
    lm[5, [15, 16]] = lm[5, [16, 15]].copy()
    assert fix_left_right(lm, 100.0, 30.0)[1]["frames_swapped"]["arms"] == 0


def test_continuity_ignores_frames_where_the_wrists_are_guessed():
    # Hands together, then MediaPipe mislabels the arms while unsure (low visibility,
    # as in a fall), then labels them right again: the guessed frames must not make
    # the right ones look crossed, or the swap carries on to every later frame.
    together = body(((250.0, 100.0), (252.0, 100.0)), FEET)
    apart = body(HANDS, FEET)
    flipped = apart.copy()
    flipped[[15, 16]] = apart[[16, 15]]
    lm = np.stack([together] * 3 + [flipped, flipped, apart] + [apart] * 4)
    lm[3:6, [15, 16], 3] = 0.3
    fixed, report = fix_left_right(lm, 100.0, 30.0)
    assert report["frames_swapped"]["arms"] == 0 and np.allclose(fixed[6:, 15, :2], HANDS[0])


def test_hip_centers():
    assert np.allclose(hip_centers(body(HANDS, FEET)[None])[0], (250, 300))
