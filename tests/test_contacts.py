from __future__ import annotations

import numpy as np
import pytest
from conftest import square_hold

from beta.contacts import Contact, busy_frames, detect_contacts, hold_distances, limb_speed

A, B = square_hold(1, 100, 100), square_hold(2, 200, 100)  # centers (120,120) and (220,120)
FAR = (0.0, 0.0)


def path(*segments) -> np.ndarray:
    """(n_frames, (x, y)) pieces -> (F, 2)."""
    return np.array([xy for n, xy in segments for _ in range(n)], float)


def run(pts, seen=None, holds=(A, B), torso=100.0, fps=30.0) -> list[Contact]:
    return detect_contacts("LH", pts, np.ones(len(pts), bool) if seen is None else seen, list(holds), torso, fps)


def test_dwell_gives_a_backdated_contact():
    assert run(path((10, FAR), (30, (120, 120)), (20, FAR))) == [Contact("LH", 1, 10, 39)]


def test_jitter_on_a_hold_gives_one_contact():
    pts = path((10, FAR), (40, (120, 120)))
    pts[10:] += np.random.default_rng(0).uniform(-2, 2, (40, 2))
    assert len(run(pts)) == 1


def test_fast_pass_through_gives_none():
    xs = np.linspace(0, 300, 30)  # 10 px/frame = 3 torso/s
    assert run(np.column_stack([xs, np.full(30, 120.0)])) == []


def test_occlusion_does_not_split_a_contact():
    pts, seen = path((10, FAR), (30, (120, 120)), (20, FAR)), np.ones(60, bool)
    pts[20:26], seen[20:26] = (400, 400), False
    assert run(pts, seen) == [Contact("LH", 1, 10, 39)]


def test_moving_to_an_adjacent_hold_switches():
    assert run(path((30, (120, 120)), (30, (220, 120)))) == [Contact("LH", 1, 0, 29), Contact("LH", 2, 30, 59)]


def test_one_frame_excursion_does_not_release():
    pts = path((10, FAR), (30, (120, 120)), (20, FAR))
    pts[20] = (400, 400)
    assert run(pts) == [Contact("LH", 1, 10, 39)]


def test_coming_back_to_a_hold_later_counts_from_the_return():
    # A fast pass over A, two seconds elsewhere, then a dwell: the contact starts at the return.
    pass_over = np.column_stack([np.linspace(0, 300, 30), np.full(30, 120.0)])  # over A at frames 10-14
    pts = np.vstack([pass_over, path((60, FAR), (30, (120, 120)))])
    assert run(pts, holds=(A,)) == [Contact("LH", 1, 90, 119)]


def test_point_in_two_overlapping_holds_goes_to_the_deeper_one():
    c = square_hold(3, 130, 100)  # overlaps A from x = 130 to 140
    assert run(path((20, (133, 120))), holds=(A, c))[0].hold == 1  # 7 px inside A, 3 px inside c


def test_unseen_frames_pause_the_onset_counter():
    # Slow seen frames inside: 13-16, then 3 unseen, then 20-26 before the leave shows up in the speed window.
    # That's 4 + 7: it only reaches the 8 needed if unseen frames pause the count instead of resetting it.
    pts, seen = path((10, FAR), (20, (120, 120)), (10, FAR)), np.ones(40, bool)
    seen[17:20] = False
    assert run(pts, seen) == [Contact("LH", 1, 10, 29)]


def test_scaling_coordinates_gives_the_same_contacts():
    pts = path((10, FAR), (30, (120, 120)), (30, (220, 120)))
    big = [square_hold(1, 200, 200, 80), square_hold(2, 400, 200, 80)]
    assert run(pts * 2, holds=big, torso=200.0) == run(pts)


def test_frame_rate_does_not_change_times():
    at30 = run(path((10, FAR), (30, (120, 120)), (20, FAR)), fps=30.0)[0]
    at60 = run(path((20, FAR), (60, (120, 120)), (40, FAR)), fps=60.0)[0]
    assert abs(at30.start_frame / 30 - at60.start_frame / 60) <= 1 / 30
    assert abs(at30.end_frame / 30 - at60.end_frame / 60) <= 1 / 30


def test_points_outside_the_canvas_are_fine():  # Review Focus 4
    neg = square_hold(1, -150, -60)
    assert run(path((10, (-500.0, -500.0)), (30, (-130, -40))), holds=(neg,)) == [Contact("LH", 1, 10, 39)]


def test_limb_speed_in_torso_lengths_per_second():
    pts = np.column_stack([np.arange(20) * 10.0, np.zeros(20)])
    assert np.allclose(limb_speed(pts, 30.0, 100.0), 3.0)


def test_hold_distances_are_signed():
    d = hold_distances(np.array([[120.0, 120.0], [90.0, 120.0], [np.nan, np.nan]]), [A], 100.0)
    assert d[0, 0] == pytest.approx(0.2) and d[1, 0] == pytest.approx(-0.1) and np.isnan(d[2, 0])


def test_busy_frames_marks_contact_spans():
    busy = busy_frames([Contact("LH", 1, 2, 4), Contact("LF", 2, 7, 12)], 10)
    assert busy.tolist() == [False, False, True, True, True, False, False, True, True, True]
