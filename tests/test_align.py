from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from beta.align import (
    STILL_TOL_PX,
    CameraTrack,
    FrameTransform,
    apply_h,
    camera_id,
    check_still,
    corner_shift,
    estimate_camera,
    load_camera,
    save_camera,
)
from beta.synthetic import make_moving_climb_video
from beta.video import probe


def shift(dx: float, dy: float) -> np.ndarray:
    return np.array([[1.0, 0, dx], [0, 1.0, dy], [0, 0, 1.0]])


def moving_cam(*matrices, canvas=(700, 600)) -> CameraTrack:
    ts = tuple(
        FrameTransform(i, m, "lost" if m is None else ("anchor" if i == 0 else "track"), 0 if m is None else 300)
        for i, m in enumerate(matrices)
    )
    return CameraTrack("moving", 30.0, len(ts), 640, 480, canvas, 0, ts, {})


def test_still_camera_maps_frame_pixels_to_themselves():
    cam = CameraTrack.still(fps=30.0, frame_count=10, width=640, height=480)
    pts = np.array([[10.0, 20.0], [600.5, 470.25]])
    assert np.allclose(apply_h(cam.frame_to_wall(3), pts), pts)
    assert cam.canvas == (640, 480) and cam.transforms == ()
    assert cam.frame_to_wall(10) is None


def test_moving_camera_round_trips_points():
    w = np.array([[1.02, 0.01, 5.0], [-0.01, 1.02, 40.0], [0, 0, 1.0]])
    cam = moving_cam(np.eye(3), w)
    pts = np.array([[100.0, 100.0], [320.0, 240.0]])
    assert np.allclose(apply_h(cam.wall_to_frame(1), apply_h(cam.frame_to_wall(1), pts)), pts, atol=1e-6)


def test_lost_frame_has_no_transform():
    cam = moving_cam(None)
    assert cam.frame_to_wall(0) is None and cam.wall_to_frame(0) is None
    assert cam.warp_to_wall(np.zeros((480, 640, 3), np.uint8), 0) is None


def test_apply_h_keeps_nan_rows():
    out = apply_h(np.eye(3), np.array([[1.0, 2.0], [np.nan, np.nan]]))
    assert np.allclose(out[0], [1, 2]) and np.isnan(out[1]).all()


def test_corner_shift():
    assert corner_shift(shift(3, -4), 640, 480) == pytest.approx(5.0)


def test_warp_to_wall_covers_only_the_frame_footprint():
    cam = moving_cam(shift(50, 0), canvas=(690, 480))
    warped, mask = cam.warp_to_wall(np.full((480, 640, 3), 200, np.uint8), 0)
    assert warped.shape == (480, 690, 3) and mask.shape == (480, 690)
    assert mask[:, :49].max() == 0 and mask[:, 60:680].min() == 255


def test_camera_json_round_trip_keeps_the_id(tmp_path):
    cam = moving_cam(np.eye(3), shift(1.5, 30.25), None)
    path = save_camera(tmp_path / "camera.json", cam, {"path": "clip.mp4"})
    loaded, doc = load_camera(path)
    assert doc["id"] == cam.id == loaded.id == camera_id(cam) and len(cam.id) == 12
    assert np.array_equal(loaded.frame_to_wall(1), cam.frame_to_wall(1)) and loaded.frame_to_wall(2) is None
    assert camera_id(replace(cam, reference_frame=1)) != cam.id
    still, _ = load_camera(save_camera(tmp_path / "still.json", CameraTrack.still(30.0, 5, 640, 480), {"path": "c.mp4"}))
    assert still.mode == "still" and np.array_equal(still.frame_to_wall(4), np.eye(3))


# --- estimating the camera movement ------------------------------------------


def frame_errors(cam: CameraTrack, truth, frames=None) -> list[float]:
    """Median px error per frame: big-wall points seen in every frame should land on the same wall
    coordinates whichever frame they are mapped from."""
    h_big, w_big = truth.wall.height, truth.wall.width
    ys, xs = np.mgrid[0.45 * h_big : 0.55 * h_big : 8j, 0.2 * w_big : 0.8 * w_big : 8j]
    grid = np.column_stack([xs.ravel(), ys.ravel()])
    ref = cam.reference_frame
    want = apply_h(cam.frame_to_wall(ref), apply_h(truth.wall_to_frame[ref], grid))
    out = []
    for t in range(cam.frame_count) if frames is None else frames:
        w = cam.frame_to_wall(t)
        if w is not None:
            got = apply_h(w, apply_h(truth.wall_to_frame[t], grid))
            out.append(float(np.median(np.linalg.norm(got - want, axis=1))))
    return out


def test_still_clip_is_detected_as_still(climb_video):
    cam = estimate_camera(climb_video[0])
    assert cam.mode == "still" and cam.transforms == () and cam.canvas == (640, 480)
    assert cam.summary["max_shift_px"] < STILL_TOL_PX


def test_check_still_reports_motion(moving_video):
    path, _ = moving_video
    check = check_still(path, probe(path))
    assert not check.still and check.max_shift_px > 20 and check.matched >= 6  # of the 8 samples besides the hub


def test_moving_clip_lines_up_with_the_wall(moving_video):
    path, truth = moving_video
    cam = estimate_camera(path)
    assert cam.mode == "moving" and cam.frame_count == len(cam.transforms) == 45
    assert all(t.matrix is not None for t in cam.transforms)
    assert max(frame_errors(cam, truth)) <= 1.0


def test_anchors_about_once_a_second(moving_video):
    cam = estimate_camera(moving_video[0])
    anchors = [t.frame for t in cam.transforms if t.method == "anchor"]
    assert anchors[0] == 0 and all(b - a <= 15 for a, b in zip(anchors, anchors[1:]))  # 15 fps
    assert sum(t.method == "track" for t in cam.transforms) > len(anchors)


def test_drift_stays_bounded_over_a_long_clip(tmp_path):
    truth = make_moving_climb_video(tmp_path / "long.avi", 320, 240, n_frames=600, fps=30.0)
    cam = estimate_camera(tmp_path / "long.avi")
    assert max(frame_errors(cam, truth)) <= 1.0
    assert cam.summary["anchor_correction_px"]["max"] <= 2.0


def test_blank_frames_are_lost_and_alignment_recovers(tmp_path):
    truth = make_moving_climb_video(tmp_path / "gap.avi", blank_frames=range(20, 24))
    cam = estimate_camera(tmp_path / "gap.avi")
    lost = [t for t in cam.transforms if t.matrix is None]
    assert [t.frame for t in lost] == [20, 21, 22, 23] and {t.method for t in lost} == {"lost"}
    assert cam.summary["lost_frames"] == 4 and cam.summary["longest_lost_run"] == [4, 20]
    assert max(frame_errors(cam, truth, range(24, 45))) <= 1.0


def test_a_big_climber_does_not_pull_the_alignment(tmp_path):
    truth = make_moving_climb_video(tmp_path / "big.avi", climber_frac=2.5)
    assert max(frame_errors(estimate_camera(tmp_path / "big.avi"), truth)) <= 1.0


def test_alignment_scales_back_from_working_size(tmp_path):  # Review Focus 3
    truth = make_moving_climb_video(tmp_path / "hd.avi", 1600, 1200, n_frames=20)
    cam = estimate_camera(tmp_path / "hd.avi")
    assert cam.canvas[0] >= 1600 and max(frame_errors(cam, truth)) <= 1.5  # 1 working px is ~1.6 full-res px
