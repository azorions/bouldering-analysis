from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from beta.align import CameraTrack, FrameTransform, apply_h, camera_id, corner_shift, load_camera, save_camera


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
