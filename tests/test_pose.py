from __future__ import annotations

import http.client
import os
import zipfile
from pathlib import Path

import numpy as np
import pytest
from conftest import HIDDEN, STICK, FakeLandmarker, FakeResponse

from beta import pose
from beta.holds import HoldSpec, detect_holds
from beta.pose import (
    CONTACT_POINTS,
    LANDMARK_NAMES,
    LEFT_COLOR,
    RIGHT_COLOR,
    PoseFrame,
    PoseTrack,
    _faint,
    draw_pose,
    estimate_poses,
    load_pose,
    save_pose,
    write_pose_video,
)
from beta.synthetic import climber_center, make_climb_video, make_wall
from beta.video import iter_frames, probe, read_frame

MODELS = Path(__file__).resolve().parents[1] / "models"


def stick_frame(i: int, n: int, w: int = 640, h: int = 480) -> np.ndarray:
    """The (33, 5) landmarks FakeLandmarker reports for frame ``i``, in pixels."""
    fake, blank = FakeLandmarker(n), np.zeros((h, w, 3), np.uint8)
    for _ in range(i):
        fake.detect(blank, 0)
    return pose.landmarks_to_array(fake.detect(blank, 0), w, h)


def test_landmark_names_match_mediapipe():
    from mediapipe.tasks.python.vision import PoseLandmark

    assert LANDMARK_NAMES == tuple(m.name.lower() for m in PoseLandmark)
    assert {LANDMARK_NAMES[i] for i in CONTACT_POINTS.values()} == {
        "left_wrist", "right_wrist", "left_foot_index", "right_foot_index"
    }  # fmt: skip


def test_estimate_poses_in_pixels_with_gaps(climb_video):
    path, _ = climb_video
    fake = FakeLandmarker(31, missing={3, 4})
    frames = estimate_poses(iter_frames(path), 15.0, fake)

    assert [p.frame for p in frames] == list(range(31))
    assert frames[3].landmarks is None and frames[4].landmarks is None
    lm = frames[10].landmarks
    assert lm.shape == (33, 5)
    cx, cy = climber_center(10, 31, 640, 480)
    dx, dy = STICK[15]
    assert lm[15, :2] == pytest.approx((cx + dx * 480, cy + dy * 480))
    assert lm[15, 3] == pytest.approx(0.9) and lm[27, 3] == pytest.approx(0.2)
    assert np.isnan(lm[1, 3])  # a missing score becomes NaN


@pytest.mark.parametrize("fps", [15.0, 29.97, 59.94, 2500.0])
def test_timestamps_strictly_increase(fps):
    fake = FakeLandmarker(100)
    frames = estimate_poses(((i, np.zeros((8, 8, 3), np.uint8)) for i in range(100)), fps, fake)
    t = [p.t_ms for p in frames]
    assert fake.calls == t
    assert all(b > a for a, b in zip(t, t[1:]))
    if fps < 1000:
        assert t == [round(i * 1000 / fps) for i in range(100)]


def test_track_summary():
    lm = stick_frame(0, 10)
    frames = [PoseFrame(i, i * 100, None if i in (2, 3, 4, 7) else lm) for i in range(10)]
    track = PoseTrack(frames, 10.0, 640, 480)
    assert track.detected() == 6
    assert track.longest_gap() == (3, 2)
    vis = track.mean_visibility()
    assert vis["LH"] == pytest.approx(0.9) and vis["LF"] == pytest.approx(0.2)
    assert PoseTrack([PoseFrame(0, 0, None)], 10.0, 640, 480).mean_visibility()["RH"] is None


def test_pose_json_round_trip(tmp_path):
    lm = stick_frame(5, 10)
    track = PoseTrack([PoseFrame(0, 0, lm), PoseFrame(1, 33, None)], 29.97, 640, 480)
    path = save_pose(tmp_path / "pose.json", track, {"path": Path("a/b.mp4"), "fps": 29.97, "width": 640, "height": 480}, {"model": "lite"})

    text = path.read_text(encoding="utf-8")
    assert "NaN" not in text
    loaded, doc = load_pose(path)
    assert doc["schema_version"] == 1 and doc["source"]["path"] == "a/b.mp4"
    assert len(doc["landmark_names"]) == 33 and doc["landmark_fields"][3] == "visibility"
    assert (loaded.fps, loaded.width, loaded.height) == (29.97, 640, 480)
    assert [p.t_ms for p in loaded.frames] == [0, 33]
    assert loaded.frames[1].landmarks is None
    np.testing.assert_allclose(loaded.frames[0].landmarks, lm, atol=0.05)  # NaN == NaN here


def test_draw_pose_colors_each_side():
    frame = np.full((480, 640, 3), 120, np.uint8)
    lm = stick_frame(15, 31)
    img = draw_pose(frame, PoseFrame(15, 1000, lm))

    def at(idx: int) -> tuple[int, ...]:
        return tuple(int(v) for v in img[round(lm[idx, 1]), round(lm[idx, 0])])

    assert at(CONTACT_POINTS["LH"]) == LEFT_COLOR
    assert at(CONTACT_POINTS["RH"]) == RIGHT_COLOR
    assert CONTACT_POINTS["LF"] in HIDDEN
    assert at(CONTACT_POINTS["LF"]) == _faint(LEFT_COLOR)  # low visibility: drawn faint
    assert at(CONTACT_POINTS["RF"]) == RIGHT_COLOR


def test_hand_in_the_header_band_keeps_its_color():
    frame = np.full((480, 640, 3), 120, np.uint8)
    lm = stick_frame(15, 31)
    lm[CONTACT_POINTS["RH"], :2] = (500, 20)  # reaching up into the darkened header
    img = draw_pose(frame, PoseFrame(15, 1000, lm))
    assert tuple(int(v) for v in img[20, 500]) == RIGHT_COLOR


def test_draw_pose_without_a_pose_still_labels_the_frame():
    frame = np.full((480, 640, 3), 120, np.uint8)
    for p in (None, PoseFrame(3, 200, None)):
        img = draw_pose(frame, p)
        assert img.shape == frame.shape
        # The header band darkens the grey frame, so only the text can be this bright.
        assert (img[:60] > 200).any()
    assert (frame == 120).all()  # the input is not drawn on


def test_draw_pose_scales_to_the_output_size():
    frame = np.full((480, 640, 3), 120, np.uint8)
    lm = stick_frame(15, 31)
    img = draw_pose(frame, PoseFrame(15, 1000, lm), size=(320, 240))
    assert img.shape == (240, 320, 3)
    x, y = lm[CONTACT_POINTS["RH"], :2] / 2
    assert tuple(int(v) for v in img[round(y), round(x)]) == RIGHT_COLOR


def test_draw_pose_draws_holds_underneath():
    image, _ = make_wall(640, 480, seed=0)
    holds = detect_holds(image, HoldSpec.from_color("green")).holds
    without = draw_pose(image, None)
    with_holds = draw_pose(image, None, holds)
    x, y, w, h = holds[0].bbox
    assert (with_holds[y : y + h, x : x + w] != without[y : y + h, x : x + w]).any()


def test_write_pose_video(climb_video, tmp_path):
    path, _ = climb_video
    frames = estimate_poses(iter_frames(path), 15.0, FakeLandmarker(31, missing={0}))
    out = write_pose_video(path, PoseTrack(frames, 15.0, 640, 480), tmp_path / "pose_debug.mp4")
    info = probe(out)
    assert (info.frame_count, info.width, info.height) == (31, 640, 480)
    assert info.fps == pytest.approx(15.0)


def test_write_pose_video_draws_holds(climb_video, tmp_path):
    path, _ = climb_video
    holds = detect_holds(make_wall(640, 480, seed=0)[0], HoldSpec.from_color("green")).holds  # the clip's wall
    track = PoseTrack(estimate_poses(iter_frames(path), 15.0, FakeLandmarker(31, missing=set(range(31)))), 15.0, 640, 480)
    plain = read_frame(write_pose_video(path, track, tmp_path / "plain.mp4"), 0)
    drawn = read_frame(write_pose_video(path, track, tmp_path / "holds.mp4", holds), 0)
    # The outline (dark, with a thin white line on top) and the number change the
    # pixels on and around each hold well beyond mp4 compression noise; away from the
    # holds the two videos match. Holds under the darkened header band change least.
    changed = np.abs(drawn.astype(int) - plain.astype(int)).max(axis=2) > 40
    near = np.zeros(changed.shape, bool)
    for hold in holds:
        x, y, w, h = hold.bbox
        box = (slice(max(0, y - 8), y + h + 8), slice(max(0, x - 8), x + w + 8))
        assert changed[box].any(), f"hold {hold.id} not drawn"
        near[box] = True
    assert changed[near].sum() > 1000
    assert changed[~near].sum() == 0


def test_write_pose_video_downscales_to_even_size(tmp_path):
    src = tmp_path / "wide.avi"
    make_climb_video(src, width=1501, height=845, n_frames=3)
    frames = estimate_poses(iter_frames(src), 15.0, FakeLandmarker(3))
    info = probe(write_pose_video(src, PoseTrack(frames, 15.0, 1501, 845), tmp_path / "out.mp4"))
    assert info.width <= 1280 and info.width % 2 == 0 and info.height % 2 == 0
    assert info.frame_count == 3


def test_download_model(tmp_path, monkeypatch):
    target = tmp_path / "models" / "pose_landmarker_lite.task"
    monkeypatch.setattr(pose.urllib.request, "urlopen", lambda url, timeout: FakeResponse(b"model", 5))
    assert pose.download_model("lite", target).read_bytes() == b"model"


def test_truncated_download_leaves_no_file(tmp_path, monkeypatch):
    target = tmp_path / "pose_landmarker_lite.task"
    monkeypatch.setattr(pose.urllib.request, "urlopen", lambda url, timeout: FakeResponse(b"mod", 5))
    with pytest.raises(ValueError, match="--model-path"):
        pose.download_model("lite", target)
    assert list(tmp_path.iterdir()) == []


def test_dropped_connection_leaves_no_file(tmp_path, monkeypatch):
    # A chunked download cut off partway raises IncompleteRead, which is not an OSError.
    dropped = FakeResponse(b"mod", fail_after=http.client.IncompleteRead(b"mod"))
    monkeypatch.setattr(pose.urllib.request, "urlopen", lambda url, timeout: dropped)
    with pytest.raises(ValueError, match="--model-path"):
        pose.download_model("lite", tmp_path / "pose_landmarker_lite.task")
    assert list(tmp_path.iterdir()) == []


def test_empty_download_leaves_no_file(tmp_path, monkeypatch):
    monkeypatch.setattr(pose.urllib.request, "urlopen", lambda url, timeout: FakeResponse(b""))
    with pytest.raises(ValueError, match="empty file"):
        pose.download_model("lite", tmp_path / "pose_landmarker_lite.task")
    assert list(tmp_path.iterdir()) == []


def test_malformed_content_length_is_ignored(tmp_path, monkeypatch):
    monkeypatch.setattr(pose.urllib.request, "urlopen", lambda url, timeout: FakeResponse(b"model", "5, 5"))
    assert pose.download_model("lite", tmp_path / "m.task").read_bytes() == b"model"


def test_broken_model_file_is_a_clear_error(tmp_path):
    bad = tmp_path / "bad.task"
    bad.write_bytes(b"not a model")
    with pytest.raises(ValueError, match="could not load pose model"):
        pose.open_landmarker(bad)


def test_empty_model_file_is_a_clear_error(tmp_path):
    empty = tmp_path / "empty.task"
    empty.write_bytes(b"")
    with pytest.raises(ValueError, match=r"empty\.task: the file is empty"):
        pose.open_landmarker(empty)


def test_model_bundle_without_pose_models_is_a_clear_error(tmp_path):
    # e.g. MediaPipe's hand landmarker passed with --model-path: a valid .task bundle, wrong models inside
    bundle = tmp_path / "hand_landmarker.task"
    with zipfile.ZipFile(bundle, "w") as z:
        z.writestr("hand_detector.tflite", b"\0" * 64)
    with pytest.raises(ValueError, match=r"could not load pose model .*hand_landmarker\.task"):
        pose.open_landmarker(bundle)


def test_converts_real_mediapipe_landmarks():
    # FakeLandmarker returns SimpleNamespace objects; check the real MediaPipe type converts the same way.
    from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark

    lms = [NormalizedLandmark(x=0.5, y=0.25, z=-0.1, visibility=0.9, presence=0.8) for _ in range(33)]
    lms[1] = NormalizedLandmark(x=0.1, y=0.2, z=0.0, visibility=None, presence=None)
    arr = pose.landmarks_to_array(lms, 640, 480)
    assert arr.shape == (33, 5)
    assert tuple(arr[0]) == pytest.approx((320.0, 120.0, -0.1, 0.9, 0.8))
    assert np.isnan(arr[1, 3]) and np.isnan(arr[1, 4])


# --- the real model (uv run pytest -m slow) ---------------------------------


def real_model(variant: str = "lite") -> Path:
    path = pose.model_file(variant, MODELS)
    return path if path.is_file() else pose.download_model(variant, path)


@pytest.mark.slow
def test_real_model_runs_on_the_synthetic_clip(climb_video):
    path, _ = climb_video
    with pose.open_landmarker(real_model()) as landmarker:
        frames = estimate_poses(iter_frames(path), 15.0, landmarker)
    assert len(frames) == 31
    assert all(p.landmarks is None for p in frames)  # an ellipse with a circle on top is not a person


@pytest.mark.slow
@pytest.mark.skipif(not os.environ.get("BETA_POSE_CLIP"), reason="set BETA_POSE_CLIP to a real climbing video")
def test_real_model_finds_the_climber():
    path = Path(os.environ["BETA_POSE_CLIP"])
    info = probe(path)
    with pose.open_landmarker(real_model()) as landmarker:
        frames = estimate_poses(iter_frames(path), info.fps, landmarker)
    found = sum(p.landmarks is not None for p in frames)
    assert found > len(frames) / 2, f"pose found in only {found} of {len(frames)} frames"
