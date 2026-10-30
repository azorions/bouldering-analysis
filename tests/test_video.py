from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np
import pytest
from conftest import inside

from beta import video
from beta.holds import HoldSpec, detect_holds
from beta.synthetic import climber_center
from beta.video import (
    _bit_depth,
    imread,
    imwrite,
    iter_frames,
    load_input,
    masked_median,
    median_background,
    probe,
    read_frame,
    read_frames,
    sample_indices,
)


def frame_marker(frame: np.ndarray) -> float:
    """The synthetic clip encodes the frame index as brightness 8*i in the top-left corner."""
    return float(frame[2:10, 2:10].mean()) / 8


def test_probe(climb_video):
    path, _ = climb_video
    info = probe(path)
    assert (info.width, info.height, info.frame_count) == (640, 480, 31)
    assert info.fps == pytest.approx(15.0)


@pytest.mark.parametrize("index", [0, 17, 30])
def test_read_frame_returns_requested_frame(climb_video, index):
    path, _ = climb_video
    assert frame_marker(read_frame(path, index)) == pytest.approx(index, abs=0.5)


def test_read_frame_out_of_range(climb_video):
    path, _ = climb_video
    with pytest.raises(IndexError):
        read_frame(path, 500)


def test_read_frames_returns_exact_indices(climb_video):  # Review Focus 1
    frames = read_frames(climb_video[0], [30, 0, 7, 99])
    assert [i for i, _ in frames] == [0, 7, 30]
    assert [round(float(f[:12, :12].mean())) for _, f in frames] == pytest.approx([0, 56, 240], abs=4)


def test_masked_median_ignores_uncovered_pixels():
    full, half, none = (np.full((2, 4), 255, np.uint8), np.zeros((2, 4), np.uint8), np.zeros((2, 4), np.uint8))
    half[:, :2] = 255
    imgs = [np.full((2, 4, 3), v, np.uint8) for v in (10, 20, 200, 99)]
    out = masked_median(imgs, [full, full, half, none])
    assert out[0, 0, 0] == 20 and out[0, 3, 0] == 15
    assert masked_median(imgs[:1], [none])[0, 0, 0] == 0


def test_iter_frames_is_sequential(climb_video):
    path, _ = climb_video
    indices = [i for i, _ in iter_frames(path)]
    assert indices == list(range(31))
    partial = list(iter_frames(path, start=10, stop=13))
    assert [i for i, _ in partial] == [10, 11, 12]
    assert frame_marker(partial[0][1]) == pytest.approx(10, abs=0.5)


def test_sample_indices():
    idx = sample_indices(31, 25)
    assert len(idx) == 25 and idx[0] == 0 and idx[-1] == 30 and idx == sorted(set(idx))
    assert sample_indices(10, 25) == list(range(10))
    assert sample_indices(1, 25) == [0]


def test_median_background_removes_the_climber(climb_video):
    path, truth = climb_video
    green = truth.of_color("green")
    middle = 15
    climber = climber_center(middle, 31, 640, 480)

    # On a raw frame, the climber's green shirt shows up as a "hold" ...
    frame_holds = detect_holds(read_frame(path, middle), HoldSpec.from_color("green")).holds
    assert any(inside(h, climber) for h in frame_holds)

    # ... but the median background has the bare wall, with exactly the route's holds.
    background, used = median_background(path, n=25)
    assert used == 25
    holds = detect_holds(background, HoldSpec.from_color("green")).holds
    assert not any(inside(h, climber) for h in holds)
    assert len(holds) == len(green)


def test_load_input_image(wall, tmp_path):
    image, _ = wall
    path = imwrite(tmp_path / "wall.png", image)
    loaded, source = load_input(path)
    assert np.array_equal(loaded, image)
    assert source["kind"] == "image" and (source["width"], source["height"]) == (1280, 960)


def test_load_input_video_frame_and_background(climb_video):
    path, _ = climb_video
    frame, source = load_input(path, frame=5)
    assert source["kind"] == "video" and source["frame"] == 5 and source["background_frames"] is None
    assert frame_marker(frame) == pytest.approx(5, abs=0.5)
    _, source = load_input(path, bg_frames=11)
    assert source["frame"] is None and source["background_frames"] == 11


def test_missing_inputs_raise(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_input(tmp_path / "nope.mp4")
    with pytest.raises(FileNotFoundError):
        imread(tmp_path / "nope.png")


def test_heic_photos_get_a_clear_error(tmp_path):
    # OpenCV can't decode HEIC; without the check it opened as a 1-frame "video" of one tile.
    path = tmp_path / "IMG_0001.HEIC"
    path.write_bytes(b"\x00" * 64)
    with pytest.raises(ValueError, match="Export it as JPEG"):
        load_input(path)


@pytest.mark.parametrize(
    "tag, depth",
    [
        (b"I420", 8),
        (b"NV12", 8),
        (b"Y800", 8),
        (b"Y3\x0b\x0a", 10),  # FFmpeg yuv420p10le: what iPhone HDR (HEVC Main10) decodes to
        (b"Y3\x0b\x0c", 12),
        (b"P010", 10),
        (b"Y210", 10),
    ],
)
def test_bit_depth_from_pixel_format_tag(tag, depth):
    assert _bit_depth(int.from_bytes(tag, "little")) == depth


def test_bit_depth_unknown_tag_defaults_to_8():
    assert _bit_depth(0) == 8 and _bit_depth(-1) == 8


def test_hdr_video_warns(climb_video, monkeypatch):
    path, _ = climb_video
    info = probe(path)
    assert info.bit_depth == 8
    monkeypatch.setattr(video, "probe", lambda p: replace(info, bit_depth=10))
    with pytest.warns(UserWarning, match="10-bit video"):
        _, source = load_input(path, frame=0)
    assert source["bit_depth"] == 10


def test_single_frame_video_warns(tmp_path):
    path = tmp_path / "still.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 15, (64, 48))
    writer.write(np.full((48, 64, 3), 120, np.uint8))
    writer.release()
    with pytest.warns(UserWarning, match="still image"):
        load_input(path)


def test_image_io_handles_non_ascii_paths(tmp_path):
    image = np.full((20, 30, 3), (10, 200, 30), np.uint8)
    path = imwrite(tmp_path / "salle d'escalade é" / "mur.png", image)
    assert np.array_equal(imread(path), image)
