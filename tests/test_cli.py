from __future__ import annotations

import subprocess
import sys
from dataclasses import replace
from urllib.error import URLError

import pytest
from conftest import FOOT_CHIPS, FakeLandmarker, foot_chip_wall

from beta import cli, pose, video
from beta.cli import main
from beta.jsonio import load_json
from beta.pose import load_pose
from beta.video import imread, imwrite, probe


def test_holds_on_image_writes_outputs(wall, tmp_path, capsys):
    image, truth = wall
    src = imwrite(tmp_path / "wall.png", image)
    out = tmp_path / "out"
    assert main(["holds", str(src), "--color", "green", "--out", str(out)]) == 0

    data = load_json(out / "holds.json")
    assert len(data["holds"]) == len(truth.of_color("green"))
    assert data["source"]["kind"] == "image"
    assert imread(out / "holds_debug.png").shape[1] > image.shape[1]
    assert imread(out / "background.png").shape == image.shape

    printed = capsys.readouterr().out
    assert f"Detected {len(data['holds'])} green holds" in printed
    assert "cut off by frame edge" in printed


def test_holds_on_video_uses_median_background(climb_video, tmp_path):
    path, truth = climb_video
    out = tmp_path / "out"
    assert main(["holds", str(path), "--color", "green", "--out", str(out)]) == 0
    data = load_json(out / "holds.json")
    assert data["source"]["kind"] == "video"
    assert data["source"]["background_frames"] == 25
    assert len(data["holds"]) == len(truth.of_color("green"))


def test_sample_prints_color_and_suggestion(wall, tmp_path, capsys):
    image, truth = wall
    src = imwrite(tmp_path / "wall.png", image)
    x, y = truth.of_color("green")[0].probes[0]
    assert main(["holds", str(src), "--color", "green", "--sample", f"{x},{y}", "--out", str(tmp_path / "o")]) == 0
    printed = capsys.readouterr().out
    assert "inside the current green range" in printed
    assert "--hsv-lower" in printed


def test_sample_on_bare_wall_says_grey_instead_of_suggesting(wall, tmp_path, capsys):
    # A brightness-only suggestion for a grey spot would match the whole wall.
    image, _ = wall
    src = imwrite(tmp_path / "wall.png", image)
    assert main(["holds", str(src), "--color", "green", "--sample", "300,300", "--out", str(tmp_path / "o")]) == 0
    printed = capsys.readouterr().out
    assert "looks grey" in printed
    assert "--hsv-lower" not in printed


def test_near_limit_hint_names_the_option(tmp_path, capsys):
    src = imwrite(tmp_path / "chips.png", foot_chip_wall())
    assert main(["holds", str(src), "--color", "green", "--min-area-frac", "5e-5", "--out", str(tmp_path / "o")]) == 0
    printed = capsys.readouterr().out
    assert f"{len(FOOT_CHIPS)} blobs were just under the size limit" in printed
    assert "--min-area-frac" in printed


def test_library_warnings_print_as_plain_messages(climb_video, tmp_path, monkeypatch, capsys):
    path, _ = climb_video
    monkeypatch.setattr(video, "probe", lambda p: replace(probe(p), bit_depth=10))
    assert main(["holds", str(path), "--color", "green", "--frame", "0", "--out", str(tmp_path / "o")]) == 0
    err = capsys.readouterr().err
    assert err.startswith("warning: climb.avi is 10-bit video")
    assert "UserWarning" not in err and ".py:" not in err


def test_custom_hsv_without_color(wall, tmp_path):
    image, _ = wall
    src = imwrite(tmp_path / "wall.png", image)
    out = tmp_path / "out"
    assert main(["holds", str(src), "--hsv-lower", "100,100,50", "--hsv-upper", "120,255,255", "--out", str(out)]) == 0
    assert load_json(out / "holds.json")["params"]["color"] == "custom"


def test_errors_are_reported_not_raised(tmp_path, capsys):
    assert main(["holds", str(tmp_path / "missing.mp4"), "--color", "green"]) == 1
    assert "error:" in capsys.readouterr().err
    assert main(["holds", str(tmp_path / "missing.png")]) == 1  # no --color and no HSV range
    assert "error:" in capsys.readouterr().err


def test_python_dash_m_entry_point():
    result = subprocess.run([sys.executable, "-m", "beta", "holds", "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "--color" in result.stdout


def test_importing_the_cli_does_not_load_mediapipe():
    # MediaPipe takes a couple of seconds to import; `beta holds` shouldn't pay for it.
    code = "import sys, beta.cli; print('mediapipe' in sys.modules)"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.stdout.strip() == "False"


@pytest.fixture
def fake_model(tmp_path, monkeypatch):
    """A dummy model file, with FakeLandmarker standing in for MediaPipe."""
    model = tmp_path / "fake.task"
    model.write_bytes(b"fake")
    opened: list[FakeLandmarker] = []

    def open_fake(path, min_confidence):
        opened.append(FakeLandmarker(31, missing={5}))
        return opened[-1]

    monkeypatch.setattr(cli, "open_landmarker", open_fake)
    return model, opened


def test_pose_writes_json_and_debug_video(climb_video, tmp_path, fake_model, capsys):
    path, _ = climb_video
    model, opened = fake_model
    out = tmp_path / "out"
    assert main(["pose", str(path), "--model-path", str(model), "--out", str(out)]) == 0

    track, doc = load_pose(out / "pose.json")
    assert len(track.frames) == 31 and track.frames[5].landmarks is None
    assert doc["source"]["fps"] == pytest.approx(15.0) and doc["params"]["model"] is None
    assert probe(out / "pose_debug.mp4").frame_count == 31
    assert opened[0].closed

    printed = capsys.readouterr().out
    assert "Pose found in 30 of 31 frames (97%)" in printed
    assert "longest run without a pose: 1 frame (0.1 s) from frame 5" in printed
    assert "LF 0.20" in printed  # the fake's hidden foot
    assert "drawn on the video" not in printed  # no holds.json in out/


def test_pose_no_video(climb_video, tmp_path, fake_model):
    path, _ = climb_video
    model, _ = fake_model
    out = tmp_path / "out"
    assert main(["pose", str(path), "--model-path", str(model), "--out", str(out), "--no-video"]) == 0
    assert (out / "pose.json").is_file() and not (out / "pose_debug.mp4").exists()


def test_pose_draws_holds_from_the_output_dir(climb_video, tmp_path, fake_model, capsys):
    path, truth = climb_video
    model, _ = fake_model
    out = tmp_path / "out"
    assert main(["holds", str(path), "--color", "green", "--out", str(out)]) == 0
    assert main(["pose", str(path), "--model-path", str(model), "--out", str(out)]) == 0
    n = len(truth.of_color("green"))
    assert f"{n} holds from {out / 'holds.json'} drawn on the video" in capsys.readouterr().out


def test_pose_skips_holds_from_an_image_of_another_size(climb_video, wall, tmp_path, fake_model, capsys):
    path, _ = climb_video
    model, _ = fake_model
    out = tmp_path / "out"
    src = imwrite(tmp_path / "wall.png", wall[0])  # 1280x960, the clip is 640x480
    assert main(["holds", str(src), "--color", "green", "--out", str(out)]) == 0
    capsys.readouterr()
    assert main(["pose", str(path), "--model-path", str(model), "--out", str(out)]) == 0
    captured = capsys.readouterr()
    assert "made from a 1280x960 image but the video is 640x480" in captured.err
    assert "drawn on the video" not in captured.out


def test_pose_rejects_photos_and_bad_options(tmp_path, capsys):
    assert main(["pose", str(tmp_path / "wall.png")]) == 1
    assert "pose needs a video" in capsys.readouterr().err
    assert main(["pose", str(tmp_path / "IMG_0001.HEIC")]) == 1
    with pytest.raises(SystemExit):
        main(["pose", "clip.mp4", "--min-confidence", "1.5"])
    with pytest.raises(SystemExit):
        main(["pose", "clip.mp4", "--model", "huge"])


def test_pose_missing_model_path(climb_video, tmp_path, capsys):
    path, _ = climb_video
    assert main(["pose", str(path), "--model-path", str(tmp_path / "nope.task")]) == 1
    assert "no such model file" in capsys.readouterr().err


def test_pose_download_failure_is_a_clear_error(climb_video, tmp_path, monkeypatch, capsys):
    path, _ = climb_video
    monkeypatch.chdir(tmp_path)  # models/ is relative to the working directory

    def offline(url, timeout):
        raise URLError("no network")

    monkeypatch.setattr(pose.urllib.request, "urlopen", offline)
    assert main(["pose", str(path), "--model", "lite"]) == 1
    captured = capsys.readouterr()
    assert "Downloading the lite pose model" in captured.out
    assert "could not download the pose model" in captured.err and "--model-path" in captured.err
    assert list((tmp_path / "models").iterdir()) == []
