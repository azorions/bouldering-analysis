from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import replace
from urllib.error import URLError

import numpy as np
import pytest
from conftest import FOOT_CHIPS, FakeLandmarker, FakeResponse, body, foot_chip_wall, inside, track_of

from beta import cli, pose, video
from beta.align import apply_h, load_camera
from beta.cli import main
from beta.holds import HoldSpec, detect_holds, load_holds
from beta.jsonio import load_json, save_json
from beta.pose import load_pose, save_pose
from beta.video import imread, imwrite, median_background, probe


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


def test_non_ascii_names_print_when_output_is_piped(wall, tmp_path):
    # Windows encodes piped or redirected output as cp1252, which has no Japanese characters.
    src = imwrite(tmp_path / "クライム.png", wall[0])
    result = subprocess.run(
        [sys.executable, "-m", "beta", "holds", str(src), "--color", "green"],
        cwd=tmp_path,  # output goes to outputs/クライム/, which is printed
        env={**os.environ, "PYTHONIOENCODING": "cp1252"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "クライム".encode("ascii", "backslashreplace").decode() in result.stdout  # printed as \u escapes


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
    assert "Pose found in 30 of 31 frames (96%)" in printed
    assert "longest run without a pose: 1 frame (0.07 s) from frame 5" in printed
    assert "LF 0.20" in printed  # the fake's hidden foot
    assert "drawn on the video" not in printed  # no holds.json in out/


def test_pose_no_video(climb_video, tmp_path, fake_model):
    path, _ = climb_video
    model, _ = fake_model
    out = tmp_path / "out"
    assert main(["pose", str(path), "--model-path", str(model), "--out", str(out), "--no-video"]) == 0
    assert (out / "pose.json").is_file() and not (out / "pose_debug.mp4").exists()


def test_pose_no_video_ignores_holds(climb_video, tmp_path, fake_model, capsys):
    path, _ = climb_video
    model, _ = fake_model
    args = ["pose", str(path), "--model-path", str(model), "--out", str(tmp_path / "out"), "--no-video"]
    assert main([*args, "--holds", str(tmp_path / "missing.json")]) == 0  # not even checked
    assert "--holds is ignored with --no-video" in capsys.readouterr().err


def test_pose_draws_holds_from_the_output_dir(climb_video, tmp_path, fake_model, monkeypatch, capsys):
    path, truth = climb_video
    model, _ = fake_model
    out = tmp_path / "out"
    assert main(["holds", str(path), "--color", "green", "--out", str(out)]) == 0
    drawn = []

    def write_video(src, track, out_path, holds=(), camera=None):
        drawn.extend(holds)
        return out_path

    monkeypatch.setattr(cli, "write_pose_video", write_video)
    assert main(["pose", str(path), "--model-path", str(model), "--out", str(out)]) == 0
    assert drawn == load_holds(out / "holds.json")[0]  # the holds themselves reach the video, not just the message
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


def _pose_with_fake_video(path, out, tmp_path, monkeypatch) -> dict:
    """Run `pose` with a fake model; returns what reached write_pose_video."""
    seen = {}
    monkeypatch.setattr(cli, "open_landmarker", lambda p, c: FakeLandmarker(45))
    monkeypatch.setattr(cli, "write_pose_video", lambda s, t, o, holds=(), camera=None: seen.update(holds=holds, camera=camera) or o)
    model = tmp_path / "fake.task"
    model.write_bytes(b"x")
    assert main(["pose", str(path), "--model-path", str(model), "--out", str(out)]) == 0
    return seen


def test_pose_uses_the_camera_for_moving_holds(moving_video, tmp_path, monkeypatch):
    path, _ = moving_video
    out = tmp_path / "out"
    assert main(["holds", str(path), "--color", "green", "--out", str(out)]) == 0
    seen = _pose_with_fake_video(path, out, tmp_path, monkeypatch)
    assert seen["holds"] and seen["camera"].mode == "moving"


def test_pose_skips_holds_from_another_alignment(moving_video, tmp_path, monkeypatch, capsys):
    path, _ = moving_video
    out = tmp_path / "out"
    assert main(["holds", str(path), "--color", "green", "--out", str(out)]) == 0
    doc = load_json(out / "holds.json")
    doc["camera"]["id"] = "000000000000"
    save_json(out / "holds.json", doc)
    capsys.readouterr()
    seen = _pose_with_fake_video(path, out, tmp_path, monkeypatch)
    assert "was made with a different camera.json" in capsys.readouterr().err
    assert seen["holds"] == []


def test_pose_rejects_photos_and_bad_options(tmp_path, capsys):
    assert main(["pose", str(tmp_path / "wall.png")]) == 1
    assert "pose needs a video" in capsys.readouterr().err
    heic = tmp_path / "IMG_0001.HEIC"  # a real file, so the error can't be "no such file"
    heic.write_bytes(b"\0" * 64)
    assert main(["pose", str(heic)]) == 1
    assert "IMG_0001.HEIC is a photo" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(["pose", "clip.mp4", "--min-confidence", "1.5"])
    with pytest.raises(SystemExit):
        main(["pose", "clip.mp4", "--model", "huge"])


def test_pose_missing_model_path(climb_video, tmp_path, capsys):
    path, _ = climb_video
    assert main(["pose", str(path), "--model-path", str(tmp_path / "nope.task"), "--out", str(tmp_path / "out")]) == 1
    assert "no such model file" in capsys.readouterr().err


def test_pose_checks_options_before_the_pose_pass(climb_video, tmp_path, fake_model, capsys):
    path, _ = climb_video
    model, opened = fake_model
    run = ["pose", str(path), "--model-path", str(model)]
    out_file = tmp_path / "out.txt"
    out_file.write_text("")
    assert main([*run, "--out", str(out_file)]) == 1
    assert "out.txt is a file" in capsys.readouterr().err

    out = tmp_path / "out"
    assert main([*run, "--out", str(out), "--holds", str(tmp_path)]) == 1  # a folder
    assert "no such holds file" in capsys.readouterr().err
    not_json = tmp_path / "holds.json"
    not_json.write_text("{not json")
    other_json = tmp_path / "pose.json"
    other_json.write_text('{"frames": []}')
    for bad in (not_json, other_json):
        assert main([*run, "--out", str(out), "--holds", str(bad)]) == 1
        assert "is not a holds.json written by `beta holds`" in capsys.readouterr().err
    assert not opened  # none of them got as far as the model


def test_pose_downloads_again_over_an_empty_model(climb_video, tmp_path, fake_model, monkeypatch, capsys):
    # An empty model file is what an interrupted download used to leave behind.
    path, _ = climb_video
    monkeypatch.chdir(tmp_path)  # models/ is relative to the working directory
    cached = pose.model_file("lite")
    cached.parent.mkdir()
    cached.write_bytes(b"")
    monkeypatch.setattr(pose.urllib.request, "urlopen", lambda url, timeout: FakeResponse(b"model", 5))
    assert main(["pose", str(path), "--model", "lite", "--no-video"]) == 0
    assert "Downloading the lite pose model" in capsys.readouterr().out
    assert cached.read_bytes() == b"model"


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


def test_align_on_still_clip(climb_video, tmp_path, capsys):
    out = tmp_path / "out"
    assert main(["align", str(climb_video[0]), "--out", str(out)]) == 0
    assert load_camera(out / "camera.json")[0].mode == "still"
    assert not (out / "align_debug.mp4").exists()
    assert "Camera is still" in capsys.readouterr().out


def test_align_on_moving_clip_writes_debug_video(moving_video, tmp_path, capsys):
    out = tmp_path / "out"
    assert main(["align", str(moving_video[0]), "--out", str(out)]) == 0
    assert probe(out / "align_debug.mp4").frame_count == 45
    printed = capsys.readouterr().out
    assert "Camera moves: up to" in printed and "Aligned 45 of 45 frames" in printed and "ms per frame" in printed


def test_align_no_video_and_bad_options(moving_video, tmp_path, capsys):
    out = tmp_path / "out"
    assert main(["align", str(moving_video[0]), "--out", str(out), "--no-video"]) == 0
    assert (out / "camera.json").is_file() and not (out / "align_debug.mp4").exists()
    assert main(["align", str(tmp_path / "wall.png")]) == 1
    assert "align needs a video" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(["align", "clip.mp4", "--anchor-every", "0"])


def test_holds_on_moving_clip_find_every_route_hold(moving_video, tmp_path, capsys):
    path, truth = moving_video
    out = tmp_path / "out"
    assert main(["holds", str(path), "--color", "green", "--out", str(out)]) == 0
    holds, doc = load_holds(out / "holds.json")
    cam, _ = load_camera(out / "camera.json")
    assert doc["camera"] == {"id": cam.id, "mode": "moving", "canvas": list(cam.canvas)}
    assert doc["source"]["width"] == 640 and "Checking whether the camera moves" in capsys.readouterr().out
    # Size limits come from the video frame, not the bigger wall view.
    assert doc["params"]["max_area_px"] == pytest.approx(0.03 * 640 * 480, abs=0.1)
    to_wall = cam.frame_to_wall(cam.reference_frame) @ truth.wall_to_frame[cam.reference_frame]
    route = [t for t in truth.wall.of_color("green") if t.kind != "border"]
    centers = apply_h(to_wall, np.array([t.center for t in route]))
    visible = [c for c in centers if 0 <= c[0] < cam.canvas[0] and 0 <= c[1] < cam.canvas[1]]
    assert visible and all(any(inside(h, c) for h in holds) for c in visible)


def test_holds_on_still_clip_are_unchanged(climb_video, tmp_path):
    path, _ = climb_video
    out = tmp_path / "out"
    assert main(["holds", str(path), "--color", "green", "--out", str(out)]) == 0
    holds, doc = load_holds(out / "holds.json")
    expected = detect_holds(median_background(path)[0], HoldSpec.from_color("green")).holds
    assert doc["camera"]["mode"] == "still" and holds == expected


def test_holds_redo_alignment_for_another_video(climb_video, moving_video, tmp_path):
    out = tmp_path / "out"
    assert main(["align", str(moving_video[0]), "--out", str(out), "--no-video"]) == 0
    assert main(["holds", str(climb_video[0]), "--color", "green", "--out", str(out)]) == 0
    assert load_camera(out / "camera.json")[0].mode == "still"


def test_holds_on_image_has_no_camera(wall, tmp_path):
    src = imwrite(tmp_path / "wall.png", wall[0])
    assert main(["holds", str(src), "--color", "green", "--out", str(tmp_path / "o")]) == 0
    assert "camera" not in load_json(tmp_path / "o" / "holds.json")
    assert not (tmp_path / "o" / "camera.json").exists()


def scripted_pose(out, path, holds, frames=31, hidden=()):
    """Write a pose.json for the clip: hands on the top three holds (RH moves at frame 15), feet on the bottom two.

    ``hidden`` landmarks get visibility 0.1 in every frame.
    """
    ordered = sorted(holds, key=lambda h: h.id)
    hand, foot = ordered[-3:], ordered[:2]
    lm = [body((hand[0].centroid, (hand[1] if t < 15 else hand[2]).centroid), (foot[0].centroid, foot[1].centroid)) for t in range(frames)]
    for f in lm:
        f[list(hidden), 3] = 0.1
    src = {"path": path, "kind": "video", "fps": 15.0, "frame_count": frames, "width": 640, "height": 480, "bit_depth": 8}
    save_pose(out / "pose.json", track_of(lm, fps=15.0), src, {})
    return hand


def test_contacts_end_to_end(climb_video, tmp_path, capsys):
    path, _ = climb_video
    out = tmp_path / "out"
    assert main(["holds", str(path), "--color", "green", "--out", str(out)]) == 0
    hand = scripted_pose(out, path, load_holds(out / "holds.json")[0])
    assert main(["contacts", str(path), "--out", str(out)]) == 0
    doc = load_json(out / "contacts.json")
    assert doc["attempts"][0]["sequence"]["RH"] == [hand[1].id, hand[2].id]
    assert imread(out / "contacts_debug.png").size > 0
    printed = capsys.readouterr().out
    assert "Attempt 1: clip ended mid-attempt" in printed and f"RH: {hand[1].id}, {hand[2].id}" in printed


def test_contacts_accepts_pose_with_fewer_frames_than_the_header(climb_video, tmp_path):  # Review Focus 2
    path, _ = climb_video
    out = tmp_path / "out"
    assert main(["holds", str(path), "--color", "green", "--out", str(out)]) == 0
    scripted_pose(out, path, load_holds(out / "holds.json")[0], frames=30)
    assert main(["contacts", str(path), "--out", str(out)]) == 0


def test_limb_never_seen(climb_video, tmp_path, capsys):  # Review Focus 5
    path, _ = climb_video
    out = tmp_path / "out"
    assert main(["holds", str(path), "--color", "green", "--out", str(out)]) == 0
    scripted_pose(out, path, load_holds(out / "holds.json")[0], hidden=(32,))  # right toe
    assert main(["contacts", str(path), "--out", str(out)]) == 0
    assert load_json(out / "contacts.json")["attempts"][0]["sequence"]["RF"] == []
    assert "RF: -" in capsys.readouterr().out


def test_contacts_refuses_mismatched_inputs(climb_video, wall, tmp_path, capsys):
    path, _ = climb_video
    out = tmp_path / "out"
    assert main(["contacts", str(path), "--out", str(out)]) == 1
    assert "run `beta holds" in capsys.readouterr().err  # nothing there yet
    assert main(["holds", str(path), "--color", "green", "--out", str(out)]) == 0
    assert main(["contacts", str(path), "--out", str(out)]) == 1
    assert "run `beta pose" in capsys.readouterr().err
    scripted_pose(out, path, load_holds(out / "holds.json")[0])
    doc = load_json(out / "holds.json")
    doc["camera"]["id"] = "000000000000"
    save_json(out / "holds.json", doc)
    assert main(["contacts", str(path), "--out", str(out)]) == 1
    assert "different camera.json" in capsys.readouterr().err
    src = imwrite(tmp_path / "wall.png", wall[0])
    assert main(["holds", str(src), "--color", "green", "--out", str(out)]) == 0
    assert main(["contacts", str(path), "--out", str(out)]) == 1
    assert "made from an image" in capsys.readouterr().err
