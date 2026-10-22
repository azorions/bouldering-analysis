from __future__ import annotations

import subprocess
import sys
from dataclasses import replace

from conftest import FOOT_CHIPS, foot_chip_wall

from beta import video
from beta.cli import main
from beta.jsonio import load_json
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
