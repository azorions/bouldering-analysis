from __future__ import annotations

import cv2
import numpy as np
import pytest

from beta.holds import Hold
from beta.synthetic import HOLD_HSV, TrueHold, WallTruth, hsv_to_bgr, make_climb_video, make_wall


@pytest.fixture(scope="session")
def wall() -> tuple[np.ndarray, WallTruth]:
    return make_wall(1280, 960, seed=0)


@pytest.fixture(scope="session")
def climb_video(tmp_path_factory: pytest.TempPathFactory):
    path = tmp_path_factory.mktemp("video") / "climb.avi"
    truth = make_climb_video(path, width=640, height=480, n_frames=31, fps=15.0)
    return path, truth


FOOT_CHIPS = [(400, 900), (700, 950), (1000, 880), (1300, 960), (1600, 910)]


def foot_chip_wall() -> np.ndarray:
    """1080p grey wall with five ~60 px^2 green foot chips (3e-5 of the frame).

    That's a 4-5 cm chip when the whole problem is in frame: kept by the
    default min_area_frac (2e-5), rejected by the old 5e-5.
    """
    rng = np.random.default_rng(3)
    image = np.clip(125 + rng.normal(0, 4, (1080, 1920, 3)), 0, 255).astype(np.uint8)
    for center in FOOT_CHIPS:
        cv2.ellipse(image, center, (5, 4), 0, 0, 360, hsv_to_bgr(HOLD_HSV["green"]), -1, cv2.LINE_AA)
    return image


def inside(hold: Hold, point: tuple[float, float]) -> bool:
    return cv2.pointPolygonTest(hold.contour(), (float(point[0]), float(point[1])), False) >= 0


def match(holds: list[Hold], truth: list[TrueHold]) -> list[list[int]]:
    """For each true hold, the ids of detected holds that contain all of its probe points."""
    return [[h.id for h in holds if all(inside(h, p) for p in t.probes)] for t in truth]


def expected_order(truth: list[TrueHold]) -> list[TrueHold]:
    """True holds in the order they should be numbered: bottom to top, then left to right."""
    return sorted(truth, key=lambda t: (-t.center[1], t.center[0]))
