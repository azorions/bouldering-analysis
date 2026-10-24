from __future__ import annotations

import io
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from beta.holds import Hold
from beta.synthetic import HOLD_HSV, TrueHold, WallTruth, climber_center, hsv_to_bgr, make_climb_video, make_wall


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


# Landmark offsets from the synthetic climber's center, as fractions of the frame
# height: a figure seen from behind (its left on the image left), arms reaching up.
STICK = {
    0: (0.0, -0.30),
    11: (-0.05, -0.22), 12: (0.05, -0.22),
    13: (-0.10, -0.30), 14: (0.10, -0.30),
    15: (-0.12, -0.38), 16: (0.12, -0.38),
    19: (-0.125, -0.40), 20: (0.125, -0.40),
    23: (-0.04, 0.0), 24: (0.04, 0.0),
    25: (-0.06, 0.10), 26: (0.06, 0.10),
    27: (-0.06, 0.20), 28: (0.06, 0.20),
    29: (-0.07, 0.21), 30: (0.07, 0.21),
    31: (-0.04, 0.22), 32: (0.04, 0.22),
}  # fmt: skip
HIDDEN = (27, 31)  # left ankle and toes get low visibility, as if behind a volume


class FakeLandmarker:
    """Stands in for beta.pose.Landmarker without the model: a stick figure that
    follows the synthetic climber, with no pose on the ``missing`` frames."""

    def __init__(self, n_frames: int, missing: set[int] | frozenset[int] = frozenset()) -> None:
        self.n_frames = n_frames
        self.missing = set(missing)
        self.calls: list[int] = []  # t_ms of each call
        self.closed = False

    def detect(self, frame_bgr: np.ndarray, t_ms: int):
        i = len(self.calls)
        self.calls.append(t_ms)
        if i in self.missing:
            return None
        h, w = frame_bgr.shape[:2]
        cx, cy = climber_center(i, self.n_frames, w, h)
        out = []
        for k in range(33):
            dx, dy = STICK.get(k, STICK[0])
            out.append(SimpleNamespace(
                x=(cx + dx * h) / w, y=(cy + dy * h) / h, z=-0.1,
                visibility=0.2 if k in HIDDEN else 0.9, presence=0.95,
            ))  # fmt: skip
        out[1].visibility = None  # MediaPipe's scores are optional
        return out

    def close(self) -> None:
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class FakeResponse(io.BytesIO):
    """Stands in for urllib's HTTP response in download tests. With ``fail_after``,
    reading past the body raises it, like a connection dropped mid-download."""

    def __init__(self, body: bytes, length: int | str | None = None, fail_after: Exception | None = None) -> None:
        super().__init__(body)
        self.headers = {} if length is None else {"Content-Length": str(length)}
        self.fail_after = fail_after

    def read(self, size: int | None = -1) -> bytes:
        data = super().read(size)
        if not data and self.fail_after is not None:
            raise self.fail_after
        return data
