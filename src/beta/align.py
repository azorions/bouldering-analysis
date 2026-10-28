"""Lining up a moving camera with one fixed view of the wall.

Friends film climbs by holding the phone, and the phone tilts to follow the
climber. Hold outlines found once for the whole clip would then drift off the
real holds. This stage works out, for every frame, the 3x3 matrix that maps the
frame's pixels onto one fixed view of the wall: **wall coordinates**. Holds are
detected in wall coordinates and pose points are mapped into them, so the later
stages work as if the camera had never moved.

Why one 3x3 matrix (a homography) per frame is enough: a phone held in a hand
mostly turns rather than moves sideways, and a camera that rotates about its
own center maps every image point by the same homography, whatever the point's
depth. The wall is close to flat anyway, which a homography also describes
exactly. (It breaks down when the person filming walks far sideways: angled
wall panels then shift by different amounts, which is called parallax.)

A still camera gets identity matrices and a canvas the size of the frame, so
its wall coordinates are exactly its frame pixels and nothing downstream
changes.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from beta.jsonio import load_json, save_json


@dataclass(frozen=True)
class FrameTransform:
    frame: int
    matrix: np.ndarray | None  # 3x3 float64, frame px -> wall px; None = lost (alignment failed on this frame)
    method: str  # "track" (followed from the previous frame) | "anchor" (matched to the reference) | "lost"
    inliers: int  # point matches that agreed with the fitted matrix


@dataclass(frozen=True)
class CameraTrack:
    mode: str  # "still" | "moving"
    fps: float
    frame_count: int  # frames actually read when moving, the video header's count when still
    width: int  # video frame size
    height: int
    canvas: tuple[int, int]  # (width, height) of wall coordinates
    reference_frame: int | None  # the frame whose view the wall coordinates follow; None when still
    transforms: tuple[FrameTransform, ...] = ()  # one per frame when moving, empty when still
    summary: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def still(cls, fps: float, frame_count: int, width: int, height: int, summary: dict[str, Any] | None = None) -> CameraTrack:
        return cls("still", fps, frame_count, width, height, (width, height), None, (), summary or {})

    @property
    def id(self) -> str:
        return camera_id(self)

    def frame_to_wall(self, frame: int) -> np.ndarray | None:
        """Matrix from frame pixels to wall coordinates, or None (lost frame, or no such frame)."""
        if not 0 <= frame < self.frame_count:
            return None
        if self.mode == "still":
            return np.eye(3)
        if frame >= len(self.transforms):
            return None
        return self.transforms[frame].matrix

    def wall_to_frame(self, frame: int) -> np.ndarray | None:
        m = self.frame_to_wall(frame)
        return None if m is None else np.linalg.inv(m)

    def warp_to_wall(self, image: np.ndarray, frame: int) -> tuple[np.ndarray, np.ndarray] | None:
        """The frame drawn into wall coordinates, plus a mask of the canvas pixels it covers (255)."""
        m = self.frame_to_wall(frame)
        if m is None:
            return None
        warped = cv2.warpPerspective(image, m, self.canvas, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        covered = np.full(image.shape[:2], 255, np.uint8)
        # Nearest-neighbour keeps the mask strictly 0/255 at the footprint's edge.
        mask = cv2.warpPerspective(covered, m, self.canvas, flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
        return warped, mask


def apply_h(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Map (N, 2) points through a 3x3 homography. Rows with NaN stay NaN."""
    pts = np.asarray(points, np.float64).reshape(-1, 2)
    out = np.full_like(pts, np.nan)
    ok = np.isfinite(pts).all(axis=1)
    if ok.any():
        out[ok] = cv2.perspectiveTransform(pts[ok].reshape(-1, 1, 2), np.asarray(matrix, np.float64)).reshape(-1, 2)
    return out


def _corners(width: int, height: int) -> np.ndarray:
    return np.array([[0.0, 0.0], [width, 0.0], [width, height], [0.0, height]])


def corner_gap(a: np.ndarray, b: np.ndarray, width: int, height: int) -> float:
    """Largest distance between where ``a`` and ``b`` send the four corners of a frame.

    The corners move the most under any rotation or zoom, so this is the
    worst-case disagreement between two transforms anywhere in the frame.
    """
    c = _corners(width, height)
    return float(np.linalg.norm(apply_h(a, c) - apply_h(b, c), axis=1).max())


def corner_shift(matrix: np.ndarray, width: int, height: int) -> float:
    """How far (px) a transform moves the frame corners: 0 for the identity."""
    return corner_gap(matrix, np.eye(3), width, height)


def camera_id(cam: CameraTrack) -> str:
    """A short fingerprint of the alignment result.

    holds.json and contacts.json record it, so a stage can tell that its input
    was made with a different camera.json. It is derived from the result rather
    than random, so rerunning ``align`` with the same settings keeps the same id
    and doesn't force ``holds`` to run again. Matrices are rounded to 4 decimals
    so float noise below that can't change it.
    """
    h = hashlib.sha1()
    h.update(f"{cam.mode}|{cam.reference_frame}|{cam.canvas[0]}x{cam.canvas[1]}".encode())
    for t in cam.transforms:
        matrix = "lost" if t.matrix is None else ",".join(f"{v:.4f}" for v in np.asarray(t.matrix).ravel())
        h.update(f"|{t.frame}:{t.method}:{matrix}".encode())
    return h.hexdigest()[:12]


def camera_ref(cam: CameraTrack) -> dict[str, Any]:
    """What holds.json records about the camera its polygons were made with."""
    return {"id": cam.id, "mode": cam.mode, "canvas": list(cam.canvas)}


# --- JSON -------------------------------------------------------------------


def camera_to_dict(cam: CameraTrack, source: dict[str, Any], params: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "source": source,
        "params": params or {},
        "id": cam.id,
        "mode": cam.mode,
        "reference_frame": cam.reference_frame,
        "canvas": list(cam.canvas),
        "fps": cam.fps,
        "frame_count": cam.frame_count,
        "width": cam.width,
        "height": cam.height,
        # Full float precision (no rounding), so a loaded camera has exactly the same id.
        "frames": [
            {"frame": t.frame, "matrix": None if t.matrix is None else np.asarray(t.matrix).tolist(), "method": t.method, "inliers": t.inliers}
            for t in cam.transforms
        ],
        "summary": cam.summary,
    }


def save_camera(path: str | Path, cam: CameraTrack, source: dict[str, Any], params: dict[str, Any] | None = None) -> Path:
    # No indentation: a minute of moving video is ~1800 matrices.
    return save_json(path, camera_to_dict(cam, source, params), indent=None)


def load_camera(path: str | Path) -> tuple[CameraTrack, dict[str, Any]]:
    """The camera track plus the full JSON document from ``camera.json``."""
    data = load_json(path)
    transforms = tuple(
        FrameTransform(int(f["frame"]), None if f["matrix"] is None else np.array(f["matrix"], np.float64), str(f["method"]), int(f["inliers"]))
        for f in data["frames"]
    )
    ref = data["reference_frame"]
    cam = CameraTrack(
        str(data["mode"]),
        float(data["fps"]),
        int(data["frame_count"]),
        int(data["width"]),
        int(data["height"]),
        (int(data["canvas"][0]), int(data["canvas"][1])),
        None if ref is None else int(ref),
        transforms,
        dict(data["summary"]),
    )
    return cam, data
