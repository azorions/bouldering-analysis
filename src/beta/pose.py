"""Body pose per frame with MediaPipe Pose.

MediaPipe's PoseLandmarker finds one person and returns 33 body landmarks
(nose, shoulders, wrists, hips, ankles, toes, ...). For each landmark it gives:

- x, y: position, as fractions of the image (0-1). This module converts them
  to pixels so they line up with the hold polygons in holds.json.
- z: depth relative to the hips, roughly on the scale of x. A single camera
  can't measure depth well, so it is noisy; it is kept for later milestones.
  It is left in MediaPipe's units (fractions of the image width), not pixels.
- visibility: how likely the landmark is visible rather than hidden (behind
  the body, round a volume, ...). MediaPipe still guesses a position for a
  hidden wrist, so the contacts stage should ignore low-visibility points.
- presence: how likely the landmark is inside the frame at all.

The landmarker runs in VIDEO mode: after the first detection it tracks the
person from frame to frame instead of searching the whole image again, which
is faster and less jittery. VIDEO mode needs strictly increasing timestamps,
so frames are read in order with :func:`beta.video.iter_frames` and stamped
from their index and the fps.
"""

from __future__ import annotations

import http.client
import os
import shutil
import sys
import tempfile
import urllib.request
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

import cv2
import numpy as np

from beta.holds import FONT, Hold, put_label
from beta.jsonio import load_json, save_json
from beta.video import iter_frames

BGR = tuple[int, int, int]

MODEL_VARIANTS = ("lite", "full", "heavy")
MODEL_SIZES_MB = {"lite": 6, "full": 9, "heavy": 31}
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_{variant}/float16/latest/pose_landmarker_{variant}.task"
)
MODEL_DIR = Path("models")

# MediaPipe's landmark order (mediapipe.tasks.python.vision.PoseLandmark), copied
# here so that loading pose.json or drawing doesn't need to import MediaPipe.
LANDMARK_NAMES = (
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right", "left_shoulder", "right_shoulder", "left_elbow",
    "right_elbow", "left_wrist", "right_wrist", "left_pinky", "right_pinky", "left_index", "right_index",
    "left_thumb", "right_thumb", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
    "left_heel", "right_heel", "left_foot_index", "right_foot_index",
)  # fmt: skip
LANDMARK_FIELDS = ("x", "y", "z", "visibility", "presence")
# The points that touch holds: wrists for hands, toes ("foot_index") for feet,
# since climbers stand on the front of the shoe. "Left" is the climber's own left.
CONTACT_POINTS = {"LH": 15, "RH": 16, "LF": 31, "RF": 32}

_LEFT_EDGES = [(11, 13), (13, 15), (15, 19), (11, 23), (23, 25), (25, 27), (27, 29), (29, 31), (27, 31)]
_RIGHT_EDGES = [(12, 14), (14, 16), (16, 20), (12, 24), (24, 26), (26, 28), (28, 30), (30, 32), (28, 32)]
_CENTER_EDGES = [(11, 12), (23, 24)]


@dataclass(frozen=True)
class PoseFrame:
    frame: int  # index in the video, counting from 0
    t_ms: int  # timestamp given to MediaPipe
    landmarks: np.ndarray | None  # (33, 5): x_px, y_px, z, visibility, presence; None = no person found


@dataclass
class PoseTrack:
    frames: list[PoseFrame]
    fps: float
    width: int
    height: int

    def detected(self) -> int:
        return sum(p.landmarks is not None for p in self.frames)

    def longest_gap(self) -> tuple[int, int | None]:
        """Length and first frame of the longest run of frames without a pose."""
        best, best_start, run, start = 0, None, 0, None
        for p in self.frames:
            if p.landmarks is None:
                start = p.frame if run == 0 else start
                run += 1
                if run > best:
                    best, best_start = run, start
            else:
                run = 0
        return best, best_start

    def mean_visibility(self) -> dict[str, float | None]:
        """Mean visibility of each contact point over the frames with a pose."""
        found = [p.landmarks for p in self.frames if p.landmarks is not None]
        out: dict[str, float | None] = {}
        for name, idx in CONTACT_POINTS.items():
            values = np.array([lm[idx, 3] for lm in found], dtype=float)
            values = values[np.isfinite(values)]
            out[name] = float(values.mean()) if values.size else None
        return out


# --- model ------------------------------------------------------------------


def model_file(variant: str, model_dir: str | Path = MODEL_DIR) -> Path:
    return Path(model_dir) / f"pose_landmarker_{variant}.task"


def download_model(variant: str, path: str | Path) -> Path:
    """Download a pose model. Writes to ``<path>.part`` first, so an interrupted
    download never leaves a truncated file that looks like a real model."""
    path = Path(path)
    url = MODEL_URL.format(variant=variant)
    part = path.with_name(path.name + ".part")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=60) as response, open(part, "wb") as f:
            length = response.headers.get("Content-Length")
            shutil.copyfileobj(response, f)
        size = part.stat().st_size
        if size == 0:
            raise OSError("the server sent an empty file")
        # A malformed length (e.g. "5, 5") is treated as unknown rather than as an error.
        expected = int(length) if length and length.strip().isdigit() else None
        if expected is not None and size != expected:
            raise OSError(f"got {size} of {expected} bytes")
        part.replace(path)
    # URLError and timeouts are OSErrors; a connection dropped partway through the
    # body raises http.client.IncompleteRead, which is an HTTPException instead.
    except (OSError, http.client.HTTPException) as exc:
        part.unlink(missing_ok=True)
        raise ValueError(
            f"could not download the pose model from {url} ({exc}). "
            "Download it by hand and pass the file with --model-path"
        ) from exc
    return path


@contextmanager
def _stderr_fd_to(file: IO[bytes]) -> Iterator[None]:
    """Send everything written to file descriptor 2 into ``file``.

    MediaPipe's C++ code prints startup notes ("Created TensorFlow Lite XNNPACK
    delegate", ...) straight to the stderr descriptor, bypassing ``sys.stderr``,
    so they can only be silenced at the descriptor level.
    """
    sys.stderr.flush()
    saved = os.dup(2)
    os.dup2(file.fileno(), 2)
    try:
        yield
    finally:
        os.dup2(saved, 2)
        os.close(saved)


class Landmarker:
    """MediaPipe PoseLandmarker in VIDEO mode that takes BGR frames.

    The model is passed to MediaPipe as bytes rather than a path, which avoids
    path encoding problems on Windows (e.g. a user folder with an accented name).
    """

    def __init__(self, model_path: str | Path, min_confidence: float = 0.5) -> None:
        # Imported here: MediaPipe takes a while to load and `beta holds` doesn't need it.
        from mediapipe import Image, ImageFormat
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode

        self._image = lambda rgb: Image(image_format=ImageFormat.SRGB, data=rgb)
        model_bytes = Path(model_path).read_bytes()
        if not model_bytes:
            raise ValueError(f"could not load pose model {model_path}: the file is empty")
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_buffer=model_bytes),
            running_mode=RunningMode.VIDEO,
            num_poses=1,  # bouldering videos have one climber; spotters are usually out of frame
            min_pose_detection_confidence=min_confidence,
            min_pose_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
        )
        with tempfile.TemporaryFile() as log:
            try:
                with _stderr_fd_to(log):
                    self._landmarker = PoseLandmarker.create_from_options(options)
            # RuntimeError: not a model at all. FileNotFoundError: a .task bundle without
            # the pose models inside (e.g. MediaPipe's hand model). ValueError: options
            # MediaPipe rejects.
            except (RuntimeError, FileNotFoundError, ValueError) as exc:
                log.seek(0)
                details = log.read().decode(errors="replace").strip()
                raise ValueError(
                    f"could not load pose model {model_path} (is it a MediaPipe pose_landmarker .task file?): {exc}\n{details}".strip()
                ) from exc

    def detect(self, frame_bgr: np.ndarray, t_ms: int) -> Sequence[Any] | None:
        """The landmarks of the first person found (MediaPipe objects with x, y, z,
        visibility, presence), or None."""
        rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        result = self._landmarker.detect_for_video(self._image(rgb), t_ms)
        return result.pose_landmarks[0] if result.pose_landmarks else None

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> Landmarker:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def open_landmarker(model_path: str | Path, min_confidence: float = 0.5) -> Landmarker:
    return Landmarker(model_path, min_confidence)


# --- estimation -------------------------------------------------------------


def landmarks_to_array(landmarks: Sequence[Any], width: int, height: int) -> np.ndarray:
    """MediaPipe landmarks -> (33, 5) array with x, y in pixels. Missing scores become NaN."""
    rows = [
        (lm.x * width, lm.y * height, lm.z, np.nan if lm.visibility is None else lm.visibility,
         np.nan if lm.presence is None else lm.presence)
        for lm in landmarks
    ]  # fmt: skip
    return np.array(rows, dtype=np.float64)


def estimate_poses(
    frames: Iterable[tuple[int, np.ndarray]],
    fps: float,
    landmarker: Any,
    progress: Callable[[int], None] | None = None,
) -> list[PoseFrame]:
    """Run ``landmarker.detect(frame_bgr, t_ms)`` on each ``(index, frame)`` in order."""
    out: list[PoseFrame] = []
    last_t = -1
    for i, frame in frames:
        # Strictly increasing even if rounding would repeat a timestamp (very high fps).
        t_ms = max(round(i * 1000 / fps), last_t + 1)
        last_t = t_ms
        found = landmarker.detect(frame, t_ms)
        height, width = frame.shape[:2]
        out.append(PoseFrame(i, t_ms, None if found is None else landmarks_to_array(found, width, height)))
        if progress:
            progress(len(out))
    return out


# --- debug video ------------------------------------------------------------

VIDEO_MAX_WIDTH = 1280  # downscaled to this, like holds_debug.png, to keep the file small
LEFT_COLOR: BGR = (255, 170, 0)  # blue
RIGHT_COLOR: BGR = (0, 140, 255)  # orange
CENTER_COLOR: BGR = (230, 230, 230)
HOLD_COLOR: BGR = (255, 255, 255)
VISIBLE_MIN = 0.5  # below this MediaPipe is mostly guessing where the landmark is; drawn faint


def _faint(color: BGR) -> BGR:
    return tuple(int(c * 0.4) for c in color)  # type: ignore[return-value]


def _draw_holds(img: np.ndarray, holds: Sequence[Hold], sx: float, sy: float) -> None:
    """Hold outlines and numbers, blended in at 60% so the skeleton stands out on top."""
    layer = img.copy()
    for hold in holds:
        pts = (np.asarray(hold.polygon, np.float64).reshape(-1, 2) * (sx, sy)).round().astype(np.int32)
        cv2.polylines(layer, [pts], True, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.polylines(layer, [pts], True, HOLD_COLOR, 1, cv2.LINE_AA)
        put_label(layer, str(hold.id), (hold.centroid[0] * sx, hold.centroid[1] * sy), 0.4)
    cv2.addWeighted(layer, 0.6, img, 0.4, 0, dst=img)


def draw_pose(
    frame_bgr: np.ndarray, pose: PoseFrame | None, holds: Sequence[Hold] = (), size: tuple[int, int] | None = None
) -> np.ndarray:
    """One debug video frame: faint hold outlines, then the skeleton.

    Left side blue, right side orange, so a left/right swap (common when the
    climber faces the wall) shows up as the colors jumping between limbs.
    Joints MediaPipe isn't sure are visible are drawn faint. ``size`` is the
    output (width, height), by default the frame's own size.
    """
    height, width = frame_bgr.shape[:2]
    size = size or (width, height)
    sx, sy = size[0] / width, size[1] / height
    img = cv2.resize(frame_bgr, size, interpolation=cv2.INTER_AREA) if size != (width, height) else frame_bgr.copy()
    if holds:
        _draw_holds(img, holds, sx, sy)

    found = pose is not None and pose.landmarks is not None
    lines = [] if pose is None else [f"frame {pose.frame}  {pose.t_ms / 1000:.2f} s"]
    lines.append("blue = left, orange = right (the climber's own), faint = low visibility" if found else "no pose")
    # Text goes on a darkened band rather than being outlined: a thick outline glyph
    # is wider than the thin one, so on a long line the outline drifts off the text.
    # The band is darkened first so a hand reaching into it is still drawn at full color.
    band = 10 + 22 * len(lines)
    img[:band] = (img[:band] * 0.35).astype(np.uint8)

    if found:
        lm = pose.landmarks
        xy = np.column_stack([lm[:, 0] * sx, lm[:, 1] * sy])
        vis = np.nan_to_num(lm[:, 3], nan=0.0)

        def pt(i: int) -> tuple[int, int]:
            return round(xy[i, 0]), round(xy[i, 1])

        for edges, color in ((_CENTER_EDGES, CENTER_COLOR), (_LEFT_EDGES, LEFT_COLOR), (_RIGHT_EDGES, RIGHT_COLOR)):
            for a, b in edges:
                if not np.isfinite(xy[[a, b]]).all():
                    continue
                sure = min(vis[a], vis[b]) >= VISIBLE_MIN
                cv2.line(img, pt(a), pt(b), color if sure else _faint(color), 2 if sure else 1, cv2.LINE_AA)
        for name, idx in CONTACT_POINTS.items():
            if not np.isfinite(xy[idx]).all():
                continue
            color = LEFT_COLOR if name[0] == "L" else RIGHT_COLOR
            sure = vis[idx] >= VISIBLE_MIN
            cv2.circle(img, pt(idx), 6, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(img, pt(idx), 5, color if sure else _faint(color), -1, cv2.LINE_AA)
            put_label(img, name, (xy[idx, 0], xy[idx, 1] - 14), 0.45, color if sure else _faint(color))

    for i, line in enumerate(lines):
        cv2.putText(img, line, (10, 22 + 22 * i), FONT, 0.55, (235, 235, 235), 1, cv2.LINE_AA)
    return img


def write_pose_video(video_path: str | Path, track: PoseTrack, out_path: str | Path, holds: Sequence[Hold] = ()) -> Path:
    """Draw every frame of the clip with its pose and write it as .mp4."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    s = min(1.0, VIDEO_MAX_WIDTH / track.width)
    # Most players reject odd frame sizes in mp4.
    size = (max(2, round(track.width * s)) // 2 * 2, max(2, round(track.height * s)) // 2 * 2)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), track.fps, size)
    if not writer.isOpened():
        raise ValueError(f"OpenCV could not open a video writer for {out_path}")
    poses = {p.frame: p for p in track.frames}
    try:
        for i, frame in iter_frames(video_path):
            writer.write(draw_pose(frame, poses.get(i), holds, size=size))
    finally:
        writer.release()
    return out_path


# --- JSON -------------------------------------------------------------------


def _rounded(lm: np.ndarray) -> np.ndarray:
    # 0.1 px is far below MediaPipe's accuracy; rounding keeps pose.json small.
    return np.column_stack([lm[:, :2].round(1), lm[:, 2:3].round(4), lm[:, 3:].round(3)])


def pose_to_dict(track: PoseTrack, source: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": source,
        "params": params,
        "landmark_names": list(LANDMARK_NAMES),
        "landmark_fields": list(LANDMARK_FIELDS),
        "landmark_units": dict(zip(LANDMARK_FIELDS, ("px", "px", "image widths", "0-1", "0-1"))),
        "frames": [
            {"frame": p.frame, "t_ms": p.t_ms, "landmarks": None if p.landmarks is None else _rounded(p.landmarks)}
            for p in track.frames
        ],
    }


def save_pose(path: str | Path, track: PoseTrack, source: dict[str, Any], params: dict[str, Any]) -> Path:
    # No indentation: a minute of video is ~60k numbers, one per line would be unreadable anyway.
    return save_json(path, pose_to_dict(track, source, params), indent=None)


def load_pose(path: str | Path) -> tuple[PoseTrack, dict[str, Any]]:
    """The pose track plus the full JSON document (source and params) from ``pose.json``."""
    data = load_json(path)
    frames = [
        PoseFrame(int(f["frame"]), int(f["t_ms"]), None if f["landmarks"] is None else np.array(f["landmarks"], dtype=np.float64))
        for f in data["frames"]
    ]  # np.array turns the nulls for missing scores back into NaN
    src = data["source"]
    return PoseTrack(frames, float(src["fps"]), int(src["width"]), int(src["height"])), data
