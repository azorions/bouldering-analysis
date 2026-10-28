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
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from beta.jsonio import load_json, save_json
from beta.video import VideoInfo, iter_frames, probe, read_frame, sample_indices

# Feature work runs on a copy downscaled to this long side, so its cost doesn't
# grow with resolution; transforms are scaled back to full-resolution pixels.
ALIGN_MAX_SIDE = 1024
STILL_TOL_PX = 1.0  # a clip whose sampled views all agree to within this (frame corners, px) is still
STILL_SAMPLES = 9  # frames compared to decide still vs moving; also the candidates for the reference view
MIN_INLIERS = 30  # fewer agreeing point matches than this and a fitted matrix isn't trusted
MAX_ZOOM = 1.5  # a frame that seems zoomed further than this from the reference is a bad fit, not a real zoom
MAX_STEP_FRAC = 0.25  # nor is one whose corners jump more than this fraction of the frame diagonal in one frame
CANVAS_MAX_SCALE = 2.0  # wall coordinates cover at most this many frame widths/heights
RANSAC_PX = 3.0  # how far (working px) a match may sit from the fitted matrix and still count as agreeing
SIFT_FEATURES = 2000
RANK_FEATURES = 500  # strongest features per sample used to rank reference candidates (all pairs, so kept small)
RATIO = 0.75  # Lowe's ratio test: keep a match only if it is clearly better than the second best
KLT_CORNERS = 800
# Tracking starts from up to KLT_CORNERS points; as the view moves away from the keyframe
# (or the climber covers the wall) fewer survive, and a fit from a few dozen is imprecise.
# Below this many, re-anchor straight away instead of waiting for the next scheduled anchor.
TRACK_MIN_INLIERS = 100
FB_MAX_PX = 1.0  # forward-backward check: a tracked point must come back to within this of where it started


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


# --- estimation -------------------------------------------------------------


@dataclass(frozen=True)
class StillCheck:
    still: bool
    samples: list[int]  # frame indices that were compared
    max_shift_px: float  # largest corner shift (full-resolution px) among samples that matched the hub
    matched: int  # samples (besides the hub) with a trusted match to it


@dataclass(frozen=True)
class _Features:
    points: np.ndarray  # (N, 2) float32, working px
    descriptors: np.ndarray | None
    responses: np.ndarray  # keypoint strength, to pick the strongest few

    def strongest(self, n: int) -> _Features:
        if len(self.points) <= n:
            return self
        keep = np.argsort(-self.responses)[:n]
        return _Features(self.points[keep], None if self.descriptors is None else self.descriptors[keep], self.responses[keep])


class _Matcher:
    """SIFT features matched with Lowe's ratio test, then a RANSAC homography.

    SIFT finds distinctive spots (bolt holes, hold edges, panel corners) and
    describes each by the gradients around it, so the same spot can be found
    again in another frame even after the camera turned or zoomed. RANSAC fits
    the matrix to random small sets of matches and keeps the one most matches
    agree with. Matches on the climber move differently from the wall, so they
    disagree with it and are left out.
    """

    def __init__(self) -> None:
        self.sift = cv2.SIFT_create(SIFT_FEATURES)
        self.bf = cv2.BFMatcher(cv2.NORM_L2)

    def features(self, gray: np.ndarray) -> _Features:
        kps, des = self.sift.detectAndCompute(gray, None)
        pts = np.array([k.pt for k in kps], np.float32).reshape(-1, 2)
        return _Features(pts, des, np.array([k.response for k in kps], np.float32))

    def match(self, a: _Features, b: _Features) -> tuple[np.ndarray | None, int]:
        """Homography from ``a``'s image to ``b``'s (working px) and its inlier count, or (None, count)."""
        if a.descriptors is None or b.descriptors is None or len(a.points) < 2 or len(b.points) < 2:
            return None, 0
        pairs = self.bf.knnMatch(a.descriptors, b.descriptors, k=2)
        good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < RATIO * p[1].distance]
        if len(good) < MIN_INLIERS:
            return None, len(good)
        src = a.points[[m.queryIdx for m in good]]
        dst = b.points[[m.trainIdx for m in good]]
        h, mask = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_PX)
        inliers = 0 if mask is None else int(mask.sum())
        return (h, inliers) if h is not None and inliers >= MIN_INLIERS else (None, inliers)


@dataclass(frozen=True)
class _Keyframe:
    """The last anchored frame: its corners are followed into every frame until the next anchor."""

    gray: np.ndarray
    points: np.ndarray  # (N, 1, 2) float32 corners, working px
    h: np.ndarray  # this frame -> reference view, working px


def _keyframe(gray: np.ndarray, h: np.ndarray) -> _Keyframe | None:
    points = cv2.goodFeaturesToTrack(gray, KLT_CORNERS, 0.01, 8)
    return None if points is None or len(points) < MIN_INLIERS else _Keyframe(gray, points, h)


def _track(key: _Keyframe, cur: np.ndarray) -> tuple[np.ndarray | None, int]:
    """Homography from ``cur`` to the keyframe (working px), by following the keyframe's corners.

    KLT optical flow looks for each corner only near where it was (through an
    image pyramid, so bigger moves are found too), which is far cheaper than
    matching features across the whole image. Every frame is measured against
    the last anchored frame rather than the previous frame, so the small error
    of each measurement doesn't add up from frame to frame. Each point is also
    tracked back again and kept only if it returns to where it started: on a
    frame that shows no wall (a hand over the lens) the trail goes nowhere
    consistent, too few points survive, and the frame is not trusted.
    """
    p1, st1, _ = cv2.calcOpticalFlowPyrLK(key.gray, cur, key.points, None, winSize=(21, 21), maxLevel=3)
    back, st2, _ = cv2.calcOpticalFlowPyrLK(cur, key.gray, p1, None, winSize=(21, 21), maxLevel=3)
    returned = np.linalg.norm((back - key.points).reshape(-1, 2), axis=1) < FB_MAX_PX
    ok = (st1.ravel() == 1) & (st2.ravel() == 1) & returned
    if ok.sum() < MIN_INLIERS:
        return None, int(ok.sum())
    h, mask = cv2.findHomography(p1[ok], key.points[ok], cv2.RANSAC, RANSAC_PX)
    inliers = 0 if mask is None else int(mask.sum())
    return (h, inliers) if h is not None and inliers >= MIN_INLIERS else (None, inliers)


def _work_scale(width: int, height: int) -> float:
    return min(1.0, ALIGN_MAX_SIDE / max(width, height))


def _gray(frame: np.ndarray, s: float) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if s < 1:
        gray = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return gray


def _to_full(h: np.ndarray, s: float) -> np.ndarray:
    """A working-scale homography as one between full-resolution pixels."""
    scale = np.diag([s, s, 1.0])
    return np.linalg.inv(scale) @ h @ scale


def _sample_features(path: str | Path, info: VideoInfo, n: int, matcher: _Matcher, s: float) -> dict[int, _Features]:
    # Seeking (read_frame) can land a frame or two off on phone video; that's fine
    # here, because a sample is only compared with other samples, never paired
    # with a transform for a specific frame number.
    out: dict[int, _Features] = {}
    for i in sample_indices(info.frame_count, n):
        try:
            out[i] = matcher.features(_gray(read_frame(path, i), s))
        except IndexError:
            continue  # the header frame count can overshoot by a frame or two
    return out


def _check_still(feats: dict[int, _Features], info: VideoInfo, matcher: _Matcher, s: float) -> StillCheck:
    samples = sorted(feats)
    if len(samples) < 2:
        return StillCheck(True, samples, 0.0, 0)
    # Compare every sample with one hub. If a hub matches fewer than half the
    # others (the climber fills it, a hand covers the lens), try its neighbours.
    mid = len(samples) // 2
    hubs = [samples[k] for k in (mid, mid - 1, mid + 1) if 0 <= k < len(samples)]
    shifts: list[float] = []
    for hub in hubs:
        shifts = []
        for other in samples:
            if other == hub:
                continue
            h, _ = matcher.match(feats[other], feats[hub])
            if h is not None:
                shifts.append(corner_shift(_to_full(h, s), info.width, info.height))
        if len(shifts) >= (len(samples) - 1) / 2:
            break
    largest = max(shifts, default=0.0)
    conclusive = len(shifts) >= (len(samples) - 1) / 2
    # Too few matches to tell: call it still (there's nothing to align with anyway); the caller adds a note.
    return StillCheck(largest < STILL_TOL_PX or not conclusive, samples, largest, len(shifts))


def check_still(path: str | Path, info: VideoInfo, n: int = STILL_SAMPLES) -> StillCheck:
    """Whether the camera stays put: compare ``n`` frames spread over the clip with one of them."""
    matcher = _Matcher()
    s = _work_scale(info.width, info.height)
    return _check_still(_sample_features(path, info, n, matcher, s), info, matcher, s)


def _choose_reference(feats: dict[int, _Features], matcher: _Matcher) -> int:
    """The sample with the most inlier matches to all the others: the most central, least covered view."""
    samples = sorted(feats)
    small = {i: feats[i].strongest(RANK_FEATURES) for i in samples}
    score = dict.fromkeys(samples, 0)
    for a_i, a in enumerate(samples):
        for b in samples[a_i + 1 :]:
            _, inliers = matcher.match(small[a], small[b])
            score[a] += inliers
            score[b] += inliers
    return max(samples, key=lambda i: (score[i], -abs(i - samples[len(samples) // 2])))


def _plausible(h: np.ndarray, h_prev: np.ndarray | None, width: int, height: int) -> bool:
    """Reject fits no real phone movement produces: a big zoom, or the view jumping between two frames."""
    zoom = math.sqrt(abs(float(np.linalg.det(h[:2, :2]))))
    if not 1 / MAX_ZOOM <= zoom <= MAX_ZOOM:
        return False
    return h_prev is None or corner_gap(h, h_prev, width, height) <= MAX_STEP_FRAC * math.hypot(width, height)


def _longest_run(flags: list[bool]) -> list[int | None]:
    best, best_start, run, start = 0, None, 0, None
    for i, flag in enumerate(flags):
        if flag:
            start = i if run == 0 else start
            run += 1
            if run > best:
                best, best_start = run, start
        else:
            run = 0
    return [best, best_start]


def estimate_camera(
    path: str | Path,
    info: VideoInfo | None = None,
    anchor_every_s: float = 1.0,
    n_samples: int = STILL_SAMPLES,
    progress: Callable[[int], None] | None = None,
) -> CameraTrack:
    """Work out whether the camera moves and, if it does, every frame's transform to wall coordinates.

    About once every ``anchor_every_s`` seconds a frame is matched straight to
    the reference view with SIFT (careful but slow): an anchor. The frames in
    between follow the last anchor's corners with KLT (cheap): like counting
    steps from the last landmark you checked. If an anchor fails, tracking
    carries on from the current frame and the anchor is retried next frame.
    """
    started = time.perf_counter()
    info = info or probe(path)
    matcher = _Matcher()
    s = _work_scale(info.width, info.height)
    feats = _sample_features(path, info, n_samples, matcher, s)
    check = _check_still(feats, info, matcher, s)
    if check.still:
        summary: dict[str, Any] = {"max_shift_px": round(check.max_shift_px, 2), "checked_frames": check.matched}
        if check.matched < (len(check.samples) - 1) / 2:
            summary["note"] = "too few wall features to check whether the camera moves"
        summary["seconds"] = round(time.perf_counter() - started, 2)
        return CameraTrack.still(info.fps, info.frame_count, info.width, info.height, summary)

    ref = _choose_reference(feats, matcher)
    ref_feats = matcher.features(_gray(read_frame(path, ref), s))
    work_w, work_h = round(info.width * s), round(info.height * s)
    anchor_every = max(1, round(anchor_every_s * info.fps))

    fits: list[tuple[np.ndarray | None, str, int]] = []  # per frame: (frame -> reference, working px), method, inliers
    corrections: list[float] = []
    h_prev: np.ndarray | None = None
    key: _Keyframe | None = None
    last_anchor = -anchor_every
    for i, frame in iter_frames(path):
        gray = _gray(frame, s)
        tracked, n_track = None, 0
        if key is not None:
            step, n_track = _track(key, gray)
            tracked = None if step is None else key.h @ step
        due = tracked is None or n_track < TRACK_MIN_INLIERS or i - last_anchor >= anchor_every
        h, method, inliers = None, "lost", 0
        if not due:
            h, method, inliers = tracked, "track", n_track
        else:
            anchored, n_anchor = matcher.match(matcher.features(gray), ref_feats)
            if anchored is not None and _plausible(anchored, None, work_w, work_h):
                h, method, inliers = anchored, "anchor", n_anchor
                last_anchor = i
                if tracked is not None:
                    corrections.append(corner_gap(_to_full(anchored, s), _to_full(tracked, s), info.width, info.height))
            elif tracked is not None:  # the anchor failed but tracking still works: keep it, retry next frame
                h, method, inliers = tracked, "track", n_track
        if h is not None and not _plausible(h, h_prev, work_w, work_h):
            h, method, inliers = None, "lost", 0
        fits.append((h, method, inliers))
        if h is None:
            key = None  # forces an anchor attempt on the next frame
        elif due:
            # A new anchor, or a failed one: either way follow points from here, so
            # tracking never has to reach far back to an old keyframe.
            key = _keyframe(gray, h)
        h_prev = h
        if progress:
            progress(i + 1)

    full = [None if h is None else _to_full(h, s) for h, _, _ in fits]
    # Wall coordinates: the reference frame's pixels, shifted so everything the camera saw is
    # at positive coordinates, and capped so one bad fit can't make the canvas huge.
    corners = _corners(info.width, info.height)
    seen = np.vstack([apply_h(h, corners) for h in full if h is not None])
    pad = (CANVAS_MAX_SCALE - 1) / 2
    x0, y0 = np.maximum(seen.min(axis=0), (-pad * info.width, -pad * info.height))
    x1, y1 = np.minimum(seen.max(axis=0), ((1 + pad) * info.width, (1 + pad) * info.height))
    x0, y0 = math.floor(x0), math.floor(y0)
    shift = np.array([[1.0, 0, -x0], [0, 1.0, -y0], [0, 0, 1.0]])
    canvas = (math.ceil(x1) - x0, math.ceil(y1) - y0)
    transforms = tuple(
        FrameTransform(i, None if h is None else shift @ h, method, inliers) for i, (h, (_, method, inliers)) in enumerate(zip(full, fits))
    )
    lost = [h is None for h in full]
    summary = {
        "max_shift_px": round(max((corner_shift(h, info.width, info.height) for h in full if h is not None), default=0.0), 1),
        "lost_frames": sum(lost),
        "longest_lost_run": _longest_run(lost),
        "anchor_correction_px": {
            "median": round(float(np.median(corrections)), 2) if corrections else 0.0,
            "max": round(max(corrections), 2) if corrections else 0.0,
        },
        "seconds": round(time.perf_counter() - started, 2),
    }
    return CameraTrack("moving", info.fps, len(fits), info.width, info.height, canvas, ref, transforms, summary)


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
