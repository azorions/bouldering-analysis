"""Synthetic walls and climbing clips with known hold positions.

These let the tests (and you, before any real footage exists) check hold
detection against a known answer. The scene includes the cases that break
naive color thresholding on real walls: chalk, shadow, holds cut off by the
frame edge, near-miss colors, specks, a big painted panel in the route color,
and red holds on both sides of the hue wrap-around.

Every position and size is a fraction of the image, and all random draws are
made in those fractions, so one seed renders the same scene at any resolution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

# OpenCV HSV (H 0-179) of each hold color in the scene.
HOLD_HSV: dict[str, tuple[int, int, int]] = {
    "green": (60, 200, 170),
    "red_low": (2, 210, 180),  # red just above 0 ...
    "red_high": (176, 210, 180),  # ... and just below 180: same color, opposite ends of the hue scale
    "orange": (14, 220, 220),
    "yellow": (28, 200, 210),  # close to green's lower hue bound (35)
    "teal": (92, 200, 170),  # close to green's upper hue bound (85); counts as blue
    "blue": (110, 200, 170),
    "purple": (140, 150, 140),
    "pink": (160, 120, 220),
}
# Which color preset each scene color should be detected as.
PRESET_OF = {"red_low": "red", "red_high": "red", "teal": "blue"}

# (scene color, x, y, radius, shape, kind). x and y are fractions of width and
# height, radius a fraction of min(width, height).
_ROUTE = [
    ("green", 0.30, 0.90, 0.030, "ellipse", "plain"),  # start hold
    ("green", 0.55, 0.82, 0.025, "poly", "plain"),
    ("green", 0.008, 0.70, 0.030, "ellipse", "border"),  # mostly off the left edge
    ("green", 0.45, 0.62, 0.022, "ellipse", "chalked"),
    ("green", 0.70, 0.55, 0.028, "poly", "shadowed"),
    ("green", 0.35, 0.44, 0.020, "ellipse", "adjacent"),  # a pair just apart
    ("green", 0.62, 0.30, 0.035, "poly", "plain"),
    ("green", 0.25, 0.18, 0.025, "ellipse", "plain"),
    ("green", 0.50, 0.08, 0.030, "poly", "plain"),  # top hold
]
_DISTRACTORS = [
    ("red_low", 0.85, 0.85, 0.028, "ellipse"),
    ("red_high", 0.15, 0.55, 0.025, "poly"),
    ("orange", 0.82, 0.68, 0.025, "ellipse"),
    ("yellow", 0.12, 0.35, 0.025, "poly"),
    ("teal", 0.88, 0.42, 0.025, "ellipse"),
    ("blue", 0.15, 0.82, 0.025, "poly"),
    ("purple", 0.40, 0.28, 0.022, "ellipse"),
    ("pink", 0.65, 0.94, 0.020, "ellipse"),
]
_LARGE_PANEL = (0.72, 0.05, 0.22, 0.18)  # x, y, w, h fractions: ~4% of the image, over the 3% limit
_ADJACENT_GAP = 0.02  # edge-to-edge gap of the adjacent pair, fraction of min(width, height)
_N_SPECKS = 40


@dataclass(frozen=True)
class TrueHold:
    color: str  # preset it should be detected as, e.g. "green" or "red"
    scene_color: str  # key into HOLD_HSV
    kind: str  # plain | border | chalked | shadowed | adjacent | distractor
    center: tuple[float, float]
    radius: float
    probes: tuple[tuple[int, int], ...]  # pixels that must all fall inside this hold's detected polygon


@dataclass
class WallTruth:
    width: int
    height: int
    holds: list[TrueHold]
    specks: list[tuple[int, int]] = field(default_factory=list)
    large_panel: tuple[int, int, int, int] = (0, 0, 0, 0)  # x, y, w, h in px

    def of_color(self, color: str) -> list[TrueHold]:
        return [h for h in self.holds if h.color == color]


def hsv_to_bgr(hsv: tuple[int, int, int]) -> tuple[int, int, int]:
    b, g, r = cv2.cvtColor(np.uint8([[hsv]]), cv2.COLOR_HSV2BGR)[0, 0]
    return int(b), int(g), int(r)


def _shape_mask(size: tuple[int, int], center: tuple[float, float], r: float, shape: str, rng: np.random.Generator) -> np.ndarray:
    """Filled hold silhouette. Random draws don't depend on resolution."""
    mask = np.zeros(size, np.uint8)
    cx, cy = center
    if shape == "ellipse":
        aspect, angle = rng.uniform(0.65, 0.9), rng.uniform(0, 180)
        cv2.ellipse(mask, (round(cx), round(cy)), (max(1, round(r)), max(1, round(r * aspect))), angle, 0, 360, 255, -1, cv2.LINE_AA)
    else:  # irregular convex polygon, like a typical crimp or sloper outline
        # Evenly spaced, jittered vertex angles keep the center well inside the outline.
        k = int(rng.integers(6, 9))
        step = 2 * np.pi / k
        angles = np.arange(k) * step + rng.uniform(-0.25, 0.25, k) * step
        radii = r * rng.uniform(0.8, 1.1, k)
        pts = np.stack([cx + radii * np.cos(angles), cy + radii * np.sin(angles)], axis=1)
        cv2.fillPoly(mask, [cv2.convexHull(pts.round().astype(np.int32))], 255, cv2.LINE_AA)
    return mask


def _paint(image: np.ndarray, mask: np.ndarray, bgr: tuple[int, int, int]) -> None:
    """Paint a color through an anti-aliased mask (soft edges, like a real photo)."""
    alpha = (mask.astype(np.float32) / 255)[..., None]
    image[:] = (image * (1 - alpha) + np.array(bgr, np.float32) * alpha).astype(np.uint8)


def make_wall(width: int = 1280, height: int = 960, seed: int = 0) -> tuple[np.ndarray, WallTruth]:
    """A grey wall with a green route, distractor holds and the awkward cases.

    Returns the BGR image and the ground truth.
    """
    rng = np.random.default_rng(seed)
    d = min(width, height)
    size = (height, width)

    # Grey plywood with a soft left-to-right lighting gradient, panel seams and bolt holes.
    gradient = np.linspace(-12, 12, width, dtype=np.float32)[None, :, None]
    image = np.clip(np.full((height, width, 3), 125, np.float32) + gradient, 0, 255).astype(np.uint8)
    for fx in (0.33, 0.66):
        cv2.line(image, (round(fx * width), 0), (round(fx * width), height), (85, 85, 85), max(1, round(0.002 * d)))
    for fy in (0.5,):
        cv2.line(image, (0, round(fy * height)), (width, round(fy * height)), (85, 85, 85), max(1, round(0.002 * d)))
    for gx in np.arange(0.05, 1.0, 0.1):
        for gy in np.arange(0.05, 1.0, 0.1):
            cv2.circle(image, (round(gx * width), round(gy * height)), max(1, round(0.003 * d)), (45, 45, 45), -1)

    truth: list[TrueHold] = []

    def add_hold(scene_color: str, fx: float, fy: float, fr: float, shape: str, kind: str) -> tuple[float, float, float]:
        cx, cy, r = fx * width, fy * height, fr * d
        mask = _shape_mask(size, (cx, cy), r, shape, rng)
        _paint(image, mask, hsv_to_bgr(HOLD_HSV[scene_color]))
        probes = [(round(cx), round(cy))]
        if kind == "border":
            probes = [(round(cx + 0.5 * r), round(cy))]  # the visible part, inside the frame
        elif kind == "chalked":
            # Chalk is white powder: it washes out saturation. A band of it across the
            # middle would split the hold in two without the morphological close.
            band = np.zeros(size, np.uint8)
            t = max(2, round(0.006 * d))
            band[max(0, round(cy - t / 2)) : round(cy + t / 2), :] = 255
            chalk = cv2.bitwise_and(mask, band)
            sel = chalk > 0
            image[sel] = (image[sel] * 0.3 + 255 * 0.7).astype(np.uint8)
        elif kind == "shadowed":
            # Darken the right side down to 45% brightness: hue and saturation stay, value drops.
            xs = np.arange(width, dtype=np.float32)
            factor = np.clip(1 - 0.55 * (xs - cx) / r, 0.45, 1.0)[None, :, None]
            sel = (mask > 0)[..., None]
            image[:] = np.where(sel, (image * factor).astype(np.uint8), image)
            probes.append((round(cx + 0.5 * r), round(cy)))  # in the dark part
        color = PRESET_OF.get(scene_color, scene_color)
        truth.append(TrueHold(color, scene_color, kind, (cx, cy), r, tuple(probes)))
        return cx, cy, r

    for scene_color, fx, fy, fr, shape, kind in _ROUTE:
        cx, cy, r = add_hold(scene_color, fx, fy, fr, shape, kind)
        if kind == "adjacent":
            # Partner hold just to the right (and slightly higher, so ids don't tie).
            partner_r = 0.018
            partner_fx = (cx + r + _ADJACENT_GAP * d + partner_r * d) / width
            add_hold(scene_color, partner_fx, fy - 0.015, partner_r, "ellipse", "adjacent")
    for scene_color, fx, fy, fr, shape in _DISTRACTORS:
        add_hold(scene_color, fx, fy, fr, shape, "distractor")

    # A large panel in the route color (a painted volume): must be rejected as too large.
    px, py, pw, ph = (round(v) for v in (_LARGE_PANEL[0] * width, _LARGE_PANEL[1] * height, _LARGE_PANEL[2] * width, _LARGE_PANEL[3] * height))
    cv2.rectangle(image, (px, py), (px + pw, py + ph), hsv_to_bgr(HOLD_HSV["green"]), -1)

    # Specks of route color (paint flecks, a dropped tape scrap), away from the holds.
    specks: list[tuple[int, int]] = []
    speck_r = max(1, round(0.0015 * d))
    while len(specks) < _N_SPECKS:
        fx, fy = rng.uniform(0.02, 0.98), rng.uniform(0.02, 0.98)
        x, y = fx * width, fy * height
        near_hold = any(np.hypot(x - h.center[0], y - h.center[1]) < h.radius + 0.06 * d for h in truth)
        on_panel = px - 0.03 * d <= x <= px + pw + 0.03 * d and py - 0.03 * d <= y <= py + ph + 0.03 * d
        if near_hold or on_panel:
            continue
        cv2.circle(image, (round(x), round(y)), speck_r, hsv_to_bgr(HOLD_HSV["green"]), -1)
        specks.append((round(x), round(y)))

    # Camera sensor noise over everything.
    noise = rng.normal(0, 5, image.shape).astype(np.float32)
    image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return image, WallTruth(width, height, truth, specks, (px, py, pw, ph))


def climber_center(i: int, n_frames: int, width: int, height: int) -> tuple[int, int]:
    """Where the fake climber is in frame ``i``: a diagonal from bottom-left to top-right."""
    t = i / max(1, n_frames - 1)
    return round((0.15 + 0.7 * t) * width), round((0.85 - 0.7 * t) * height)


def make_climb_video(
    path: str | Path, width: int = 640, height: int = 480, n_frames: int = 31, fps: float = 15.0, seed: int = 0
) -> WallTruth:
    """Write a static-camera clip of the synthetic wall with a "climber" moving across it.

    The climber wears a route-colored shirt, so if it survived into the median
    background it would be detected as an extra hold. The top-left 12x12 px
    square has brightness ``8 * frame_index`` so tests can check which frame
    was read. Written as MJPG .avi, which OpenCV can always encode.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wall, truth = make_wall(width, height, seed)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV could not open a video writer for {path}")
    try:
        for i in range(n_frames):
            frame = wall.copy()
            _draw_climber(frame, i, n_frames)
            writer.write(frame)
    finally:
        writer.release()
    return truth


def _draw_climber(frame: np.ndarray, i: int, n_frames: int, size: float = 1.0) -> None:
    """The fake climber (a route-colored body and a head) plus the frame-number marker square."""
    height, width = frame.shape[:2]
    cx, cy = climber_center(i, n_frames, width, height)
    ax, ay = round(0.06 * width * size), round(0.10 * height * size)
    cv2.ellipse(frame, (cx, cy), (ax, ay), 0, 0, 360, hsv_to_bgr(HOLD_HSV["green"]), -1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy - ay - round(0.04 * height * size)), round(0.035 * height * size), (120, 160, 210), -1, cv2.LINE_AA)
    frame[:12, :12] = min(255, 8 * i)


def add_wall_texture(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Grey scuffs, chalk dots and screw holes over the bare wall (not on the holds).

    A real gym wall is covered in these, and they are what camera alignment
    locks on to: the plain synthetic wall gives a 640x480 frame only ~200
    feature points, too few to align by. They are colorless, so they don't
    change which pixels count as a hold color.
    """
    height, width = image.shape[:2]
    d = min(height, width)
    layer = image.copy()
    for _ in range(round(width * height / 150)):
        x, y = int(rng.integers(0, width)), int(rng.integers(0, height))
        r = max(1, round(rng.uniform(0.002, 0.009) * d))
        v = int(rng.choice([rng.integers(40, 90), rng.integers(170, 230)]))  # dark holes or light chalk
        if rng.random() < 0.7:
            cv2.circle(layer, (x, y), r, (v, v, v), -1, cv2.LINE_AA)
        else:
            angle = rng.uniform(0, math.pi)
            dx, dy = round(3 * r * math.cos(angle)), round(3 * r * math.sin(angle))
            cv2.line(layer, (x - dx, y - dy), (x + dx, y + dy), (v, v, v), max(1, r // 2), cv2.LINE_AA)
    # Keep the holds (anything colorful) and a thin rim around them as they were.
    sat = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[..., 1]
    colorful = cv2.dilate((sat > 40).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    return np.where(colorful[..., None], image, layer)


@dataclass
class MovingTruth:
    wall: WallTruth  # the big wall the camera looks at
    wall_to_frame: list[np.ndarray]  # per frame: 3x3 from big-wall px to that frame's px (the true camera)


def camera_view(i: int, n_frames: int, width: int, height: int, wall_size: tuple[int, int], margin: int, tilt_frac: float) -> np.ndarray:
    """3x3 from big-wall px to frame px for frame ``i``: a handheld phone following a climber.

    The view pans up the wall by up to ``tilt_frac`` of the frame height and
    back down, rolls by up to 1.5 degrees and zooms by up to 4%, all smoothly,
    like the first real clip.
    """
    s = i / max(1, n_frames - 1)
    pan = tilt_frac * height * math.sin(math.pi * s)
    angle = 1.5 * math.sin(2 * math.pi * s)
    zoom = 1 + 0.04 * math.sin(math.pi * s)
    # The wall point at the middle of the view: starts at the bottom of the big wall and moves up by `pan`.
    cx, cy = wall_size[0] / 2, wall_size[1] - margin - 1 - height / 2 - pan
    m = cv2.getRotationMatrix2D((cx, cy), angle, zoom)  # rotate and zoom about that point (it stays put)
    m[:, 2] += (width / 2 - cx, height / 2 - cy)  # then move it to the middle of the frame
    return np.vstack([m, [0.0, 0.0, 1.0]])


def make_moving_climb_video(
    path: str | Path,
    width: int = 640,
    height: int = 480,
    n_frames: int = 45,
    fps: float = 15.0,
    seed: int = 0,
    tilt_frac: float = 0.2,
    blank_frames: range | tuple[int, ...] = (),
    climber_frac: float = 1.0,
) -> MovingTruth:
    """Like :func:`make_climb_video`, but filmed by a handheld phone that follows the climber.

    The camera looks at a wall taller than the frame (see :func:`camera_view`),
    so the true frame-to-wall transform of every frame is known. Frames in
    ``blank_frames`` are plain grey, like a hand over the lens, so alignment
    must mark them lost. ``climber_frac`` scales the climber, to check that a
    big moving body doesn't drag the alignment along with it.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # A margin around the area the view sweeps, so rotated corners never see past the wall's edge.
    margin = math.ceil(0.04 * max(width, height)) + 2
    inner_height = round(height * (1 + tilt_frac)) + 2
    wall, truth = make_wall(width + 2 * margin, inner_height + 2 * margin, seed)
    rng = np.random.default_rng(seed + 1)
    wall = add_wall_texture(wall, rng)
    blank = set(blank_frames)
    views = [camera_view(i, n_frames, width, height, (truth.width, truth.height), margin, tilt_frac) for i in range(n_frames)]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV could not open a video writer for {path}")
    try:
        for i, view in enumerate(views):
            if i in blank:
                writer.write(np.full((height, width, 3), 125, np.uint8))
                continue
            frame = cv2.warpPerspective(wall, view, (width, height), flags=cv2.INTER_LINEAR)
            # Fresh sensor noise every frame, unlike the wall's own texture, which moves with the wall.
            frame = np.clip(frame + rng.normal(0, 2, frame.shape), 0, 255).astype(np.uint8)
            _draw_climber(frame, i, n_frames, climber_frac)
            writer.write(frame)
    finally:
        writer.release()
    return MovingTruth(truth, views)
