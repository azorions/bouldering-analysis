"""Route hold detection by color.

Gyms set each boulder problem in one hold color, so "the holds of the green
route" is roughly "the green blobs on the wall". This module finds them with
HSV color thresholding:

1. Blur slightly, so sensor noise and hold texture don't punch holes in the mask.
2. Convert BGR -> HSV. Hue is the color itself, separate from saturation
   (how vivid) and value (how bright), so a hold half in shadow keeps roughly
   the same hue while its RGB values change a lot. That's why thresholding in
   HSV is far more robust than in RGB.
3. Keep pixels inside the route color's HSV range.
4. Morphological *open* (erode, then dilate) deletes specks smaller than the
   kernel; *close* (dilate, then erode) bridges thin gaps, e.g. a chalk smear
   across a hold that would otherwise split it in two.
5. Each connected blob becomes a candidate. Blobs that are too small (noise) or
   too large (a painted wall panel or mat) are rejected.
6. Outlines are simplified to polygons and numbered bottom to top.

All sizes are fractions of the image size, so the same settings work on 720p
and 4K footage. Later milestones may swap this for a segmentation model:
:func:`detect_holds` (image in, :class:`HoldDetection` out) is the interface
to keep.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from beta.jsonio import load_json, save_json

BGR = tuple[int, int, int]
HSV = tuple[int, int, int]


@dataclass(frozen=True)
class HSVRange:
    """Inclusive OpenCV HSV bounds (H 0-179, S and V 0-255).

    Hue is an angle, so red sits at both ends of the scale. A range whose
    lower hue is greater than its upper hue wraps around, e.g. 172..8 covers
    172-179 and 0-8.
    """

    lower: HSV
    upper: HSV

    def bands(self) -> list[tuple[HSV, HSV]]:
        """The range as one or two non-wrapping ``(lower, upper)`` bands for ``cv2.inRange``."""
        (h0, s0, v0), (h1, s1, v1) = self.lower, self.upper
        if h0 <= h1:
            return [(self.lower, self.upper)]
        return [((h0, s0, v0), (179, s1, v1)), ((0, s0, v0), (h1, s1, v1))]

    def contains(self, hsv: HSV) -> bool:
        h, s, v = hsv
        return any(lo[0] <= h <= hi[0] and lo[1] <= s <= hi[1] and lo[2] <= v <= hi[2] for lo, hi in self.bands())

    def describe(self) -> str:
        (h0, s0, v0), (h1, s1, v1) = self.lower, self.upper
        wrap = " (wraps)" if h0 > h1 else ""
        return f"H {h0}-{h1}{wrap}, S {s0}-{s1}, V {v0}-{v1}"


# Starting points tuned for typical gym lighting. Real footage often needs a
# nudge: use `beta holds ... --sample X,Y` to read the HSV of a real hold, then
# pass --hsv-lower/--hsv-upper.
PRESETS: dict[str, HSVRange] = {
    "red": HSVRange((172, 100, 60), (8, 255, 255)),
    "orange": HSVRange((9, 120, 80), (20, 255, 255)),  # high S min: plywood walls are a dull orange
    "yellow": HSVRange((21, 100, 100), (34, 255, 255)),
    "green": HSVRange((35, 70, 50), (85, 255, 255)),
    "blue": HSVRange((86, 80, 50), (125, 255, 255)),
    "purple": HSVRange((126, 50, 40), (150, 255, 255)),
    "pink": HSVRange((151, 60, 90), (171, 255, 255)),
    "white": HSVRange((0, 0, 190), (179, 40, 255)),
    "black": HSVRange((0, 0, 0), (179, 255, 50)),
}

# Colors defined by brightness alone, not hue, so they also match light walls,
# chalk, shadows and gaps between panels.
UNRELIABLE_COLORS: dict[str, str] = {
    "white": "white holds are found by brightness alone, so light walls and chalk will match too",
    "black": "black holds are found by darkness alone, so shadows and panel gaps will match too",
}


# Below ~4x4 px a blob is about the size of the blur and morphology kernels, so
# it can't be told apart from noise (e.g. JPEG/video compression specks). This
# only matters at low resolution: at 1080p and up, min_area_frac is larger.
MIN_AREA_FLOOR_PX = 16.0


@dataclass(frozen=True)
class HoldSpec:
    """What to detect and how. Size settings are fractions of the image (of one video frame for a wall view)."""

    color: str
    ranges: tuple[HSVRange, ...]
    # Of image area. 2e-5 is ~40 px^2 at 1080p: about a 4 cm foot chip when the
    # whole problem is in frame. Smaller specks are mostly gone after the open.
    # Never below MIN_AREA_FLOOR_PX, whatever the resolution (see min_area_px).
    min_area_frac: float = 2e-5
    max_area_frac: float = 0.03  # bigger blobs are wall panels, volumes or mats
    blur_frac: float = 0.004  # kernel sizes, as fractions of min(height, width)
    open_frac: float = 0.003
    close_frac: float = 0.008
    approx_eps_frac: float = 0.01  # polygon simplification tolerance, fraction of perimeter
    roi: tuple[int, int, int, int] | None = None  # x, y, w, h

    @classmethod
    def from_color(cls, color: str | None, hsv_lower: HSV | None = None, hsv_upper: HSV | None = None, **kwargs: Any) -> HoldSpec:
        """A spec from a preset name, or from explicit HSV bounds (which override the preset)."""
        if (hsv_lower is None) != (hsv_upper is None):
            raise ValueError("--hsv-lower and --hsv-upper must be given together")
        if hsv_lower is not None and hsv_upper is not None:
            for name, bound in (("lower", hsv_lower), ("upper", hsv_upper)):
                if not (0 <= bound[0] <= 179 and 0 <= bound[1] <= 255 and 0 <= bound[2] <= 255):
                    raise ValueError(f"HSV {name} {bound} out of range (H 0-179, S 0-255, V 0-255)")
            if hsv_lower[1] > hsv_upper[1] or hsv_lower[2] > hsv_upper[2]:
                raise ValueError(
                    f"HSV lower {hsv_lower} is above upper {hsv_upper} in S or V, which matches nothing "
                    "(only H may wrap around)"
                )
            return cls(color or "custom", (HSVRange(tuple(hsv_lower), tuple(hsv_upper)),), **kwargs)
        if color is None:
            raise ValueError("give a --color or --hsv-lower/--hsv-upper")
        if color not in PRESETS:
            raise ValueError(f"unknown color {color!r}; choose from {', '.join(PRESETS)}")
        return cls(color, (PRESETS[color],), **kwargs)

    def describe_ranges(self) -> str:
        return " | ".join(r.describe() for r in self.ranges)

    def min_area_px(self, width: int, height: int) -> float:
        return max(MIN_AREA_FLOOR_PX, self.min_area_frac * width * height)

    def max_area_px(self, width: int, height: int) -> float:
        return self.max_area_frac * width * height


@dataclass(frozen=True)
class Hold:
    id: int  # 1 = lowest hold on the wall
    polygon: tuple[tuple[int, int], ...]  # simplified outline, pixel coordinates
    centroid: tuple[float, float]
    area: float  # px^2 of the detected blob
    bbox: tuple[int, int, int, int]  # x, y, w, h
    touches_border: bool  # cut off by the frame (or ROI) edge, so only partly visible

    def contour(self) -> np.ndarray:
        """The polygon in the ``(N, 1, 2) int32`` layout OpenCV functions expect."""
        return np.asarray(self.polygon, dtype=np.int32).reshape(-1, 1, 2)


@dataclass(frozen=True)
class Rejected:
    contour: np.ndarray  # (N, 2) int32 outline
    reason: str  # "too_small" | "too_large"
    area: float


@dataclass
class HoldDetection:
    holds: list[Hold]
    mask: np.ndarray  # cleaned binary mask (0/255), same size as the image
    rejected: list[Rejected]
    spec: HoldSpec
    kernels: dict[str, int] = field(default_factory=dict)  # actual blur/open/close sizes in px
    frame_size: tuple[int, int] | None = None  # (width, height) the size fractions refer to; None = the image's

    def rejected_counts(self) -> dict[str, int]:
        counts = Counter(r.reason for r in self.rejected)
        return {"too_small": counts.get("too_small", 0), "too_large": counts.get("too_large", 0)}

    def _size(self) -> tuple[int, int]:
        return self.frame_size or (self.mask.shape[1], self.mask.shape[0])

    @property
    def min_area_px(self) -> float:
        return self.spec.min_area_px(*self._size())

    @property
    def max_area_px(self) -> float:
        return self.spec.max_area_px(*self._size())

    def near_limit(self, fraction: float = 0.5) -> list[Rejected]:
        """Too-small blobs of at least ``fraction`` of the minimum hold area, largest first.

        Noise specks are far smaller than the limit; blobs just under it are
        often real footholds, so the CLI and debug image point them out.
        """
        floor = fraction * self.min_area_px
        small = [r for r in self.rejected if r.reason == "too_small" and r.area >= floor]
        return sorted(small, key=lambda r: r.area, reverse=True)


def odd_kernel(frac: float, size: int, minimum: int = 3) -> int:
    """Kernel side length ``frac * size``, at least ``minimum`` and odd (so it has a center pixel)."""
    k = max(minimum, round(frac * size))
    return k if k % 2 else k + 1


def color_mask(image_bgr: np.ndarray, ranges: tuple[HSVRange, ...] | list[HSVRange], blur_px: int = 0) -> np.ndarray:
    """Binary mask (0/255) of pixels inside any of the HSV ranges."""
    if blur_px > 1:
        image_bgr = cv2.GaussianBlur(image_bgr, (blur_px, blur_px), 0)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)  # not HSV_FULL: hue stays 0-179
    mask = np.zeros(hsv.shape[:2], np.uint8)
    for r in ranges:
        for lo, hi in r.bands():
            mask |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
    return mask


def clean_mask(mask: np.ndarray, open_px: int, close_px: int) -> np.ndarray:
    """Open (remove specks), then close (bridge chalk and shadow gaps)."""
    ellipse = cv2.MORPH_ELLIPSE  # round kernels don't leave square artifacts on curved holds
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(ellipse, (open_px, open_px)))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(ellipse, (close_px, close_px)))


def number_holds(holds: list[Hold]) -> list[Hold]:
    """Renumber holds bottom to top (image y grows downward), ties left to right."""
    ordered = sorted(holds, key=lambda h: (-h.centroid[1], h.centroid[0]))
    return [replace(h, id=i) for i, h in enumerate(ordered, start=1)]


def detect_holds(image_bgr: np.ndarray, spec: HoldSpec, frame_size: tuple[int, int] | None = None) -> HoldDetection:
    """Find the holds matching ``spec`` in a BGR image of the wall.

    ``frame_size`` is the (width, height) that ``spec``'s size fractions are
    relative to, when that isn't the image: a moving camera's wall view is
    bigger than one video frame, but a hold in it is as big as in the frame.
    """
    height, width = image_bgr.shape[:2]
    ref_w, ref_h = frame_size or (width, height)
    d = min(ref_h, ref_w)
    kernels = {
        "blur_px": odd_kernel(spec.blur_frac, d),
        "open_px": odd_kernel(spec.open_frac, d),
        "close_px": odd_kernel(spec.close_frac, d),
    }

    mask = color_mask(image_bgr, spec.ranges, kernels["blur_px"])
    x0, y0, x1, y1 = 0, 0, width, height
    if spec.roi is not None:
        rx, ry, rw, rh = spec.roi
        x0, y0 = max(0, rx), max(0, ry)
        x1, y1 = min(width, rx + rw), min(height, ry + rh)
        if x0 >= x1 or y0 >= y1:
            raise ValueError(f"ROI {spec.roi} does not overlap the {width}x{height} image")
        roi_mask = np.zeros_like(mask)
        roi_mask[y0:y1, x0:x1] = 255
        mask &= roi_mask
    mask = clean_mask(mask, kernels["open_px"], kernels["close_px"])

    # Area limits are relative to the whole frame, not the ROI or the wall view:
    # a hold's size on screen doesn't depend on how much of the wall is shown.
    min_area = spec.min_area_px(ref_w, ref_h)
    max_area = spec.max_area_px(ref_w, ref_h)

    # RETR_TREE returns every outline with its nesting: blob outlines sit at
    # even depths (0, 2, ...) and the holes in them at odd depths. contourArea
    # of an outline includes its holes, so holes (a chalk patch, a bolt hole)
    # are effectively filled. A blob inside a hole is either part of the
    # enclosing hold (a chalk ring around a hold's middle: skip it) or a
    # separate hold enclosed by something too large to be a hold (a black hold
    # framed by dark wall edges, holds inside a painted frame: keep it).
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    parents = hierarchy.reshape(-1, 4)[:, 3] if hierarchy is not None else np.empty(0, int)
    areas = [float(cv2.contourArea(c)) for c in contours]

    def depth(i: int) -> int:
        d = 0
        while parents[i] >= 0:
            i, d = parents[i], d + 1
        return d

    kept: list[Hold] = []
    rejected: list[Rejected] = []
    for i, c in enumerate(contours):
        level = depth(i)
        if level % 2:
            continue  # a hole, not a blob
        if level > 0 and areas[parents[parents[i]]] <= max_area:
            continue  # inside a hole of a hold-sized blob: part of that hold
        area = areas[i]
        outline = c.reshape(-1, 2).astype(np.int32)
        if area < min_area:
            rejected.append(Rejected(outline, "too_small", area))
            continue
        if area > max_area:
            rejected.append(Rejected(outline, "too_large", area))
            continue

        eps = max(1.0, spec.approx_eps_frac * cv2.arcLength(c, True))
        poly = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
        if len(poly) < 3:
            poly = cv2.convexHull(c).reshape(-1, 2)
        m = cv2.moments(c)
        bx, by, bw, bh = (int(v) for v in cv2.boundingRect(c))
        if m["m00"] > 0:
            cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
        else:
            cx, cy = bx + bw / 2, by + bh / 2
        kept.append(
            Hold(
                id=0,
                polygon=tuple((int(x), int(y)) for x, y in poly),
                centroid=(round(cx, 1), round(cy, 1)),
                area=round(area, 1),
                bbox=(bx, by, bw, bh),
                touches_border=bx <= x0 or by <= y0 or bx + bw >= x1 or by + bh >= y1,
            )
        )

    return HoldDetection(number_holds(kept), mask, rejected, spec, kernels, frame_size)


@dataclass(frozen=True)
class HSVSample:
    hsv: HSV
    suggested: HSVRange
    note: str = ""


def sample_hsv(image_bgr: np.ndarray, x: int, y: int, radius: int = 4) -> HSVSample:
    """Typical HSV of the patch around ``(x, y)`` and a range to try for that hold color."""
    height, width = image_bgr.shape[:2]
    if not (0 <= x < width and 0 <= y < height):
        raise ValueError(f"sample point ({x}, {y}) is outside the {width}x{height} image")
    patch = image_bgr[max(0, y - radius) : y + radius + 1, max(0, x - radius) : x + radius + 1]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float64)

    # Hue is an angle: the plain average of 178 and 2 would be 90 (cyan), not red.
    # Average it as unit vectors on the circle instead. Only the more saturated
    # half of the patch counts: on a small hold or an off-center click, some
    # pixels are wall, and a tinted wall (plywood, warm white) would otherwise
    # drag the hue away from the hold's.
    sat_all = hsv[:, 1]
    vivid = hsv[sat_all >= np.median(sat_all), 0]
    angle = vivid * (2 * math.pi / 180)
    hue = round(math.degrees(math.atan2(np.sin(angle).mean(), np.cos(angle).mean())) / 2) % 180
    sat, val = int(np.median(sat_all)), int(np.median(hsv[:, 2]))

    if val < 60:
        note = "very dark: hue is unreliable here, so the suggestion matches by darkness only"
        suggested = HSVRange((0, 0, 0), (179, 255, min(255, val + 30)))
    elif sat < 40:
        note = "low saturation (white/grey): hue is unreliable here, so the suggestion matches by brightness only"
        suggested = HSVRange((0, 0, max(0, val - 50)), (179, min(255, sat + 40), 255))
    else:
        note = ""
        suggested = HSVRange(((hue - 10) % 180, max(30, sat // 2), max(30, int(val * 0.4))), ((hue + 10) % 180, 255, 255))
    return HSVSample((hue, sat, val), suggested, note)


# --- debug image ------------------------------------------------------------

PANEL_MAX_WIDTH = 1200  # each panel is downscaled to this, so labels stay readable on 4K input
KEPT_COLOR: BGR = (0, 220, 0)
REJECTED_COLOR: BGR = (0, 0, 255)
ROI_COLOR: BGR = (0, 220, 255)
SAMPLE_COLOR: BGR = (255, 255, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def outline_color(spec: HoldSpec) -> BGR:
    """A bright color opposite the route color on the hue wheel, so outlines stand out on the holds."""
    if spec.color in UNRELIABLE_COLORS:
        return (255, 0, 255)
    h0, h1 = spec.ranges[0].lower[0], spec.ranges[0].upper[0]
    mid = (h0 + h1) / 2 if h0 <= h1 else ((h0 + h1 + 180) / 2) % 180
    complement = int((mid + 90) % 180)
    b, g, r = cv2.cvtColor(np.uint8([[[complement, 255, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return int(b), int(g), int(r)


def put_label(img: np.ndarray, text: str, center: tuple[float, float], scale: float = 0.5, color: BGR = (255, 255, 255)) -> None:
    """Text centered on ``center`` with a dark outline, readable on any background."""
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, 1)
    org = (int(center[0] - tw / 2), int(center[1] + th / 2))
    cv2.putText(img, text, org, FONT, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, color, 1, cv2.LINE_AA)


def dashed_polyline(img: np.ndarray, pts: np.ndarray, color: BGR, thickness: int = 2, dash: int = 8) -> None:
    """Closed polyline drawn as dashes (cv2 has no dashed line style)."""
    pts = np.asarray(pts, np.float64).reshape(-1, 2)
    for a, b in zip(pts, np.roll(pts, -1, axis=0)):
        length = float(np.hypot(*(b - a)))
        steps = max(1, int(length // dash))
        for i in range(0, steps, 2):
            p = a + (b - a) * (i / steps)
            q = a + (b - a) * (min(i + 1, steps) / steps)
            cv2.line(img, (round(p[0]), round(p[1])), (round(q[0]), round(q[1])), color, thickness, cv2.LINE_AA)


def _fit_text(lines: list[str], width: int, scale: float = 0.5) -> float:
    widest = max(cv2.getTextSize(line, FONT, scale, 1)[0][0] for line in lines)
    return scale if widest <= width else max(0.3, scale * width / widest)


def draw_crosshair(img: np.ndarray, center: tuple[float, float], color: BGR = SAMPLE_COLOR, size: int = 10) -> None:
    x, y = round(center[0]), round(center[1])
    cv2.line(img, (x - size, y), (x + size, y), (0, 0, 0), 3)
    cv2.line(img, (x, y - size), (x, y + size), (0, 0, 0), 3)
    cv2.line(img, (x - size, y), (x + size, y), color, 1)
    cv2.line(img, (x, y - size), (x, y + size), color, 1)


def draw_holds_debug(
    image_bgr: np.ndarray, det: HoldDetection, note: str = "", marks: list[tuple[int, int]] | tuple = ()
) -> np.ndarray:
    """Side-by-side debug image.

    Left: the frame with numbered hold outlines (dashed = cut off by the frame
    edge). Right: the cleaned color mask, with kept blobs outlined in green and
    rejected ones in red. Blobs just under the size limit are circled in red on
    both panels; ``marks`` (e.g. a --sample point, in full-resolution pixels)
    get a crosshair. A header lists the settings and counts.
    """
    height, width = image_bgr.shape[:2]
    s = min(1.0, PANEL_MAX_WIDTH / width)
    size = (max(1, round(width * s)), max(1, round(height * s)))
    left = cv2.resize(image_bgr, size, interpolation=cv2.INTER_AREA) if s < 1 else image_bgr.copy()
    mask = cv2.resize(det.mask, size, interpolation=cv2.INTER_NEAREST) if s < 1 else det.mask
    right = np.zeros_like(left)
    right[mask > 0] = (110, 110, 110)

    def scaled(points: Any) -> np.ndarray:
        return (np.asarray(points, np.float64).reshape(-1, 2) * s).round().astype(np.int32)

    color = outline_color(det.spec)
    for hold in det.holds:
        pts = scaled(hold.polygon)
        if hold.touches_border:
            dashed_polyline(left, pts, color, 2)
        else:
            cv2.polylines(left, [pts], True, color, 2, cv2.LINE_AA)
        cv2.polylines(right, [pts], True, KEPT_COLOR, 2, cv2.LINE_AA)
    for r in det.rejected:
        pts = scaled(r.contour)
        cv2.polylines(right, [pts], True, REJECTED_COLOR, 1, cv2.LINE_AA)
        if r.reason == "too_large":
            # Anchor on the blob's topmost point: the vertex mean can fall in a hole,
            # e.g. the middle of a ring, on top of a hold that was kept.
            top = pts[pts[:, 1].argmin()]
            put_label(right, "too large", (float(top[0]), float(top[1]) + 12), 0.5, REJECTED_COLOR)
    near = det.near_limit()
    for r in near:
        (cx, cy), radius = cv2.minEnclosingCircle(scaled(r.contour).astype(np.float32))
        for panel in (left, right):
            cv2.circle(panel, (round(cx), round(cy)), round(radius) + 6, REJECTED_COLOR, 1, cv2.LINE_AA)
    if det.spec.roi is not None:
        rx, ry, rw, rh = det.spec.roi
        for panel in (left, right):
            cv2.rectangle(panel, (round(rx * s), round(ry * s)), (round((rx + rw) * s), round((ry + rh) * s)), ROI_COLOR, 1)
    for mx, my in marks:
        for panel in (left, right):
            draw_crosshair(panel, (mx * s, my * s))
    # Labels go on last so no line is drawn across a number.
    for hold in det.holds:
        center = (hold.centroid[0] * s, hold.centroid[1] * s)
        put_label(left, str(hold.id), center, 0.55)
        put_label(right, str(hold.id), center, 0.45)

    gap = np.full((left.shape[0], 8, 3), 30, np.uint8)
    body = np.hstack([left, gap, right])
    counts = det.rejected_counts()
    lines = [
        f"{det.spec.color}: {det.spec.describe_ranges()}    kept {len(det.holds)} holds    "
        f"rejected {counts['too_small']} too small, {counts['too_large']} too large",
        "left: holds numbered bottom to top (dashed = cut off by frame edge)    "
        "right: color mask, green = kept, red = rejected",
        "--sample X,Y takes pixel coordinates in background.png (full resolution), not in this image",
    ]
    if near:
        lines.insert(
            -1,
            f"red circles: {len(near)} {'blob' if len(near) == 1 else 'blobs'} just under the size limit "
            f"({det.min_area_px:.0f} px^2), maybe footholds: lower --min-area-frac to keep them",
        )
    if marks:
        lines[-1] = "crosshair = --sample point.  " + lines[-1]
    if note:
        lines.insert(0, note)
    scale = _fit_text(lines, body.shape[1] - 20)
    line_h = int(28 * scale / 0.5)
    header = np.full((line_h * len(lines) + 12, body.shape[1], 3), 30, np.uint8)
    for i, line in enumerate(lines):
        cv2.putText(header, line, (10, line_h * (i + 1)), FONT, scale, (235, 235, 235), 1, cv2.LINE_AA)
    return np.vstack([header, body])


# --- JSON -------------------------------------------------------------------


def hold_to_dict(hold: Hold) -> dict[str, Any]:
    return {
        "id": hold.id,
        "polygon": [list(p) for p in hold.polygon],
        "centroid": list(hold.centroid),
        "area": hold.area,
        "bbox": list(hold.bbox),
        "touches_border": hold.touches_border,
    }


def hold_from_dict(d: dict[str, Any]) -> Hold:
    return Hold(
        id=int(d["id"]),
        polygon=tuple((int(x), int(y)) for x, y in d["polygon"]),
        centroid=(float(d["centroid"][0]), float(d["centroid"][1])),
        area=float(d["area"]),
        bbox=tuple(int(v) for v in d["bbox"]),
        touches_border=bool(d["touches_border"]),
    )


def detection_to_dict(det: HoldDetection, source: dict[str, Any], camera: dict[str, Any] | None = None) -> dict[str, Any]:
    """``camera`` (from ``align.camera_ref``) records which wall coordinates the polygons are in."""
    spec = det.spec
    return {
        "source": source,
        **({"camera": camera} if camera is not None else {}),
        "params": {
            "color": spec.color,
            "hsv_ranges": [{"lower": list(r.lower), "upper": list(r.upper)} for r in spec.ranges],
            "min_area_frac": spec.min_area_frac,
            "max_area_frac": spec.max_area_frac,
            "blur_frac": spec.blur_frac,
            "open_frac": spec.open_frac,
            "close_frac": spec.close_frac,
            "approx_eps_frac": spec.approx_eps_frac,
            "roi": list(spec.roi) if spec.roi else None,
            "min_area_px": round(det.min_area_px, 1),
            "max_area_px": round(det.max_area_px, 1),
            **det.kernels,
        },
        "holds": [hold_to_dict(h) for h in det.holds],
        "rejected_counts": det.rejected_counts(),
    }


def save_holds(path: str | Path, det: HoldDetection, source: dict[str, Any], camera: dict[str, Any] | None = None) -> Path:
    return save_json(path, detection_to_dict(det, source, camera))


def load_holds(path: str | Path) -> tuple[list[Hold], dict[str, Any]]:
    """Holds plus the full JSON document (source and params) from ``holds.json``."""
    data = load_json(path)
    return [hold_from_dict(h) for h in data["holds"]], data
