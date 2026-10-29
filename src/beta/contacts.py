"""The contacts stage: which hold each hand and foot is on, frame by frame.

A limb is on a hold when its point (see :mod:`beta.limbs`) sits on the
hold's outline, or just outside it, and stays there, slow, for a moment. The
"slow" part matters: a hand reaching past a hold crosses it for a few frames
without using it. Distances are in torso lengths and times in seconds, so the
same settings work at any resolution, camera distance and frame rate.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from beta.align import CameraTrack
from beta.attempts import (
    DROP_TORSOS,
    DROP_WINDOW_S,
    FLOOR_TOLERANCE,
    LANDED_S,
    LAST_HOLDS_S,
    MAX_GAP_S,
    SETTLE_S,
    TOP_MATCH_S,
    Attempt,
    estimate_floor,
    feet_on_floor,
    find_attempts,
)
from beta.holds import FONT, Hold, put_label
from beta.jsonio import save_json
from beta.limbs import LIMBS, effector_points, fix_left_right, hip_centers, torso_length, wall_landmarks
from beta.pose import LEFT_COLOR, RIGHT_COLOR, PoseTrack


@dataclass(frozen=True)
class ContactParams:
    min_hold_s: float = 0.25  # slow on a hold this long before it counts as a contact
    release_s: float = 0.15  # off the hold this long before the contact ends
    onset_margin: float = 0.05  # a contact can start this close outside a hold's outline (torso lengths)
    release_margin: float = 0.25  # ... and only ends beyond this: a hand shifting its grip stays on the hold
    max_speed: float = 0.5  # torso lengths per second; faster frames are a limb passing by, not holding
    speed_window_s: float = 0.1
    min_visibility: float = 0.5

    def onset_frames(self, fps: float) -> int:
        return math.ceil(self.min_hold_s * fps)

    def release_frames(self, fps: float) -> int:
        return math.ceil(self.release_s * fps)


@dataclass(frozen=True)
class Contact:
    limb: str  # "LH" | "RH" | "LF" | "RF"
    hold: int  # hold id
    start_frame: int
    end_frame: int  # inclusive


def hold_distances(points: np.ndarray, holds: Sequence[Hold], torso: float) -> np.ndarray:
    """Signed distance from each point to each hold's outline, in torso lengths: (F, H).

    Positive inside the hold, negative outside, NaN for a NaN point.
    """
    out = np.full((len(points), len(holds)), np.nan)
    contours = [h.contour() for h in holds]
    for t, (x, y) in enumerate(points):
        if np.isfinite(x) and np.isfinite(y):
            for j, c in enumerate(contours):
                out[t, j] = cv2.pointPolygonTest(c, (float(x), float(y)), True) / torso
    return out


def limb_speed(points: np.ndarray, fps: float, torso: float, window_s: float = 0.1) -> np.ndarray:
    """Speed of a point in torso lengths per second, (F,).

    Measured over a window, ``|p[t+k] - p[t-k]|`` divided by the time between
    them, which smooths the pose's frame-to-frame jitter as it differentiates.
    At the ends of the clip the window is cut short. NaN if either end is NaN.
    """
    n = len(points)
    k = max(1, round(window_s * fps))
    t = np.arange(n)
    a, b = np.clip(t - k, 0, n - 1), np.clip(t + k, 0, n - 1)
    with np.errstate(invalid="ignore", divide="ignore"):  # a one-frame clip has a == b: NaN
        return np.linalg.norm(points[b] - points[a], axis=1) / ((b - a) / fps) / torso


def detect_contacts(
    limb: str,
    points: np.ndarray,
    seen: np.ndarray,
    holds: Sequence[Hold],
    torso: float,
    fps: float,
    params: ContactParams = ContactParams(),
) -> list[Contact]:
    """One limb's contacts, in time order.

    A small state machine, run over the frames where the limb is seen (other
    frames change nothing, so a hand hidden for a moment keeps its contact):

    - **Onset.** The point stays on the same hold (within ``onset_margin``),
      slow, for ``min_hold_s``. The contact is backdated to when the point
      arrived on the hold, not when it was confirmed.
    - **Release.** The point is beyond ``release_margin`` of the hold for
      ``release_s``. The contact ends at the last frame it was still on the
      hold. Speed never releases a contact, because hands rotate and slide
      on holds while holding them.
    - **Switch.** A new contact on another hold ends the current one.
    """
    if not holds:
        return []
    onset_n, release_n = params.onset_frames(fps), params.release_frames(fps)
    dist = hold_distances(points, holds, torso)
    speed = limb_speed(points, fps, torso, params.speed_window_s)
    ids = [h.id for h in holds]

    contacts: list[Contact] = []
    on: int | None = None  # index into holds of the current contact
    start = last_inside = out_count = 0
    run: int | None = None  # the hold the point has been arriving on
    run_start = slow_count = off_count = 0
    for t in range(len(points)):
        if not seen[t] or not np.isfinite(speed[t]) or not np.isfinite(dist[t]).all():
            continue
        d = dist[t]

        if on is not None:
            if d[on] >= -params.release_margin:
                out_count, last_inside = 0, t
            else:
                out_count += 1
                if out_count >= release_n:
                    contacts.append(Contact(limb, ids[on], start, last_inside))
                    on = None

        best = int(np.argmax(d))
        cand = best if d[best] >= -params.onset_margin else None
        if cand is None:
            # Off every hold. A single stray frame (a pose glitch) shouldn't make the
            # point "arrive" again later, but a real absence must: same tolerance as a release.
            slow_count, off_count = 0, off_count + 1
            if off_count >= release_n:
                run = None
            continue
        off_count = 0
        if cand != run:
            run, run_start, slow_count = cand, t, 0
        slow_count = slow_count + 1 if speed[t] < params.max_speed else 0
        if slow_count >= onset_n and cand != on:
            if on is not None:
                contacts.append(Contact(limb, ids[on], start, min(last_inside, run_start - 1)))
            on, start, last_inside, out_count = cand, run_start, t, 0
    if on is not None:
        contacts.append(Contact(limb, ids[on], start, last_inside))
    return contacts


def busy_frames(contacts: Sequence[Contact], frame_count: int) -> np.ndarray:
    """(F,) True on every frame some contact covers."""
    busy = np.zeros(frame_count, bool)
    for c in contacts:
        busy[max(0, c.start_frame) : min(frame_count, c.end_frame + 1)] = True
    return busy


# --- the stage ----------------------------------------------------------------


@dataclass
class ContactResult:
    fps: float
    frame_count: int
    torso_px: float
    floor_y: float | None  # wall coordinates
    floor_source: str  # "estimated" | "--floor-y" | "not found"
    pose_cleaning: dict[str, Any]  # what fix_left_right changed
    contacts: dict[str, list[Contact]]  # per limb, over the whole clip
    seen: np.ndarray  # (F, 4) in LIMBS order
    attempts: list[Attempt]
    top_hold: int | None = None  # the hold that counts as the top


def analyze(
    track: PoseTrack,
    holds: Sequence[Hold],
    camera: CameraTrack,
    params: ContactParams = ContactParams(),
    top_hold: int | None = None,
    floor_y: float | None = None,
) -> ContactResult:
    """Pose + holds + camera -> each limb's contacts, split into attempts.

    Everything happens in wall coordinates, where the holds are, so a moving
    camera doesn't move the climber relative to the holds.
    """
    fps = track.fps
    lm = wall_landmarks(track, camera)
    torso = torso_length(lm)
    lm, cleaning = fix_left_right(lm, torso, fps)
    points, seen = effector_points(lm, params.min_visibility)
    contacts = {limb: detect_contacts(limb, points[:, k], seen[:, k], holds, torso, fps, params) for k, limb in enumerate(LIMBS)}

    n = len(lm)
    if floor_y is not None:
        source = "--floor-y"
    else:
        hands_busy = busy_frames(contacts["LH"] + contacts["RH"], n)
        floor_y = estimate_floor(points[:, 2:], seen[:, 2:], hands_busy, holds, torso, fps)
        source = "estimated" if floor_y is not None else "not found"
    on_floor = None
    if floor_y is not None:
        feet_busy = np.column_stack([busy_frames(contacts["LF"], n), busy_frames(contacts["RF"], n)])
        on_floor = feet_on_floor(points[:, 2:], seen[:, 2:], feet_busy, floor_y, torso)
    attempts = find_attempts(contacts, on_floor, hip_centers(lm), holds, torso, fps, top_hold)
    top = top_hold if top_hold is not None else max((h.id for h in holds), default=None)
    return ContactResult(fps, n, torso, floor_y, source, cleaning, contacts, seen, attempts, top)


# --- JSON -------------------------------------------------------------------


def _seconds(frame: int, fps: float) -> float:
    return round(frame / fps, 2)


def _attempt_to_dict(n: int, a: Attempt, fps: float) -> dict[str, Any]:
    fall = None
    if a.fall is not None:
        fall = {
            "frame": a.fall.frame,
            "t_s": _seconds(a.fall.frame, fps),
            "last_holds": a.fall.last_holds,
            "release_order": [{"limb": r.limb, "hold": r.hold, "frame": r.frame, "t_s": _seconds(r.frame, fps)} for r in a.fall.release_order],
        }
    return {
        "attempt": n,
        "start_frame": a.start_frame,
        "start_s": _seconds(a.start_frame, fps),
        "end_frame": a.end_frame,
        "end_s": _seconds(a.end_frame, fps),
        "outcome": a.outcome,
        "high_point": a.high_point,
        "sequence": a.sequence(),
        "contacts": {
            limb: [
                {"hold": c.hold, "start_frame": c.start_frame, "end_frame": c.end_frame, "start_s": _seconds(c.start_frame, fps), "end_s": _seconds(c.end_frame, fps)}
                for c in cs
            ]
            for limb, cs in a.contacts.items()
        },
        "fall": fall,
    }


def contacts_to_dict(result: ContactResult, source: dict[str, Any], inputs: dict[str, Any], params: ContactParams) -> dict[str, Any]:
    return {
        "source": source,
        "inputs": inputs,
        "params": {
            **asdict(params),
            "top_hold": result.top_hold,
            "top_match_s": TOP_MATCH_S,
            "drop_torsos": DROP_TORSOS,
            "drop_window_s": DROP_WINDOW_S,
            "max_gap_s": MAX_GAP_S,
            "landed_s": LANDED_S,
            "last_holds_s": LAST_HOLDS_S,
            "floor_tolerance": FLOOR_TOLERANCE,
            "settle_s": SETTLE_S,
        },
        "body_scale_px": round(result.torso_px, 1),
        "floor_y": None if result.floor_y is None else round(result.floor_y, 1),
        "floor_source": result.floor_source,
        "pose_cleaning": result.pose_cleaning,
        "attempts": [_attempt_to_dict(i + 1, a, result.fps) for i, a in enumerate(result.attempts)],
    }


def save_contacts(path: str | Path, result: ContactResult, source: dict[str, Any], inputs: dict[str, Any], params: ContactParams) -> Path:
    return save_json(path, contacts_to_dict(result, source, inputs, params))


# --- debug image ------------------------------------------------------------

TIMELINE_W = 1200
LABEL_W = 48  # the limb names left of the rows
ROW_H = 40
AXIS_H = 30
LINE_H = 22
BG_COLOR = (25, 25, 25)
ATTEMPT_COLOR = (70, 52, 38)  # a dark blue tint behind each attempt
UNSEEN_COLOR = (60, 60, 60)
FALL_COLOR = (0, 0, 255)
FLOOR_COLOR = (0, 255, 255)
TEXT_COLOR = (235, 235, 235)
OUTCOME_TEXT = {"topped": "topped", "fell": "fell", "came_down": "came down", "clip_ended": "clip ended mid-attempt"}


def _runs(flags: np.ndarray) -> list[tuple[int, int]]:
    """(first, last) of each run of True."""
    out, start = [], None
    for i, f in enumerate(list(flags) + [False]):
        if f and start is None:
            start = i
        elif not f and start is not None:
            out.append((start, i - 1))
            start = None
    return out


def _header(result: ContactResult) -> list[str]:
    floor = "not found" if result.floor_y is None else f"y={result.floor_y:.0f} ({result.floor_source})"
    lines = [f"torso {result.torso_px:.0f} px   floor: {floor}"]
    for i, a in enumerate(result.attempts):
        hp = "-" if a.high_point is None else a.high_point
        lines.append(
            f"attempt {i + 1}: {OUTCOME_TEXT[a.outcome]}, {a.start_frame / result.fps:.1f}-{a.end_frame / result.fps:.1f} s, high point hold {hp}"
        )
    lines.append("blue = left, orange = right (the climber's own); grey = not seen; shaded = attempt; red = fall")
    return lines


def _timeline(result: ContactResult) -> np.ndarray:
    n, fps = max(1, result.frame_count), result.fps
    x0, x1 = LABEL_W, TIMELINE_W - 12

    def fx(frame: float) -> int:
        return int(round(x0 + (x1 - x0) * frame / n))

    header = _header(result)
    top = 12 + LINE_H * len(header)
    bottom = top + ROW_H * len(LIMBS)
    img = np.full((bottom + AXIS_H, TIMELINE_W, 3), BG_COLOR, np.uint8)
    for i, line in enumerate(header):
        cv2.putText(img, line, (10, 22 + LINE_H * i), FONT, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)
    for a in result.attempts:
        cv2.rectangle(img, (fx(a.start_frame), top), (fx(a.end_frame + 1) - 1, bottom - 1), ATTEMPT_COLOR, -1)
    for k, limb in enumerate(LIMBS):
        y = top + k * ROW_H
        color = LEFT_COLOR if limb[0] == "L" else RIGHT_COLOR
        for a, b in _runs(~result.seen[:, k]):
            cv2.rectangle(img, (fx(a), y + 4), (fx(b + 1) - 1, y + ROW_H - 5), UNSEEN_COLOR, -1)
        for c in result.contacts[limb]:
            left, right = fx(c.start_frame), fx(c.end_frame + 1) - 1
            # Filled without anti-aliasing, so the bars keep exactly the limb colours of pose_debug.mp4.
            cv2.rectangle(img, (left, y + 8), (max(left, right), y + ROW_H - 9), color, -1)
            if right - left > 18:
                put_label(img, str(c.hold), ((left + right) / 2, y + ROW_H / 2), 0.45)
        cv2.putText(img, limb, (10, y + ROW_H // 2 + 6), FONT, 0.55, color, 1, cv2.LINE_AA)
    for a in result.attempts:
        if a.fall is not None:
            x = fx(a.fall.frame)
            cv2.line(img, (x, top - 4), (x, bottom + 3), FALL_COLOR, 2)
    # Seconds axis, with a tick step that gives at most ~15 ticks.
    seconds = result.frame_count / fps
    step = next((s for s in (1, 2, 5, 10, 15, 30, 60) if seconds / s <= 15), 120)
    for s in range(0, int(seconds) + 1, step):
        x = fx(s * fps)
        cv2.line(img, (x, bottom), (x, bottom + 5), TEXT_COLOR, 1)
        cv2.putText(img, f"{s}s", (x - 8, bottom + 22), FONT, 0.45, TEXT_COLOR, 1, cv2.LINE_AA)
    return img


def _wall_panel(result: ContactResult, holds: Sequence[Hold], background: np.ndarray | None) -> np.ndarray:
    if background is None:  # no image of the wall: a dark panel big enough for the holds
        w = max((h.bbox[0] + h.bbox[2] for h in holds), default=640) + 20
        h = max((h.bbox[1] + h.bbox[3] for h in holds), default=480) + 20
        background = np.full((h, w, 3), BG_COLOR, np.uint8)
    s = TIMELINE_W / background.shape[1]
    panel = (cv2.resize(background, (TIMELINE_W, round(background.shape[0] * s)), interpolation=cv2.INTER_AREA) * 0.6).astype(np.uint8)
    sides: dict[int, set[str]] = {}
    for a in result.attempts:
        for limb, cs in a.contacts.items():
            for c in cs:
                sides.setdefault(c.hold, set()).add(limb[0])
    for hold in holds:
        if hold.id not in sides:
            continue
        side = sides[hold.id]
        color = LEFT_COLOR if side == {"L"} else RIGHT_COLOR if side == {"R"} else (255, 255, 255)
        pts = (np.asarray(hold.polygon, np.float64) * s).round().astype(np.int32)
        cv2.polylines(panel, [pts], True, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.polylines(panel, [pts], True, color, 2, cv2.LINE_AA)
        put_label(panel, str(hold.id), (hold.centroid[0] * s, hold.centroid[1] * s), 0.6, color)
    by_id = {h.id: h for h in holds}
    for a in result.attempts:
        if a.high_point in by_id:
            h = by_id[a.high_point]
            r = int(max(h.bbox[2], h.bbox[3]) * s / 2 + 12)
            cv2.circle(panel, (round(h.centroid[0] * s), round(h.centroid[1] * s)), r, FLOOR_COLOR, 2, cv2.LINE_AA)
    if result.floor_y is not None:
        y = round(result.floor_y * s)
        cv2.line(panel, (0, y), (TIMELINE_W - 1, y), FLOOR_COLOR, 2)
        cv2.putText(panel, "floor", (8, y - 6), FONT, 0.5, FLOOR_COLOR, 1, cv2.LINE_AA)
    return panel


def draw_contacts_debug(result: ContactResult, holds: Sequence[Hold], background: np.ndarray | None = None) -> np.ndarray:
    """The timeline of every limb's contacts, above the wall with the holds used.

    Timeline: one row per limb, a bar per contact labelled with its hold, grey
    where the limb wasn't seen (which explains most missing contacts), attempts
    shaded and falls marked in red. Wall: the holds used in an attempt in their
    limb's colour (white if both sides used it), the high point ringed and the
    floor in yellow.
    """
    return np.vstack([_timeline(result), _wall_panel(result, holds, background)])
