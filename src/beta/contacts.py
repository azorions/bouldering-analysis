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
from dataclasses import dataclass

import cv2
import numpy as np

from beta.holds import Hold


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
