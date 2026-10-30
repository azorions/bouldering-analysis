"""Hand and foot points in wall coordinates, cleaned up from the raw pose.

Contacts are decided on four points: the climber's left and right hand and
foot. They are taken in wall coordinates (see :mod:`beta.align`), so a camera
that tilts doesn't move them. Before that, two things the raw pose gets wrong
for climbing are fixed:

- MediaPipe mixes up the climber's left and right (:func:`fix_left_right`).
- The wrist and toe landmarks aren't the parts that touch a hold: fingers
  curl over a hold past the knuckles, and climbers stand on the front of the
  shoe (:func:`effector_points`).

Arrays here are (frames, 33, 5) landmark stacks, one row per frame in the
layout of ``PoseFrame.landmarks`` (x, y, z, visibility, presence), with NaN
for frames that have no pose.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from beta.align import CameraTrack, apply_h
from beta.pose import CONTACT_POINTS, PoseTrack

LIMBS = tuple(CONTACT_POINTS)  # ("LH", "RH", "LF", "RF")

# Landmark index pairs (the climber's left, right).
FACE_PAIRS = ((1, 4), (2, 5), (3, 6), (7, 8), (9, 10))
ARM_PAIRS = ((13, 14), (15, 16), (17, 18), (19, 20), (21, 22))  # elbow, wrist, pinky, index, thumb
LEG_PAIRS = ((25, 26), (27, 28), (29, 30), (31, 32))  # knee, ankle, heel, toe
TORSO_PAIRS = ((11, 12), (23, 24))  # shoulders, hips
ALL_PAIRS = FACE_PAIRS + ARM_PAIRS + LEG_PAIRS + TORSO_PAIRS

CONFIDENT_VIS = 0.5  # below this visibility MediaPipe is mostly guessing where a landmark is
GLOBAL_SWAP_FRAC = 0.7  # swap every frame's labels when more than this share of confident frames face the camera
CONTINUITY_S = 0.2  # a frame is compared with the previous clear one only if that is at most this recent
SWAP_COST_RATIO = 0.5  # swap a limb pair when crossing it over matches the last frame this much better
SWAP_MIN_APART = 0.3  # ... and the two limbs are this far apart (torso lengths); closer, a swap is a coin toss
HAND_REACH = 1.2  # hand point: this far along wrist -> knuckles
FOREARM_REACH = 0.4  # without knuckles: this far past the wrist along elbow -> wrist
TOE_TO_HEEL = 0.15  # foot point: this far from the toe towards the heel


def wall_landmarks(track: PoseTrack, camera: CameraTrack) -> np.ndarray:
    """Every frame's landmarks with x, y mapped into wall coordinates, (F, 33, 5).

    Row ``i`` is video frame ``i``. Frames with no pose, or whose camera
    transform was lost, are all NaN.
    """
    out = np.full((len(track.frames), 33, 5), np.nan)
    for p in track.frames:
        if p.landmarks is None or not 0 <= p.frame < len(out):
            continue
        to_wall = camera.frame_to_wall(p.frame)
        if to_wall is None:
            continue
        out[p.frame] = p.landmarks
        out[p.frame, :, :2] = apply_h(to_wall, p.landmarks[:, :2])
    return out


def hip_centers(lm: np.ndarray) -> np.ndarray:
    """Mid-hip per frame, (F, 2)."""
    return (lm[:, 23, :2] + lm[:, 24, :2]) / 2


def torso_length(lm: np.ndarray) -> float:
    """The climber's size: median mid-shoulder to mid-hip distance over the frames that show both.

    Distances and speeds downstream are measured in torso lengths, so the
    thresholds don't depend on resolution or on how far away the camera is.
    The median ignores the odd frame where the pose is badly off.
    """
    shoulders = (lm[:, 11, :2] + lm[:, 12, :2]) / 2
    d = np.linalg.norm(shoulders - hip_centers(lm), axis=1)
    d = d[np.isfinite(d)]
    if not d.size:
        raise ValueError("no frame shows the climber's shoulders and hips")
    return float(np.median(d))


def _swap(lm: np.ndarray, frames: Any, pairs: tuple[tuple[int, int], ...]) -> None:
    for left, right in pairs:
        lm[frames, [left, right]] = lm[frames, [right, left]]  # fancy indexing copies the right-hand side first


def fix_left_right(lm: np.ndarray, torso: float, fps: float) -> tuple[np.ndarray, dict[str, Any]]:
    """A copy of ``lm`` with MediaPipe's left/right mix-ups undone, plus a report of what changed.

    MediaPipe learned mostly from people facing the camera, and a climber seen
    from behind looks much like a mirrored person facing it. So it sometimes
    labels the whole body the wrong way round, and sometimes flips just the
    arms or legs for a few frames. Two fixes:

    - **Whole clip.** A climber faces the wall, so their right shoulder and hip
      are on the image right. If most frames where the torso is clearly seen
      say otherwise, every label in every frame is swapped.
    - **Frame by frame.** If crossing over the two wrists (or ankles) matches
      the previous frame far better than keeping them, that arm (or leg) pair
      is swapped. Not when the two are close together: then either labelling
      fits about as well, and swapping would flip-flop. Only frames where both
      are clearly seen are compared.
    """
    out = lm.copy()
    facing = (out[:, 12, 0] - out[:, 11, 0]) + (out[:, 24, 0] - out[:, 23, 0])
    confident = (out[:, [11, 12, 23, 24], 3] >= CONFIDENT_VIS).all(axis=1) & np.isfinite(facing)
    n = int(confident.sum())
    global_swap = bool(n and (facing[confident] <= 0).sum() > GLOBAL_SWAP_FRAC * n)
    if global_swap:
        _swap(out, slice(None), ALL_PAIRS)

    swapped = {"arms": 0, "legs": 0}
    max_gap = math.ceil(CONTINUITY_S * fps)
    for chain, (l_i, r_i), pairs in (("arms", (15, 16), ARM_PAIRS), ("legs", (27, 28), LEG_PAIRS)):
        prev = None
        for t in range(len(out)):
            # Only frames where both wrists (ankles) are clearly seen take part. Where
            # MediaPipe is guessing, a wrong label would make the next good frame look
            # crossed, and its "fix" would then carry on to every frame after it.
            # Guessed points don't count as seen for contacts anyway.
            if not (out[t, [l_i, r_i], 3] >= CONFIDENT_VIS).all():  # also skips frames with no pose (NaN)
                continue
            if prev is not None and t - prev <= max_gap:
                left, right = out[t, l_i, :2], out[t, r_i, :2]
                left_before, right_before = out[prev, l_i, :2], out[prev, r_i, :2]  # already fixed
                keep = np.linalg.norm(left - left_before) + np.linalg.norm(right - right_before)
                cross = np.linalg.norm(left - right_before) + np.linalg.norm(right - left_before)
                if cross < SWAP_COST_RATIO * keep and np.linalg.norm(left - right) > SWAP_MIN_APART * torso:
                    _swap(out, t, pairs)
                    swapped[chain] += 1
            prev = t
    return out, {"global_swap": global_swap, "frames_swapped": swapped}


def effector_points(lm: np.ndarray, min_visibility: float = CONFIDENT_VIS) -> tuple[np.ndarray, np.ndarray]:
    """The point of each hand and foot that touches a hold, (F, 4, 2) in ``LIMBS`` order, and whether it is seen, (F, 4).

    - Hand: past the knuckles (the index and pinky landmarks), where curled
      fingers wrap over a hold. If the knuckles are hidden, a little past the
      wrist along the forearm instead.
    - Foot: the front of the shoe, just behind the toe landmark.

    A point is seen when its wrist or toe landmark has at least
    ``min_visibility``: below that MediaPipe still places it, but mostly guesses.
    """
    pts = np.full((len(lm), 4, 2), np.nan)
    seen = np.zeros((len(lm), 4), bool)
    for k, (wrist, elbow, index, pinky) in enumerate(((15, 13, 19, 17), (16, 14, 20, 18))):
        w = lm[:, wrist, :2]
        knuckles = (lm[:, index, :2] + lm[:, pinky, :2]) / 2
        fingers = (lm[:, [index, pinky], 3] >= CONFIDENT_VIS).all(axis=1) & np.isfinite(knuckles).all(axis=1)
        forearm = w + FOREARM_REACH * (w - lm[:, elbow, :2])
        forearm = np.where(np.isfinite(forearm), forearm, w)  # no elbow either: the wrist itself
        pts[:, k] = np.where(fingers[:, None], w + HAND_REACH * (knuckles - w), forearm)
        seen[:, k] = lm[:, wrist, 3] >= min_visibility
    for k, (toe, heel) in enumerate(((31, 29), (32, 30)), start=2):
        t = lm[:, toe, :2]
        front = t + TOE_TO_HEEL * (lm[:, heel, :2] - t)
        pts[:, k] = np.where(np.isfinite(front), front, t)
        seen[:, k] = lm[:, toe, 3] >= min_visibility
    return pts, seen & np.isfinite(pts).all(axis=2)
