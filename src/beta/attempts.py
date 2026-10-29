"""Splitting a clip into attempts, and how each one ended.

A clip often holds more than one go at the problem, with the climber walking
back and standing around in between. An attempt runs from both feet leaving
the floor until the climber tops out, falls or climbs down. Only contacts
inside an attempt count, so a foot brushing a hold while walking back to the
wall doesn't end up in the hold sequence. Distances are in torso lengths and
times in seconds, like in :mod:`beta.contacts`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

from beta.holds import Hold

if TYPE_CHECKING:  # contacts.py imports this module
    from beta.contacts import Contact

TOP_MATCH_S = 1.0  # both hands on the top hold together this long: topped
DROP_TORSOS = 1.0  # mid-hip dropping this far ...
DROP_WINDOW_S = 0.5  # ... within this long is a fall (or a jump down)
MAX_GAP_S = 0.5  # hip gaps up to this long are bridged: a fast fall often blurs the pose away
LANDED_S = 2.0  # a fall lands (feet on the floor, or out of view) within this long
LAST_HOLDS_S = 1.5  # contacts ending this long before the fall count as the holds it fell from
FLOOR_TOLERANCE = 0.15  # torso lengths: how flat the floor is taken to be
SETTLE_S = 0.25  # feet on (or off) the floor this long to count, so a single bad frame doesn't


@dataclass(frozen=True)
class Release:
    limb: str
    hold: int
    frame: int  # the contact's last frame


@dataclass(frozen=True)
class Fall:
    frame: int  # the last hand release before the drop
    last_holds: dict[str, int | None]  # per limb, the hold it was on just before (None: none recently)
    release_order: tuple[Release, ...]  # those contacts, the first to let go first


@dataclass(frozen=True)
class Attempt:
    start_frame: int
    end_frame: int  # inclusive
    outcome: str  # "topped" | "fell" | "came_down" | "clip_ended"
    high_point: int | None  # highest hold id a hand reached
    contacts: dict[str, list[Contact]]  # per limb, clipped to the attempt
    fall: Fall | None = None

    def sequence(self) -> dict[str, list[int]]:
        """Each limb's holds in order. Letting go and grabbing the same hold again (a regrip) counts once."""
        out = {}
        for limb, cs in self.contacts.items():
            holds: list[int] = []
            for c in cs:
                if not holds or holds[-1] != c.hold:
                    holds.append(c.hold)
            out[limb] = holds
        return out


def _first_run(flags: np.ndarray, start: int, length: int) -> int | None:
    """First index >= ``start`` that begins ``length`` True values in a row."""
    run = 0
    for i in range(max(0, start), len(flags)):
        run = run + 1 if flags[i] else 0
        if run >= length:
            return i - length + 1
    return None


def estimate_floor(
    feet: np.ndarray, feet_seen: np.ndarray, hands_busy: np.ndarray, holds: Sequence[Hold], torso: float, fps: float
) -> float | None:
    """The floor's y in wall coordinates, from where the feet stand still with the hands free.

    ``feet`` is (F, 2, 2): left and right foot points. Candidates are frames
    where a foot is seen, still, below every hold, and no hand is on a hold
    (so the climber isn't on the wall). They are grouped into levels
    ``FLOOR_TOLERANCE`` torso lengths tall, and the floor is the highest
    level with enough of them (``SETTLE_S`` worth of frames).

    Why the highest well-supported level and not, say, a percentile: someone
    walking towards the camera has their feet lower in the image only because
    they're closer to it. On the real clip that walk is much longer than any
    rest at the wall, so it would drag a percentile down. The floor at the
    wall is the highest place feet stand still when they're not on a hold.

    None if feet never rest like that (the clip starts and ends on the wall).
    """
    from beta.contacts import ContactParams, limb_speed  # contacts.py imports this module

    lowest_hold = max((h.bbox[1] + h.bbox[3] for h in holds), default=-np.inf)
    max_speed = ContactParams().max_speed
    candidates = []
    for k in range(2):
        speed = limb_speed(feet[:, k], fps, torso)
        y = feet[:, k, 1]
        with np.errstate(invalid="ignore"):  # NaN points are never candidates
            ok = feet_seen[:, k] & ~hands_busy & (speed < max_speed) & (y > lowest_hold)
        candidates.append(y[ok])
    ys = np.concatenate(candidates)
    if not ys.size:
        return None
    level_h = FLOOR_TOLERANCE * torso
    levels = np.floor(ys / level_h).astype(int)
    need = math.ceil(SETTLE_S * fps)
    for level in sorted(set(levels.tolist())):  # top of the image down
        if (levels == level).sum() >= need:
            return float(np.median(ys[np.abs(levels - level) <= 1]))
    return None


def feet_on_floor(feet: np.ndarray, feet_seen: np.ndarray, feet_busy: np.ndarray, floor_y: float, torso: float) -> np.ndarray:
    """(F, 2): whether each foot is on the floor.

    A foot is on the floor when it is seen, not on a hold, and no higher than
    a little above the floor. While a foot is unseen it keeps its last state:
    a foot hidden behind the other leg hasn't left the floor.
    """
    out = np.zeros(feet_seen.shape, bool)
    low_enough = floor_y - FLOOR_TOLERANCE * torso
    for k in range(2):
        state = False
        for t in range(len(out)):
            if feet_seen[t, k]:
                state = bool(not feet_busy[t, k] and feet[t, k, 1] >= low_enough)
            out[t, k] = state
    return out


def _fill_gaps(y: np.ndarray, max_gap: int) -> np.ndarray:
    """``y`` with runs of up to ``max_gap`` NaNs between two values filled by a straight line."""
    out = y.copy()
    known = np.flatnonzero(np.isfinite(y))
    for a, b in zip(known[:-1], known[1:]):
        if 1 < b - a <= max_gap + 1:
            out[a + 1 : b] = np.linspace(y[a], y[b], b - a + 1)[1:-1]
    return out


def _topped(contacts: dict[str, list[Contact]], top: int | None, start: int, need: int) -> int | None:
    """The frame where both hands have matched on the top hold for ``need`` frames."""
    best = None
    for a in contacts.get("LH", []):
        for b in contacts.get("RH", []):
            if a.hold != top or b.hold != top:
                continue
            s, e = max(a.start_frame, b.start_frame, start), min(a.end_frame, b.end_frame)
            if e - s + 1 >= need and (best is None or s + need - 1 < best):
                best = s + need - 1
    return best


def _fall(
    y: np.ndarray, raw_y: np.ndarray, start: int, hands: list[Contact], on_floor: np.ndarray | None, fps: float, torso: float
) -> tuple[int, int] | None:
    """(drop start, fall frame) of the first fall after ``start``, or None.

    A fall is the hips dropping ``DROP_TORSOS`` within ``DROP_WINDOW_S``,
    with no hand still on a hold by the end of that window (otherwise it was
    feet cutting loose while the hands held on) and a landing soon after: a
    foot on the floor, or the climber out of view. The fall frame is the
    last hand release around the drop, which is when the climber came off.
    """
    window, landed, back = round(DROP_WINDOW_S * fps), round(LANDED_S * fps), round(LAST_HOLDS_S * fps)
    for t in range(start + 1, len(y)):
        if not np.isfinite(y[t]):
            continue
        with np.errstate(invalid="ignore"):
            dropped = (y[t + 1 : t + window + 1] - y[t] >= DROP_TORSOS * torso).any()
        if not dropped:
            continue
        check = t + window
        if any(c.start_frame <= check <= c.end_frame for c in hands):
            continue
        after = slice(t, min(len(y), t + landed + 1))
        if not (on_floor is None or on_floor[after].any() or np.isnan(raw_y[after]).any()):
            continue
        ends = [c.end_frame for c in hands if t - back <= c.end_frame <= t + window]
        return t, max(start, max(ends) if ends else t)
    return None


def _clip(contacts: dict[str, list[Contact]], start: int, end: int) -> dict[str, list[Contact]]:
    return {
        limb: [replace(c, start_frame=max(c.start_frame, start), end_frame=min(c.end_frame, end)) for c in cs if c.end_frame >= start and c.start_frame <= end]
        for limb, cs in contacts.items()
    }


def _fall_details(contacts: dict[str, list[Contact]], frame: int, fps: float) -> Fall:
    back = round(LAST_HOLDS_S * fps)
    last: dict[str, int | None] = {}
    released = []
    for limb, cs in contacts.items():
        recent = [c for c in cs if c.end_frame >= frame - back and c.start_frame <= frame]
        last[limb] = recent[-1].hold if recent else None
        if recent:
            released.append(Release(limb, recent[-1].hold, recent[-1].end_frame))
    return Fall(frame, last, tuple(sorted(released, key=lambda r: r.frame)))


def _attempt_from(
    start: int,
    contacts: dict[str, list[Contact]],
    on_floor: np.ndarray | None,
    y: np.ndarray,
    raw_y: np.ndarray,
    top: int | None,
    torso: float,
    fps: float,
) -> Attempt:
    """The attempt starting at ``start``: it ends at the first of topped, fell or came down."""
    settle = math.ceil(SETTLE_S * fps)
    hands = [c for limb in ("LH", "RH") for c in contacts.get(limb, []) if c.end_frame >= start]
    ends: list[tuple[int, int, str]] = []  # (frame, priority, outcome)
    topped = _topped(contacts, top, start, math.ceil(TOP_MATCH_S * fps))
    if topped is not None:
        ends.append((topped, 0, "topped"))
    fall = _fall(y, raw_y, start, hands, on_floor, fps, torso)
    if fall is not None:
        ends.append((fall[1], 1, "fell"))
    if on_floor is not None:
        down = _first_run(on_floor.all(axis=1), start, settle)
        if down is not None:
            ends.append((down, 2, "came_down"))
    end, _, outcome = min(ends, default=(len(y) - 1, 3, "clip_ended"))
    clipped = _clip(contacts, start, end)
    hand_holds = [c.hold for limb in ("LH", "RH") for c in clipped.get(limb, [])]
    details = _fall_details(clipped, end, fps) if outcome == "fell" else None
    return Attempt(start, end, outcome, max(hand_holds, default=None), clipped, details)


def find_attempts(
    contacts: dict[str, list[Contact]],
    on_floor: np.ndarray | None,
    hips: np.ndarray,
    holds: Sequence[Hold],
    torso: float,
    fps: float,
    top_hold: int | None = None,
) -> list[Attempt]:
    """Split the clip into attempts, each with its outcome.

    - **Start:** both feet off the floor for ``SETTLE_S``. Without a floor
      estimate there is one attempt, from the first contact.
    - **End**, whichever comes first:
      - topped: both hands on the top hold (``top_hold``, by default the
        highest-numbered hold) together for ``TOP_MATCH_S``;
      - fell: see :func:`_fall`;
      - came down: both feet back on the floor for ``SETTLE_S``;
      - otherwise the clip ends.
    - The next attempt is looked for only once both feet have been back on
      the floor, so walking back to the wall starts nothing.

    Attempts with no contacts at all are dropped.
    """
    n = len(hips)
    settle = math.ceil(SETTLE_S * fps)
    top = top_hold if top_hold is not None else max((h.id for h in holds), default=None)
    raw_y = hips[:, 1]
    y = _fill_gaps(raw_y, round(MAX_GAP_S * fps))

    def has_contacts(a: Attempt) -> bool:
        return any(a.contacts.values())

    if on_floor is None:
        starts = [c.start_frame for cs in contacts.values() for c in cs]
        if not starts:
            return []
        attempt = _attempt_from(min(starts), contacts, None, y, raw_y, top, torso, fps)
        return [attempt] if has_contacts(attempt) else []

    off, both = ~on_floor.any(axis=1), on_floor.all(axis=1)
    attempts = []
    t = 0
    while t < n:
        start = _first_run(off, t, settle)
        if start is None:
            break
        attempt = _attempt_from(start, contacts, on_floor, y, raw_y, top, torso, fps)
        if has_contacts(attempt):
            attempts.append(attempt)
        back_down = _first_run(both, attempt.end_frame, settle)
        if back_down is None:
            break
        t = back_down
    return attempts
