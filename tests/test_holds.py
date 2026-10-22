from __future__ import annotations

import cv2
import numpy as np
import pytest
from conftest import FOOT_CHIPS, expected_order, foot_chip_wall, inside, match

from beta.holds import (
    PRESETS,
    Hold,
    HoldSpec,
    HSVRange,
    detect_holds,
    draw_holds_debug,
    load_holds,
    number_holds,
    odd_kernel,
    sample_hsv,
    save_holds,
)
from beta.synthetic import HOLD_HSV, hsv_to_bgr, make_wall


def detect(image: np.ndarray, color: str, **kwargs) -> list[Hold]:
    return detect_holds(image, HoldSpec.from_color(color, **kwargs)).holds


def test_finds_every_route_hold_and_nothing_else(wall):
    image, truth = wall
    holds = detect(image, "green")
    green = truth.of_color("green")
    assert len(holds) == len(green)
    hits = match(holds, green)
    assert all(len(ids) == 1 for ids in hits), hits
    assert sorted(i for ids in hits for i in ids) == [h.id for h in holds]  # one-to-one


@pytest.mark.parametrize("color", ["red", "orange", "yellow", "blue", "purple", "pink"])
def test_other_presets_find_only_their_color(wall, color):
    image, truth = wall
    holds = detect(image, color)
    expected = truth.of_color(color)
    assert len(holds) == len(expected)
    assert all(len(ids) == 1 for ids in match(holds, expected))


def test_holds_numbered_bottom_to_top(wall):
    image, truth = wall
    holds = detect(image, "green")
    assert [h.id for h in holds] == list(range(1, len(holds) + 1))
    ys = [h.centroid[1] for h in holds]
    assert ys == sorted(ys, reverse=True)
    # and the numbering matches the true layout, hold for hold
    for i, t in enumerate(expected_order(truth.of_color("green")), start=1):
        assert match(holds, [t]) == [[i]]


def test_equal_heights_numbered_left_to_right():
    def hold(x: float, y: float) -> Hold:
        return Hold(0, ((0, 0), (1, 0), (0, 1)), (x, y), 1.0, (0, 0, 1, 1), False)

    numbered = number_holds([hold(50, 100), hold(10, 100), hold(30, 200)])
    assert [(h.id, h.centroid) for h in numbered] == [(1, (30, 200)), (2, (10, 100)), (3, (50, 100))]


def test_red_wraps_around_the_hue_scale(wall):
    image, truth = wall
    holds = detect(image, "red")
    red = truth.of_color("red")
    assert {t.scene_color for t in red} == {"red_low", "red_high"}
    assert all(len(ids) == 1 for ids in match(holds, red))


def test_specks_are_not_holds(wall):
    image, truth = wall
    holds = detect(image, "green")
    assert truth.specks
    assert not any(inside(h, s) for h in holds for s in truth.specks)


def test_chalked_hold_stays_one_hold(wall):
    image, truth = wall
    (chalked,) = [t for t in truth.holds if t.kind == "chalked"]
    assert len(match(detect(image, "green"), [chalked])[0]) == 1
    # Sanity check that the chalk band really splits the hold without the close step.
    split = detect(image, "green", close_frac=0.0)
    assert match(split, [chalked]) == [[]]


def test_shadowed_hold_detected_including_dark_side(wall):
    image, truth = wall
    (shadowed,) = [t for t in truth.holds if t.kind == "shadowed"]
    assert len(shadowed.probes) == 2  # center and a point in the shadow
    assert len(match(detect(image, "green"), [shadowed])[0]) == 1


def test_adjacent_holds_stay_separate(wall):
    image, truth = wall
    pair = [t for t in truth.holds if t.kind == "adjacent"]
    assert len(pair) == 2
    ids = match(detect(image, "green"), pair)
    assert all(len(i) == 1 for i in ids) and ids[0] != ids[1]


def test_large_region_rejected_as_too_large(wall):
    image, truth = wall
    det = detect_holds(image, HoldSpec.from_color("green"))
    (big,) = [r for r in det.rejected if r.reason == "too_large"]
    x, y, w, h = cv2.boundingRect(big.contour)
    px, py, pw, ph = truth.large_panel
    assert abs(x - px) <= 3 and abs(y - py) <= 3 and abs(w - pw) <= 6 and abs(h - ph) <= 6
    assert not any(inside(hold, (px + pw / 2, py + ph / 2)) for hold in det.holds)


def _grey_canvas(width: int = 640, height: int = 480) -> np.ndarray:
    rng = np.random.default_rng(1)
    return np.clip(125 + rng.normal(0, 4, (height, width, 3)), 0, 255).astype(np.uint8)


def test_holds_inside_a_too_large_ring_are_kept():
    # A route-colored frame (trim, tape loop, dark wall edges for black holds) encloses
    # the holds. The frame is too large to be a hold; the holds inside must survive.
    image = _grey_canvas()
    green = hsv_to_bgr(HOLD_HSV["green"])
    cv2.rectangle(image, (40, 40), (600, 440), green, 12)
    centers = [(200, 300), (320, 200), (450, 330)]
    for c in centers:
        cv2.circle(image, c, 18, green, -1)
    det = detect_holds(image, HoldSpec.from_color("green"))
    assert det.rejected_counts()["too_large"] == 1
    assert len(det.holds) == 3
    assert all(any(inside(h, c) for h in det.holds) for c in centers)


def test_chalk_ring_does_not_split_a_hold():
    # A closed ring of chalk inside a hold leaves a separate route-colored island in the
    # middle. That island is part of the same hold, not a second one.
    image = _grey_canvas()
    cv2.circle(image, (320, 240), 40, hsv_to_bgr(HOLD_HSV["green"]), -1)
    ring = np.zeros(image.shape[:2], np.uint8)
    cv2.circle(ring, (320, 240), 24, 255, 8)
    sel = ring > 0
    image[sel] = (image[sel] * 0.3 + 255 * 0.7).astype(np.uint8)
    holds = detect(image, "green")
    assert len(holds) == 1
    assert inside(holds[0], (320, 240))


def test_small_footholds_kept_by_default_and_flagged_when_limit_is_raised():
    image = foot_chip_wall()
    holds = detect(image, "green")
    assert len(holds) == len(FOOT_CHIPS)
    assert all(any(inside(h, c) for h in holds) for c in FOOT_CHIPS)
    # With a stricter limit they're rejected, but reported as near the limit rather than lost among specks.
    det = detect_holds(image, HoldSpec.from_color("green", min_area_frac=5e-5))
    assert det.holds == []
    assert len(det.near_limit()) == len(FOOT_CHIPS)


def test_border_hold_flagged_not_dropped(wall):
    image, truth = wall
    holds = detect(image, "green")
    (border,) = [t for t in truth.holds if t.kind == "border"]
    (ids,) = match(holds, [border])
    assert len(ids) == 1
    by_id = {h.id: h for h in holds}
    assert by_id[ids[0]].touches_border
    assert sum(h.touches_border for h in holds) == 1


def test_custom_hsv_range_overrides_preset(wall):
    image, truth = wall
    # The blue preset also catches the teal hold; a tighter custom range doesn't.
    assert len(detect(image, "blue")) == 2
    spec = HoldSpec.from_color("blue", hsv_lower=(100, 100, 50), hsv_upper=(120, 255, 255))
    holds = detect_holds(image, spec).holds
    (blue,) = [t for t in truth.holds if t.scene_color == "blue"]
    assert len(holds) == 1 and match(holds, [blue]) == [[1]]
    assert HoldSpec.from_color(None, (100, 100, 50), (120, 255, 255)).color == "custom"


def test_roi_limits_detection(wall):
    image, truth = wall
    h, w = image.shape[:2]
    holds = detect(image, "green", roi=(0, h // 2, w, h - h // 2))
    lower_half = [t for t in truth.of_color("green") if t.center[1] > h / 2]
    assert len(holds) == len(lower_half)
    assert all(len(ids) == 1 for ids in match(holds, lower_half))


def test_polygons_are_valid(wall):
    image, _ = wall
    for hold in detect(image, "green"):
        assert len(hold.polygon) >= 3
        assert inside(hold, hold.centroid)
        x, y, bw, bh = hold.bbox
        assert all(x <= px <= x + bw and y <= py <= y + bh for px, py in hold.polygon)


@pytest.mark.parametrize("size", [(1280, 720), (3840, 2160)])
def test_same_scene_same_result_at_any_resolution(size):
    image, truth = make_wall(*size, seed=0)
    holds = detect(image, "green")
    green = truth.of_color("green")
    assert len(holds) == len(green)
    for i, t in enumerate(expected_order(green), start=1):
        assert match(holds, [t]) == [[i]]


def test_json_round_trip(wall, tmp_path):
    image, _ = wall
    det = detect_holds(image, HoldSpec.from_color("green"))
    source = {"path": "wall.png", "kind": "image", "frame": None, "background_frames": None, "width": 1280, "height": 960}
    path = save_holds(tmp_path / "holds.json", det, source)
    holds, data = load_holds(path)
    assert holds == det.holds
    assert data["schema_version"] == 1
    assert data["source"] == source
    assert data["params"]["color"] == "green"
    assert data["params"]["hsv_ranges"] == [{"lower": [35, 70, 50], "upper": [85, 255, 255]}]
    assert data["rejected_counts"] == det.rejected_counts()


def test_sample_hsv_reads_hold_color(wall):
    image, truth = wall
    plain = [t for t in truth.of_color("green") if t.kind == "plain"][0]
    smp = sample_hsv(image, *plain.probes[0])
    h, s, v = smp.hsv
    true_h, true_s, true_v = HOLD_HSV["green"]
    assert abs(h - true_h) <= 3 and abs(s - true_s) <= 20 and abs(v - true_v) <= 25
    assert smp.suggested.contains(smp.hsv)
    assert smp.note == ""


def test_sample_hsv_handles_red_wraparound(wall):
    image, truth = wall
    (red_high,) = [t for t in truth.holds if t.scene_color == "red_high"]
    smp = sample_hsv(image, *red_high.probes[0])
    assert smp.hsv[0] >= 170
    assert smp.suggested.lower[0] > smp.suggested.upper[0]  # wrapping range
    assert smp.suggested.contains(smp.hsv)


@pytest.mark.parametrize("radius, offset", [(4, 0), (12, 9)])  # a 9 px foothold; a click near a hold's edge
def test_sample_hsv_ignores_tinted_wall_pixels(radius, offset):
    # On a plywood-colored wall, wall pixels in the 9x9 patch used to drag the hue
    # off the hold's color, so the suggested range excluded the hold.
    true_hsv = HOLD_HSV["green"]
    image = np.full((200, 200, 3), hsv_to_bgr((15, 90, 170)), np.uint8)
    cv2.circle(image, (100, 100), radius, hsv_to_bgr(true_hsv), -1)
    rng = np.random.default_rng(2)
    image = np.clip(image + rng.normal(0, 4, image.shape), 0, 255).astype(np.uint8)
    image = cv2.imdecode(cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])[1], cv2.IMREAD_COLOR)
    smp = sample_hsv(image, 100 + offset, 100)
    assert abs(smp.hsv[0] - true_hsv[0]) <= 4
    assert smp.suggested.contains(true_hsv)


def test_sample_hsv_rejects_points_outside_image(wall):
    image, _ = wall
    with pytest.raises(ValueError):
        sample_hsv(image, image.shape[1], 0)


def test_hsv_range_bands():
    assert HSVRange((35, 70, 50), (85, 255, 255)).bands() == [((35, 70, 50), (85, 255, 255))]
    assert PRESETS["red"].bands() == [((172, 100, 60), (179, 255, 255)), ((0, 100, 60), (8, 255, 255))]
    assert PRESETS["red"].contains((2, 200, 200)) and PRESETS["red"].contains((176, 200, 200))
    assert not PRESETS["red"].contains((90, 200, 200))


@pytest.mark.parametrize(
    "args",
    [
        ("chartreuse", None, None),  # unknown preset
        (None, None, None),  # nothing to detect
        ("green", (35, 70, 50), None),  # only one bound
        ("green", (200, 70, 50), (85, 255, 255)),  # hue out of range
        ("green", (35, 255, 50), (85, 70, 255)),  # S bounds swapped: would match nothing
        ("green", (35, 70, 255), (85, 255, 50)),  # V bounds swapped
    ],
)
def test_bad_specs_raise(args):
    with pytest.raises(ValueError):
        HoldSpec.from_color(*args)


def test_min_area_has_a_pixel_floor_at_low_resolution():
    spec = HoldSpec.from_color("green")
    assert spec.min_area_px(1920, 1080) == pytest.approx(2e-5 * 1920 * 1080)  # fraction decides at HD
    assert spec.min_area_px(640, 480) == 16  # 2e-5 would be only ~6 px^2: compression specks pass


def test_odd_kernel():
    assert odd_kernel(0.004, 1080) == 5
    assert odd_kernel(0.008, 1080) == 9
    assert odd_kernel(0.0, 1080) == 3
    assert all(odd_kernel(f, d) % 2 == 1 for f in (0.001, 0.0035, 0.01) for d in (240, 719, 2160))


def test_debug_image_layout(wall):
    image, _ = wall
    det = detect_holds(image, HoldSpec.from_color("green"))
    debug = draw_holds_debug(image, det, note="source: test")
    assert debug.shape[1] == 2 * 1200 + 8  # two panels capped at 1200 px wide, plus the gap
    assert debug.shape[0] > 960 * 1200 / 1280  # header strip on top of the scaled panels
    small = cv2.resize(image, (640, 480))
    small_debug = draw_holds_debug(small, detect_holds(small, HoldSpec.from_color("green")))
    assert small_debug.shape[1] == 2 * 640 + 8  # small inputs are not resized
    big = cv2.resize(image, (3840, 2880))
    big_debug = draw_holds_debug(big, detect_holds(big, HoldSpec.from_color("green")))
    assert big_debug.shape[1] <= 2 * 1200 + 8  # 4K input is downscaled for viewing


def test_debug_image_marks_sample_point(wall):
    image, _ = wall
    det = detect_holds(image, HoldSpec.from_color("green"))
    plain = draw_holds_debug(image, det)
    marked = draw_holds_debug(image, det, marks=[(100, 100)])
    assert plain.shape == marked.shape
    s = 1200 / 1280
    header = marked.shape[0] - round(960 * s)
    y, x = header + round(100 * s), round(100 * s)
    assert not np.array_equal(plain[y - 3 : y + 4, x - 3 : x + 4], marked[y - 3 : y + 4, x - 3 : x + 4])
