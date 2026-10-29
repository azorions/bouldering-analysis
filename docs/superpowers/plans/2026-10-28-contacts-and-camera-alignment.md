# Camera Alignment and Contacts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn a still or handheld bouldering video into per-limb hold sequences and attempt outcomes (topped, fell, came down), with fall details.

**Architecture:** A new `align` stage maps every frame onto one fixed view of the wall and saves the result to `camera.json`. For still clips, the frame-to-wall transforms are identities. For moving clips, the stage tracks KLT points from frame to frame and re-anchors with SIFT about once a second.

The other stages use `camera.json` as follows:
- `holds` builds its median background from the aligned frames.
- `pose` stays in frame pixels. Only its debug video uses the camera, to draw the holds in the right place.
- A new `contacts` stage maps the pose into wall coordinates and runs a per-limb state machine. It then splits the clip into attempts.

**Tech Stack:** Python 3.11+, OpenCV (`opencv-contrib-python`: SIFT, KLT, homographies), NumPy, pytest, uv.

**Spec:** `docs/superpowers/specs/2026-10-28-contacts-and-camera-alignment-design.md`

## Global Constraints

**Dependencies and imports**
- No new dependencies. The timeline image is drawn with OpenCV; matplotlib is not installed.
- `align.py`, `limbs.py`, `contacts.py` and `attempts.py` never import MediaPipe. `tests/test_cli.py::test_importing_the_cli_does_not_load_mediapipe` must keep passing.

**Data and outputs**
- Stage outputs go through `beta.jsonio.save_json` / `load_json` (`schema_version` 1).
- The new files live in `outputs/<stem>/`: `camera.json`, `align_debug.mp4`, `contacts.json` and `contacts_debug.png`.
- Wall coordinates for a still camera are exactly the frame pixels: identity transforms, and the canvas is the frame size.

**Alignment thresholds**

| Setting | Value |
|---|---|
| Still check samples | 25 |
| Still tolerance | ≤ 1 px corner shift (`STILL_TOL_PX = 1.0`) |
| Re-anchor | every 1 s (`--anchor-every`) |
| Canvas size cap | 2x the frame in each direction |
| Working size | long side ≤ 1024 px (`ALIGN_MAX_SIDE`) |

**Contact thresholds**

| Setting | Value |
|---|---|
| Onset | 0.25 s (`--min-hold-s`) |
| Release | 0.15 s |
| Onset margin | 0.05 torso |
| Release margin | 0.25 torso |
| Speed limit | 0.5 torso/s |
| Speed window k | 0.1 s |
| Visibility | ≥ 0.5 (`--min-visibility`) |
| Hand point | wrist + 1.2·(mean(index, pinky) − wrist); fallback wrist + 0.4·(wrist − elbow) |
| Foot point | toe + 0.15·(heel − toe) |
| Left/right vote | > 70% of confident frames disagree → swap all |
| Left/right continuity | swap if < 0.5x the keep cost and limbs > 0.3 torso apart |

**Attempt thresholds**

| Setting | Value |
|---|---|
| Topped | both hands on the top hold for ≥ 1 s |
| Fall | mid-hip drops ≥ 1 torso within 0.5 s, hands released, then on the floor or out of view |
| Pose gap tolerated in a fall | ≤ 0.5 s |
| Last holds window | 1.5 s |

**Code, tests and commits**
- Comments explain non-obvious CV/ML choices briefly; the reader is a 3rd-year CS student. Match the docstring and comment style of `holds.py` and `pose.py`.
- A plain `uv run pytest` stays offline and fast. The `slow` marker is only for MediaPipe model tests.
- Commits use Conventional Commits (`feat(align): ...`) with a plain-language body and no attribution trailers.

## Review Focus

These failure modes are implied by the spec but not covered by any of its tests. Each is pinned by a test in the task that owns the code.

1. **Variable-frame-rate phone video.** Seeking with `read_frame` can land a few frames away from the sequential index, which would pair a frame with the wrong transform. Anything that pairs a frame image with its transform must read sequentially. Pinned by Task 4: `test_read_frames_returns_exact_indices`.
2. **The header frame count can overshoot what's actually read.** `pose.json` then has a frame or two fewer than the header says, and `contacts` must not refuse a still-camera clip for that. Pinned by Task 10: `test_contacts_accepts_pose_with_fewer_frames_than_the_header`.
3. **Input larger than `ALIGN_MAX_SIDE`.** 1080p and 4K phone footage is aligned on a downscaled copy, and the transforms must come back in full-resolution pixels. Pinned by Task 2: `test_alignment_scales_back_from_working_size`.
4. **Wall points outside the canvas.** This happens when the canvas is capped, or a hand reaches beyond everything the camera saw. Negative or out-of-canvas coordinates must just work in the distance math. Pinned by Task 8: `test_points_outside_the_canvas_are_fine`.
5. **A limb that's never seen.** For example, a foot hidden all clip, or the climber's shoulders never visible. The output must be valid with an empty sequence, or the command must fail with a clear message rather than a traceback. Pinned by Task 7: `test_torso_length_needs_shoulders_and_hips`. Pinned by Task 10: `test_limb_never_seen`.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/beta/align.py` (new) | `CameraTrack` data, the still check, KLT tracking with SIFT anchors, `camera.json`, `align_debug.mp4` |
| `src/beta/limbs.py` (new) | Pose in wall coordinates, torso length, the left/right fix, hand and foot effector points, hip centers |
| `src/beta/contacts.py` (new) | The contacts stage: per-limb state machine, `analyze()` orchestration, `contacts.json`, `contacts_debug.png` |
| `src/beta/attempts.py` (new) | Floor estimate, attempt segmentation, outcomes and fall details |
| `src/beta/video.py` | Add `read_frames`, `masked_median`, and a `camera` parameter for `load_input` |
| `src/beta/holds.py` | `save_holds` / `detection_to_dict` gain an optional `camera` key |
| `src/beta/pose.py` | Hold overlay mapped through the camera (`wall_to_frame`) |
| `src/beta/synthetic.py` | `make_moving_climb_video` with known camera motion |
| `src/beta/cli.py` | `align` and `contacts` commands; `holds` auto-aligns; `pose` overlay uses the camera |
| `tests/conftest.py` | `moving_video` fixture; `body`, `track_of` and `square_hold` helpers |
| `tests/test_align.py`, `tests/test_limbs.py`, `tests/test_contacts.py`, `tests/test_attempts.py` (new) | Unit tests per module |
| `README.md` | Align and contacts usage; the camera assumption updated |

---

### Task 1: Camera track data and `camera.json`

**Files:**
- Create: `src/beta/align.py`
- Test: `tests/test_align.py`

**Interfaces:**
- Produces:
  - `FrameTransform(frame: int, matrix: np.ndarray | None, method: str, inliers: int)`. This is a frozen dataclass.
    - `matrix` is a 3x3 float64 mapping frame px to wall px; `None` means lost.
    - `method` is `"track" | "anchor" | "lost"`.
  - `CameraTrack`, a frozen dataclass with fields, in order:
    - `mode: str` (`"still"` or `"moving"`)
    - `fps: float`
    - `frame_count: int`: frames actually read when moving, the header count when still
    - `width: int`, `height: int`: the video frame size
    - `canvas: tuple[int, int]`: (w, h)
    - `reference_frame: int | None`
    - `transforms: tuple[FrameTransform, ...]`: empty when still
    - `summary: dict`
  - Members of `CameraTrack`:
    - `CameraTrack.still(fps, frame_count, width, height, summary=None) -> CameraTrack`
    - `.frame_to_wall(frame) -> np.ndarray | None`: identity when still; `None` when lost or out of range
    - `.wall_to_frame(frame) -> np.ndarray | None`
    - `.warp_to_wall(image, frame) -> tuple[np.ndarray, np.ndarray] | None`: the canvas image and a uint8 coverage mask (255 = covered)
    - `.id -> str`: a property equal to `camera_id(self)`
  - `apply_h(matrix, points: np.ndarray) -> np.ndarray`: (N, 2) to (N, 2); NaN rows stay NaN.
  - `corner_gap(a, b, width, height) -> float`: the largest distance between where `a` and `b` send the four frame corners. `corner_shift(m, w, h) = corner_gap(m, np.eye(3), w, h)`.
  - `camera_id(cam) -> str`: 12 hex characters. It is the sha1 of the mode, reference frame, canvas and, for each transform, its frame, method and matrix rounded to 4 decimals.
  - `camera_ref(cam) -> dict`: `{"id", "mode", "canvas": [w, h]}`, which goes into `holds.json`.
  - `save_camera(path, cam, source) -> Path`, `load_camera(path) -> tuple[CameraTrack, dict]`.

- [x] **Step 1: Write the failing tests** in `tests/test_align.py`

```python
import numpy as np
import pytest
from dataclasses import replace
from beta.align import CameraTrack, FrameTransform, apply_h, camera_id, corner_shift, load_camera, save_camera

def shift(dx, dy):
    return np.array([[1.0, 0, dx], [0, 1.0, dy], [0, 0, 1.0]])

def moving_cam(*matrices, canvas=(700, 600)):
    ts = tuple(FrameTransform(i, m, "lost" if m is None else ("anchor" if i == 0 else "track"), 0 if m is None else 300)
               for i, m in enumerate(matrices))
    return CameraTrack("moving", 30.0, len(ts), 640, 480, canvas, 0, ts, {})

def test_still_camera_maps_frame_pixels_to_themselves():
    cam = CameraTrack.still(fps=30.0, frame_count=10, width=640, height=480)
    pts = np.array([[10.0, 20.0], [600.5, 470.25]])
    assert np.allclose(apply_h(cam.frame_to_wall(3), pts), pts)
    assert cam.canvas == (640, 480) and cam.transforms == ()
    assert cam.frame_to_wall(10) is None

def test_moving_camera_round_trips_points():
    w = np.array([[1.02, 0.01, 5.0], [-0.01, 1.02, 40.0], [0, 0, 1.0]])
    cam = moving_cam(np.eye(3), w)
    pts = np.array([[100.0, 100.0], [320.0, 240.0]])
    assert np.allclose(apply_h(cam.wall_to_frame(1), apply_h(cam.frame_to_wall(1), pts)), pts, atol=1e-6)

def test_lost_frame_has_no_transform():
    cam = moving_cam(None)
    assert cam.frame_to_wall(0) is None and cam.wall_to_frame(0) is None
    assert cam.warp_to_wall(np.zeros((480, 640, 3), np.uint8), 0) is None

def test_apply_h_keeps_nan_rows():
    out = apply_h(np.eye(3), np.array([[1.0, 2.0], [np.nan, np.nan]]))
    assert np.allclose(out[0], [1, 2]) and np.isnan(out[1]).all()

def test_corner_shift():
    assert corner_shift(shift(3, -4), 640, 480) == pytest.approx(5.0)

def test_warp_to_wall_covers_only_the_frame_footprint():
    cam = moving_cam(shift(50, 0), canvas=(690, 480))
    warped, mask = cam.warp_to_wall(np.full((480, 640, 3), 200, np.uint8), 0)
    assert warped.shape == (480, 690, 3) and mask.shape == (480, 690)
    assert mask[:, :49].max() == 0 and mask[:, 60:680].min() == 255

def test_camera_json_round_trip_keeps_the_id(tmp_path):
    cam = moving_cam(np.eye(3), shift(1.5, 30.25), None)
    path = save_camera(tmp_path / "camera.json", cam, {"path": "clip.mp4"})
    loaded, doc = load_camera(path)
    assert doc["id"] == cam.id == loaded.id == camera_id(cam) and len(cam.id) == 12
    assert np.array_equal(loaded.frame_to_wall(1), cam.frame_to_wall(1)) and loaded.frame_to_wall(2) is None
    assert camera_id(replace(cam, reference_frame=1)) != cam.id
    still, _ = load_camera(save_camera(tmp_path / "still.json", CameraTrack.still(30.0, 5, 640, 480), {"path": "c.mp4"}))
    assert still.mode == "still" and np.array_equal(still.frame_to_wall(4), np.eye(3))
```

- [x] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_align.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'beta.align'`

- [x] **Step 3: Implement the Task 1 interfaces in `src/beta/align.py`**

- Module docstring: explain wall coordinates and why a homography fits a handheld phone. A camera that rotates about its own center maps every image point by one homography, whatever its depth.
- `warp_to_wall`:
  - Uses `cv2.warpPerspective(image, W, canvas)` with `INTER_LINEAR` and a zero border.
  - The mask is a `255` image warped with `INTER_NEAREST`.
- `save_camera` writes matrices at full float precision, so `load_camera` restores them exactly and the id survives a round trip. It also writes `"id"`. `indent=None`, like `pose.json`.
- JSON shape: `{source, params, id, mode, reference_frame, canvas: [w, h], fps, frame_count, width, height, frames: [{frame, matrix | null, method, inliers}], summary}`.

- [x] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_align.py -v`
Expected: 7 passed

- [x] **Step 5: Commit**

```bash
git add src/beta/align.py tests/test_align.py
git commit -m "feat(align): add camera track data and camera.json"
```

---

### Task 2: Estimate camera movement

**Files:**
- Modify: `src/beta/align.py` (add the estimation)
- Modify: `src/beta/synthetic.py` (add `make_moving_climb_video`)
- Modify: `tests/conftest.py` (add the `moving_video` fixture)
- Test: `tests/test_align.py`

**Interfaces:**
- Consumes: Task 1 (`CameraTrack`, `FrameTransform`, `corner_gap`, `corner_shift`, `apply_h`); `beta.video.probe`, `read_frame`, `iter_frames`, `sample_indices`, `VideoInfo`.
- Produces:
  - `StillCheck(still: bool, samples: list[int], max_shift_px: float, matched: int)`, a frozen dataclass.
  - `check_still(path, info: VideoInfo, n: int = 25) -> StillCheck`.
  - `estimate_camera(path, info: VideoInfo | None = None, anchor_every_s: float = 1.0, n_samples: int = 25, progress: Callable[[int], None] | None = None) -> CameraTrack`.
  - Summary keys for moving clips:
    - `max_shift_px`
    - `lost_frames`
    - `longest_lost_run`: `[length, first_frame]`, or `[0, None]`
    - `anchor_correction_px`: `{"median", "max"}`
    - `seconds`
  - Summary keys for still clips: `max_shift_px`, `checked_frames`, `seconds`, and `note` when too few samples matched.
  - Constants: `ALIGN_MAX_SIDE = 1024`, `STILL_TOL_PX = 1.0`, `MIN_INLIERS = 30`, `MAX_ZOOM = 1.5`, `MAX_STEP_FRAC = 0.25`, `CANVAS_MAX_SCALE = 2.0`.
  - `synthetic.MovingTruth(wall: WallTruth, wall_to_frame: list[np.ndarray])`:
    - `wall_to_frame[t]` maps big-wall px to frame-t px.
    - The frame for a blank frame is uniform grey, but its matrix is still the true camera.
  - `synthetic.make_moving_climb_video(path, width=640, height=480, n_frames=45, fps=15.0, seed=0, tilt_frac=0.2, blank_frames=(), climber_frac=1.0) -> MovingTruth`.
  - conftest: a session fixture `moving_video -> (path, MovingTruth)`, with 640x480, 45 frames, 15 fps.

- [x] **Step 1: Add `make_moving_climb_video`** to `src/beta/synthetic.py`

- Render `make_wall(width, round(height * (1 + tilt_frac)) + 2, seed)` once.
- Each frame is `cv2.warpPerspective(big_wall, M_t, (width, height))`, with a smooth path like the real clip:
  - The view pans up by up to `tilt_frac · height` and back, along `sin(π s)` with `s = t / (n − 1)`.
  - Rotation is ±1.5° along `sin(2π s)`.
  - Zoom is ±4% along `sin(π s)`.
  - Rotation and zoom are about the frame center.
- Then draw the green climber ellipse and head exactly as in `make_climb_video`, scaled by `climber_frac`, and set the top-left marker square to `min(255, 8 * i)`.
- Frames in `blank_frames` are uniform `(125, 125, 125)`.
- Write MJPG `.avi`.
- This is test-support code, so it has no separate test; the Task 2 tests exercise it.

- [x] **Step 2: Write the failing tests** (append to `tests/test_align.py`; add `moving_video` to conftest)

```python
from beta.align import STILL_TOL_PX, check_still, estimate_camera
from beta.synthetic import make_moving_climb_video
from beta.video import probe

def frame_errors(cam, truth, frames=None):
    """Median px error per frame: big-wall points seen in every frame should land on the same wall
    coordinates whichever frame they are mapped from."""
    h_big, w_big = truth.wall.height, truth.wall.width
    ys, xs = np.mgrid[0.45 * h_big:0.55 * h_big:8j, 0.2 * w_big:0.8 * w_big:8j]
    grid = np.column_stack([xs.ravel(), ys.ravel()])
    ref = cam.reference_frame
    want = apply_h(cam.frame_to_wall(ref), apply_h(truth.wall_to_frame[ref], grid))
    out = []
    for t in (range(cam.frame_count) if frames is None else frames):
        w = cam.frame_to_wall(t)
        if w is not None:
            out.append(float(np.median(np.linalg.norm(apply_h(w, apply_h(truth.wall_to_frame[t], grid)) - want, axis=1))))
    return out

def test_still_clip_is_detected_as_still(climb_video):
    cam = estimate_camera(climb_video[0])
    assert cam.mode == "still" and cam.transforms == () and cam.canvas == (640, 480)
    assert cam.summary["max_shift_px"] < STILL_TOL_PX

def test_check_still_reports_motion(moving_video):
    path, _ = moving_video
    check = check_still(path, probe(path))
    assert not check.still and check.max_shift_px > 20 and check.matched >= 12

def test_moving_clip_lines_up_with_the_wall(moving_video):
    path, truth = moving_video
    cam = estimate_camera(path)
    assert cam.mode == "moving" and cam.frame_count == len(cam.transforms) == 45
    assert all(t.matrix is not None for t in cam.transforms)
    assert max(frame_errors(cam, truth)) <= 1.0

def test_anchors_about_once_a_second(moving_video):
    cam = estimate_camera(moving_video[0])
    anchors = [t.frame for t in cam.transforms if t.method == "anchor"]
    assert anchors[0] == 0 and all(b - a <= 15 for a, b in zip(anchors, anchors[1:]))  # 15 fps
    assert sum(t.method == "track" for t in cam.transforms) > len(anchors)

def test_drift_stays_bounded_over_a_long_clip(tmp_path):
    truth = make_moving_climb_video(tmp_path / "long.avi", 320, 240, n_frames=600, fps=30.0)
    cam = estimate_camera(tmp_path / "long.avi")
    assert max(frame_errors(cam, truth)) <= 1.0
    assert cam.summary["anchor_correction_px"]["max"] <= 2.0

def test_blank_frames_are_lost_and_alignment_recovers(tmp_path):
    truth = make_moving_climb_video(tmp_path / "gap.avi", blank_frames=range(20, 24))
    cam = estimate_camera(tmp_path / "gap.avi")
    lost = [t for t in cam.transforms if t.matrix is None]
    assert [t.frame for t in lost] == [20, 21, 22, 23] and {t.method for t in lost} == {"lost"}
    assert cam.summary["lost_frames"] == 4 and cam.summary["longest_lost_run"] == [4, 20]
    assert max(frame_errors(cam, truth, range(24, 45))) <= 1.0

def test_a_big_climber_does_not_pull_the_alignment(tmp_path):
    truth = make_moving_climb_video(tmp_path / "big.avi", climber_frac=2.5)
    assert max(frame_errors(estimate_camera(tmp_path / "big.avi"), truth)) <= 1.0

def test_alignment_scales_back_from_working_size(tmp_path):  # Review Focus 3
    truth = make_moving_climb_video(tmp_path / "hd.avi", 1600, 1200, n_frames=20)
    cam = estimate_camera(tmp_path / "hd.avi")
    assert cam.canvas[0] >= 1600 and max(frame_errors(cam, truth)) <= 1.5  # 1 working px is ~1.6 full-res px
```

- [x] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/test_align.py -v`
Expected: the new tests FAIL with `ImportError: cannot import name 'check_still'`. The Task 1 tests still pass.

- [x] **Step 4: Implement `check_still` and `estimate_camera`**

All feature work is done on grayscale copies scaled by `s = min(1, ALIGN_MAX_SIDE / max(w, h))`. The full-resolution transform is `S⁻¹·H·S`, with `S = diag(s, s, 1)`. Matching uses:
- SIFT with `nfeatures=2000`, a ratio test at 0.75 and a BF L2 kNN matcher.
- `cv2.findHomography` with RANSAC at 3.0 working px. A fit counts only when it has at least `MIN_INLIERS` inliers.

`check_still`:
- Samples come from `sample_indices(info.frame_count, n)`, read with `read_frame`. Seeking is fine here: a still view is still at any index.
- **Hub:** the middle sample. If fewer than half the other samples match it, try the neighbours of the middle, up to 3 hubs.
- `max_shift_px` is the largest `corner_shift` (full-resolution px) among matched samples.
- `still = max_shift_px < STILL_TOL_PX`.
- If no hub reaches half, the result is `still=True` with `matched` recorded. `estimate_camera` then adds `summary["note"] = "too few wall features to check whether the camera moves"`.

`estimate_camera`:
- Returns `CameraTrack.still(...)` if the check says still.
- Otherwise:
  1. **Reference:** the sample with the most total inliers in all-pairs matching among the samples.
  2. **Pass:** run sequentially with `iter_frames`, following this per-frame logic:

```text
due = previous H unknown or (i - last_anchor) >= round(anchor_every_s * fps)
if not due:
    KLT: goodFeaturesToTrack(prev, 800, 0.01, 8) -> calcOpticalFlowPyrLK(prev, cur) -> RANSAC H(cur -> prev)
    if inliers >= MIN_INLIERS: H = H_prev @ step, method "track"; else due = True
if due:
    SIFT match cur -> reference; if inliers >= MIN_INLIERS: H = that, method "anchor", last_anchor = i
    if a chained prediction existed: record corner_gap(H_anchor, H_chained) as the anchor correction
reject H (lost) if zoom sqrt(|det H[:2,:2]|) is outside [1/MAX_ZOOM, MAX_ZOOM],
    or corner_gap(H, H_prev) > MAX_STEP_FRAC * frame diagonal
lost -> matrix None, method "lost", H_prev = None (forces an anchor attempt next frame)
```

  3. **Canvas:** the bounding box of the frame corners mapped through every valid full-resolution `H`, clipped to `[-w/2, 1.5w] × [-h/2, 1.5h]` (`CANVAS_MAX_SCALE`). `T` translates by `(-floor(x0), -floor(y0))`, the canvas size is `(ceil(x1) - floor(x0), ceil(y1) - floor(y0))`, and `W_t = T @ H_t`.
- The climber moves differently from the wall, so RANSAC rejects matches on the climber. Comment on this where you match.
- `progress(i + 1)` is called after each frame, if given.

- [x] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_align.py -v`
Expected: 15 passed. The long-clip test may take a few seconds.

- [x] **Step 6: Run the full suite**

Run: `uv run pytest`
Expected: all pass

- [x] **Step 7: Commit**

```bash
git add src/beta/align.py src/beta/synthetic.py tests/conftest.py tests/test_align.py
git commit -m "feat(align): estimate camera movement with tracked points and anchors"
```

---

### Task 3: `beta align` command and `align_debug.mp4`

**Files:**
- Modify: `src/beta/align.py` (add `write_align_video`)
- Modify: `src/beta/cli.py` (add `cmd_align`, `_video_source`, `_print_camera`, the `align` parser and the `positive_arg` type)
- Modify: `src/beta/video.py` (add `read_frames`, also needed by Task 4)
- Modify: `README.md`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: Task 2 `estimate_camera`; Task 1 `save_camera`, `CameraTrack`.
- Produces:
  - `video.read_frames(path, indices: Iterable[int]) -> list[tuple[int, np.ndarray]]`. It reads sequentially and returns exactly those indices, in order, skipping any past the end.
  - `align.write_align_video(video_path, cam, out_path, holds: Sequence[Hold] = ()) -> Path`.
  - `cli._video_source(path, info) -> dict`: `{"path", "kind": "video", "fps", "frame_count", "width", "height", "bit_depth"}`, with the header frame count.
  - `cli._print_camera(cam) -> None`.

- [x] **Step 1: Write the failing tests** (append to `tests/test_cli.py`)

```python
from beta.align import load_camera

def test_align_on_still_clip(climb_video, tmp_path, capsys):
    out = tmp_path / "out"
    assert main(["align", str(climb_video[0]), "--out", str(out)]) == 0
    assert load_camera(out / "camera.json")[0].mode == "still"
    assert not (out / "align_debug.mp4").exists()
    assert "Camera is still" in capsys.readouterr().out

def test_align_on_moving_clip_writes_debug_video(moving_video, tmp_path, capsys):
    out = tmp_path / "out"
    assert main(["align", str(moving_video[0]), "--out", str(out)]) == 0
    assert probe(out / "align_debug.mp4").frame_count == 45
    printed = capsys.readouterr().out
    assert "Camera moves: up to" in printed and "Aligned 45 of 45 frames" in printed and "ms per frame" in printed

def test_align_no_video_and_bad_options(moving_video, tmp_path, capsys):
    out = tmp_path / "out"
    assert main(["align", str(moving_video[0]), "--out", str(out), "--no-video"]) == 0
    assert (out / "camera.json").is_file() and not (out / "align_debug.mp4").exists()
    assert main(["align", str(tmp_path / "wall.png")]) == 1
    assert "align needs a video" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(["align", "clip.mp4", "--anchor-every", "0"])
```

- [x] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_cli.py -k align -v`
Expected: FAIL (`invalid choice: 'align'`)

- [x] **Step 3: Implement the Task 3 interfaces**

**`write_align_video`**
- Size and codec as in `pose.write_pose_video`: scaled to ≤ 1280 px wide, even dimensions, `mp4v`.
- **Backdrop:** the reference frame, read with `read_frames` so it is paired with its own transform. Warp it to the canvas at 40% brightness.
- **Each frame:** warp it onto the backdrop where it is covered, and draw hold outlines when given.
- **Header:**
  - Normal frames show `frame {i}  {method}  {inliers} inliers`.
  - Lost frames show `frame {i}  LOST` in red.

**`cmd_align`**
- Reject images and HEIC files with `"{name} is a photo: align needs a video of the climb"`.
- Run `estimate_camera` with `_progress`, then `save_camera(out / "camera.json", cam, _video_source(...))`.
- If the clip is moving and `--no-video` wasn't given, write the debug video.

**`_print_camera` output, still:**

```
Camera is still (largest shift {x:.1f} px over {n} sampled frames)
```

If `summary` has a `note`, print it as a warning on stderr.

**`_print_camera` output, moving:**

```
Camera moves: up to {max_shift:.0f} px from frame {ref}'s view
  Aligned {ok} of {total} frames[ ({lost} lost, longest run {n} from frame {f})]
  Drift corrected at anchors: median {m:.1f} px, max {x:.1f} px
  Took {s:.1f} s ({ms:.0f} ms per frame)
```

**Parser:** `align` takes `input`, `--out`, `--no-video` and `--anchor-every S`. `positive_arg` rejects values ≤ 0 with `ArgumentTypeError`.

**`README.md`**
- Add a "Camera movement" section before "2. Track the climber's pose". It covers what `beta align` does, its outputs, how to read `align_debug.mp4`, and that `holds` runs it automatically.
- Under "Assumptions", replace "The camera is static" with:
  - the camera may be still or handheld;
  - alignment fails when the filmer walks far sideways, or when the climber fills the frame.

- [x] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_cli.py -k align -v`
Expected: 3 passed

- [x] **Step 5: Commit**

```bash
git add src/beta/align.py src/beta/cli.py src/beta/video.py README.md tests/test_cli.py
git commit -m "feat(cli): add the align command and its debug video"
```

---

### Task 4: Holds on aligned frames

**Files:**
- Modify: `src/beta/video.py` (add `masked_median`, a `camera` parameter for `load_input`, and an updated module docstring)
- Modify: `src/beta/holds.py:519-543` (`detection_to_dict`, `save_holds`)
- Modify: `src/beta/cli.py` (`cmd_holds`, `_source_note`, new `_camera_for`)
- Modify: `README.md` (section 1)
- Test: `tests/test_video.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: Task 3 `read_frames`, `_video_source`, `_print_camera`; Tasks 1–2 `CameraTrack.warp_to_wall`, `estimate_camera`, `save_camera`, `load_camera`, `camera_ref`.
- Produces:
  - `video.masked_median(images: Sequence[np.ndarray], masks: Sequence[np.ndarray]) -> np.ndarray`. It is the per-pixel median over the images whose mask is > 0. Uncovered pixels are 0.
  - `video.load_input(path, frame=None, bg_frames=25, camera=None)`:
    - Behavior is unchanged when `camera` is `None` or still.
    - When the camera is moving, it warps the frames onto the canvas.
    - `source["width"]` and `source["height"]` are the video's size.
  - `holds.detection_to_dict(det, source, camera: dict | None = None)` and `holds.save_holds(path, det, source, camera=None)`. When `camera` is given, the output gains a `"camera"` key.
  - `cli._camera_for(path, out_dir, info) -> CameraTrack`. It reuses `out_dir/camera.json` when its `source` has the same file name, header frame count, width and height. Otherwise it prints `Checking whether the camera moves ...`, estimates, saves, and calls `_print_camera`.

- [x] **Step 1: Write the failing tests**

In `tests/test_video.py`:

```python
from beta.video import masked_median, read_frames

def test_read_frames_returns_exact_indices(climb_video):  # Review Focus 1
    frames = read_frames(climb_video[0], [30, 0, 7, 99])
    assert [i for i, _ in frames] == [0, 7, 30]
    assert [round(float(f[:12, :12].mean())) for _, f in frames] == pytest.approx([0, 56, 240], abs=4)

def test_masked_median_ignores_uncovered_pixels():
    full, half, none = (np.full((2, 4), 255, np.uint8), np.zeros((2, 4), np.uint8), np.zeros((2, 4), np.uint8))
    half[:, :2] = 255
    imgs = [np.full((2, 4, 3), v, np.uint8) for v in (10, 20, 200, 99)]
    out = masked_median(imgs, [full, full, half, none])
    assert out[0, 0, 0] == 20 and out[0, 3, 0] == 15
    assert masked_median(imgs[:1], [none])[0, 0, 0] == 0
```

In `tests/test_cli.py`:

```python
from beta.align import apply_h, load_camera
from beta.holds import detect_holds, HoldSpec
from beta.video import median_background
from conftest import inside

def test_holds_on_moving_clip_find_every_route_hold(moving_video, tmp_path, capsys):
    path, truth = moving_video
    out = tmp_path / "out"
    assert main(["holds", str(path), "--color", "green", "--out", str(out)]) == 0
    holds, doc = load_holds(out / "holds.json")
    cam, _ = load_camera(out / "camera.json")
    assert doc["camera"] == {"id": cam.id, "mode": "moving", "canvas": list(cam.canvas)}
    assert doc["source"]["width"] == 640 and "Checking whether the camera moves" in capsys.readouterr().out
    to_wall = cam.frame_to_wall(cam.reference_frame) @ truth.wall_to_frame[cam.reference_frame]
    route = [t for t in truth.wall.of_color("green") if t.kind != "border"]
    centers = apply_h(to_wall, np.array([t.center for t in route]))
    visible = [c for c in centers if 0 <= c[0] < cam.canvas[0] and 0 <= c[1] < cam.canvas[1]]
    assert visible and all(any(inside(h, c) for h in holds) for c in visible)

def test_holds_on_still_clip_are_unchanged(climb_video, tmp_path):
    path, _ = climb_video
    out = tmp_path / "out"
    assert main(["holds", str(path), "--color", "green", "--out", str(out)]) == 0
    holds, doc = load_holds(out / "holds.json")
    expected = detect_holds(median_background(path)[0], HoldSpec.from_color("green")).holds
    assert doc["camera"]["mode"] == "still" and holds == expected

def test_holds_redo_alignment_for_another_video(climb_video, moving_video, tmp_path):
    out = tmp_path / "out"
    assert main(["align", str(moving_video[0]), "--out", str(out), "--no-video"]) == 0
    assert main(["holds", str(climb_video[0]), "--color", "green", "--out", str(out)]) == 0
    assert load_camera(out / "camera.json")[0].mode == "still"

def test_holds_on_image_has_no_camera(wall, tmp_path):
    src = imwrite(tmp_path / "wall.png", wall[0])
    assert main(["holds", str(src), "--color", "green", "--out", str(tmp_path / "o")]) == 0
    assert "camera" not in load_json(tmp_path / "o" / "holds.json")
    assert not (tmp_path / "o" / "camera.json").exists()
```

- [x] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_video.py tests/test_cli.py -k "read_frames or masked or holds_on or redo" -v`
Expected: FAIL (`ImportError: masked_median`, and `KeyError: 'camera'`)

- [x] **Step 3: Implement the Task 4 interfaces**

**`masked_median`**
- Stack the images as float32, with NaN where the mask is 0.
- Take `np.nanmedian` in row chunks of about 64 rows, to bound memory on big canvases.
- Fill NaN with 0 and convert to uint8.
- Suppress numpy's all-NaN `RuntimeWarning` locally. The CLI turns warnings into messages, so it would otherwise print.

**`load_input` with a moving camera**
- Read `sample_indices(info.frame_count, bg_frames)` with `read_frames`.
- Warp each with `camera.warp_to_wall`, skipping lost frames, then take `masked_median`.
- `--frame N` warps that single frame.
- Set `source["width"], source["height"] = info.width, info.height` for every video input. For still clips that equals the image size, as before.
- Update the module docstring: the median needs a still view, and a moving camera is first aligned by `beta.align`.

**`_source_note`:** for moving holds it reads `source: {name}, median of {n} aligned frames ({canvas w}x{canvas h} wall view)`. Pass the camera in.

**`cmd_holds`**
- For video input, `camera = _camera_for(path, out_dir, probe(path))`.
- Then `load_input(..., camera=camera)` and `save_holds(..., camera=camera_ref(camera))`.

**`README.md` section 1:** for videos, `holds` checks whether the camera moved and aligns the frames first. Holds are numbered on the aligned wall view.

- [x] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_video.py tests/test_cli.py -v`
Expected: all pass, including the existing `test_holds_on_video_uses_median_background`.

- [x] **Step 5: Commit**

```bash
git add src/beta/video.py src/beta/holds.py src/beta/cli.py README.md tests/test_video.py tests/test_cli.py
git commit -m "feat(holds): detect holds on camera-aligned frames"
```

---

### Task 5: Hold overlay in `pose_debug.mp4` follows the camera

**Files:**
- Modify: `src/beta/pose.py:277-358` (`_draw_holds`, `draw_pose`, `write_pose_video`)
- Modify: `src/beta/cli.py:174-221` (`_overlay_holds`, `cmd_pose`)
- Test: `tests/test_pose.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: Task 1 `CameraTrack.wall_to_frame`, `apply_h`, `load_camera`.
- Produces:
  - `pose.draw_pose(frame_bgr, pose, holds=(), size=None, wall_to_frame: np.ndarray | None = None)`. `None` means hold coordinates are already frame pixels.
  - `pose.write_pose_video(video_path, track, out_path, holds=(), camera: CameraTrack | None = None)`. On frames where `camera.wall_to_frame(i)` is `None`, holds are not drawn.
  - `cli._overlay_holds(path, width, height) -> tuple[list[Hold], CameraTrack | None]`:
    - Holds with a moving `camera` key load `camera.json` from the same folder.
    - If it's missing or its id differs, print `warning: {path} was made with a different camera.json (run `beta holds` again), so its holds are not drawn` and return `([], None)`.

- [x] **Step 1: Write the failing tests**

In `tests/test_pose.py`:

```python
from beta.holds import Hold

def test_pose_video_draws_holds_where_the_camera_sees_them():
    hold = Hold(1, ((100, 100), (140, 100), (140, 140), (100, 140)), (120.0, 120.0), 1600.0, (100, 100, 40, 40), False)
    frame = np.zeros((480, 640, 3), np.uint8)
    img = draw_pose(frame, None, [hold], wall_to_frame=np.array([[1.0, 0, -30], [0, 1.0, 0], [0, 0, 1.0]]))
    assert img[105:135, 69:72].max() > 60       # the left edge now at x = 70
    assert img[105:135, 138:143].max() < 30     # nothing left at the old right edge (x = 140)

def test_pose_video_skips_holds_on_lost_frames(climb_video, tmp_path, monkeypatch):
    ...  # camera with frame 3 lost; monkeypatch pose.draw_pose to record (frame index, holds); assert frame 3 got ()
```

Spell out the second test fully when implementing:
- Build a `CameraTrack` for the 31-frame clip with every transform the identity except `FrameTransform(3, None, "lost", 0)`.
- Record the `holds` argument of each `draw_pose` call.
- Assert that frame 3 got `()` and frame 2 got the hold.

In `tests/test_cli.py`, update `test_pose_draws_holds_from_the_output_dir`'s fake to `def write_video(src, track, out_path, holds=(), camera=None)`, and add:

```python
def test_pose_uses_the_camera_for_moving_holds(moving_video, tmp_path, monkeypatch):
    path, _ = moving_video
    out = tmp_path / "out"
    assert main(["holds", str(path), "--color", "green", "--out", str(out)]) == 0
    seen = {}
    monkeypatch.setattr(cli, "open_landmarker", lambda p, c: FakeLandmarker(45))
    monkeypatch.setattr(cli, "write_pose_video", lambda s, t, o, holds=(), camera=None: seen.update(holds=holds, camera=camera) or o)
    model = tmp_path / "fake.task"; model.write_bytes(b"x")
    assert main(["pose", str(path), "--model-path", str(model), "--out", str(out)]) == 0
    assert seen["holds"] and seen["camera"].mode == "moving"

def test_pose_skips_holds_from_another_alignment(moving_video, tmp_path, monkeypatch, capsys):
    ...  # after `holds`, overwrite holds.json["camera"]["id"] with "000000000000" (load_json/save_json);
         # `pose` prints "was made with a different camera.json" on stderr and passes holds == []
```

- [x] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_pose.py tests/test_cli.py -k "camera or lost or alignment or draws_holds" -v`
Expected: FAIL (`unexpected keyword argument 'wall_to_frame'`)

- [x] **Step 3: Implement the Task 5 interfaces**

- `_draw_holds` maps the polygon and centroid through `wall_to_frame` with `apply_h` before scaling by `(sx, sy)`.
- `cmd_pose` unpacks `holds, camera = _overlay_holds(...)` and passes `camera` to `write_pose_video`.
- For holds without a `camera` key (made from an image), keep the existing size check.

- [x] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest`
Expected: all pass

- [x] **Step 5: Commit**

```bash
git add src/beta/pose.py src/beta/cli.py tests/test_pose.py tests/test_cli.py
git commit -m "feat(pose): draw holds where a moving camera sees them"
```

---

### Task 6: CHECKPOINT on the real clip (user)

No code. Stop here and work through it with the user.

- [x] **Step 1: Align**

Run: `uv run beta align samples/bouldering_video_sample.mp4`
Expected:
- `Camera moves: up to ~115 px`.
- 1000 or more of 1030 frames aligned. Any lost frames should be near the end, where the phone is picked up.
- Drift median ≤ ~1 px.
- About 20–25 ms per frame.

Open `outputs/bouldering_video_sample/align_debug.mp4`. The wall should stand still.

- [x] **Step 2: Holds**

Run: `uv run beta holds samples/bouldering_video_sample.mp4 --color white`
Expected: `holds_debug.png` is sharp, not smeared.

With the user, tune with `--sample X,Y` (pixels in `background.png`) and `--roi` until the route's holds are outlined:
- False holds on the lit ceiling don't matter.
- Missing parts of touched holds do.

- [x] **Step 3: Pose debug with aligned holds**

Run: `uv run beta pose samples/bouldering_video_sample.mp4` (about 1.5 min)
Expected: in `pose_debug.mp4` the hold outlines stay on the holds while the camera moves.

- [x] **Step 4: The user writes the answer key** to `samples/bouldering_video_sample.truth.json` (gitignored, like the clip)

```json
{"holds_command": "beta holds ... (the exact tuned command)", "top_hold": 0,
 "outcome": "fell", "sequence": {"LH": [], "RH": [], "LF": [], "RF": []}}
```

Use the hold numbers from `holds_debug.png`.

- [x] **Step 5: Ask the user** whether CLAUDE.md's "Assume a static camera for now" line should change. CLAUDE.md is theirs, so don't edit it unasked. Get sign-off before Task 7.

---

### Task 7: Hand and foot points with the left/right fix

**Files:**
- Create: `src/beta/limbs.py`
- Modify: `tests/conftest.py` (add the `body` and `track_of` helpers)
- Test: `tests/test_limbs.py`

**Interfaces:**
- Consumes:
  - `pose.PoseTrack`, `pose.PoseFrame`, `pose.CONTACT_POINTS`
  - Task 1 `CameraTrack.frame_to_wall`, `apply_h`
- Produces:
  - `LIMBS = ("LH", "RH", "LF", "RF")`, equal to `tuple(CONTACT_POINTS)`.
  - `wall_landmarks(track, camera) -> np.ndarray`:
    - Shape (F, 33, 5), with F = `len(track.frames)`, indexed by `PoseFrame.frame`.
    - x and y are mapped through `frame_to_wall`.
    - Frames with no pose or a lost transform are all NaN.
  - `torso_length(lm) -> float`. It raises `ValueError("no frame shows the climber's shoulders and hips")` when none do.
  - `fix_left_right(lm, torso, fps) -> tuple[np.ndarray, dict]`:
    - Returns a copy.
    - The report is `{"global_swap": bool, "frames_swapped": {"arms": int, "legs": int}}`.
  - `effector_points(lm, min_visibility=0.5) -> tuple[np.ndarray, np.ndarray]`. Points are (F, 4, 2) in `LIMBS` order; `seen` is (F, 4) bool.
  - `hip_centers(lm) -> np.ndarray` of shape (F, 2).
  - conftest helpers:
    - `body(hands, feet, torso=100.0, vis=0.9) -> np.ndarray` (33, 5).
      - `hands = ((LHx, LHy), (RHx, RHy))`, `feet = ((LFx, LFy), (RFx, RFy))`.
      - Wrist, index, pinky and thumb sit on the hand point, so the hand effector equals the hand point.
      - Toe and heel sit on the foot point, and the ankle is 10 px above.
      - Mid-hip is `(mean foot x, mean foot y − torso)`, with the hips ±0.2·torso in x and the climber's left on the image left.
      - Shoulders are 1 torso above the hips at ±0.25·torso.
      - Elbows and knees are at the midpoints, and the face points 0.4·torso above mid-shoulder.
      - Visibility is `vis`, presence 0.95.
    - `track_of(frames: list[np.ndarray | None], fps=30.0, width=640, height=480) -> PoseTrack`.

- [x] **Step 1: Write the failing tests** in `tests/test_limbs.py`

```python
import numpy as np, pytest
from conftest import body, track_of
from beta.align import CameraTrack, FrameTransform
from beta.limbs import effector_points, fix_left_right, hip_centers, torso_length, wall_landmarks

HANDS, FEET = ((200.0, 100.0), (300.0, 100.0)), ((220.0, 400.0), (280.0, 400.0))

def test_hand_point_reaches_past_the_knuckles():
    lm = body(HANDS, FEET)
    lm[15, :2], lm[19, :2], lm[17, :2] = (100, 100), (110, 100), (110, 110)  # wrist, index, pinky
    pts, seen = effector_points(lm[None])
    assert np.allclose(pts[0, 0], (112, 106)) and seen[0, 0]

def test_hand_falls_back_to_the_forearm_when_fingers_are_hidden():
    lm = body(HANDS, FEET)
    lm[13, :2], lm[15, :2] = (80, 100), (100, 100)  # elbow, wrist
    lm[[17, 19], 3] = 0.2
    assert np.allclose(effector_points(lm[None])[0][0, 0], (108, 100))

def test_foot_point_is_the_front_of_the_shoe():
    lm = body(HANDS, FEET)
    lm[31, :2], lm[29, :2] = (200, 300), (180, 300)  # left toe, left heel
    assert np.allclose(effector_points(lm[None])[0][0, 2], (197, 300))

def test_seen_follows_wrist_and_toe_visibility():
    lm = body(HANDS, FEET)
    lm[15, 3], lm[32, 3] = 0.4, 0.6
    assert effector_points(lm[None])[1][0].tolist() == [False, True, True, True]

def test_wall_landmarks_apply_the_camera():
    w = np.array([[1.0, 0, 10], [0, 1.0, -5], [0, 0, 1.0]])
    cam = CameraTrack("moving", 30.0, 3, 640, 480, (700, 500), 0,
                      (FrameTransform(0, w, "anchor", 99), FrameTransform(1, None, "lost", 0), FrameTransform(2, w, "track", 99)), {})
    lm = wall_landmarks(track_of([body(HANDS, FEET), body(HANDS, FEET), None]), cam)
    assert np.allclose(lm[0, :, :2], body(HANDS, FEET)[:, :2] + (10, -5)) and np.allclose(lm[0, :, 3:], body(HANDS, FEET)[:, 3:])
    assert np.isnan(lm[1]).all() and np.isnan(lm[2]).all()

def test_torso_length_is_the_median():
    frames = [body(HANDS, FEET), body(HANDS, FEET), body(HANDS, FEET, torso=300.0)]
    assert torso_length(np.stack(frames)) == pytest.approx(100.0)

def test_torso_length_needs_shoulders_and_hips():  # Review Focus 5
    with pytest.raises(ValueError, match="shoulders and hips"):
        torso_length(np.full((3, 33, 5), np.nan))

PAIRS = [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16), (17, 18), (19, 20), (21, 22),
         (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)]

def mirrored(lm):  # every left landmark labelled right and vice versa, as MediaPipe does when it gets the facing wrong
    out = lm.copy()
    for l, r in PAIRS:
        out[[l, r]] = lm[[r, l]]
    return out

def test_global_swap_when_most_frames_face_the_camera():
    lm = np.stack([mirrored(body(HANDS, FEET))] * 10)
    fixed, report = fix_left_right(lm, 100.0, 30.0)
    assert report["global_swap"] and np.allclose(fixed[0, 15, :2], HANDS[0])
    assert report["frames_swapped"] == {"arms": 0, "legs": 0}

def test_no_global_swap_when_facing_the_wall():
    assert not fix_left_right(np.stack([body(HANDS, FEET)] * 10), 100.0, 30.0)[1]["global_swap"]

def test_continuity_swaps_a_flipped_arm_pair():
    lm = np.stack([body(HANDS, FEET)] * 10)
    lm[5, [13, 15, 17, 19, 21]], lm[5, [14, 16, 18, 20, 22]] = lm[5, [14, 16, 18, 20, 22]].copy(), lm[5, [13, 15, 17, 19, 21]].copy()
    fixed, report = fix_left_right(lm, 100.0, 30.0)
    assert report["frames_swapped"] == {"arms": 1, "legs": 0} and np.allclose(fixed[5, 15, :2], HANDS[0])

def test_continuity_leaves_close_hands_alone():
    close = ((240.0, 100.0), (260.0, 100.0))  # 0.2 torso apart: a match, not a swap
    lm = np.stack([body(close, FEET)] * 10)
    lm[5, [15, 16]] = lm[5, [16, 15]].copy()
    assert fix_left_right(lm, 100.0, 30.0)[1]["frames_swapped"]["arms"] == 0

def test_hip_centers():
    assert np.allclose(hip_centers(body(HANDS, FEET)[None])[0], (250, 300))
```

- [x] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_limbs.py -v`
Expected: FAIL (`ModuleNotFoundError: beta.limbs`)

- [x] **Step 3: Implement `src/beta/limbs.py`**

**Left/right pairs** (left, right):
- Face: (1,4), (2,5), (3,6), (7,8), (9,10).
- Arms: (13,14), (15,16), (17,18), (19,20), (21,22).
- Legs: (25,26), (27,28), (29,30), (31,32).
- Torso: (11,12), (23,24).

**Global swap** (all pairs):
- Confident frames are those where shoulders and hips all have visibility ≥ 0.5.
- A frame agrees when `(x12 − x11) + (x24 − x23) > 0`: the climber faces the wall, so their right side is on the image right.
- Swap all pairs when more than 70% of confident frames disagree.

**Continuity**, run after the global swap:
- Compare each frame with the previous frame that has a pose, if it is no more than `ceil(0.2·fps)` frames back. Compare against already-fixed values.
- Arms use the wrists (15, 16) and legs the ankles (27, 28):
  - `keep = |L−L'| + |R−R'|`
  - `swap = |L−R'| + |R−L'|`
- Swap that chain's pairs when `swap < 0.5·keep` and `|L−R| > 0.3·torso`.

The docstring explains why: MediaPipe confuses sides on climbers seen from behind.

- [x] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_limbs.py -v`
Expected: 12 passed

- [x] **Step 5: Commit**

```bash
git add src/beta/limbs.py tests/conftest.py tests/test_limbs.py
git commit -m "feat(contacts): add hand and foot points with a left/right fix"
```

---

### Task 8: Contact state machine

**Files:**
- Create: `src/beta/contacts.py` (the state machine part)
- Modify: `tests/conftest.py` (add `square_hold`)
- Test: `tests/test_contacts.py`

**Interfaces:**
- Consumes: `holds.Hold` (`contour()`).
- Produces:
  - `ContactParams`, a frozen dataclass:
    - Fields: `min_hold_s=0.25`, `release_s=0.15`, `onset_margin=0.05`, `release_margin=0.25`, `max_speed=0.5`, `speed_window_s=0.1`, `min_visibility=0.5`.
    - Methods: `onset_frames(fps) = ceil(min_hold_s·fps)`, `release_frames(fps) = ceil(release_s·fps)`.
  - `Contact(limb: str, hold: int, start_frame: int, end_frame: int)`, a frozen dataclass.
  - `hold_distances(points (F,2), holds, torso) -> (F, H)`: the signed `pointPolygonTest` distance divided by torso. It is positive inside and NaN for NaN points.
  - `limb_speed(points, fps, torso, window_s=0.1) -> (F,)`:
    - In torso/s, using `k = max(1, round(window_s·fps))`.
    - Indices are clipped to `[0, F−1]`, and the result is divided by the actual time between the two frames used.
    - NaN if either point is NaN.
  - `detect_contacts(limb, points, seen, holds, torso, fps, params=ContactParams()) -> list[Contact]`.
  - `busy_frames(contacts: list[Contact], frame_count) -> np.ndarray` (F,) bool.
  - conftest: `square_hold(id, x, y, size=40) -> Hold`, a square with its top-left corner at (x, y).

- [ ] **Step 1: Write the failing tests** in `tests/test_contacts.py`

```python
import numpy as np, pytest
from conftest import square_hold
from beta.contacts import Contact, ContactParams, detect_contacts, hold_distances, limb_speed

A, B = square_hold(1, 100, 100), square_hold(2, 200, 100)  # centers (120,120) and (220,120)
FAR = (0.0, 0.0)

def path(*segments):  # (n_frames, (x, y)) pieces -> (F, 2)
    return np.array([xy for n, xy in segments for _ in range(n)], float)

def run(pts, seen=None, holds=(A, B), torso=100.0, fps=30.0):
    return detect_contacts("LH", pts, np.ones(len(pts), bool) if seen is None else seen, list(holds), torso, fps)

def test_dwell_gives_a_backdated_contact():
    assert run(path((10, FAR), (30, (120, 120)), (20, FAR))) == [Contact("LH", 1, 10, 39)]

def test_jitter_on_a_hold_gives_one_contact():
    pts = path((10, FAR), (40, (120, 120)))
    pts[10:] += np.random.default_rng(0).uniform(-2, 2, (40, 2))
    assert len(run(pts)) == 1

def test_fast_pass_through_gives_none():
    xs = np.linspace(0, 300, 30)  # 10 px/frame = 3 torso/s
    assert run(np.column_stack([xs, np.full(30, 120.0)])) == []

def test_occlusion_does_not_split_a_contact():
    pts, seen = path((10, FAR), (30, (120, 120)), (20, FAR)), np.ones(60, bool)
    pts[20:26], seen[20:26] = (400, 400), False
    assert run(pts, seen) == [Contact("LH", 1, 10, 39)]

def test_moving_to_an_adjacent_hold_switches():
    assert run(path((30, (120, 120)), (30, (220, 120)))) == [Contact("LH", 1, 0, 29), Contact("LH", 2, 30, 59)]

def test_one_frame_excursion_does_not_release():
    pts = path((10, FAR), (30, (120, 120)), (20, FAR))
    pts[20] = (400, 400)
    assert run(pts) == [Contact("LH", 1, 10, 39)]

def test_point_in_two_overlapping_holds_goes_to_the_deeper_one():
    c = square_hold(3, 130, 100)  # overlaps A from x = 130 to 140
    assert run(path((20, (133, 120))), holds=(A, c))[0].hold == 1  # 7 px inside A, 3 px inside c

def test_unseen_frames_pause_the_onset_counter():
    # Slow seen frames inside: 13-16, then 3 unseen, then 20-26 before the leave shows up in the speed window.
    # That's 4 + 7: it only reaches the 8 needed if unseen frames pause the count instead of resetting it.
    pts, seen = path((10, FAR), (20, (120, 120)), (10, FAR)), np.ones(40, bool)
    seen[17:20] = False
    assert run(pts, seen) == [Contact("LH", 1, 10, 29)]

def test_scaling_coordinates_gives_the_same_contacts():
    pts = path((10, FAR), (30, (120, 120)), (30, (220, 120)))
    big = [square_hold(1, 200, 200, 80), square_hold(2, 400, 200, 80)]
    assert run(pts * 2, holds=big, torso=200.0) == run(pts)

def test_frame_rate_does_not_change_times():
    at30 = run(path((10, FAR), (30, (120, 120)), (20, FAR)), fps=30.0)[0]
    at60 = run(path((20, FAR), (60, (120, 120)), (40, FAR)), fps=60.0)[0]
    assert abs(at30.start_frame / 30 - at60.start_frame / 60) <= 1 / 30
    assert abs(at30.end_frame / 30 - at60.end_frame / 60) <= 1 / 30

def test_points_outside_the_canvas_are_fine():  # Review Focus 4
    neg = square_hold(1, -150, -60)
    assert run(path((10, (-500.0, -500.0)), (30, (-130, -40))), holds=(neg,)) == [Contact("LH", 1, 10, 39)]

def test_limb_speed_in_torso_lengths_per_second():
    pts = np.column_stack([np.arange(20) * 10.0, np.zeros(20)])
    assert np.allclose(limb_speed(pts, 30.0, 100.0), 3.0)

def test_hold_distances_are_signed():
    d = hold_distances(np.array([[120.0, 120.0], [90.0, 120.0], [np.nan, np.nan]]), [A], 100.0)
    assert d[0, 0] == pytest.approx(0.2) and d[1, 0] == pytest.approx(-0.1) and np.isnan(d[2, 0])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_contacts.py -v`
Expected: FAIL (`ModuleNotFoundError: beta.contacts`)

- [ ] **Step 3: Implement `detect_contacts`**

The tests don't settle the onset backdating, so it is pinned here. It refines the spec's "backdated to the first frame of the run" so the start is when the hand arrived. The speed window still sees the reach for about 0.1 s after the hand lands, so the first slow frame comes slightly later.

Per frame t, on seen frames only. Unseen frames, or frames with NaN speed, change nothing.

```text
cand(t)   = hold with the largest distance among those with distance >= -onset_margin, else None
on        = current contact hold or None
inside_run: run of consecutive seen frames whose cand == h (its first frame = run_start); a different cand starts a new run
slow_count: seen frames in the current inside_run with speed < max_speed, reset to 0 by a fast frame
onset: slow_count == onset_frames and h != on
       -> end the current contact at min(last_confirmed_inside, run_start - 1); start Contact(h, run_start)
release (while on): distance to `on` < -release_margin counts; a seen frame within the margin resets the count
       and becomes last_confirmed_inside; at release_frames -> end at last_confirmed_inside
end of clip: an open contact ends at last_confirmed_inside
```

The docstring states: speed never releases a contact, because hands rotate and slide on holds.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_contacts.py -v`
Expected: 13 passed

- [ ] **Step 5: Commit**

```bash
git add src/beta/contacts.py tests/conftest.py tests/test_contacts.py
git commit -m "feat(contacts): detect when each limb is on a hold"
```

---

### Task 9: Attempts and falls

**Files:**
- Create: `src/beta/attempts.py`
- Test: `tests/test_attempts.py`

**Interfaces:**
- Consumes: Task 8 `Contact` (imported under `TYPE_CHECKING`, because contacts.py imports attempts in Task 10), `holds.Hold`.
- Produces:
  - Constants: `TOP_MATCH_S = 1.0`, `DROP_TORSOS = 1.0`, `DROP_WINDOW_S = 0.5`, `MAX_GAP_S = 0.5`, `LANDED_S = 2.0`, `LAST_HOLDS_S = 1.5`, `FLOOR_TOLERANCE = 0.15`, `SETTLE_S = 0.25`.
  - Frozen dataclasses:
    - `Release(limb: str, hold: int, frame: int)`
    - `Fall(frame: int, last_holds: dict[str, int | None], release_order: tuple[Release, ...])`
    - `Attempt(start_frame: int, end_frame: int, outcome: str, high_point: int | None, contacts: dict[str, list[Contact]], fall: Fall | None = None)`. `outcome` is `"topped" | "fell" | "came_down" | "clip_ended"`.
  - `Attempt.sequence() -> dict[str, list[int]]`: holds in order, with back-to-back repeats merged.
  - `estimate_floor(feet (F,2,2), feet_seen (F,2), hands_busy (F,), holds, torso, fps) -> float | None`.
  - `feet_on_floor(feet, feet_seen, feet_busy (F,2), floor_y, torso) -> (F,2) bool`.
  - `find_attempts(contacts: dict[str, list[Contact]], on_floor: np.ndarray | None, hips (F,2), holds, torso, fps, top_hold: int | None = None) -> list[Attempt]`.

- [ ] **Step 1: Write the failing tests** in `tests/test_attempts.py`

These use direct arrays at 30 fps with torso 100. Holds come from `square_hold`, ids 1–20, placed so that a higher id is higher on the wall. Hold 20 is the top. Every hold's bbox bottom is ≤ 850.

```python
import numpy as np, pytest
from conftest import square_hold
from beta.attempts import Attempt, estimate_floor, feet_on_floor, find_attempts
from beta.contacts import Contact

HOLDS = [square_hold(i, 300, 800 - 30 * i) for i in range(1, 21)]
F, FPS, TORSO = 240, 30.0, 100.0

def C(limb, hold, a, b): return Contact(limb, hold, a, b)
def contacts(*cs): return {l: [c for c in cs if c.limb == l] for l in ("LH", "RH", "LF", "RF")}
def floor(on_from=()):
    """(F,2) on-floor mask: both feet on the floor in the given [a, b) spans."""
    m = np.zeros((F, 2), bool)
    for a, b in on_from: m[a:b] = True
    return m
def hips(y0=400.0, drops=()):  # drops: (start, end, dy) linear changes in y
    h = np.column_stack([np.full(F, 300.0), np.full(F, y0)])
    for a, b, dy in drops:
        h[a:b, 1] += np.linspace(0, dy, b - a); h[b:, 1] += dy
    return h

def test_sequence_merges_regrips():
    a = Attempt(0, 10, "clip_ended", 7, contacts(C("LH", 7, 0, 3), C("LH", 7, 5, 8), C("LH", 9, 9, 10)))
    assert a.sequence()["LH"] == [7, 9]

def test_floor_is_where_feet_rest_with_hands_free():
    # A short rest at the wall (900), climbing (600), then a much longer stand near the camera (1050).
    feet = np.full((F, 2, 2), 900.0); feet[30:90, :, 1] = 600.0; feet[90:, :, 1] = 1050.0
    busy = np.zeros(F, bool); busy[30:90] = True
    assert estimate_floor(feet, np.ones((F, 2), bool), busy, HOLDS, TORSO, FPS) == pytest.approx(900.0, abs=1)

def test_no_floor_without_resting_feet():
    feet = np.full((F, 2, 2), 600.0)
    assert estimate_floor(feet, np.ones((F, 2), bool), np.zeros(F, bool), HOLDS, TORSO, FPS) is None

def test_feet_on_floor_carry_through_unseen_frames():
    feet = np.full((F, 2, 2), 900.0); seen = np.ones((F, 2), bool); seen[10:20] = False; feet[10:20] = 0.0
    on = feet_on_floor(feet, seen, np.zeros((F, 2), bool), 900.0, TORSO)
    assert on[5:25].all()

def test_attempt_starts_when_both_feet_leave_the_floor():
    cs = contacts(C("LF", 1, 2, 8), C("LH", 3, 10, 50), C("RH", 4, 30, 60))
    a = find_attempts(cs, floor([(0, 20)]), hips(), HOLDS, TORSO, FPS)[0]
    assert a.start_frame == 20 and a.contacts["LH"] == [C("LH", 3, 20, 50)] and a.contacts["LF"] == []

def test_attempt_from_the_first_frame_when_the_clip_starts_on_the_wall():
    assert find_attempts(contacts(C("LH", 3, 0, 50)), floor(), hips(), HOLDS, TORSO, FPS)[0].start_frame == 0

def test_topped():
    cs = contacts(C("LH", 18, 0, 99), C("LH", 20, 100, 200), C("RH", 20, 110, 200))
    a = find_attempts(cs, floor(), hips(), HOLDS, TORSO, FPS)[0]
    assert (a.outcome, a.end_frame, a.high_point) == ("topped", 139, 20)

def test_top_hold_override():
    cs = contacts(C("LH", 18, 100, 200), C("RH", 18, 100, 200))
    assert find_attempts(cs, floor(), hips(), HOLDS, TORSO, FPS, top_hold=18)[0].outcome == "topped"

def test_fall_with_a_pose_gap():
    cs = contacts(C("RF", 2, 0, 55), C("LF", 9, 0, 57), C("LH", 14, 0, 58), C("RH", 15, 0, 59))
    h = hips(drops=[(59, 69, 130.0)]); h[60:69] = np.nan  # 0.3 s without a pose mid-fall
    a = find_attempts(cs, floor([(75, F)]), h, HOLDS, TORSO, FPS)[0]
    assert a.outcome == "fell" and a.end_frame == 59 and a.fall.frame == 59
    assert a.fall.last_holds == {"LH": 14, "RH": 15, "LF": 9, "RF": 2}
    assert [r.limb for r in a.fall.release_order] == ["RF", "LF", "LH", "RH"]

def test_cutting_feet_is_not_a_fall():
    cs = contacts(C("LH", 14, 0, 200), C("RH", 15, 0, 200))
    assert find_attempts(cs, floor(), hips(drops=[(50, 59, 120.0)]), HOLDS, TORSO, FPS)[0].outcome == "clip_ended"

def test_came_down():
    cs = contacts(C("LH", 5, 0, 85), C("RH", 6, 0, 85))
    a = find_attempts(cs, floor([(90, F)]), hips(drops=[(60, 120, 150.0)]), HOLDS, TORSO, FPS)[0]
    assert (a.outcome, a.end_frame) == ("came_down", 90)

def test_two_attempts_and_no_contacts_from_walking_back():
    cs = contacts(C("LH", 14, 0, 59), C("RH", 15, 0, 59), C("RF", 1, 90, 100),  # RF brushes hold 1 walking back
                  C("LH", 3, 160, 230), C("RH", 4, 160, 230))
    on = floor([(75, 150)]); on[90:101, 1] = False  # the busy foot isn't "on the floor", the other one is
    attempts = find_attempts(cs, on, hips(drops=[(59, 69, 130.0)]), HOLDS, TORSO, FPS)
    assert [a.outcome for a in attempts] == ["fell", "clip_ended"] and attempts[1].start_frame == 150
    assert attempts[0].contacts["RF"] == []

def test_single_attempt_without_a_floor():
    a = find_attempts(contacts(C("LH", 3, 40, 100)), None, hips(), HOLDS, TORSO, FPS)
    assert len(a) == 1 and a[0].start_frame == 40 and a[0].outcome == "clip_ended"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_attempts.py -v`
Expected: FAIL (`ModuleNotFoundError: beta.attempts`)

- [ ] **Step 3: Implement `src/beta/attempts.py`**

Put the reasoning for each heuristic in its docstring.

**`estimate_floor`**
- Candidate foot values are foot y in frames where:
  - the foot is seen;
  - no hand is busy;
  - the foot's speed is < 0.5 torso/s (reuse `contacts.limb_speed` via a local import);
  - the foot is below the lowest hold's bbox bottom.
- Bin the candidates (each foot in each frame counts once) into levels `FLOOR_TOLERANCE·torso` tall.
- Walk the levels from the top of the image down. The first level with at least `ceil(SETTLE_S·fps)` candidates is the floor: return the median of the candidates in that level and its two neighbours.
- Return `None` if no level has that much support.
- Why the highest well-supported level, and not a percentile:
  - Someone walking toward the camera has feet lower in the image only because they're closer to it.
  - On the real clip that walk is much longer than any rest at the wall, so it would drag a percentile down.
  - The floor at the wall is the highest place feet stand still when they're not on a hold.

**`feet_on_floor`**
- A foot is on the floor when it is seen, not busy, and `y >= floor_y - FLOOR_TOLERANCE·torso`.
- Unseen frames keep the previous state, starting from `False`.

**`find_attempts`**, scanning forward from frame 0:

```text
settle = ceil(SETTLE_S*fps)
start  = first frame s where neither foot is on the floor for settle frames in a row (with on_floor None:
         first frame of any contact). Attempts with no contacts are dropped.
end    = earliest of:
  topped:    first overlap of LH and RH contacts on top (top_hold or max hold id) reaching ceil(TOP_MATCH_S*fps)
             frames -> end = overlap_start + that - 1
  fell:      hip y with gaps <= MAX_GAP_S filled by linear interpolation; drop_start = first t > start with
             y[t'] - y[t] >= DROP_TORSOS*torso for some t' in (t, t + DROP_WINDOW_S*fps]; confirmed if no hand
             contact covers drop_start + DROP_WINDOW_S*fps and (on_floor None, or a foot is on the floor, or hips
             are NaN) within LANDED_S after it; fall frame = latest hand-contact end in
             [drop_start - LAST_HOLDS_S*fps, drop_start + DROP_WINDOW_S*fps], else drop_start
  came_down: first frame both feet are on the floor for settle frames (the run's first frame)
  clip_ended: last frame
next attempt: search again only after both feet have been on the floor for settle frames since the end
contacts in an attempt: those overlapping [start, end], clipped to it
high_point: max hold id among hand contacts; fall.last_holds: per limb, the hold of its last contact with
end >= fall_frame - LAST_HOLDS_S*fps and start <= fall_frame; release_order: those contacts by end frame
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_attempts.py -v`
Expected: 13 passed

- [ ] **Step 5: Commit**

```bash
git add src/beta/attempts.py tests/test_attempts.py
git commit -m "feat(contacts): split the clip into attempts and describe falls"
```

---

### Task 10: The contacts stage: `analyze`, `contacts.json`, timeline image, `beta contacts`

**Files:**
- Modify: `src/beta/contacts.py` (add `ContactResult`, `analyze`, the JSON functions, `draw_contacts_debug`)
- Modify: `src/beta/cli.py` (add `cmd_contacts` and the `contacts` parser; update the module docstring's stage list)
- Modify: `README.md` (section "3. Hand and foot contacts")
- Test: `tests/test_contacts.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: Task 7 (`wall_landmarks`, `torso_length`, `fix_left_right`, `effector_points`, `hip_centers`, `LIMBS`); Task 8; Task 9; Task 1 `load_camera`; `holds.load_holds`; `pose.load_pose`, `LEFT_COLOR`, `RIGHT_COLOR`.
- Produces:
  - `ContactResult`, a dataclass with fields: `fps`, `frame_count`, `torso_px`, `floor_y: float | None`, `floor_source: str` (`"estimated" | "--floor-y" | "not found"`), `pose_cleaning: dict`, `contacts: dict[str, list[Contact]]`, `seen: np.ndarray` (F,4), `attempts: list[Attempt]`.
  - `analyze(track, holds, camera, params=ContactParams(), top_hold=None, floor_y=None) -> ContactResult`.
  - `contacts_to_dict(result, source, inputs, params) -> dict`, with the spec's JSON shape. Times are rounded to 0.01 s.
  - `save_contacts(path, result, source, inputs, params) -> Path`.
  - `draw_contacts_debug(result, holds, background=None) -> np.ndarray`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_contacts.py`, with a scripted climb at 30 fps on square holds in 640x480. Holds 1–2 are the feet holds and 3–5 the hand holds.

```python
from conftest import body, track_of
from beta.align import CameraTrack, FrameTransform
from beta.contacts import analyze, contacts_to_dict, draw_contacts_debug
from beta.pose import LEFT_COLOR, RIGHT_COLOR

CLIMB_HOLDS = [square_hold(1, 200, 340), square_hold(2, 280, 340), square_hold(3, 180, 160),
               square_hold(4, 300, 160), square_hold(5, 310, 80)]
c = {h.id: h.centroid for h in CLIMB_HOLDS}

def scripted_climb():
    """Stand (0-19), pull on (20), RH 4 -> 5 at 50, fall (90-99), lie on the mat (100-149)."""
    frames = []
    for t in range(150):
        if t < 20 or t >= 100:
            frames.append(body(((200.0, 300.0), (300.0, 300.0)), ((230.0, 450.0), (290.0, 450.0))))
        elif t < 90:
            frames.append(body((c[3], c[5] if t >= 50 else c[4]), (c[1], c[2])))
        else:
            lm = body((c[3], c[5]), (c[1], c[2])); lm[:, 1] += 15.0 * (t - 89); frames.append(lm)
    return track_of(frames)

def test_analyze_a_scripted_climb():
    r = analyze(scripted_climb(), CLIMB_HOLDS, CameraTrack.still(30.0, 150, 640, 480))
    assert r.floor_source == "estimated" and r.floor_y == pytest.approx(450, abs=2)
    (a,) = r.attempts
    assert (a.start_frame, a.outcome) == (20, "fell")
    assert a.sequence() == {"LH": [3], "RH": [4, 5], "LF": [1], "RF": [2]}
    assert a.fall.last_holds == {"LH": 3, "RH": 5, "LF": 1, "RF": 2}

def test_same_contacts_with_and_without_camera_motion():
    track = scripted_climb()
    shifts = [np.array([[1.0, 0, 0], [0, 1.0, 2.0 * t], [0, 0, 1.0]]) for t in range(150)]
    moved = track_of([None if p.landmarks is None else p.landmarks - [0, 2.0 * p.frame, 0, 0, 0] for p in track.frames])
    cam = CameraTrack("moving", 30.0, 150, 640, 480, (640, 800), 0,
                      tuple(FrameTransform(t, s, "track", 99) for t, s in enumerate(shifts)), {})
    assert analyze(moved, CLIMB_HOLDS, cam).contacts == analyze(track, CLIMB_HOLDS, CameraTrack.still(30.0, 150, 640, 480)).contacts

def test_contacts_json_shape():
    r = analyze(scripted_climb(), CLIMB_HOLDS, CameraTrack.still(30.0, 150, 640, 480))
    d = contacts_to_dict(r, {"path": "clip.mp4"}, {"camera_id": "abc"}, ContactParams())
    a = d["attempts"][0]
    assert a["outcome"] == "fell" and a["sequence"]["RH"] == [4, 5] and a["start_s"] == pytest.approx(0.67)
    assert a["contacts"]["LH"][0]["hold"] == 3 and a["fall"]["release_order"][0]["limb"] in ("LF", "RF", "LH", "RH")
    assert d["params"]["min_hold_s"] == 0.25 and d["body_scale_px"] == pytest.approx(100.0)

def test_contacts_debug_image_has_limb_colors():
    r = analyze(scripted_climb(), CLIMB_HOLDS, CameraTrack.still(30.0, 150, 640, 480))
    img = draw_contacts_debug(r, CLIMB_HOLDS, np.zeros((480, 640, 3), np.uint8))
    assert (img == LEFT_COLOR).all(axis=2).any() and (img == RIGHT_COLOR).all(axis=2).any()
```

In `tests/test_cli.py` (also import `body`, `track_of` from conftest and `save_json` from `beta.jsonio`):

```python
from beta.pose import save_pose

def scripted_pose(out, path, holds, frames=31):
    hand, foot = sorted(holds, key=lambda h: h.id)[-3:], sorted(holds, key=lambda h: h.id)[:2]
    lm = [body((hand[0].centroid, (hand[1] if t < 15 else hand[2]).centroid), (foot[0].centroid, foot[1].centroid))
          for t in range(frames)]
    src = {"path": path, "kind": "video", "fps": 15.0, "frame_count": frames, "width": 640, "height": 480, "bit_depth": 8}
    save_pose(out / "pose.json", track_of(lm, fps=15.0), src, {})
    return hand

def test_contacts_end_to_end(climb_video, tmp_path, capsys):
    path, _ = climb_video
    out = tmp_path / "out"
    assert main(["holds", str(path), "--color", "green", "--out", str(out)]) == 0
    hand = scripted_pose(out, path, load_holds(out / "holds.json")[0])
    assert main(["contacts", str(path), "--out", str(out)]) == 0
    doc = load_json(out / "contacts.json")
    assert doc["attempts"][0]["sequence"]["RH"] == [hand[1].id, hand[2].id]
    assert imread(out / "contacts_debug.png").size > 0
    printed = capsys.readouterr().out
    assert "Attempt 1: clip ended mid-attempt" in printed and f"RH: {hand[1].id}, {hand[2].id}" in printed

def test_contacts_accepts_pose_with_fewer_frames_than_the_header(climb_video, tmp_path):  # Review Focus 2
    path, _ = climb_video
    out = tmp_path / "out"
    assert main(["holds", str(path), "--color", "green", "--out", str(out)]) == 0
    scripted_pose(out, path, load_holds(out / "holds.json")[0], frames=30)
    assert main(["contacts", str(path), "--out", str(out)]) == 0

def test_limb_never_seen(climb_video, tmp_path, capsys):  # Review Focus 5
    ...  # as end_to_end, but set landmark 32 (right toe) visibility to 0.1 in every frame before save_pose;
         # assert exit 0, doc["attempts"][0]["sequence"]["RF"] == [] and "RF: -" printed

def test_contacts_refuses_mismatched_inputs(climb_video, wall, tmp_path, capsys):
    path, _ = climb_video
    out = tmp_path / "out"
    assert main(["contacts", str(path), "--out", str(out)]) == 1
    assert "run `beta holds" in capsys.readouterr().err                 # nothing there yet
    assert main(["holds", str(path), "--color", "green", "--out", str(out)]) == 0
    assert main(["contacts", str(path), "--out", str(out)]) == 1
    assert "run `beta pose" in capsys.readouterr().err
    scripted_pose(out, path, load_holds(out / "holds.json")[0])
    doc = load_json(out / "holds.json"); doc["camera"]["id"] = "000000000000"; save_json(out / "holds.json", doc)
    assert main(["contacts", str(path), "--out", str(out)]) == 1
    assert "different camera.json" in capsys.readouterr().err
    src = imwrite(tmp_path / "wall.png", wall[0])
    assert main(["holds", str(src), "--color", "green", "--out", str(out)]) == 0
    assert main(["contacts", str(path), "--out", str(out)]) == 1
    assert "made from an image" in capsys.readouterr().err
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_contacts.py tests/test_cli.py -k "analyze or camera_motion or json_shape or debug_image or contacts" -v`
Expected: FAIL (`ImportError: cannot import name 'analyze'`)

- [ ] **Step 3: Implement `analyze`, the JSON functions and `draw_contacts_debug`**

**`analyze`**, in order:
1. `wall_landmarks`
2. `torso_length`
3. `fix_left_right`
4. `effector_points`
5. per-limb `detect_contacts`
6. the floor: `floor_y` if given (`"--floor-y"`), else `estimate_floor(points[:, 2:], seen[:, 2:], hands_busy, ...)`
7. `feet_on_floor` if there is a floor
8. `find_attempts`

**`draw_contacts_debug`**
- **Timeline:** about 1200 px wide.
  - One 40 px row per limb, labelled LH/RH/LF/RF.
  - Filled `cv2.rectangle` bars, without anti-aliasing, so the colours stay exact: `LEFT_COLOR` for L limbs and `RIGHT_COLOR` for R limbs. Label each bar with its hold id using `holds.put_label`.
  - Unseen stretches in grey (60, 60, 60).
  - Attempt spans shaded, a red marker at each fall, and a seconds axis.
- **Wall panel:** below the timeline, the background scaled to the timeline width.
  - Holds used in any attempt are outlined in their limb's colour and numbered.
  - The high point gets a ring.
  - The floor line is drawn in yellow when it is known.
- **Header:** torso px, the floor and its source, and the outcome of each attempt.

- [ ] **Step 4: Implement `cmd_contacts` and the parser**

**Inputs:** `--pose`, `--holds` and `--camera` default to `out_dir/{pose,holds,camera}.json`. The checks, in order, each fail with a `ValueError` whose message names the command to run:

| Problem | Message |
|---|---|
| `holds.json` missing | `no holds.json in {dir}: run \`beta holds {input} --color ...\` first` |
| `holds.json` has no `camera` key | `{path} was made from an image; contacts needs holds detected on this video: run \`beta holds {input}\`` |
| `pose.json` missing | `no pose.json in {dir}: run \`beta pose {input}\` first` |
| `camera.json` missing | `... run \`beta holds {input}\` ...` |
| Camera id ≠ the holds' camera id | `{holds} was made with a different camera.json: run \`beta holds {input}\` again` |
| Pose and camera describe different videos | `... different videos ...` |

"Different videos" means:
- the file names (`Path(source["path"]).name`) differ, or
- width or height differ, or
- the camera is moving and `camera.frame_count != len(pose frames)`.

A still camera has no frame-count check (Review Focus 2).

**Options:**
- `--top-hold N` must exist in `holds.json`, else `no hold {N} in holds.json`.
- `--floor-y Y` is a float.
- `--min-hold-s S` uses `positive_arg`.
- `--min-visibility V` uses `unit_arg`.

**Background:** read `out_dir/background.png` when its size equals the canvas.

**Output:**

```
Climber's torso: {t:.0f} px. Floor: y={f:.0f} ({source})
Attempt {n}: {topped | fell | came down | clip ended mid-attempt}, {start:.1f}-{end:.1f} s, high point hold {hp or -}
  LH: 8, 11, 14
  RH: -
  ...
  Fell at {t:.1f} s. Last holds: LH 14, RH 15, LF 9, RF -
  Let go: RF 7 at 25.1 s, LH 14 at 25.3 s
Wrote:
  {paths}
```

- When the floor isn't found, print `Floor: not found (pass --floor-y Y to set it), so the clip is treated as one attempt`.
- When there are no attempts, print `No attempts found: no hand or foot stayed on a hold for {min_hold_s} s`.

**`README.md` section 3:** cover the command, the outputs, how to read `contacts_debug.png`, the options, and the known risks from the spec (hidden hands, ceiling false holds needing `--top-hold`).

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest`
Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add src/beta/contacts.py src/beta/cli.py README.md tests/test_contacts.py tests/test_cli.py
git commit -m "feat(cli): add the contacts command, contacts.json and the timeline"
```

---

### Task 11: Real-clip acceptance and tuning

**Files:**
- Modify (only if tuning is needed): `src/beta/contacts.py` (`ContactParams` defaults), `src/beta/attempts.py` (constants)
- Modify: `docs/superpowers/specs/2026-10-28-contacts-and-camera-alignment-design.md` (append a "Tuning log")

- [ ] **Step 1: Run contacts on the real clip**

Run: `uv run beta contacts samples/bouldering_video_sample.mp4 --top-hold <top_hold from the truth file>`

Expected:
- `Attempt 1: fell` at about 25 s (frame ~750–760), and no second attempt.
- Nothing counted after the fall.
- Each limb's sequence equals `samples/bouldering_video_sample.truth.json`.

- [ ] **Step 2: Diagnose each mismatch with `contacts_debug.png` and `pose_debug.mp4`**

For each mismatch, find which of these caused it:
- a grey (unseen) stretch;
- a hold outline missing part of the hold;
- a threshold.

Change a threshold only when that is the cause. Then rerun `uv run pytest`, since the defaults are pinned in tests, so update those tests together with the default.

- [ ] **Step 3: Record each change**

Append a `## Tuning log` section to the design doc. Each entry gives the date, the setting, the old and new value, the evidence (frame numbers and what the debug output showed) and the reason.

- [ ] **Step 4: Check the timing goals**

- In the Task 6 `beta align` output, the moving clip should show ≤ ~25 ms per frame.
- Run `uv run beta align samples/synthetic_climb.avi`. Expected: `Camera is still`, taking about 2 s or less.

- [ ] **Step 5: Commit**

```bash
git add -A src docs tests
git commit -m "test(contacts): check contacts against the first real clip"
```
