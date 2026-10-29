# bouldering-analysis
<lb>

A computer vision and data analytics tool designed to analyze bouldering video footage, track climber movement, assess body positioning, and evaluate problem difficulty/grade metrics.

## Setup

Requires [uv](https://docs.astral.sh/uv/). From the repo root:

```
uv sync
uv run pytest
```

Commands run inside the project environment with `uv run python -m beta ...` (or `uv run beta ...`).

`uv run pytest` skips the tests that need the real pose model. Run those with `uv run pytest -m slow` (downloads the 6 MB lite model to `models/` once), or just one with `uv run pytest -m slow -k <test name>` (naming it as `tests/test_pose.py::<test name>` alone still skips it). Set `BETA_POSE_CLIP=path/to/clip.mp4` to also check that the model finds the climber in a real video.

## Usage

### 1. Detect route holds

Hold detection uses color: it finds the holds of one route by their HSV color.

```
uv run python -m beta holds <video-or-image> --color green
```

- For a video, `holds` first checks whether the camera moved (see [Camera movement](#camera-movement)), or reuses the folder's `camera.json` if it was made from the same clip. If the camera moved, each frame is aligned onto one view of the wall first, so holds are found and numbered on that aligned wall view, which can be bigger than one frame.
- Holds are detected on the **median background**: the per-pixel median of 25 frames spread over the clip. The wall stands still (or has been aligned to), so this erases the moving climber, and holds hidden under hands are still found. Use `--frame N` to detect on a single frame instead.
- Color presets: red, orange, yellow, green, blue, purple, pink, white, black. White and black are unreliable, because they match on brightness alone.
- Output goes to `outputs/<input name>/`:
  - `holds.json`: hold polygons, numbered bottom to top.
  - `holds_debug.png`: numbered outlines on the left, the color mask on the right (green = kept, red = rejected).
  - `background.png`: the image the holds were detected on (the aligned wall view for a moving camera).

The presets are only starting points: lighting, white balance and the camera all shift colors. If holds are missed or extra ones appear, read a hold's real color and use a custom range. Take X,Y from **`background.png`** (full-resolution pixels), not from the downscaled `holds_debug.png`, and click on a colorful part of the hold rather than chalk or shadow:

```
uv run python -m beta holds clip.mp4 --color green --sample 812,1004
# HSV at (812, 1004): H=58 S=143 V=121  (inside the current green range)
#   to match this color try: --hsv-lower 48,71,48 --hsv-upper 68,255,255
uv run python -m beta holds clip.mp4 --hsv-lower 48,71,48 --hsv-upper 68,255,255
```

Tune on the video's own `background.png`. A range tuned on a photo can be a few hue units off on video, because OpenCV decodes phone video with a different color matrix.

Other knobs:
- `--min-area-frac F`: smallest hold, as a fraction of the frame (default 2e-5, about 40 px^2 at 1080p). Blobs just under the limit are circled in red in `holds_debug.png` and counted in the output. If those are footholds, lower it.
- `--max-area-frac F`: largest hold (default 0.03). Raise it for close-up shots where one hold fills more of the frame.
- `--roi X,Y,W,H`: only look inside this rectangle, for example to skip a neighbouring route.

To try it without footage:

```
uv run python scripts/make_synthetic.py
uv run python -m beta holds samples/synthetic_wall.png --color green
uv run python -m beta holds samples/synthetic_climb.avi --color green
```

### Camera movement

Friends usually film by holding the phone, and the phone tilts to follow the climber. `beta align` works out how the camera moved, so every frame can be mapped onto one fixed view of the wall (**wall coordinates**). Holds are found in wall coordinates, so their outlines stay on the real holds while the view moves.

```
uv run python -m beta align clip.mp4
```

- It first compares a few frames spread over the clip. If they all show the same view (to within 1 px), the camera is still: wall coordinates are just the frame pixels, and it stops there after about a second.
- For a moving camera it follows points on the wall from frame to frame (cheap), and about once a second (`--anchor-every S`) matches a frame straight to a reference view, so small errors don't add up (drift). On a laptop this takes about 20 ms per frame.
- Output goes to `outputs/<input name>/`:
  - `camera.json`: still or moving and, for a moving camera, one 3x3 matrix per frame that maps the frame's pixels to wall coordinates (`null` for a frame that couldn't be aligned).
  - `align_debug.mp4` (moving camera only; skip it with `--no-video`): every frame warped onto the reference view, over a faded copy of that view.
- The summary shows how far the camera moved, how many frames were aligned, how much drift the anchors corrected, and the time taken.
- `beta holds` runs the alignment itself when the output folder has no `camera.json` for the clip. Run `beta align` on its own to look at the debug video or to change `--anchor-every`.

Things to check in `align_debug.mp4`:
- **The wall stands still.** If the alignment is right, only the climber moves. A wall that jumps, or slides against the faded copy around the frame, means badly aligned frames.
- **The header** shows the frame number, how the frame was aligned (`track`: followed from the last anchor, `anchor`: matched to the reference view) and how many points agreed. Frames that couldn't be aligned say LOST in red, and only the faded copy is shown.

### 2. Track the climber's pose

Pose uses [MediaPipe Pose](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker): for every frame it finds 33 body landmarks (shoulders, wrists, hips, ankles, toes, ...).

```
uv run python -m beta holds clip.mp4 --color green
uv run python -m beta pose clip.mp4
```

- The model is downloaded to `models/` on the first run. `--model heavy` (default, 31 MB) is the most accurate; `--model lite` (6 MB) is about 4x faster, for quick looks. On a laptop CPU heavy takes about 55 ms per frame, so a one-minute 30 fps clip takes under two minutes. Without internet, download the `.task` file by hand and pass it with `--model-path`.
- Output goes to the same `outputs/<input name>/` folder:
  - `pose.json`: per frame, the 33 landmarks as `[x, y, z, visibility, presence]`, or `null` when no climber was found. x and y are pixels in the same coordinates as the hold polygons in `holds.json`. z is depth relative to the hips, left in MediaPipe's units (fractions of the image width, not pixels) and noisy, since one camera can't measure depth well; `landmark_units` in the file lists the units of each field. visibility (0 to 1) says how sure MediaPipe is that the point isn't hidden; it still guesses positions for hidden wrists and feet, so treat low-visibility points with suspicion.
  - `pose_debug.mp4`: the clip with the skeleton drawn on it. The climber's left side is blue and right side is orange; points with low visibility are drawn faint. If `holds.json` is in the folder, the numbered holds are drawn too.
- The summary shows how many frames had a pose, the longest run without one, and the mean visibility of each hand (LH, RH) and foot (LF, RF).

Things to check in `pose_debug.mp4`:
- **Left/right swaps.** From behind, MediaPipe sometimes mixes up the climber's left and right; the blue and orange limbs then jump sides for a few frames.
- **Missed frames.** A climber who is small in a wide shot may not be found. Film closer, or lower `--min-confidence` (default 0.5).

### Assumptions and known limitations

- The camera may be still (tripod or propped phone) or handheld. A phone that turns and tilts to follow the climber is fine.
- Alignment fails when the person filming walks far sideways (angled wall panels then shift by different amounts) or when the climber fills the frame (too little wall left to match).
- Routes are marked by hold color. Gyms that mark routes with tape on multicolor holds aren't supported yet.
- If the climber stays in one spot for more than half the clip, they end up in the median background. Use `--frame N` with a frame where they're out of the way.
- iPhone HEIC photos can't be read. Export as JPEG, or set Settings > Camera > Formats > Most Compatible.
- iPhone HDR video (10-bit, on by default since the iPhone 12) decodes with washed-out colors, so presets may miss holds. The tool warns when it sees one. Tune with `--sample`, or record with Settings > Camera > Record Video > HDR Video off.
