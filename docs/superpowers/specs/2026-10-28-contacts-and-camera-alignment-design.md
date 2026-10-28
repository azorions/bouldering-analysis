# Milestone 1 step 3: camera alignment and hold contacts

Status: draft for review, 2026-10-28

## Intent

Turn a bouldering video into what the climber did on the route:

- Which holds each hand and foot used, in order and with times (`LH: 7,10,12,16`).
- How each attempt ended. For a fall, also when it happened and what let go first.

The first real clip (`samples/bouldering_video_sample.mp4`, a failed attempt on a
white route) was filmed by a friend holding the phone, and the camera follows the
climber up the wall. Videos filmed like that are normal, so the pipeline has to
work with a moving camera as well as a still one. A still-camera video must not
get worse results or become noticeably slower because of this.

The fall details feed Milestone 2, which compares a failed attempt with
successful sends.

## What the real clip showed (measured 2026-10-28)

- **Video:** 576x1024 portrait, 30 fps, 1030 frames (34 s).
  - The climber is on the wall until about frame 750 (25 s).
  - Then comes the fall: mid-hip drops ~500 px, about 4.6 torso lengths, between
    frames 750 and 780.
  - After that the climber walks back past the low holds and picks up the phone.
- **Camera motion:**
  - The phone tilts to follow the climber, by up to 115 px vertically (11% of the
    frame), with about 1.5° of rotation and 4% of zoom.
  - Holds are 20-40 px across, so this motion is several holds' worth.
  - The current median background is smeared, and only 6 holds were found.
- **Alignment works:** matching SIFT features to one reference frame and fitting a
  homography (RANSAC) lines every climbing frame up to within
  - 0.6 px (median), and
  - 1.5 px (90th percentile).

  There were 230-640 inlier matches per frame.
- **Timing per frame on this laptop:**

  | Step | Time per frame |
  |---|---|
  | SIFT match | 83 ms |
  | ORB match | 25 ms |
  | KLT optical flow between consecutive frames | 17 ms |
  | Heavy pose model | ~80 ms (84 s for the whole clip) |

- **Pose quality:**
  - Pose is found in 98% of frames.
  - Wrists have visibility ≥ 0.5 in 59-66% of climbing frames, and toes in 73-82%.
  - The clip-level left/right labelling is correct: 99% of frames show the climber
    from behind. The labels flip for only a few frames (around 274-343), plus
    while the climber is on the mat after the fall.
  - The median torso is 108 px.
- **White holds:** on a single sharp frame the white preset finds 21 holds.
  - About 6 of them are the lit ceiling and wall edges at the top.
  - Some shaded halves of cream holds (e.g. the crescent) are missed.

## Goals

1. **Camera alignment.** Work out, for each frame, how it maps onto one fixed view
   of the wall ("wall coordinates").
   - Detect automatically whether the camera is still or moving. There is no flag
     for this.
   - Still clips: about 1-2 s extra, and results identical to today.
   - Moving clips: about 20-25 ms per frame extra, with a median error of 1 px or
     less.
2. **Holds on a moving camera.** Detect holds in wall coordinates, on a median
   background built from aligned frames.
3. **Contacts.** Per limb (LH, RH, LF, RF), find which hold it is on, with start
   and end times, in wall coordinates.
4. **Attempts.** Split the clip into attempts. Each attempt has a start, an
   outcome, a high point and its own per-limb sequence. A fall also records the
   last holds and the order in which limbs let go.
5. **Left/right fix.** Correct MediaPipe's left/right swaps before contacts are
   computed.
6. **Debug visualizations** for both new stages, following the project
   convention.

## Non-goals (this step)

- **Comparing videos taken from different spots.** Aligning several videos to one
  shared photo of the wall is Milestone 2. This step keeps the wall-coordinate
  design so it can grow into that.
- **The annotated video and `track`.** The annotated output video with the holds
  in use highlighted, and the `track` command, are step 4 (render).
- **Better white-hold detection.** It is tuned with the existing `--sample` and
  `--roi` options at the checkpoint below; a segmentation model comes later.
- **Telling a deliberate jump-off from a fall.**
- **Filming styles that break alignment.** These are not supported:
  - the person filming walks far enough sideways that the angled wall panels move
    differently (parallax);
  - views that don't overlap the reference view at all.

## Anti-goals (would make this a failure even if it runs)

- A still-camera clip gives different holds or contacts than it would without
  alignment, or takes noticeably longer.
- Silent wrong answers.
  - When alignment fails on a frame, that frame is marked lost and treated as
    unseen. It never gets a guessed transform.
  - When `holds.json` was built against a different `camera.json` than the one
    contacts is using, contacts refuses to run.
- The walk back after a fall, or standing next to the wall, produces contacts.

## Constraints

- The CLAUDE.md conventions apply:
  - separate stage modules;
  - plain JSON between stages;
  - every stage can be rerun on its own;
  - a debug visualization per stage;
  - short comments explaining non-obvious CV/ML choices.
- Everything runs on the CPU (no MediaPipe GPU delegate on Windows).
- No new dependencies. SIFT and KLT are in the existing `opencv-contrib-python`.
  The timeline image is drawn with OpenCV (matplotlib is not installed).
- Existing behavior for image input to `beta holds` is unchanged.

## Approaches considered

**A. A separate `align` stage that saves per-frame transforms (chosen).**
- `beta align` writes `camera.json`, which holds one 3x3 matrix per frame mapping
  frame pixels to wall coordinates.
- `holds` builds its median from aligned frames.
- `pose` is unchanged and runs on the original frames.
- `contacts` maps pose points into wall coordinates.
- Debug videos map wall-coordinate holds back into each frame.

Why: it fits the stage and JSON convention, pose keeps the original pixels, and a
still clip just records "still".

**B. Write a stabilized copy of the video and run everything unchanged on it.**
This needs less new code, but it was rejected:
- Re-encoding a clip that is already low resolution loses detail.
- Black borders appear wherever the camera pointed away from the reference view.
- Pose would run on warped frames.
- It adds an extra video file and an encoding pass, which works against the speed
  goal.

**C. Align inside each stage that needs it.** Dropped:
- It duplicates work and blurs the boundaries between stages.

**Per-frame alignment method: option 2 was chosen.**
- Option 1, a full SIFT match against the reference on every frame, costs 83 ms
  per frame. That more than doubles the run time.
- Option 2 follows points from frame to frame with KLT optical flow (17 ms) and
  chains the frame-to-frame transforms. The small per-step errors would add up, so
  roughly once a second a full SIFT match against the reference resets the chain.

## Design

### Wall coordinates

- Wall coordinates are the pixels of a canvas that covers everything the camera
  saw, laid out in the reference frame's view.
- For each frame t, `W_t` (3x3) maps frame-t pixels to wall coordinates:
  `W_t = T · H_t`.
  - `H_t` maps frame t onto the reference frame.
  - `T` is a translation that keeps all of the canvas positive.
- **Still camera:** `W_t` is the identity and the canvas is the frame size. Wall
  coordinates are then exactly today's frame pixels, which is what keeps
  still-camera results identical.
- **Moving camera:**
  - The canvas is the bounding box of all aligned frame footprints. That way a
    hold seen only while the camera tilts up still gets detected.
  - The canvas is capped at 2x the frame size in each direction, so one bad
    transform can't blow it up.

Why a homography: a handheld phone mostly rotates rather than moves sideways, and
a pure camera rotation maps the image by a homography exactly, whatever the depth
of each point. The measured 0.6 px error confirms that here.

### `align.py` → `camera.json`, `align_debug.mp4`

1. **Sample.** Take about 25 frames spread evenly over the clip (the same frames
   the median background uses). Work on grayscale copies downscaled to a long side
   of at most 1024 px, so cost doesn't grow with resolution. Scale the transforms
   back to full-resolution pixels.
2. **Still check.**
   - Match each sample to one fixed sample with SIFT and a RANSAC homography.
   - If every sample with enough matches moves the frame corners by less than
     ~1 px, the clip is **still**: write identity transforms and stop.
   - Samples with too few matches are skipped as inconclusive. An example is the
     climber filling the frame.
   - Known gap: a brief wobble that returns to the same view between two samples
     isn't caught.
3. **Reference frame.** Use the sample with the most inlier matches to the other
   samples, which is the most "central" view. This avoids picking a frame where a
   hand covers the lens.
4. **Full pass over every frame (moving clips only).**
   - Track corner features from one frame to the next with KLT optical flow.
   - Fit a RANSAC homography between each pair of consecutive frames and chain it
     onto the previous frame's transform.
   - **Re-anchor** about once a second (`--anchor-every`, default 1 s): match the
     frame directly to the reference with SIFT and replace the chained transform.
     - Record the size of the correction (`anchor_correction_px`) as a drift and
       quality measure.
   - Re-anchor immediately whenever tracking is weak (too few tracked points or
     inliers).
   - **Lost frames:**
     - If re-anchoring also fails, the frame is lost: its `W_t` is null.
     - While lost, keep trying to re-anchor until it succeeds.
     - A transform with an implausible jump also counts as lost. Examples are a
       scale change or rotation far beyond the frame-to-frame motion.
   - The climber moves differently from the wall, so RANSAC treats matches on the
     climber as outliers. Align does not use pose, so the two stages stay
     independent.
5. **`camera.json`** contains:
   - `source`;
   - `params`;
   - `id`: derived from the result, e.g. a hash of the reference frame, the canvas
     and the rounded transforms, so rerunning `align` with the same settings keeps
     the same id and doesn't force `holds` to rerun; later stages record and
     check it;
   - `mode`: `"still"` or `"moving"`;
   - `reference_frame`;
   - `canvas`: `{width, height}`;
   - `frames`: a list of `{frame, W | null, method: "track" | "anchor", inliers}`;
     for still clips, a loader returns the identity transform for every frame;
   - `summary`:
     - max shift in px;
     - lost frame count and longest run of lost frames;
     - median and max anchor correction in px.
6. **`align_debug.mp4`** (moving clips only):
   - Each frame is warped onto the canvas over a faded median background. If the
     alignment is right, the wall stands still and only the climber moves.
   - Hold outlines are drawn when `holds.json` exists.
   - A header shows the frame number, method and inlier count. Lost frames are
     marked in red.
   - For a still clip the CLI just prints that the camera is still.
7. **`beta align <video> [--out DIR] [--no-video] [--anchor-every S]`** prints:
   - still or moving;
   - the max shift;
   - lost frames;
   - the drift correction at anchors;
   - the time taken.

### Changes to existing stages

- **`beta holds <video>`:**
  - If `outputs/<stem>/camera.json` is missing, it runs the alignment first and
    writes `camera.json` (the still check alone takes about a second), and prints
    what it did. This keeps the old one-command usage working.
  - For a moving clip, the median background is the per-pixel median of the
    sampled frames warped onto the canvas. Each pixel only uses the frames that
    actually cover it.
  - `--frame N` warps that one frame onto the canvas.
  - Hold polygons, `background.png` and `holds_debug.png` are all in wall
    coordinates.
  - `holds.json` records the camera `id`, `mode` and canvas size under a new
    `camera` key.
  - `source.width` and `source.height` stay the video's size.
  - If `camera.json` exists but describes a different video (size or frame count
    differ), alignment is rerun.
  - Image input is unchanged and has no camera.
- **`pose`:**
  - `pose.json` is unchanged and stays in frame pixels.
  - For `pose_debug.mp4`, hold outlines are mapped into each frame with `W_t⁻¹`,
    using `camera.json`.
  - With a moving camera and no `camera.json`, holds are not drawn and a warning
    is printed.

### `contacts.py` → `contacts.json`, `contacts_debug.png`

**Inputs:**
- `pose.json`, `holds.json` and `camera.json` from the output folder, each
  overridable with a flag.
- Contacts refuses to run when:
  - the camera ids don't match;
  - `holds.json` has no camera id, because it predates this change; or
  - the files describe different videos (file name or frame count differ).

The error says which stage to rerun.

**1. Wall coordinates.**
- Map all landmarks of each frame through `W_t`.
- Frames with a lost transform count as "no pose".

**2. Left/right fix** (from the original Milestone 1 plan, moved here because
step 2 didn't build it):
- **Clip-level vote.**
  - A climber facing the wall has their right shoulder and hip on the image right.
  - If more than 70% of confident frames disagree, swap all left/right labels.
  - The expected direction is fixed to "facing the wall", with no flag.
- **Per-frame continuity.**
  - Swap an arm pair or a leg pair when matching it to the other side's previous
    position costs less than half as much as keeping it.
  - Only when the two limbs are more than 0.3 torso lengths apart.
- The counts go in `contacts.json` under `pose_cleaning`.

**3. Body scale.**
- Torso length is the median mid-shoulder to mid-hip distance in wall
  coordinates.
- All distances are in torso lengths and all times are in seconds (converted with
  fps), so the thresholds work at any resolution, distance from the wall and frame
  rate.

**4. Effector points.**
- **Hand** = wrist + 1.2·(mean(index, pinky) − wrist).
  - The index and pinky landmarks sit at the knuckles, and curled fingers touch the
    hold just past them.
  - If those landmarks have visibility < 0.5, fall back to
    wrist + 0.4·(wrist − elbow).
- **Foot** = toe + 0.15·(heel − toe), the front of the shoe.
- A point counts as **seen** when its main landmark has visibility ≥ 0.5
  (`--min-visibility`).

**5. Speed.**
- `|p[t+k] − p[t−k]| / (2k/fps)` with k ≈ 0.1 s, in torso lengths per second.
- A windowed displacement both smooths and differentiates the signal.

**6. On which hold.**
- Signed distance to each polygon (`cv2.pointPolygonTest` with distance on),
  divided by torso length.
- A contact can start within 0.05 of a hold's outline, and is released only
  beyond 0.25.
- When a point is inside several holds, the deepest one wins.

**7. State machine per limb.**
- An onset counter counts consecutive seen frames that are inside the same
  candidate hold and slower than 0.5 torso/s.
- When the count reaches `N = ceil(0.25 s · fps)` (`--min-hold-s`, default 0.25),
  the contact starts, backdated to the first frame of the run. Any previous
  contact of that limb ends.
- A contact releases after `M = ceil(0.15 s · fps)` consecutive seen frames beyond
  the release margin.
  - Speed never releases a contact, because hands rotate and slide on holds.
- Frames where the limb isn't seen never start or end a contact, so occlusions are
  bridged.
  - Whether an unseen frame pauses or resets the onset counter is left to the
    builder.
- `end_frame` is the last frame the limb was confirmed inside.

**8. Attempts.**
- **Start:** an attempt starts when both feet are off the floor.
  - Contacts already in progress at that moment are the start holds.
  - Contacts that ended earlier don't count; setting up and false starts are
    ignored.
  - If the clip starts with the climber already on the wall, the attempt starts at
    the first frame.
- **The floor** is a horizontal line in wall coordinates, estimated automatically.
  - Suggested method: where the feet rest, below the lowest hold, while no hand is
    on a hold.
  - The method is left to the builder, but the estimate must be printed and drawn
    on the debug image.
  - It must ignore frames where the climber walks toward the camera. Feet are
    lower in the image there because they're closer to the camera, not because the
    floor is lower. The real clip ends like this.
  - `--floor-y Y` (wall coordinates) overrides the estimate.
- **Outcomes:**

  | Outcome | Condition |
  |---|---|
  | `topped` | Both hands on the top hold together for ≥ 1 s. |
  | `fell` | Mid-hip drops ≥ 1 torso length within 0.5 s (wall coordinates), the hands lose their contacts, and the climber ends up on the floor (or out of view). |
  | `came_down` | Back on the floor without a fast drop. |
  | `clip_ended` | The video ends during the attempt. |

  - **Top hold:** the highest hold, unless `--top-hold N` is given.
  - **Pose gaps during a fall:** pose may drop out briefly from motion blur. A gap
    of up to ~0.5 s during the drop must not break fall detection.
- **Fall details:**
  - the fall time, which is when the last hand comes off;
  - the hold each limb was on just before;
  - the release order: contact ends in the 1.5 s before the fall, ordered by end
    frame.
- **High point:** the highest hold any hand reached in the attempt.
- Everything after an attempt ends is ignored until the next start. Getting back
  on makes attempt 2.

**9. `contacts.json`.** Field names beyond these are left to the builder.

```
{
  "source": {...}, "inputs": {"pose": ..., "holds": ..., "camera": ..., "camera_id": ...},
  "params": {...every threshold above...},
  "body_scale_px": 108.0, "floor_y": ...,
  "pose_cleaning": {"global_swap": false, "frames_swapped": {"arms": 0, "legs": 0}},
  "attempts": [{
    "attempt": 1, "start_frame": 0, "start_s": 0.0, "end_frame": 758, "end_s": 25.3,
    "outcome": "fell", "high_point": 16,
    "sequence": {"LH": [8, 11, 14], "RH": [...], "LF": [...], "RF": [...]},
    "contacts": {"LH": [{"hold": 8, "start_frame": 0, "end_frame": 120, "start_s": 0.0, "end_s": 4.0}, ...], ...},
    "fall": {"frame": 758, "t_s": 25.3, "last_holds": {"LH": 14, "RH": 15, "LF": 9, "RF": null},
             "release_order": [{"limb": "RF", "hold": 7, "frame": 752, "t_s": 25.1}, ...]}
  }]
}
```

- `sequence` merges back-to-back repeats of the same hold, since a re-grip shows
  up as "7" rather than "7, 7".
- `contacts` keeps every contact separately.
- `fall` is null unless the outcome is `fell`.

**10. `contacts_debug.png`.**
- **Timeline:** one row per limb.
  - Left limbs are blue and right limbs orange, matching `pose_debug.mp4`.
  - Bars are labelled with the hold id.
  - Attempt spans are shaded, with the fall marked.
  - Stretches where a limb wasn't seen are greyed, so a missing contact can be
    explained at a glance.
- **Wall panel:** the wall background with the holds used in each attempt outlined
  and numbered, the high point marked and the floor line drawn.

**11. `beta contacts <video> [--out DIR] [--top-hold N] [--floor-y Y] [--min-hold-s S] [--min-visibility V]`**
prints, per attempt:
- the outcome and time span;
- each limb's sequence;
- for a fall: the last holds and the release order.

## Build order and checkpoint

1. `align.py`, `beta align` and its tests (synthetic moving clip, below).
2. `holds` on aligned frames, and the holds overlay in `pose_debug.mp4`.
3. **Checkpoint with the user on the real clip:**
   - Review `holds_debug.png`.
   - Tune the white range (`--sample`, `--roi`) until the route's holds are right.
     False holds at the ceiling don't matter, but missed parts of holds that get
     touched do.
   - Write down the actual sequence per limb using those hold numbers.
   - Name the top hold.
   This is the answer key.
4. `contacts.py`: effectors, left/right fix, the state machine.
5. Attempts and falls.
6. `contacts_debug.png`, `beta contacts`, the README section.
7. Run the full real clip against the answer key and tune thresholds if needed.
   Every threshold change is recorded with its reason.

## Verification

**Synthetic moving clip.**
- Extend `synthetic.py` so `make_climb_video` can render each frame as a warp of a
  larger synthetic wall along a known camera path: a tilt with small rotation and
  zoom, like the real clip. This gives the true `W_t`.
- Tests:
  - Transforms are recovered within 1 px.
  - A still clip is reported as still, with identity transforms.
  - Drift stays bounded over a 600-frame clip because of re-anchoring.
  - Blank frames inserted mid-clip are marked lost, and alignment recovers
    afterwards.
  - The moving climber blob doesn't pull the alignment off.

**Still-camera regression.**
- The existing tests pass unchanged.
- `holds` on the existing still synthetic clip gives the same holds as before.

**Contacts on synthetic keypoint tracks (no video).**

These checks come from the original plan:
- the hand offset and its forearm fallback;
- a dwell gives a backdated contact;
- jitter on a hold gives one contact;
- a fast pass-through gives none;
- an occlusion doesn't split a contact;
- moving to an adjacent hold switches holds;
- a one-frame excursion doesn't release;
- coordinates ×2 give the same contacts;
- 30 and 60 fps give the same times.

New checks:
- left/right vote and continuity swap;
- the same track, with and without a simulated camera transform, gives the same
  contacts;
- attempt start;
- a fall with a pose gap;
- topped;
- came down;
- clip ended;
- two attempts in one clip;
- walking past low holds after a fall gives no contacts;
- `contacts` refuses mismatched camera ids.

**CLI smoke tests** for `align` and `contacts`, like the existing ones.

**Real clip (manual acceptance, since `samples/` isn't in git):**
- The outcome is `fell`, at about 25 s.
- Nothing is counted after the fall.
- Each limb's sequence matches the answer key.
- Alignment adds at most ~25 ms per frame.
- The median anchor correction is at most ~1 px.

## Left to the builder

- **Alignment details:** KLT and SIFT parameters, RANSAC thresholds, the lost-frame
  and implausible-jump thresholds, and how features are re-detected.
- **Background median:** the memory approach for the masked median over the
  canvas, e.g. float32 or chunking.
- **Attempts:** the floor estimation method, within the requirements above; and
  whether unseen frames pause or reset the onset counter.
- **Outputs:** the exact debug image layouts, and JSON field names beyond those
  listed.

## Known risks

- **Hidden hands.** Hands are seen in only about 60% of climbing frames. A hand
  hidden for a whole grip will be missed, and the timeline greys those stretches
  so it shows.
- **Ceiling false holds.** False holds above the real top make "highest hold"
  wrong, so `--top-hold` is needed on this clip.
- **Partly detected holds.** The release margin forgives a lot, but the onset
  margin is only ~5 px here.
- **Parallax and large view changes** (see Non-goals).
