"""Command line interface: ``python -m beta <stage> ...`` (or ``beta <stage> ...``).

Each stage reads and writes files in ``outputs/<input name>/`` so it can be
rerun on its own. Milestone 1 stages: ``align``, ``holds``, ``pose``, ``contacts``.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Any

from beta.align import CameraTrack, camera_ref, estimate_camera, load_camera, save_camera, write_align_video
from beta.contacts import OUTCOME_TEXT, ContactParams, analyze, draw_contacts_debug, save_contacts
from beta.jsonio import load_json
from beta.holds import (
    PRESETS,
    UNRELIABLE_COLORS,
    Hold,
    HoldSpec,
    detect_holds,
    draw_holds_debug,
    load_holds,
    sample_hsv,
    save_holds,
)
from beta.pose import (
    MODEL_SIZES_MB,
    MODEL_VARIANTS,
    PoseTrack,
    download_model,
    estimate_poses,
    load_pose,
    model_file,
    open_landmarker,
    save_pose,
    write_pose_video,
)
from beta.video import UNSUPPORTED_PHOTO_EXTS, VideoInfo, imread, imwrite, is_image, iter_frames, load_input, probe


def _ints(text: str, count: int, shape: str) -> tuple[int, ...]:
    parts = [p.strip() for p in text.split(",")]
    try:
        values = tuple(int(p) for p in parts)
    except ValueError:
        values = ()
    if len(values) != count:
        raise argparse.ArgumentTypeError(f"expected {shape} ({count} comma-separated integers), got {text!r}")
    return values


def hsv_arg(text: str) -> tuple[int, int, int]:
    return _ints(text, 3, "H,S,V")  # type: ignore[return-value]


def xy_arg(text: str) -> tuple[int, int]:
    return _ints(text, 2, "X,Y")  # type: ignore[return-value]


def unit_arg(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        value = -1.0
    if not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError(f"expected a number from 0 to 1, got {text!r}")
    return value


def positive_arg(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        value = 0.0
    if not value > 0:  # also rejects nan
        raise argparse.ArgumentTypeError(f"expected a number above 0, got {text!r}")
    return value


def roi_arg(text: str) -> tuple[int, int, int, int]:
    x, y, w, h = _ints(text, 4, "X,Y,W,H")
    if w <= 0 or h <= 0:
        raise argparse.ArgumentTypeError(f"ROI width and height must be positive, got {text!r}")
    return x, y, w, h


def default_out_dir(input_path: Path) -> Path:
    return Path("outputs") / input_path.stem


def _blobs(n: int) -> str:
    return "1 blob was" if n == 1 else f"{n} blobs were"


def _source_note(source: dict[str, Any], camera: CameraTrack | None = None) -> str:
    name = Path(source["path"]).name
    size = f"{source['width']}x{source['height']}"
    if source["kind"] == "image":
        return f"source: {name} (image, {size})"
    if camera is not None and camera.mode == "moving":
        view = f"{camera.canvas[0]}x{camera.canvas[1]} wall view"
        if source["frame"] is not None:
            return f"source: {name} frame {source['frame']}, aligned ({view})"
        return f"source: {name}, median of {source['background_frames']} aligned frames ({view})"
    if source["frame"] is not None:
        return f"source: {name} frame {source['frame']} ({size})"
    return f"source: {name}, median of {source['background_frames']} frames ({size})"


def cmd_holds(args: argparse.Namespace) -> int:
    spec = HoldSpec.from_color(
        args.color,
        args.hsv_lower,
        args.hsv_upper,
        min_area_frac=args.min_area_frac,
        max_area_frac=args.max_area_frac,
        roi=args.roi,
    )
    if spec.color in UNRELIABLE_COLORS:
        print(f"warning: {UNRELIABLE_COLORS[spec.color]}", file=sys.stderr)
    if args.frame is not None and is_image(args.input):
        print("warning: --frame is ignored for image input", file=sys.stderr)

    out_dir = args.out or default_out_dir(args.input)
    camera = None
    if not is_image(args.input) and args.input.suffix.lower() not in UNSUPPORTED_PHOTO_EXTS:  # load_input explains HEIC
        camera = _camera_for(args.input, out_dir, probe(args.input))
    image, source = load_input(args.input, frame=args.frame, bg_frames=args.bg_frames, camera=camera)

    if args.sample is not None:
        x, y = args.sample
        smp = sample_hsv(image, x, y)
        lo, hi = smp.suggested.lower, smp.suggested.upper
        inside = any(r.contains(smp.hsv) for r in spec.ranges)
        print(f"HSV at ({x}, {y}): H={smp.hsv[0]} S={smp.hsv[1]} V={smp.hsv[2]}  ({'inside' if inside else 'OUTSIDE'} the current {spec.color} range)")
        if smp.note and spec.color in PRESETS and spec.color not in UNRELIABLE_COLORS:
            # Suggesting a brightness-only range here would match the whole wall.
            print("  this spot looks grey or dark (wall, chalk, glare or shadow), not a colored hold:")
            print("  sample a more colorful part of the hold. X,Y are pixels in background.png, not holds_debug.png.")
        else:
            print(f"  to match this color try: --hsv-lower {lo[0]},{lo[1]},{lo[2]} --hsv-upper {hi[0]},{hi[1]},{hi[2]}")
            if smp.note:
                print(f"  note: {smp.note}")

    # Size limits are fractions of the video frame, even on a bigger wall view.
    det = detect_holds(image, spec, frame_size=(source["width"], source["height"]))

    marks = [args.sample] if args.sample is not None else []
    written = [
        save_holds(out_dir / "holds.json", det, source, camera=None if camera is None else camera_ref(camera)),
        imwrite(out_dir / "holds_debug.png", draw_holds_debug(image, det, note=_source_note(source, camera), marks=marks)),
        imwrite(out_dir / "background.png", image),
    ]

    counts = det.rejected_counts()
    print(f"Detected {len(det.holds)} {spec.color} holds (rejected: {counts['too_small']} too small, {counts['too_large']} too large)")
    if det.holds:
        print(f"  {'id':>4} {'x':>6} {'y':>6} {'area px':>9}")
        for h in det.holds:
            note = "  cut off by frame edge" if h.touches_border else ""
            print(f"  {h.id:>4} {h.centroid[0]:>6.0f} {h.centroid[1]:>6.0f} {h.area:>9.0f}{note}")
    else:
        print("  No holds found. Check holds_debug.png, then read a hold's color with --sample X,Y")
        print("  (X,Y = a hold's pixel coordinates in background.png).")
    near = det.near_limit()
    if near:
        print(
            f"  {_blobs(len(near))} just under the size limit (largest {near[0].area:.0f} px^2, limit "
            f"{det.min_area_px:.0f} px^2; red circles in holds_debug.png). If footholds are missing, lower "
            f"--min-area-frac (now {spec.min_area_frac:g})."
        )
    if counts["too_large"]:
        print(
            f"  {_blobs(counts['too_large'])} over the size limit ({det.max_area_px:.0f} px^2). If a real hold "
            f"was rejected (close-up shot), raise --max-area-frac (now {spec.max_area_frac:g})."
        )
    print("Wrote:")
    for path in written:
        print(f"  {path}")
    return 0


def _progress(total: int):
    """A frame counter on the terminal; nothing when output is redirected (logs, tests)."""
    if not sys.stderr.isatty():
        return None

    def report(done: int) -> None:
        if done % 10 == 0 or done == total:
            print(f"\r  frame {done}/{total}", end="", file=sys.stderr, flush=True)

    return report


def _video_source(path: Path, info: VideoInfo) -> dict[str, Any]:
    """What a stage that reads the whole clip records about it (frame count from the header)."""
    return {
        "path": path,
        "kind": "video",
        "fps": info.fps,
        "frame_count": info.frame_count,
        "width": info.width,
        "height": info.height,
        "bit_depth": info.bit_depth,
    }


def _print_camera(cam: CameraTrack) -> None:
    s = cam.summary
    if cam.mode == "still":
        print(f"Camera is still (largest shift {s['max_shift_px']:.1f} px over {s['checked_frames']} sampled frames)")
        if s.get("note"):
            print(f"warning: {s['note']}", file=sys.stderr)
        return
    print(f"Camera moves: up to {s['max_shift_px']:.0f} px from frame {cam.reference_frame}'s view")
    total, lost = cam.frame_count, s["lost_frames"]
    line = f"  Aligned {total - lost} of {total} frames"
    if lost:
        run, start = s["longest_lost_run"]
        line += f" ({lost} lost, longest run {run} from frame {start})"
    print(line)
    fix = s["anchor_correction_px"]
    print(f"  Drift corrected at anchors: median {fix['median']:.1f} px, max {fix['max']:.1f} px")
    print(f"  Took {s['seconds']:.1f} s ({1000 * s['seconds'] / max(total, 1):.0f} ms per frame)")


def _align(path: Path, info: VideoInfo, out_dir: Path, anchor_every_s: float = 1.0) -> tuple[CameraTrack, Path]:
    """Estimate the camera and save it to ``out_dir/camera.json``."""
    print(f"Checking whether the camera moves in {path.name} ...")
    cam = estimate_camera(path, info, anchor_every_s=anchor_every_s, progress=_progress(info.frame_count))
    if cam.mode == "moving" and sys.stderr.isatty():
        print(file=sys.stderr)  # end the frame counter's line
    return cam, save_camera(out_dir / "camera.json", cam, _video_source(path, info), {"anchor_every_s": anchor_every_s})


def _camera_for(path: Path, out_dir: Path, info: VideoInfo) -> CameraTrack:
    """The camera from ``out_dir/camera.json`` if it was made from this clip, otherwise estimated and saved now."""
    saved = out_dir / "camera.json"
    if saved.is_file():
        try:
            cam, doc = load_camera(saved)
            src = doc["source"]
            made_from = (Path(src["path"]).name, src["frame_count"], src["width"], src["height"])
        except (KeyError, TypeError, ValueError):  # an old format or a hand edit gone wrong: make a new one
            made_from = None
        if made_from == (path.name, info.frame_count, info.width, info.height):
            return cam
    cam, _ = _align(path, info, out_dir)
    _print_camera(cam)
    return cam


def cmd_align(args: argparse.Namespace) -> int:
    path: Path = args.input
    if is_image(path) or path.suffix.lower() in UNSUPPORTED_PHOTO_EXTS:
        raise ValueError(f"{path.name} is a photo: align needs a video of the climb")
    info = probe(path)
    out_dir = args.out or default_out_dir(path)
    if out_dir.exists() and not out_dir.is_dir():
        raise ValueError(f"output folder {out_dir} is a file")

    cam, camera_path = _align(path, info, out_dir, args.anchor_every)
    written = [camera_path]
    if cam.mode == "moving" and not args.no_video:
        written.append(write_align_video(path, cam, out_dir / "align_debug.mp4"))

    _print_camera(cam)
    print("Wrote:")
    for p in written:
        print(f"  {p}")
    return 0


def _overlay_holds(path: Path, width: int, height: int) -> tuple[list[Hold], CameraTrack | None]:
    """The holds in ``path`` to draw on the debug video, plus the moving camera
    their wall coordinates belong to (None: they are frame pixels).

    No holds (with a warning) if they come from an image of a different size
    than the video, or from a moving camera other than the camera.json next to them.
    """
    try:
        holds, doc = load_holds(path)
        size = (doc["source"]["width"], doc["source"]["height"])
        ref = doc.get("camera") or {}
    except (KeyError, TypeError, ValueError) as exc:  # another JSON file, a hand edit gone wrong, an old format
        raise ValueError(f"{path} is not a holds.json written by `beta holds` ({exc})") from exc
    if size != (width, height):
        print(
            f"warning: {path} was made from a {size[0]}x{size[1]} image but the video is "
            f"{width}x{height}, so its holds are not drawn",
            file=sys.stderr,
        )
        return [], None
    if ref.get("mode") != "moving":  # an image, or a still camera: wall coordinates are frame pixels
        return holds, None
    camera = None
    try:
        camera, _ = load_camera(path.parent / "camera.json")
    except (OSError, KeyError, TypeError, ValueError):
        pass
    if camera is None or camera.id != ref.get("id"):
        print(
            f"warning: {path} was made with a different camera.json (run `beta holds` again), so its holds are not drawn",
            file=sys.stderr,
        )
        return [], None
    return holds, camera


def cmd_pose(args: argparse.Namespace) -> int:
    path: Path = args.input
    if is_image(path) or path.suffix.lower() in UNSUPPORTED_PHOTO_EXTS:
        raise ValueError(f"{path.name} is a photo: pose needs a video of the climb")
    info = probe(path)
    # Everything that can fail on bad options is checked before the slow pose pass.
    out_dir = args.out or default_out_dir(path)
    if out_dir.exists() and not out_dir.is_dir():
        raise ValueError(f"output folder {out_dir} is a file")

    if args.model_path is not None:
        model = args.model_path
        if not model.is_file():
            raise FileNotFoundError(f"no such model file: {model}")
    else:
        model = model_file(args.model)
        if not model.is_file() or model.stat().st_size == 0:  # an empty file is a failed earlier download
            print(f"Downloading the {args.model} pose model (~{MODEL_SIZES_MB[args.model]} MB) to {model} ...")
            download_model(args.model, model)

    # Holds are only drawn on the debug video.
    holds_path = args.holds or out_dir / "holds.json"
    holds, camera = [], None
    if args.no_video:
        if args.holds is not None:
            print("warning: --holds is ignored with --no-video", file=sys.stderr)
    elif args.holds is not None and not args.holds.is_file():
        raise FileNotFoundError(f"no such holds file: {args.holds}")
    elif holds_path.is_file():
        holds, camera = _overlay_holds(holds_path, info.width, info.height)
    out_dir.mkdir(parents=True, exist_ok=True)  # e.g. no write permission: fail now, not after the pose pass

    name = model.name if args.model_path is not None else f"{args.model} model"
    print(f"Finding the climber's pose in {info.frame_count} frames of {path.name} ({name}) ...")
    with open_landmarker(model, args.min_confidence) as landmarker:
        frames = estimate_poses(iter_frames(path), info.fps, landmarker, _progress(info.frame_count))
    if sys.stderr.isatty():
        print(file=sys.stderr)
    if not frames:
        raise ValueError(f"could not read any frames from {path}")
    track = PoseTrack(frames, info.fps, info.width, info.height)

    source = {
        "path": path,
        "kind": "video",
        "fps": info.fps,
        "frame_count": len(frames),
        "width": info.width,
        "height": info.height,
        "bit_depth": info.bit_depth,
    }
    params = {
        "model": None if args.model_path is not None else args.model,
        "model_file": model,
        "min_confidence": args.min_confidence,
        "running_mode": "video",
        "num_poses": 1,
    }
    written = [save_pose(out_dir / "pose.json", track, source, params)]
    if not args.no_video:
        written.append(write_pose_video(path, track, out_dir / "pose_debug.mp4", holds, camera=camera))

    found = track.detected()
    # Rounded down, so a clip with a missed frame never shows 100%.
    print(f"Pose found in {found} of {len(frames)} frames ({100 * found // len(frames)}%)")
    if found == 0:
        print(f"  Is the whole climber in frame? A lower --min-confidence (now {args.min_confidence:g}) may help.")
    else:
        gap, start = track.longest_gap()
        if gap:
            print(f"  longest run without a pose: {gap} frame{'' if gap == 1 else 's'} ({gap / info.fps:.2f} s) from frame {start}")
        vis = track.mean_visibility()
        print("  mean visibility (0-1): " + "  ".join(f"{k} {'-' if v is None else f'{v:.2f}'}" for k, v in vis.items()))
    if holds:
        print(f"  {len(holds)} holds from {holds_path} drawn on the video")
    print("Wrote:")
    for p in written:
        print(f"  {p}")
    return 0


def _contact_inputs(args: argparse.Namespace, out_dir: Path) -> tuple[Path, Path, Path]:
    """holds.json, pose.json and camera.json for the contacts stage, checked to belong together.

    Each problem names the command that fixes it.
    """
    path: Path = args.input
    holds_path = args.holds or out_dir / "holds.json"
    pose_path = args.pose or out_dir / "pose.json"
    camera_path = args.camera or out_dir / "camera.json"
    if not holds_path.is_file():
        raise ValueError(f"no holds.json in {holds_path.parent}: run `beta holds {path} --color ...` first")
    if "camera" not in load_json(holds_path):
        raise ValueError(f"{holds_path} was made from an image; contacts needs holds detected on this video: run `beta holds {path}`")
    if not pose_path.is_file():
        raise ValueError(f"no pose.json in {pose_path.parent}: run `beta pose {path}` first")
    if not camera_path.is_file():
        raise ValueError(f"no camera.json in {camera_path.parent}: run `beta holds {path}` (it checks the camera) first")
    return holds_path, pose_path, camera_path


def cmd_contacts(args: argparse.Namespace) -> int:
    path: Path = args.input
    out_dir = args.out or default_out_dir(path)
    holds_path, pose_path, camera_path = _contact_inputs(args, out_dir)
    try:
        holds, holds_doc = load_holds(holds_path)
        camera, camera_doc = load_camera(camera_path)
        track, pose_doc = load_pose(pose_path)
        same_file = Path(pose_doc["source"]["path"]).name == Path(camera_doc["source"]["path"]).name
    except (KeyError, TypeError) as exc:  # a hand edit gone wrong, an old format
        raise ValueError(f"could not read the inputs in {out_dir} ({exc}): rerun the stages that made them") from exc
    if camera.id != holds_doc["camera"].get("id"):
        raise ValueError(f"{holds_path} was made with a different camera.json: run `beta holds {path}` again")
    # A still camera's frame count is the header's, which can overshoot what pose read by a frame or two.
    if (
        not same_file
        or (track.width, track.height) != (camera.width, camera.height)
        or (camera.mode == "moving" and camera.frame_count != len(track.frames))
    ):
        raise ValueError(f"{pose_path} and {camera_path} are from different videos: run `beta pose {path}` and `beta holds {path}` again")
    if args.top_hold is not None and args.top_hold not in {h.id for h in holds}:
        raise ValueError(f"no hold {args.top_hold} in holds.json")

    params = ContactParams(min_hold_s=args.min_hold_s, min_visibility=args.min_visibility)
    result = analyze(track, holds, camera, params, top_hold=args.top_hold, floor_y=args.floor_y)
    background = None
    background_path = out_dir / "background.png"
    if background_path.is_file():
        image = imread(background_path)
        if (image.shape[1], image.shape[0]) == tuple(camera.canvas):
            background = image
    inputs = {"pose": pose_path, "holds": holds_path, "camera": camera_path, "camera_id": camera.id}
    written = [
        save_contacts(out_dir / "contacts.json", result, pose_doc["source"], inputs, params),
        imwrite(out_dir / "contacts_debug.png", draw_contacts_debug(result, holds, background)),
    ]

    floor = (
        f"y={result.floor_y:.0f} ({result.floor_source})"
        if result.floor_y is not None
        else "not found (pass --floor-y Y to set it), so the clip is treated as one attempt"
    )
    print(f"Climber's torso: {result.torso_px:.0f} px. Floor: {floor}")
    if not result.attempts:
        print(f"No attempts found: no hand or foot stayed on a hold for {params.min_hold_s:g} s")
    for n, a in enumerate(result.attempts, 1):
        hp = "-" if a.high_point is None else a.high_point
        print(f"Attempt {n}: {OUTCOME_TEXT[a.outcome]}, {a.start_frame / result.fps:.1f}-{a.end_frame / result.fps:.1f} s, high point hold {hp}")
        for limb, seq in a.sequence().items():
            print(f"  {limb}: {', '.join(map(str, seq)) or '-'}")
        if a.fall is not None:
            last = ", ".join(f"{limb} {'-' if h is None else h}" for limb, h in a.fall.last_holds.items())
            print(f"  Fell at {a.fall.frame / result.fps:.1f} s. Last holds: {last}")
            order = ", ".join(f"{r.limb} {r.hold} at {r.frame / result.fps:.1f} s" for r in a.fall.release_order)
            print(f"  Let go: {order or '-'}")
    print("Wrote:")
    for p in written:
        print(f"  {p}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="beta", description="Bouldering beta tracker: which holds does the climber use?")
    sub = parser.add_subparsers(dest="command", required=True, metavar="<stage>")

    p = sub.add_parser(
        "holds",
        help="detect the route's holds by color",
        description=(
            "Detect route holds by color on an image, a single video frame, or (default for videos) "
            "the median background of the clip, which removes the climber. Writes holds.json, "
            "holds_debug.png and background.png."
        ),
    )
    p.add_argument("input", type=Path, help="video or image of the wall")
    p.add_argument("--color", choices=list(PRESETS), help="route hold color preset")
    p.add_argument("--hsv-lower", type=hsv_arg, metavar="H,S,V", help="custom lower HSV bound (OpenCV scale: H 0-179, S and V 0-255); overrides the preset")
    p.add_argument("--hsv-upper", type=hsv_arg, metavar="H,S,V", help="custom upper HSV bound; a lower H above the upper H wraps around (for reds)")
    p.add_argument("--frame", type=int, metavar="N", help="detect on frame N instead of the median background (video only)")
    p.add_argument("--bg-frames", type=int, default=25, metavar="N", help="frames sampled for the median background (default: 25)")
    p.add_argument("--roi", type=roi_arg, metavar="X,Y,W,H", help="only look for holds inside this rectangle (pixels)")
    p.add_argument("--min-area-frac", type=float, default=HoldSpec.min_area_frac, metavar="F", help="smallest hold as a fraction of the image area (default: %(default)s)")
    p.add_argument("--max-area-frac", type=float, default=HoldSpec.max_area_frac, metavar="F", help="largest hold as a fraction of the image area (default: %(default)s)")
    p.add_argument(
        "--sample",
        type=xy_arg,
        metavar="X,Y",
        help="print the HSV color at this pixel and a range that matches it. X,Y are pixel coordinates in "
        "background.png (full resolution), not in the downscaled holds_debug.png",
    )
    p.add_argument("--out", type=Path, metavar="DIR", help="output directory (default: outputs/<input name>)")
    p.set_defaults(func=cmd_holds)

    p = sub.add_parser(
        "align",
        help="line up a handheld camera's frames with one view of the wall",
        description=(
            "Check whether the camera moves and, if it does, work out for every frame how to map it onto one fixed "
            "view of the wall (wall coordinates), so holds stay put while the phone tilts. Writes camera.json and, "
            "for a moving camera, align_debug.mp4 (every frame warped onto that view)."
        ),
    )
    p.add_argument("input", type=Path, help="video of the climb")
    p.add_argument("--anchor-every", type=positive_arg, default=1.0, metavar="S", help="re-match a frame to the reference view every S seconds, to stop drift (default: %(default)s)")
    p.add_argument("--no-video", action="store_true", help="only write camera.json, skip align_debug.mp4")
    p.add_argument("--out", type=Path, metavar="DIR", help="output directory (default: outputs/<input name>)")
    p.set_defaults(func=cmd_align)

    p = sub.add_parser(
        "pose",
        help="find the climber's body pose in every frame",
        description=(
            "Run MediaPipe Pose on every frame of a video. Writes pose.json (33 body landmarks per frame, in "
            "pixels) and pose_debug.mp4 (the skeleton drawn on the clip, with the holds from holds.json when "
            "the output directory has one). The model is downloaded to models/ on first use."
        ),
    )
    p.add_argument("input", type=Path, help="video of the climb")
    p.add_argument("--model", choices=MODEL_VARIANTS, default="heavy", help="model size: lite is fastest, heavy is most accurate (default: %(default)s)")
    p.add_argument("--model-path", type=Path, metavar="FILE", help="use this .task model file instead of downloading one")
    p.add_argument("--min-confidence", type=unit_arg, default=0.5, metavar="C", help="MediaPipe's detection, presence and tracking thresholds, 0 to 1 (default: %(default)s)")
    p.add_argument("--holds", type=Path, metavar="FILE", help="holds.json to draw on the video (default: holds.json in the output directory, if there is one)")
    p.add_argument("--no-video", action="store_true", help="only write pose.json, skip pose_debug.mp4")
    p.add_argument("--out", type=Path, metavar="DIR", help="output directory (default: outputs/<input name>)")
    p.set_defaults(func=cmd_pose)

    p = sub.add_parser(
        "contacts",
        help="which hold each hand and foot is on, and how each attempt ended",
        description=(
            "Combine pose.json, holds.json and camera.json from the output directory into each hand's and foot's "
            "hold sequence, split into attempts (topped, fell, came down). Writes contacts.json and "
            "contacts_debug.png (a timeline of every contact above the wall)."
        ),
    )
    p.add_argument("input", type=Path, help="video of the climb (names the output directory)")
    p.add_argument("--top-hold", type=int, metavar="N", help="the route's top hold, matched with both hands to top out (default: the highest-numbered hold)")
    p.add_argument("--floor-y", type=float, metavar="Y", help="the floor's y in background.png pixels, if the estimate is wrong")
    p.add_argument("--min-hold-s", type=positive_arg, default=ContactParams.min_hold_s, metavar="S", help="how long a hand or foot must stay on a hold to count (default: %(default)s)")
    p.add_argument("--min-visibility", type=unit_arg, default=ContactParams.min_visibility, metavar="V", help="ignore hands and feet MediaPipe is less sure of than this, 0 to 1 (default: %(default)s)")
    p.add_argument("--pose", type=Path, metavar="FILE", help="pose.json to use (default: in the output directory)")
    p.add_argument("--holds", type=Path, metavar="FILE", help="holds.json to use (default: in the output directory)")
    p.add_argument("--camera", type=Path, metavar="FILE", help="camera.json to use (default: in the output directory)")
    p.add_argument("--out", type=Path, metavar="DIR", help="output directory (default: outputs/<input name>)")
    p.set_defaults(func=cmd_contacts)

    return parser


def _print_warning(message, category, filename, lineno, file=None, line=None) -> None:
    """Show library warnings (e.g. "this is HDR video") as plain CLI messages, without file and line."""
    print(f"warning: {message}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    # On Windows, output redirected to a file or pipe is encoded as cp1252, which
    # can't hold e.g. a Japanese file name. Print a \u escape instead of crashing.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    args = build_parser().parse_args(argv)
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        warnings.showwarning = _print_warning
        try:
            return args.func(args)
        except (OSError, ValueError, IndexError) as exc:  # OSError includes FileNotFoundError and PermissionError
            print(f"error: {exc}", file=sys.stderr)
            return 1
