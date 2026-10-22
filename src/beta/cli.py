"""Command line interface: ``python -m beta <stage> ...`` (or ``beta <stage> ...``).

Each stage reads and writes files in ``outputs/<input name>/`` so it can be
rerun on its own. Milestone 1 stages so far: ``holds``.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Any

from beta.holds import (
    PRESETS,
    UNRELIABLE_COLORS,
    HoldSpec,
    detect_holds,
    draw_holds_debug,
    sample_hsv,
    save_holds,
)
from beta.video import imwrite, is_image, load_input


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


def roi_arg(text: str) -> tuple[int, int, int, int]:
    x, y, w, h = _ints(text, 4, "X,Y,W,H")
    if w <= 0 or h <= 0:
        raise argparse.ArgumentTypeError(f"ROI width and height must be positive, got {text!r}")
    return x, y, w, h


def default_out_dir(input_path: Path) -> Path:
    return Path("outputs") / input_path.stem


def _blobs(n: int) -> str:
    return "1 blob was" if n == 1 else f"{n} blobs were"


def _source_note(source: dict[str, Any]) -> str:
    name = Path(source["path"]).name
    size = f"{source['width']}x{source['height']}"
    if source["kind"] == "image":
        return f"source: {name} (image, {size})"
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

    image, source = load_input(args.input, frame=args.frame, bg_frames=args.bg_frames)

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

    det = detect_holds(image, spec)

    out_dir = args.out or default_out_dir(args.input)
    marks = [args.sample] if args.sample is not None else []
    written = [
        save_holds(out_dir / "holds.json", det, source),
        imwrite(out_dir / "holds_debug.png", draw_holds_debug(image, det, note=_source_note(source), marks=marks)),
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

    return parser


def _print_warning(message, category, filename, lineno, file=None, line=None) -> None:
    """Show library warnings (e.g. "this is HDR video") as plain CLI messages, without file and line."""
    print(f"warning: {message}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        warnings.showwarning = _print_warning
        try:
            return args.func(args)
        except (FileNotFoundError, ValueError, IndexError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
