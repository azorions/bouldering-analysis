"""Reading frames from videos and images.

:func:`median_background` needs a still view of the wall: the wall stays put
while the climber moves, so the per-pixel median over many frames is an image
of the empty wall. A moving (handheld) camera is first aligned by
:mod:`beta.align`, and its frames are warped onto one view of the wall before
taking the median (see :func:`load_input`).
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

if TYPE_CHECKING:  # only for type hints: beta.align imports this module
    from beta.align import CameraTrack

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
# Photo formats this OpenCV build can't decode. Without a check, OpenCV's video
# reader opens them as a one-frame "video" (sometimes just one 512 px tile).
UNSUPPORTED_PHOTO_EXTS = {".heic", ".heif", ".avif"}
DEFAULT_FPS = 30.0


@dataclass(frozen=True)
class VideoInfo:
    path: str
    fps: float
    frame_count: int
    width: int
    height: int
    bit_depth: int = 8  # bits per color sample in the encoded video; >8 usually means HDR


def _bit_depth(pixel_format: int) -> int:
    """Bits per sample from OpenCV's CAP_PROP_CODEC_PIXEL_FORMAT tag.

    FFmpeg reports 8-bit formats with ASCII fourccs (b"I420", b"NV12"), and
    high-bit-depth ones either as b"Y3" + two small integers whose last byte is
    the bit depth (b"Y3\\x0b\\x0a" = 10-bit 4:2:0) or as fourccs like b"P010".
    """
    if pixel_format <= 0:
        return 8
    tag = int(pixel_format).to_bytes(4, "little")
    if tag[3] < 32:  # not printable ASCII: a raw bit depth
        return tag[3] or 8
    if tag[:1] in (b"P", b"Y") and tag[1:2] in (b"0", b"2", b"4") and tag[2:].isdigit():
        return int(tag[2:])  # P010, P016, Y210, Y410, ...
    return 8


def is_image(path: str | Path) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTS


def imread(path: str | Path) -> np.ndarray:
    """Read a BGR image. Goes through ``imdecode`` because ``cv2.imread`` fails on
    non-ASCII Windows paths (e.g. a user folder with an accented name)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no such image: {path}")
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"could not decode image: {path}")
    return image


def imwrite(path: str | Path, image: np.ndarray) -> Path:
    """Write an image (format from the extension), creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix, image)
    if not ok:
        raise ValueError(f"could not encode image as {path.suffix}: {path}")
    buf.tofile(path)
    return path


def _open(path: str | Path) -> cv2.VideoCapture:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no such video: {path}")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"OpenCV could not open video: {path}")
    return cap


def probe(path: str | Path) -> VideoInfo:
    """Read fps, frame count and size.

    OpenCV 5 returns -1 (4.x returned 0) for properties a backend can't report,
    so anything <= 0 is treated as unknown and measured directly instead.
    """
    cap = _open(path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        bit_depth = _bit_depth(int(cap.get(cv2.CAP_PROP_CODEC_PIXEL_FORMAT)))
        if width <= 0 or height <= 0:
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f"video has no readable frames: {path}")
            height, width = frame.shape[:2]
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        if count <= 0:
            count = 0
            while cap.grab():
                count += 1
    finally:
        cap.release()
    if not fps or fps <= 0:
        warnings.warn(f"{path}: fps unknown, assuming {DEFAULT_FPS}", stacklevel=2)
        fps = DEFAULT_FPS
    if count <= 0:
        raise ValueError(f"video has no readable frames: {path}")
    return VideoInfo(str(path), float(fps), count, width, height, bit_depth)


def iter_frames(path: str | Path, start: int = 0, stop: int | None = None) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(index, frame_bgr)`` sequentially from ``start`` up to ``stop`` (exclusive)."""
    cap = _open(path)
    try:
        if start:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        i = start
        while stop is None or i < stop:
            ok, frame = cap.read()
            if not ok:
                break
            yield i, frame
            i += 1
    finally:
        cap.release()


def read_frame(path: str | Path, index: int) -> np.ndarray:
    """Read a single frame by index (seeks, so it is fine for a handful of frames).

    On variable-frame-rate phone video, seeking by index can land a few frames
    away from what sequential reading calls frame ``index``. That's harmless
    for hold detection (static camera), but stages that need exact frame
    numbers must read sequentially with :func:`iter_frames`.
    """
    if index < 0:
        raise IndexError(f"frame index must be >= 0, got {index}")
    cap = _open(path)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok:
        raise IndexError(f"could not read frame {index} of {path}")
    return frame


def read_frames(path: str | Path, indices: Iterable[int]) -> list[tuple[int, np.ndarray]]:
    """Read the frames at ``indices`` by reading the clip from the start, never seeking.

    Slower than :func:`read_frame` for a frame near the end, but frame ``i`` is
    exactly what :func:`iter_frames` calls frame ``i``, even on variable-frame-rate
    phone video. Anything that pairs a frame with that frame's camera transform
    needs this. Returns ``(index, frame_bgr)`` in increasing index order; indices
    past the end are skipped (the header frame count can overshoot).
    """
    wanted = {int(i) for i in indices if i >= 0}
    if not wanted:
        return []
    out = []
    cap = _open(path)
    try:
        for i in range(max(wanted) + 1):
            # grab() moves on without converting the frame to BGR; retrieve() only for the wanted ones.
            if not cap.grab():
                break
            if i in wanted:
                ok, frame = cap.retrieve()
                if ok:
                    out.append((i, frame))
    finally:
        cap.release()
    return out


def sample_indices(frame_count: int, n: int) -> list[int]:
    """``n`` frame indices spread evenly over the whole clip (fewer if the clip is short)."""
    n = max(1, min(n, frame_count))
    return sorted({int(i) for i in np.linspace(0, frame_count - 1, n).round()})


def median_background(path: str | Path, n: int = 25, info: VideoInfo | None = None) -> tuple[np.ndarray, int]:
    """Per-pixel temporal median of ``n`` evenly spaced frames.

    With a static camera each wall pixel shows the wall in most frames and the
    climber in only a few, so the median removes the climber and leaves a clean
    wall to detect holds on. It fails if the climber covers the same spot in
    more than half the sampled frames (a long rest); use a single ``--frame``
    then. An odd ``n`` keeps the median an actual pixel value rather than an
    average of two.

    Returns the background and the number of frames actually used.
    """
    info = info or probe(path)
    frames = []
    for i in sample_indices(info.frame_count, n):
        try:
            frames.append(read_frame(path, i))
        except IndexError:
            continue  # the header frame count can overshoot by a frame or two
    if not frames:
        raise ValueError(f"could not read any frames from {path}")
    stack = np.stack(frames)
    return np.median(stack, axis=0).astype(np.uint8), len(frames)


MEDIAN_ROWS = 64  # rows per chunk in masked_median, so the float copy stays small on a big wall view


def masked_median(images: Sequence[np.ndarray], masks: Sequence[np.ndarray]) -> np.ndarray:
    """Per-pixel median over only the images whose mask is > 0 there; 0 where none cover the pixel.

    Each aligned frame covers only part of a moving camera's wall view. The
    black outside its footprint is "no data", not a vote for black, so it is
    turned into NaN, which ``np.nanmedian`` skips.
    """
    out = np.zeros(images[0].shape, np.uint8)
    with warnings.catch_warnings():
        # A pixel no frame covers is all NaN; numpy warns, and the CLI would print it.
        warnings.simplefilter("ignore", RuntimeWarning)
        for y in range(0, out.shape[0], MEDIAN_ROWS):
            rows = slice(y, y + MEDIAN_ROWS)
            stack = np.stack(
                [np.where((m[rows] > 0)[..., None], img[rows].astype(np.float32), np.nan) for img, m in zip(images, masks)]
            )
            out[rows] = np.nan_to_num(np.nanmedian(stack, axis=0), nan=0.0).astype(np.uint8)
    return out


def _aligned_median(path: Path, n: int, info: VideoInfo, camera: CameraTrack) -> tuple[np.ndarray, int]:
    """Like :func:`median_background`, with each sampled frame first warped onto the wall view."""
    images, masks = [], []
    # Read in order, not by seeking: each frame must be paired with its own transform.
    for i, frame in read_frames(path, sample_indices(info.frame_count, n)):
        warped = camera.warp_to_wall(frame, i)
        if warped is not None:  # lost frames have no place on the wall
            images.append(warped[0])
            masks.append(warped[1])
    if not images:
        raise ValueError(f"none of the sampled frames of {path.name} could be aligned with the wall view")
    return masked_median(images, masks), len(images)


def _aligned_frame(path: Path, index: int, camera: CameraTrack) -> np.ndarray:
    got = read_frames(path, [index])
    if not got:
        raise IndexError(f"could not read frame {index} of {path}")
    warped = camera.warp_to_wall(got[0][1], index)
    if warped is None:
        raise ValueError(f"frame {index} of {path.name} couldn't be aligned with the wall view: pick another --frame")
    return warped[0]


def load_input(
    path: str | Path, frame: int | None = None, bg_frames: int = 25, camera: CameraTrack | None = None
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load the image that holds are detected on.

    - image file: the image itself
    - video + ``frame``: that frame
    - video: the median background (see :func:`median_background`)

    With a moving ``camera`` (see :mod:`beta.align`), video frames are warped
    onto its wall view first, so the image is the size of the wall view.
    ``source["width"]`` and ``["height"]`` are always the video frame's size.

    Returns ``(image_bgr, source)`` where ``source`` describes what was used.
    Problems the user should know about (an HDR clip, a still image passed as
    a video) are raised as warnings; the CLI prints them.
    """
    path = Path(path)
    if path.suffix.lower() in UNSUPPORTED_PHOTO_EXTS:
        raise ValueError(
            f"{path.name}: OpenCV can't read {path.suffix.upper()[1:]} photos. Export it as JPEG or PNG "
            "(iPhone: Settings > Camera > Formats > Most Compatible)"
        )
    if is_image(path):
        image = imread(path)
        source = {"path": path, "kind": "image", "frame": None, "background_frames": None}
        source["width"], source["height"] = int(image.shape[1]), int(image.shape[0])
    else:
        info = probe(path)
        if info.frame_count == 1:
            warnings.warn(f"{path.name} has a single frame: it looks like a still image, not a video", stacklevel=2)
        if info.bit_depth > 8:
            # Phones record HDR (10-bit HLG / BT.2020) by default. OpenCV decodes it
            # without tone mapping, so colors come out washed out (lower saturation).
            warnings.warn(
                f"{path.name} is {info.bit_depth}-bit video (probably HDR). OpenCV decodes it with washed-out "
                "colors, so the color presets may miss holds: tune with --sample on background.png, or record "
                "with HDR Video off (iPhone: Settings > Camera > Record Video > HDR Video)",
                stacklevel=2,
            )
        moving = camera is not None and camera.mode == "moving"
        if frame is not None:
            image = _aligned_frame(path, frame, camera) if moving else read_frame(path, frame)
            source = {"path": path, "kind": "video", "frame": frame, "background_frames": None}
        else:
            image, used = _aligned_median(path, bg_frames, info, camera) if moving else median_background(path, bg_frames, info)
            source = {"path": path, "kind": "video", "frame": None, "background_frames": used}
        source["bit_depth"] = info.bit_depth
        source["width"], source["height"] = info.width, info.height
    return image, source
