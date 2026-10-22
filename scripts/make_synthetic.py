"""Write a synthetic wall photo and climbing clip to samples/ for trying out `beta holds`.

    uv run python scripts/make_synthetic.py
    uv run python -m beta holds samples/synthetic_wall.png --color green
    uv run python -m beta holds samples/synthetic_climb.avi --color green
"""

from __future__ import annotations

from pathlib import Path

from beta.synthetic import make_climb_video, make_wall
from beta.video import imwrite

if __name__ == "__main__":
    samples = Path("samples")
    image, truth = make_wall(1920, 1080, seed=0)
    print(f"wrote {imwrite(samples / 'synthetic_wall.png', image)}  ({len(truth.of_color('green'))} green holds)")
    make_climb_video(samples / "synthetic_climb.avi", width=960, height=540, n_frames=45, fps=15.0)
    print(f"wrote {samples / 'synthetic_climb.avi'}  (45 frames, a green 'climber' crossing the wall)")
