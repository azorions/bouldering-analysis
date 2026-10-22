"""JSON helpers shared by every pipeline stage.

Each stage saves its result as JSON so later stages can be rerun without
redoing earlier ones. Two rules keep those files portable:

- NaN is not valid JSON (``json.dumps`` writes a bare ``NaN`` by default, which
  most other parsers reject), so missing values are written as ``null``.
- Every file carries ``schema_version``, so an output from an older format
  fails loudly instead of being silently misread.
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = 1


def to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy types, tuples, paths and dataclasses to plain JSON types."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return to_jsonable(dataclasses.asdict(obj))
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, np.generic):
        return to_jsonable(obj.item())
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, Path):
        return obj.as_posix()
    return obj


def save_json(path: str | Path, obj: dict[str, Any], indent: int | None = 2) -> Path:
    """Write ``obj`` with ``schema_version`` first. Creates parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": SCHEMA_VERSION, **to_jsonable(obj)}
    text = json.dumps(payload, indent=indent, allow_nan=False)
    path.write_text(text + "\n", encoding="utf-8")
    return path


def load_json(path: str | Path) -> dict[str, Any]:
    """Read a file written by :func:`save_json`, rejecting other schema versions."""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    version = data.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"{path}: schema_version {version!r} is not supported (expected {SCHEMA_VERSION}); "
            "rerun the stage that produced it"
        )
    return data
