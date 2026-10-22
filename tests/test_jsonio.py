from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from beta.jsonio import load_json, save_json


def test_round_trip_converts_numpy_and_nan(tmp_path):
    path = save_json(
        tmp_path / "nested" / "out.json",
        {
            "missing": float("nan"),
            "scalar": np.float32(1.5),
            "array": np.array([[1, 2], [3, 4]]),
            "tuple": (1, 2),
            "path": Path("a") / "b.mp4",
            "nested": [{"inf": float("inf")}],
        },
    )
    data = load_json(path)
    assert data == {
        "schema_version": 1,
        "missing": None,
        "scalar": 1.5,
        "array": [[1, 2], [3, 4]],
        "tuple": [1, 2],
        "path": "a/b.mp4",
        "nested": [{"inf": None}],
    }
    assert "NaN" not in path.read_text(encoding="utf-8")


def test_rejects_other_schema_versions(tmp_path):
    path = tmp_path / "old.json"
    path.write_text(json.dumps({"schema_version": 0}), encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        load_json(path)
