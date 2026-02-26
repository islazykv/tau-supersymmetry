from __future__ import annotations

import awkward as ak
import numpy as np

from src.processing.io import load_samples, save_samples


def test_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(42)
    samples = {
        "sample_a": ak.Array({"x": rng.random(5), "y": rng.integers(0, 10, 5)}),
        "sample_b": ak.Array({"x": rng.random(3), "y": rng.integers(0, 10, 3)}),
    }

    save_samples(samples, tmp_path)

    assert (tmp_path / "sample_a.parquet").exists()
    assert (tmp_path / "sample_b.parquet").exists()

    loaded = load_samples(tmp_path, ["sample_a", "sample_b"])
    assert len(loaded["sample_a"]) == 5
    assert len(loaded["sample_b"]) == 3
    assert set(loaded["sample_a"].fields) == {"x", "y"}
