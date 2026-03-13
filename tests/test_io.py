from __future__ import annotations

import awkward as ak
import numpy as np
import pandas as pd

from src.processing.io import load_dataframe, load_samples, save_dataframe, save_samples


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


def test_save_load_dataframe_roundtrip(tmp_path):
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": ["a", "b", "c"]})
    path = tmp_path / "test.parquet"

    save_dataframe(df, path)

    assert path.exists()

    loaded = load_dataframe(path)
    assert len(loaded) == 3
    assert list(loaded.columns) == ["x", "y"]
    assert loaded["x"].tolist() == [1.0, 2.0, 3.0]
