from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

from src.processing.analysis import (
    Sample,
    _apply_excludes,
    get_output_paths,
    resolve_samples,
)


def test_apply_excludes_filters_ids():
    samples = [
        {"id": "ttbar", "label": "top"},
        {"id": "higgs", "label": "higgs"},
        {"id": "diboson", "label": "diboson"},
    ]
    result = _apply_excludes(samples, ["higgs"])
    assert len(result) == 2
    assert all(isinstance(s, Sample) for s in result)
    assert {s.id for s in result} == {"ttbar", "diboson"}


def test_apply_excludes_empty_excludes():
    samples = [{"id": "ttbar", "label": "top"}, {"id": "higgs", "label": "higgs"}]
    result = _apply_excludes(samples, [])
    assert len(result) == 2


def test_resolve_samples_disabled_returns_empty(ml_cfg):
    samples = resolve_samples(ml_cfg)
    assert samples["data"] == []
    assert samples["signal"] == []


def test_resolve_samples_enabled_background(ml_cfg):
    samples = resolve_samples(ml_cfg)
    bg = samples["background"]
    assert len(bg) == 2  # higgs excluded
    assert {s.id for s in bg} == {"ttbar", "diboson"}


def test_get_output_paths(ml_cfg):
    OmegaConf.update(ml_cfg, "analysis.region", "SR")
    OmegaConf.update(ml_cfg, "analysis.channel", "1")
    paths = get_output_paths(ml_cfg)
    assert set(paths) == {"samples_dir", "dataframes_dir", "plots_dir", "models_dir"}
    base = paths["plots_dir"].parent
    assert isinstance(base, Path)
    assert "ML" in str(base)
    assert "run2" in str(base)
    assert "SR" in str(base)
    assert paths["samples_dir"] == base / "samples"
    assert paths["dataframes_dir"] == base / "dataframes"
    assert paths["plots_dir"] == base / "plots"
    assert paths["models_dir"] == base / "models"
