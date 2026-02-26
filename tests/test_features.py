from __future__ import annotations

from omegaconf import OmegaConf

from src.processing.features import resolve_features


def test_ml_channel1_includes_channel1_features(ml_cfg):
    features = resolve_features(ml_cfg)
    assert "mu_n" in features
    assert "mu_pt" in features
    assert "Mt2" not in features
    assert "sumMT" not in features


def test_ml_channel2_includes_channel2_features(ml_cfg):
    OmegaConf.update(ml_cfg, "analysis.channel", "2")
    features = resolve_features(ml_cfg)
    assert "Mt2" in features
    assert "sumMT" in features
    assert "mu_n" not in features


def test_ml_includes_base_groups(ml_cfg):
    features = resolve_features(ml_cfg)
    for f in [
        "eventClean",
        "isBadTile",
        "tau_isTruthMatchedTau",
        "mcEventWeight",
        "nVtx",
        "tau_n",
        "jet_n",
        "met",
    ]:
        assert f in features


def test_ntuples_scope_returns_flat_list():
    cfg = OmegaConf.create(
        {
            "analysis": {"scope": "NTuples", "channel": None},
            "features": {"features": ["feat_a", "feat_b", "feat_c"]},
        }
    )
    features = resolve_features(cfg)
    assert features == ["feat_a", "feat_b", "feat_c"]
