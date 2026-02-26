from __future__ import annotations

import awkward as ak
import numpy as np
from omegaconf import OmegaConf

from src.processing.merger import (
    assign_class,
    combine_background_signal,
    dict_to_array,
    merge_backgrounds,
)


def _make_array(n: int, seed: int = 0) -> ak.Array:
    rng = np.random.default_rng(seed)
    return ak.Array({"met": rng.random(n), "weight": rng.random(n)})


def test_merge_backgrounds_as_one():
    samples = {"ttbar": _make_array(5), "diboson": _make_array(3)}
    cfg = OmegaConf.create({"merge": {"background_strategy": "as_one"}})
    result = merge_backgrounds(samples, cfg)
    assert list(result.keys()) == ["background"]
    assert len(result["background"]) == 8


def test_merge_backgrounds_as_primary():
    samples = {
        "ttbar": _make_array(5),
        "singletop": _make_array(3),
        "diboson": _make_array(4),
    }
    cfg = OmegaConf.create(
        {
            "merge": {
                "background_strategy": "as_primary",
                "primary_groups": {
                    "topquarks": ["ttbar", "singletop"],
                    "diboson": ["diboson"],
                },
            }
        }
    )
    result = merge_backgrounds(samples, cfg)
    assert set(result.keys()) == {"topquarks", "diboson"}
    assert len(result["topquarks"]) == 8
    assert len(result["diboson"]) == 4


def test_combine_background_signal():
    bg = {"background": _make_array(5)}
    sig = {"signal_100_200": _make_array(3)}
    result = combine_background_signal(bg, sig)
    assert set(result.keys()) == {"background", "signal_100_200"}


def test_assign_class():
    samples = {"bg": _make_array(3), "sig": _make_array(2)}
    assign_class(samples)
    assert ak.all(samples["bg"]["class"] == 0)
    assert ak.all(samples["sig"]["class"] == 1)


def test_dict_to_array():
    samples = {"a": _make_array(3), "b": _make_array(4)}
    result = dict_to_array(samples)
    assert len(result) == 7
