import logging

import hydra
import pyrootutils
from omegaconf import DictConfig

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    cwd=True,
)

from src.processing.analysis import get_output_paths, resolve_samples  # noqa: E402
from src.processing.io import save_samples  # noqa: E402
from src.processing.merger import (  # noqa: E402
    assign_class,
    combine_background_signal,
    merge_backgrounds,
)
from src.processing.processor import process_samples  # noqa: E402

log = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig):
    samples = resolve_samples(cfg)

    log.info(
        "Resolved samples — background: %d, signal: %d",
        len(samples["background"]),
        len(samples["signal"]),
    )

    # --- process ---
    processed = {}

    if samples["background"]:
        log.info("Processing %d background samples", len(samples["background"]))
        processed["background"] = process_samples(
            cfg, "background", [s.id for s in samples["background"]]
        )

    if samples["signal"]:
        log.info("Processing %d signal samples", len(samples["signal"]))
        processed["signal"] = process_samples(
            cfg, "signal", [s.id for s in samples["signal"]]
        )

    # --- merge ---
    if "background" in processed:
        merged = merge_backgrounds(processed["background"], cfg)
        log.info("Merged backgrounds into %d group(s)", len(merged))
    else:
        merged = {}

    if "signal" in processed:
        merged = combine_background_signal(merged, processed["signal"])
        log.info("Combined %d signal sample(s)", len(processed["signal"]))

    assign_class(merged)
    log.info("Assigned class labels: %s", list(merged.keys()))

    # --- save ---
    output_paths = get_output_paths(cfg)
    output_dir = output_paths["output_dir"]
    save_samples(merged, output_dir)
    log.info("Preprocessing complete — output saved to %s", output_dir)


if __name__ == "__main__":
    main()
