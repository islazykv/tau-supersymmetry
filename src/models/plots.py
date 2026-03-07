from __future__ import annotations

import atlas_mpl_style as ampl
import matplotlib.pyplot as plt
import xgboost as xgb


def plot_training_curves(
    evals_result: dict[str, dict[str, list[float]]],
    metric: str,
) -> plt.Figure:
    """Plot training and validation loss curves with the best iteration marked."""
    train_loss = evals_result["validation_0"][metric]
    val_loss = evals_result["validation_1"][metric]
    best_iter = int(val_loss.index(min(val_loss)))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(train_loss, label="Training loss", linewidth=1.5)
    ax.plot(val_loss, label="Validation loss", linewidth=1.5)
    ax.axvline(
        best_iter,
        color="blue",
        linestyle="--",
        linewidth=1.0,
        label=f"Optimal tree number: {best_iter}",
    )
    ax.set_xlabel("Number of Trees")
    ax.set_ylabel(f"Loss ({metric})")
    ax.set_title("Loss Plot")
    ax.legend(
        loc="upper right",
        prop={"size": 14},
        labelspacing=0.1,
        handlelength=1,
        handleheight=1,
    )
    ax.grid(True, alpha=0.3)
    ampl.draw_atlas_label(0.02, 0.97, ax=ax)
    fig.tight_layout()
    return fig


def plot_kfold_training_curves(
    models: list[xgb.XGBClassifier],
    metric: str,
) -> plt.Figure:
    """Plot per-fold validation loss curves with a mean curve overlaid in black."""
    import numpy as np

    fold_curves: list[list[float]] = []
    min_len = None

    fig, ax = plt.subplots(figsize=(9, 5))

    palette = plt.cm.tab10.colors  # type: ignore[attr-defined]
    for fold_idx, model in enumerate(models):
        val_loss = model.evals_result()["validation_1"][metric]
        best_iter = int(val_loss.index(min(val_loss)))
        color = palette[fold_idx % len(palette)]
        ax.plot(
            val_loss,
            label=f"Fold {fold_idx + 1} (best: {best_iter})",
            color=color,
            linewidth=1.2,
            alpha=0.7,
        )
        fold_curves.append(val_loss)
        min_len = len(val_loss) if min_len is None else min(min_len, len(val_loss))

    # Mean curve truncated to the shortest fold
    mean_curve = np.mean([c[:min_len] for c in fold_curves], axis=0)
    ax.plot(
        mean_curve,
        label="Mean",
        color="black",
        linewidth=2.0,
        linestyle="--",
    )

    ax.set_xlabel("Number of Trees")
    ax.set_ylabel(f"Loss ({metric})")
    ax.set_title("Loss Plot")
    ax.legend(
        loc="upper right",
        prop={"size": 14},
        labelspacing=0.1,
        handlelength=1,
        handleheight=1,
    )
    ax.grid(True, alpha=0.3)
    ampl.draw_atlas_label(0.02, 0.97, ax=ax)
    fig.tight_layout()
    return fig
