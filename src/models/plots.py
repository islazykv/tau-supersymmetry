from __future__ import annotations

import matplotlib.pyplot as plt
import xgboost as xgb


def plot_training_curves(
    evals_result: dict[str, dict[str, list[float]]],
    metric: str,
) -> plt.Figure:
    """Plot training and validation loss curves for a single model.

    Parameters
    ----------
    evals_result : dict
        ``model.evals_result()`` output — keys ``"validation_0"`` (train) and
        ``"validation_1"`` (validation), each holding ``{metric: [values]}``.
    metric : str
        Name of the metric to plot (e.g. ``"mlogloss"``, ``"logloss"``).

    Returns
    -------
    plt.Figure
    """
    train_loss = evals_result["validation_0"][metric]
    val_loss = evals_result["validation_1"][metric]
    best_iter = int(val_loss.index(min(val_loss)))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_loss, label="Train", color="royalblue", linewidth=1.5)
    ax.plot(val_loss, label="Validation", color="firebrick", linewidth=1.5)
    ax.axvline(
        best_iter,
        color="grey",
        linestyle="--",
        linewidth=1.0,
        label=f"Best iter: {best_iter}",
    )
    ax.set_xlabel("Boosting Round")
    ax.set_ylabel(metric)
    ax.set_title(f"Training Curves — {metric}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_kfold_training_curves(
    models: list[xgb.XGBClassifier],
    metric: str,
) -> plt.Figure:
    """Plot per-fold validation loss curves for K-fold cross-validation.

    Each fold is drawn as a separate line.  The mean validation curve across
    all folds is overlaid in black.

    Parameters
    ----------
    models : list[xgb.XGBClassifier]
        Fitted models, one per fold (output of :func:`~src.models.bdt.train_kfold`).
    metric : str
        Name of the metric to plot.

    Returns
    -------
    plt.Figure
    """
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

    ax.set_xlabel("Boosting Round")
    ax.set_ylabel(metric)
    ax.set_title(f"K-Fold Validation Curves — {metric}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
