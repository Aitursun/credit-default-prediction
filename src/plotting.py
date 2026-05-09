from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve


def plot_roc_curves(
    results: dict[str, dict],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot OOF ROC curves for all models on one axes."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")

    colors = ["steelblue", "forestgreen", "darkorange", "crimson", "purple"]
    for (name, res), color in zip(results.items(), colors):
        fpr, tpr, _ = roc_curve(res["oof_true"], res["oof_preds"])
        label = f"{name.upper()}  (AUC={res['mean_auc']:.4f} ± {res['std_auc']:.4f})"
        ax.plot(fpr, tpr, lw=2, color=color, label=label)

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("OOF ROC Curves — Stratified 5-Fold CV")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_feature_importance(
    model,
    feature_names: list[str],
    top_n: int = 30,
    model_name: str = "",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Horizontal bar chart of top-N feature importances (LightGBM / RF / CatBoost)."""
    if not hasattr(model, "feature_importances_"):
        raise AttributeError(
            f"{type(model).__name__} does not have feature_importances_. "
            "Use model-specific explanation methods instead."
        )
    importances = model.feature_importances_
    df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(top_n)
    )

    fig, ax = plt.subplots(figsize=(9, top_n * 0.3 + 1))
    ax.barh(df["feature"][::-1], df["importance"][::-1], color="steelblue")
    ax.set_xlabel("Importance")
    ax.set_title(f"Feature Importance — {model_name} (Top {top_n})")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_cv_score_distribution(
    results: dict[str, dict],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Box plot of per-fold AUC scores across all models."""
    names = list(results.keys())
    scores = [results[n]["auc_scores"] for n in names]

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(scores, labels=[n.upper() for n in names], patch_artist=True)

    colors = ["steelblue", "forestgreen", "darkorange", "crimson", "purple"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    for i, score_list in enumerate(scores, 1):
        ax.scatter([i] * len(score_list), score_list, color="black", zorder=3, s=20)

    ax.set_ylabel("ROC-AUC")
    ax.set_title("CV Score Distribution (5 Folds)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_pr_curves(
    results: dict[str, dict],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot OOF Precision-Recall curves for all models."""
    from sklearn.metrics import precision_recall_curve

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["steelblue", "forestgreen", "darkorange", "crimson", "purple"]

    for (name, res), color in zip(results.items(), colors):
        prec_vals, rec_vals, _ = precision_recall_curve(res["oof_true"], res["oof_preds"])
        label = f"{name.upper()}  (PR-AUC={res['mean_pr_auc']:.4f})"
        ax.plot(rec_vals, prec_vals, lw=2, color=color, label=label)

    baseline = float(np.mean(list(results.values())[0]["oof_true"]))
    ax.axhline(baseline, color="black", linestyle="--", lw=1, label=f"Baseline ({baseline:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("OOF Precision-Recall Curves — Stratified 5-Fold CV")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_metrics_comparison(
    results: dict[str, dict],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Grouped bar chart comparing all key metrics across models."""
    metrics = [
        ("mean_auc",         "ROC-AUC"),
        ("mean_f1",          "F1\n(class-1)"),
        ("mean_f1_weighted", "F1\n(weighted)"),
        ("mean_precision",   "Precision"),
        ("mean_recall",      "Recall"),
        ("mean_accuracy",    "Accuracy"),
    ]

    names = list(results.keys())
    n_models = len(names)
    x = np.arange(len(metrics))
    width = 0.14
    colors = ["steelblue", "forestgreen", "darkorange", "crimson", "purple"]

    fig, ax = plt.subplots(figsize=(13, 6))
    for i, (name, color) in enumerate(zip(names, colors)):
        vals = [results[name][key] for key, _ in metrics]
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=name.upper(), color=color, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{v:.3f}",
                ha="center", va="bottom", fontsize=6.5, rotation=90,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in metrics], fontsize=10)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — All Metrics (Threshold = 0.5)")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.axhline(0.5, color="black", linestyle="--", lw=0.8, alpha=0.4)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_fold_heatmap(
    results: dict[str, dict],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Heatmap of per-fold ROC-AUC scores (models × folds)."""
    names = list(results.keys())
    n_folds = len(results[names[0]]["auc_scores"])
    data = np.array([results[n]["auc_scores"] for n in names])

    vmin = data.min() - 0.002
    vmax = data.max() + 0.002
    fig, ax = plt.subplots(figsize=(n_folds * 1.6 + 1, len(names) * 0.9 + 1.5))
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=vmin, vmax=vmax)

    ax.set_xticks(range(n_folds))
    ax.set_xticklabels([f"Fold {i + 1}" for i in range(n_folds)])
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([n.upper() for n in names])

    for i in range(len(names)):
        for j in range(n_folds):
            ax.text(j, i, f"{data[i, j]:.4f}", ha="center", va="center", fontsize=10, fontweight="bold")

    mean_per_model = data.mean(axis=1)
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(range(len(names)))
    ax2.set_yticklabels([f"μ={v:.4f}" for v in mean_per_model], fontsize=9)

    plt.colorbar(im, ax=ax, label="ROC-AUC", pad=0.12)
    ax.set_title("Per-Fold ROC-AUC Heatmap — Stratified 5-Fold CV")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_metrics_radar(
    results: dict[str, dict],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Radar (spider) chart comparing all models across key metrics."""
    metric_keys   = ["mean_auc", "mean_f1", "mean_f1_weighted", "mean_precision", "mean_recall", "mean_accuracy"]
    metric_labels = ["ROC-AUC", "F1\n(class-1)", "F1\n(weighted)", "Precision", "Recall", "Accuracy"]

    names = list(results.keys())
    colors = ["steelblue", "forestgreen", "darkorange", "crimson", "purple"]

    n = len(metric_keys)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})

    for name, color in zip(names, colors):
        vals = [results[name][k] for k in metric_keys]
        vals += vals[:1]
        ax.plot(angles, vals, lw=2, color=color, label=name.upper())
        ax.fill(angles, vals, color=color, alpha=0.07)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title("Model Comparison — Radar Chart", pad=22, fontsize=13)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.12), fontsize=10)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def build_comparison_table(results: dict[str, dict]) -> pd.DataFrame:
    """Build a summary DataFrame sorted by Mean CV AUC descending."""
    rows = []
    for name, res in results.items():
        row = {
            "Model": name.upper(),
            "ROC-AUC (mean)": round(res["mean_auc"], 4),
            "ROC-AUC (std)": round(res["std_auc"], 4),
            "F1 class-1": round(res["mean_f1"], 4),
            "F1 weighted": round(res["mean_f1_weighted"], 4),
            "Precision": round(res["mean_precision"], 4),
            "Recall": round(res["mean_recall"], 4),
            "Accuracy": round(res["mean_accuracy"], 4),
        }
        for i, auc in enumerate(res["auc_scores"], 1):
            row[f"AUC Fold {i}"] = round(auc, 4)
        rows.append(row)

    return pd.DataFrame(rows).sort_values("ROC-AUC (mean)", ascending=False).reset_index(drop=True)
