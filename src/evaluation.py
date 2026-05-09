"""Evaluation utilities: optimal threshold, confusion matrix,
calibration, lift curve, business metric."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_recall_curve,
)
from sklearn.calibration import calibration_curve


def find_optimal_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    beta: float = 1.0,
) -> tuple[float, float]:
    """Find the decision threshold that maximises F-beta on OOF predictions.

    Returns
    -------
    threshold : float
    best_fbeta : float
    """
    prec, rec, thresholds = precision_recall_curve(y_true, y_score)
    # precision_recall_curve adds one extra precision/recall point at threshold=1
    denom = beta**2 * prec[:-1] + rec[:-1] + 1e-9
    fbeta = (1 + beta**2) * prec[:-1] * rec[:-1] / denom
    best_idx = int(np.argmax(fbeta))
    return float(thresholds[best_idx]), float(fbeta[best_idx])


def plot_confusion_matrix_at_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    model_name: str = "",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Confusion matrix at a custom decision threshold."""
    y_pred = (y_score >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["No Default", "Default"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix — {model_name}\n(threshold = {threshold:.3f})")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_calibration(
    y_true: np.ndarray,
    y_score: np.ndarray,
    model_name: str = "",
    n_bins: int = 10,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Reliability diagram (calibration curve)."""
    prob_true, prob_pred = calibration_curve(y_true, y_score, n_bins=n_bins, strategy="uniform")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    ax.plot(prob_pred, prob_true, "o-", lw=2, label=model_name)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(f"Calibration Curve — {model_name}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_lift_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    model_name: str = "",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Cumulative Lift curve (sorted by predicted score descending)."""
    sorted_idx = np.argsort(y_score)[::-1]
    y_sorted = np.asarray(y_true)[sorted_idx]
    n = len(y_sorted)
    base_rate = float(y_sorted.mean())
    cum_positive_rate = np.cumsum(y_sorted) / np.arange(1, n + 1)
    lift = cum_positive_rate / base_rate
    pct_pop = np.arange(1, n + 1) / n * 100

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(pct_pop, lift, lw=2, label=model_name)
    ax.axhline(1.0, color="black", linestyle="--", lw=1, label="Random (lift = 1)")
    ax.set_xlabel("% Population contacted (by score desc)")
    ax.set_ylabel("Lift")
    ax.set_title(f"Cumulative Lift Curve — {model_name}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def business_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cost_fn: float = 1.0,
    cost_fp: float = 5.0,
) -> dict:
    """Cost-based business metric for credit default risk.

    Parameters
    ----------
    cost_fn : cost per false negative (missed default — bank loses loan)
    cost_fp : cost per false positive (rejected good customer — lost revenue)

    Returns a dict with total_cost, fn_cost, fp_cost, fn, fp, tn, tp.
    Lower total_cost is better.
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fn_cost = float(fn) * cost_fn
    fp_cost = float(fp) * cost_fp
    return {
        "total_cost": fn_cost + fp_cost,
        "fn_cost": fn_cost,
        "fp_cost": fp_cost,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def evaluate_at_threshold(
    results: dict[str, dict],
    beta: float = 1.0,
) -> pd.DataFrame:
    """For each model find optimal threshold and report all metrics at that threshold.

    Returns a DataFrame with one row per model.
    """
    rows = []
    for name, res in results.items():
        thr, fbeta = find_optimal_threshold(res["oof_true"], res["oof_preds"], beta=beta)
        y_pred = (res["oof_preds"] >= thr).astype(int)
        f1 = f1_score(res["oof_true"], y_pred, pos_label=1, zero_division=0)
        bm = business_metric(res["oof_true"], y_pred)
        rows.append({
            "Model": name.upper(),
            "Optimal threshold": round(thr, 4),
            f"F{beta} score": round(fbeta, 4),
            "F1 at threshold": round(f1, 4),
            "FP": bm["fp"],
            "FN": bm["fn"],
            "Business cost (FN×1 + FP×5)": round(bm["total_cost"], 0),
        })
    return pd.DataFrame(rows).sort_values("F1 at threshold", ascending=False).reset_index(drop=True)
