"""Model interpretation utilities: SHAP (tree/linear) and EBM explanations."""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_shap_tree(
    model,
    X_sample: pd.DataFrame,
    check_additivity: bool = False,
) -> np.ndarray:
    """SHAP values for tree-based models: LightGBM, CatBoost, RandomForest.

    Returns positive-class SHAP array of shape (n_samples, n_features).
    """
    import shap

    explainer = shap.TreeExplainer(model)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sv = explainer.shap_values(X_sample, check_additivity=check_additivity)

    if isinstance(sv, list):
        return sv[1]
    if hasattr(sv, "ndim") and sv.ndim == 3:
        return sv[:, :, 1]
    return sv


def compute_shap_linear(
    model,
    X_sample: pd.DataFrame,
    scaler=None,
) -> np.ndarray:
    """SHAP values for linear models (LogisticRegression).

    If scaler is None and model has _scaler attribute (set by train_final_model),
    it is used automatically.

    Returns SHAP array of shape (n_samples, n_features).
    """
    import shap

    X_arr = X_sample.to_numpy(dtype=np.float32)

    if scaler is None and hasattr(model, "_scaler"):
        scaler = model._scaler

    if scaler is not None:
        X_arr = scaler.transform(X_arr)

    explainer = shap.LinearExplainer(model, X_arr, feature_perturbation="interventional")
    sv = explainer.shap_values(X_arr)

    if isinstance(sv, list):
        return sv[1]
    return sv


def plot_shap_summary(
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
    model_name: str = "",
    max_display: int = 30,
    save_path_beeswarm: Optional[Path] = None,
    save_path_bar: Optional[Path] = None,
) -> None:
    """Beeswarm summary plot + mean |SHAP| bar plot."""
    import shap

    shap.summary_plot(shap_values, X_sample, show=False, max_display=max_display)
    plt.title(f"SHAP Summary — {model_name} (beeswarm, n={len(X_sample):,})")
    plt.tight_layout()
    if save_path_beeswarm:
        plt.savefig(save_path_beeswarm, dpi=150, bbox_inches="tight")
    plt.show()

    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False, max_display=max_display)
    plt.title(f"SHAP Feature Importance — {model_name} (mean |SHAP|)")
    plt.tight_layout()
    if save_path_bar:
        plt.savefig(save_path_bar, dpi=150, bbox_inches="tight")
    plt.show()


def plot_shap_dependence(
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
    top_n: int = 3,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Dependence plots for the top-N features by mean |SHAP|."""
    import shap

    mean_abs = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:top_n]
    top_features = [X_sample.columns[i] for i in top_idx]

    fig, axes = plt.subplots(1, top_n, figsize=(6 * top_n, 5))
    if top_n == 1:
        axes = [axes]

    for ax, feat in zip(axes, top_features):
        try:
            shap.dependence_plot(feat, shap_values, X_sample, ax=ax, show=False)
            ax.set_title(f"SHAP dependence: {feat}", fontsize=9)
        except Exception as e:
            ax.set_title(f"{feat}\n(unavailable: {e})", fontsize=8)
            ax.axis("off")

    plt.suptitle(f"SHAP Dependence — Top {top_n} Features", y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def compute_ebm_importances(model) -> tuple[pd.DataFrame, object]:
    """Extract EBM global term importances.

    Returns
    -------
    df : DataFrame sorted by importance descending (columns: term, importance)
    ebm_global : raw explain_global() object for interactive dashboard
    """
    ebm_global = model.explain_global(name="EBM Global")
    df = pd.DataFrame({
        "term": ebm_global.data()["names"],
        "importance": ebm_global.data()["scores"],
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    return df, ebm_global
