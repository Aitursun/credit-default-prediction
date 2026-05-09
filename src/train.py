"""
Model training utilities for Home Credit Default Risk.

Supports 5 models:
  - Logistic Regression  (logreg)
  - Random Forest        (rf)
  - LightGBM             (lgbm)
  - CatBoost             (catboost)
  - Explainable Boosting Machine (ebm)

All CV is done with StratifiedKFold + OOF predictions.
LogReg and RF receive StandardScaler applied per-fold (fit on train only).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src import config

# Models that require feature scaling
_NEEDS_SCALING = {"logreg", "rf"}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(path: Optional[Path] = None) -> tuple[pd.DataFrame, pd.Series]:
    """Load clean training data.

    Returns
    -------
    X : DataFrame of shape (n_samples, n_features)
    y : Series with binary target
    """
    if path is None:
        path = config.PROCESSED_FILES["final_train_clean"]

    if not Path(path).exists():
        raise FileNotFoundError(
            f"Clean dataset not found: {path}\n"
            "Run notebook 03_build_dataset.ipynb first to generate it."
        )

    df = pd.read_parquet(path)

    missing_cols = [c for c in [config.TARGET_COL, config.ID_COL] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Required columns missing from dataset: {missing_cols}")

    y = df[config.TARGET_COL]
    X = df.drop(columns=[config.TARGET_COL, config.ID_COL])

    # Sanity checks
    inf_count = np.isinf(X.select_dtypes("number")).sum().sum()
    if inf_count > 0:
        print(f"  [WARNING] {inf_count} Inf values found — replacing with NaN then median")
        X = X.replace([np.inf, -np.inf], np.nan)

    nan_count = X.isnull().sum().sum()
    if nan_count > 0:
        print(f"  [WARNING] {nan_count} NaN values remain — filling with column median")
        X = X.fillna(X.median(numeric_only=True))

    print(f"Loaded: {X.shape[0]:,} rows × {X.shape[1]} features")
    print(f"Default rate: {y.mean():.2%}  |  scale_pos_weight: {(y==0).sum()/(y==1).sum():.4f}")
    return X, y


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def get_models(
    scale_pos_weight: float = 11.3872,
    random_state: int = 42,
) -> dict:
    """Return a dict of configured, unfitted estimators.

    Keys: 'logreg', 'rf', 'lgbm', 'catboost', 'ebm'
    """
    from lightgbm import LGBMClassifier
    from catboost import CatBoostClassifier
    from interpret.glassbox import ExplainableBoostingClassifier

    models = {
        "logreg": LogisticRegression(
            C=0.01,
            max_iter=1000,
            class_weight="balanced",
            solver="saga",
            n_jobs=-1,
            random_state=random_state,
        ),
        "rf": RandomForestClassifier(
            n_estimators=500,
            max_depth=20,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        ),
        "lgbm": LGBMClassifier(
            n_estimators=1000,
            learning_rate=0.05,
            num_leaves=63,
            max_depth=-1,
            min_child_samples=50,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            scale_pos_weight=scale_pos_weight,
            n_jobs=-1,
            random_state=random_state,
            verbose=-1,
        ),
        "catboost": CatBoostClassifier(
            iterations=1000,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=3,
            auto_class_weights="Balanced",
            random_seed=random_state,
            verbose=0,
            thread_count=-1,
        ),
        "ebm": ExplainableBoostingClassifier(
            max_bins=256,
            max_interaction_bins=32,
            interactions=10,
            outer_bags=8,
            inner_bags=0,
            learning_rate=0.01,
            max_rounds=5000,
            n_jobs=-1,
            random_state=random_state,
        ),
    }
    return models


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def cross_validate_model(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str = "",
    n_splits: int = 5,
    random_state: int = 42,
) -> dict:
    """Stratified K-Fold CV with OOF predictions.

    Parameters
    ----------
    model_name : str
        Used to decide whether StandardScaler is needed ('logreg', 'rf').

    Returns
    -------
    dict with keys:
        auc_scores, pr_auc_scores, f1_scores, f1_weighted_scores,
        precision_scores, recall_scores, accuracy_scores,
        mean_auc, std_auc, mean_pr_auc,
        mean_f1, mean_f1_weighted, mean_precision, mean_recall, mean_accuracy,
        oof_preds, oof_true, fold_models
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    needs_scaling = model_name in _NEEDS_SCALING

    auc_scores: list[float] = []
    pr_auc_scores: list[float] = []
    f1_scores: list[float] = []
    f1_weighted_scores: list[float] = []
    precision_scores: list[float] = []
    recall_scores: list[float] = []
    accuracy_scores: list[float] = []
    fold_models: list = []
    oof_preds = np.zeros(len(y))
    oof_true = y.to_numpy()

    X_arr = X.to_numpy(dtype=np.float32)
    y_arr = y.to_numpy()

    # Guard: catch any leftover NaN/Inf that slipped through preprocessing
    if not np.isfinite(X_arr).all():
        bad = (~np.isfinite(X_arr)).sum()
        print(f"  [WARNING] {bad} non-finite values in X — replacing with 0")
        X_arr = np.nan_to_num(X_arr, nan=0.0, posinf=0.0, neginf=0.0)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_arr, y_arr), 1):
        X_train, X_val = X_arr[train_idx], X_arr[val_idx]
        y_train, y_val = y_arr[train_idx], y_arr[val_idx]

        if needs_scaling:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)

        fold_model = clone(model)
        fold_model.fit(X_train, y_train)

        proba = fold_model.predict_proba(X_val)[:, 1]
        labels = (proba >= 0.5).astype(int)
        oof_preds[val_idx] = proba

        auc = roc_auc_score(y_val, proba)
        pr_auc = average_precision_score(y_val, proba)
        f1 = f1_score(y_val, labels, pos_label=1, zero_division=0)
        f1_w = f1_score(y_val, labels, average="weighted", zero_division=0)
        prec = precision_score(y_val, labels, pos_label=1, zero_division=0)
        rec = recall_score(y_val, labels, pos_label=1, zero_division=0)
        acc = accuracy_score(y_val, labels)

        auc_scores.append(auc)
        pr_auc_scores.append(pr_auc)
        f1_scores.append(f1)
        f1_weighted_scores.append(f1_w)
        precision_scores.append(prec)
        recall_scores.append(rec)
        accuracy_scores.append(acc)
        fold_models.append(fold_model)

        print(
            f"  Fold {fold}: AUC={auc:.4f}  F1={f1:.4f}  "
            f"Prec={prec:.4f}  Rec={rec:.4f}  Acc={acc:.4f}"
        )

    mean_auc = float(np.mean(auc_scores))
    std_auc = float(np.std(auc_scores))
    mean_pr_auc = float(np.mean(pr_auc_scores))
    mean_f1 = float(np.mean(f1_scores))
    mean_f1_weighted = float(np.mean(f1_weighted_scores))
    mean_precision = float(np.mean(precision_scores))
    mean_recall = float(np.mean(recall_scores))
    mean_accuracy = float(np.mean(accuracy_scores))

    print(
        f"  → Mean AUC: {mean_auc:.4f} ± {std_auc:.4f}  |  "
        f"F1: {mean_f1:.4f}  Prec: {mean_precision:.4f}  "
        f"Rec: {mean_recall:.4f}  Acc: {mean_accuracy:.4f}"
    )

    return {
        "auc_scores": auc_scores,
        "pr_auc_scores": pr_auc_scores,
        "f1_scores": f1_scores,
        "f1_weighted_scores": f1_weighted_scores,
        "precision_scores": precision_scores,
        "recall_scores": recall_scores,
        "accuracy_scores": accuracy_scores,
        "mean_auc": mean_auc,
        "std_auc": std_auc,
        "mean_pr_auc": mean_pr_auc,
        "mean_f1": mean_f1,
        "mean_f1_weighted": mean_f1_weighted,
        "mean_precision": mean_precision,
        "mean_recall": mean_recall,
        "mean_accuracy": mean_accuracy,
        "oof_preds": oof_preds,
        "oof_true": oof_true,
        "fold_models": fold_models,
    }


# ---------------------------------------------------------------------------
# Final model training & persistence
# ---------------------------------------------------------------------------

def train_final_model(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str = "",
) -> object:
    """Train model on full dataset. Applies scaling for logreg/rf.

    For logreg and rf, a StandardScaler is fit on the full training data
    and stored as ``model._scaler``. When loading the model for inference,
    apply this scaler before calling predict_proba:
        X_scaled = model._scaler.transform(X)
        proba = model.predict_proba(X_scaled)[:, 1]
    """
    needs_scaling = model_name in _NEEDS_SCALING
    X_arr = X.to_numpy(dtype=np.float32)
    y_arr = y.to_numpy()

    # Guard: catch any leftover NaN/Inf
    if not np.isfinite(X_arr).all():
        X_arr = np.nan_to_num(X_arr, nan=0.0, posinf=0.0, neginf=0.0)

    if needs_scaling:
        scaler = StandardScaler()
        X_arr = scaler.fit_transform(X_arr)
        model._scaler = scaler  # attached for inference — see docstring

    model.fit(X_arr, y_arr)
    return model


def save_model(model, name: str, models_dir: Optional[Path] = None) -> Path:
    if models_dir is None:
        models_dir = config.MODELS_DIR
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / f"{name}.joblib"
    joblib.dump(model, path)
    size_kb = path.stat().st_size / 1024
    print(f"  Saved {name} → {path}  ({size_kb:.0f} KB)")
    return path


def load_model(name: str, models_dir: Optional[Path] = None) -> object:
    if models_dir is None:
        models_dir = config.MODELS_DIR
    return joblib.load(models_dir / f"{name}.joblib")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

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

    for i, (name, score_list) in enumerate(zip(names, scores), 1):
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
        ("mean_auc",        "ROC-AUC"),
        ("mean_f1",         "F1\n(class-1)"),
        ("mean_f1_weighted","F1\n(weighted)"),
        ("mean_precision",  "Precision"),
        ("mean_recall",     "Recall"),
        ("mean_accuracy",   "Accuracy"),
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


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def build_comparison_table(results: dict[str, dict]) -> pd.DataFrame:
    """Build a summary DataFrame sorted by Mean CV AUC descending.

    Columns
    -------
    ROC-AUC      — основная метрика ранжирования (threshold-free)
    F1 (class 1) — гармоническое среднее Precision/Recall для класса дефолт
    F1 (weighted)— F1 с учётом поддержки каждого класса
    Precision    — доля истинных дефолтов среди предсказанных дефолтов
    Recall       — доля выявленных дефолтов среди всех реальных дефолтов
    Accuracy     — вспомогательная метрика (интерпретировать с осторожностью
                   при дисбалансе классов)
    Threshold = 0.5 применяется для всех пороговых метрик.
    """
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

    df = pd.DataFrame(rows).sort_values("ROC-AUC (mean)", ascending=False).reset_index(drop=True)
    return df
