from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src import config

_NEEDS_SCALING = {"logreg", "rf"}


def load_data(path: Optional[Path] = None) -> tuple[pd.DataFrame, pd.Series]:
    """Load clean training data. Returns (X, y)."""
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


def cross_validate_model(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str = "",
    n_splits: int = 5,
    random_state: int = 42,
) -> dict:
    """Stratified K-Fold CV with OOF predictions."""
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


def train_final_model(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str = "",
) -> object:
    """Train model on full dataset. Applies scaling for logreg/rf.

    For logreg and rf, scaler is stored as model._scaler for inference:
        X_scaled = model._scaler.transform(X)
        proba = model.predict_proba(X_scaled)[:, 1]
    """
    needs_scaling = model_name in _NEEDS_SCALING
    X_arr = X.to_numpy(dtype=np.float32)
    y_arr = y.to_numpy()

    if not np.isfinite(X_arr).all():
        X_arr = np.nan_to_num(X_arr, nan=0.0, posinf=0.0, neginf=0.0)

    if needs_scaling:
        scaler = StandardScaler()
        X_arr = scaler.fit_transform(X_arr)
        model._scaler = scaler

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
