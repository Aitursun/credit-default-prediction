"""Backward-compatible re-export facade.

All logic has been moved to dedicated modules:
  src/models.py       — get_models()
  src/training.py     — load_data, cross_validate_model, train_final_model, save_model, load_model
  src/plotting.py     — plot_* functions, build_comparison_table
  src/evaluation.py   — find_optimal_threshold, confusion matrix, calibration, lift, business_metric
  src/interpretation.py — compute_shap_tree, compute_shap_linear, plot_shap_*, compute_ebm_importances

Importing from src.train still works — nothing breaks.
"""

from src.models import get_models
from src.training import (
    load_data,
    cross_validate_model,
    train_final_model,
    save_model,
    load_model,
)
from src.plotting import (
    plot_roc_curves,
    plot_feature_importance,
    plot_cv_score_distribution,
    plot_pr_curves,
    plot_metrics_comparison,
    plot_fold_heatmap,
    plot_metrics_radar,
    build_comparison_table,
)
from src.evaluation import (
    find_optimal_threshold,
    plot_confusion_matrix_at_threshold,
    plot_calibration,
    plot_lift_curve,
    business_metric,
    evaluate_at_threshold,
)
from src.interpretation import (
    compute_shap_tree,
    compute_shap_linear,
    plot_shap_summary,
    plot_shap_dependence,
    compute_ebm_importances,
)

__all__ = [
    "get_models",
    "load_data",
    "cross_validate_model",
    "train_final_model",
    "save_model",
    "load_model",
    "plot_roc_curves",
    "plot_feature_importance",
    "plot_cv_score_distribution",
    "plot_pr_curves",
    "plot_metrics_comparison",
    "plot_fold_heatmap",
    "plot_metrics_radar",
    "build_comparison_table",
    "find_optimal_threshold",
    "plot_confusion_matrix_at_threshold",
    "plot_calibration",
    "plot_lift_curve",
    "business_metric",
    "evaluate_at_threshold",
    "compute_shap_tree",
    "compute_shap_linear",
    "plot_shap_summary",
    "plot_shap_dependence",
    "compute_ebm_importances",
]
