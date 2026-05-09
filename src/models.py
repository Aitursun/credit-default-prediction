from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


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

    return {
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
