from typing import Optional
import numpy as np
import pandas as pd


def analyze_missing(df: pd.DataFrame) -> pd.DataFrame:
    missing = df.isnull().sum()
    missing_pct = missing / len(df) * 100
    result = pd.DataFrame({
        "missing_count": missing,
        "missing_pct": missing_pct.round(2),
        "dtype": df.dtypes,
        "nunique": df.nunique(),
    })
    return result[result["missing_count"] > 0].sort_values("missing_pct", ascending=False)


def get_cols_to_drop(df: pd.DataFrame, target_col: str = "TARGET") -> list:
    dropped = []
    y = df[target_col]

    for col in df.columns:
        if col in (target_col, "SK_ID_CURR"):
            continue

        series = df[col]

        if series.nunique() <= 1:
            dropped.append(col)
            continue

        top_freq = series.value_counts(normalize=True).iloc[0]
        if top_freq >= 0.99:
            dropped.append(col)
            continue

        missing_pct = series.isnull().mean()
        if missing_pct > 0.8:
            ext_sources = ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3")
            if any(col.startswith(s) for s in ext_sources):
                continue
            try:
                corr = series.corr(y)
                if pd.isna(corr) or abs(corr) < 0.01:
                    dropped.append(col)
            except Exception:
                dropped.append(col)
            continue

    seen = {}
    for col in df.columns:
        if col in dropped or col in (target_col, "SK_ID_CURR"):
            continue
        key = tuple(df[col].dropna().values[:500])
        if key in seen:
            dropped.append(col)
        else:
            seen[key] = col

    return list(set(dropped))


_HISTORY_PREFIXES = (
    "BUREAU_", "PREV_", "INS_", "CC_", "POS_",
    "BB_",
)
_EXT_SOURCES = ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3")


def _is_binary(series: pd.Series) -> bool:
    vals = series.dropna().unique()
    return set(vals).issubset({0, 1})


def fill_missing(df: pd.DataFrame, train_stats: Optional[dict] = None) -> tuple:
    df = df.copy()
    is_train = train_stats is None
    stats_dict = train_stats.copy() if train_stats else {}

    for col in df.columns:
        if df[col].isnull().sum() == 0:
            continue
        if df[col].dtype == object:
            df[col] = df[col].fillna("Unknown")
            continue

        if any(col.startswith(p) for p in _HISTORY_PREFIXES):
            df[col] = df[col].fillna(0)
            continue

        if any(col == s for s in _EXT_SOURCES):
            if is_train:
                med = df[col].median()
                stats_dict[f"{col}_median"] = med
            else:
                med = stats_dict.get(f"{col}_median", df[col].median())
            flag_col = f"{col}_MISSING"
            df[flag_col] = df[col].isnull().astype(np.int8)
            df[col] = df[col].fillna(med)
            continue

        if _is_binary(df[col]):
            if is_train:
                mode_val = df[col].mode(dropna=True)
                fill_val = mode_val.iloc[0] if len(mode_val) > 0 else 0
                stats_dict[f"{col}_fill"] = fill_val
            else:
                fill_val = stats_dict.get(f"{col}_fill", 0)
            df[col] = df[col].fillna(fill_val)
            continue

        if is_train:
            skewness = df[col].skew()
            if abs(skewness) > 1:
                fill_val = df[col].median()
                stats_dict[f"{col}_fill"] = fill_val
                stats_dict[f"{col}_method"] = "median"
            else:
                fill_val = df[col].mean()
                stats_dict[f"{col}_fill"] = fill_val
                stats_dict[f"{col}_method"] = "mean"
        else:
            fill_val = stats_dict.get(f"{col}_fill", df[col].median())

        df[col] = df[col].fillna(fill_val)

    return df, stats_dict


def winsorize_features(df: pd.DataFrame, percentiles: Optional[dict] = None) -> tuple:
    is_train = percentiles is None
    pct_dict = percentiles.copy() if percentiles else {}

    num_cols = [c for c in df.select_dtypes(include=np.number).columns
                if c not in ['SK_ID_CURR', 'TARGET']
                and not c.startswith('FLAG_')
                and not c.startswith('IS_')
                and df[c].nunique() > 2]

    for col in num_cols:
        if is_train:
            pct_dict[col] = float(df[col].quantile(0.99))
        if col in pct_dict:
            df[col] = df[col].clip(upper=pct_dict[col])

    return df, pct_dict


def remove_multicollinear(
    df: pd.DataFrame,
    target_col: str = "TARGET",
    threshold: float = 0.95,
) -> tuple[pd.DataFrame, list]:
    numeric = df.select_dtypes("number").drop(columns=[target_col, "SK_ID_CURR"], errors="ignore")
    corr_matrix = numeric.corr().abs()

    target_corr = numeric.corrwith(df[target_col]).abs()

    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = []

    for col in upper.columns:
        if col in to_drop:
            continue
        high_corr_cols = upper.index[upper[col] > threshold].tolist()
        for partner in high_corr_cols:
            if partner in to_drop:
                continue
            col_corr = target_corr.get(col, 0)
            partner_corr = target_corr.get(partner, 0)
            to_drop.append(col if col_corr < partner_corr else partner)

    to_drop = list(set(to_drop))
    df = df.drop(columns=to_drop, errors="ignore")
    return df, to_drop


def validate(df: pd.DataFrame, target_col: Optional[str] = None, original_rows: Optional[int] = None) -> dict:
    issues = {}

    missing = df.isnull().sum().sum()
    if missing > 0:
        issues["missing_values"] = missing

    inf_count = np.isinf(df.select_dtypes("number")).sum().sum()
    if inf_count > 0:
        issues["inf_values"] = inf_count

    const_cols = [c for c in df.columns if df[c].nunique() <= 1]
    if const_cols:
        issues["constant_cols"] = const_cols

    if original_rows is not None and len(df) != original_rows:
        issues["row_count_changed"] = {"expected": original_rows, "got": len(df)}

    return issues


def preprocess_pipeline(
    df: pd.DataFrame,
    target_col: str = "TARGET",
    mode: str = "train",
    params: Optional[dict] = None,
) -> tuple:
    is_train = mode == "train"
    p = params.copy() if params else {}

    original_rows = len(df)

    if is_train:
        cols_to_drop = get_cols_to_drop(df, target_col)
        p["cols_to_drop"] = cols_to_drop
    else:
        cols_to_drop = p.get("cols_to_drop", [])

    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")

    df, fill_stats = fill_missing(df, train_stats=p.get("fill_stats") if not is_train else None)
    if is_train:
        p["fill_stats"] = fill_stats

    df, pct_dict = winsorize_features(df, percentiles=p.get("pct_dict") if not is_train else None)
    if is_train:
        p["pct_dict"] = pct_dict

    if is_train:
        df, dropped_mc = remove_multicollinear(df, target_col, threshold=0.95)
        p["multicollinear_dropped"] = dropped_mc
    else:
        df = df.drop(columns=[c for c in p.get("multicollinear_dropped", []) if c in df.columns], errors="ignore")

    if is_train and target_col in df.columns:
        n0 = (df[target_col] == 0).sum()
        n1 = (df[target_col] == 1).sum()
        p["class_weight_balanced"] = True
        p["scale_pos_weight"] = round(n0 / n1, 4)

    issues = validate(df, original_rows)
    if issues:
        print(f"[preprocess_pipeline] Validation issues: {issues}")

    if is_train:
        p["feature_cols"] = [c for c in df.columns if c not in (target_col, "SK_ID_CURR")]

    return df, p
