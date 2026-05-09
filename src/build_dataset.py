import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.config import RAW_FILES, INTERIM_DIR, PROC_DATA_DIR
from src.features.application_features import build_application_features
from src.features.bureau_features import build_bureau_features
from src.features.previous_features import build_previous_features
from src.features.installment_features import build_installment_features
from src.features.pos_cash_features import build_pos_cash_features
from src.features.credit_card_features import build_credit_card_features


def _load_or_build(cache_path: Path, builder, *csv_paths):
    """Load cached aggregate from interim/ or rebuild and cache it."""
    if cache_path.exists():
        print(f"  [cache] {cache_path.name}")
        return pd.read_parquet(cache_path)
    dfs = [pd.read_csv(p) for p in csv_paths]
    result = builder(*dfs)
    result.to_parquet(cache_path, index=False)
    return result


def build_dataset(mode: str = "train") -> pd.DataFrame:
    print(f"[1/8] Loading application_{mode}.csv...")
    df = pd.read_csv(RAW_FILES[mode])

    print("[2/8] Building application features...")
    df = build_application_features(df)

    print("[3/8] Building bureau features...")
    bureau_feats = _load_or_build(
        INTERIM_DIR / "bureau_feats.parquet",
        build_bureau_features,
        RAW_FILES["bureau"],
        RAW_FILES["bureau_bal"],
    )

    print("[4/8] Building previous application features...")
    prev_feats = _load_or_build(
        INTERIM_DIR / "prev_feats.parquet",
        build_previous_features,
        RAW_FILES["prev_app"],
    )

    print("[5/8] Building installment payment features...")
    ins_feats = _load_or_build(
        INTERIM_DIR / "ins_feats.parquet",
        build_installment_features,
        RAW_FILES["installments"],
    )

    print("[6/8] Building POS CASH features...")
    pos_feats = _load_or_build(
        INTERIM_DIR / "pos_feats.parquet",
        build_pos_cash_features,
        RAW_FILES["pos_cash"],
    )

    print("[7/8] Building credit card features...")
    cc_feats = _load_or_build(
        INTERIM_DIR / "cc_feats.parquet",
        build_credit_card_features,
        RAW_FILES["credit_card"],
    )

    print("[8/8] Joining all features...")
    for feats in [bureau_feats, prev_feats, ins_feats, pos_feats, cc_feats]:
        df = df.merge(feats, on="SK_ID_CURR", how="left")

    cat_cols = df.select_dtypes("object").columns.tolist()
    df = pd.get_dummies(df, columns=cat_cols, dummy_na=False)

    bool_cols = df.select_dtypes(bool).columns
    df[bool_cols] = df[bool_cols].astype(np.int8)

    out_path = PROC_DATA_DIR / f"final_{mode}.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Saved → {out_path}  shape={df.shape}")

    return df


if __name__ == "__main__":
    build_dataset("train")