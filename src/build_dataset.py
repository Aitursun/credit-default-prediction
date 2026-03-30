import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
DATA = ROOT / "data"

from src.features.application_features import build_application_features
from src.features.bureau_features import build_bureau_features
from src.features.previous_features import build_previous_features
from src.features.installment_features import build_installment_features
from src.features.pos_cash_features import build_pos_cash_features
from src.features.credit_card_features import build_credit_card_features


def build_dataset(mode: str = "train") -> pd.DataFrame:
    print(f"[1/8] Loading application_{mode}.csv...")
    df = pd.read_csv(DATA / f"application_{mode}.csv")

    print("[2/8] Building application features...")
    df = build_application_features(df)

    print("[3/8] Building bureau features...")
    bureau = pd.read_csv(DATA / "bureau.csv")
    bureau_balance = pd.read_csv(DATA / "bureau_balance.csv")
    bureau_feats = build_bureau_features(bureau, bureau_balance)
    del bureau, bureau_balance

    print("[4/8] Building previous application features...")
    prev = pd.read_csv(DATA / "previous_application.csv")
    prev_feats = build_previous_features(prev)
    del prev

    print("[5/8] Building installment payment features...")
    ins = pd.read_csv(DATA / "installments_payments.csv")
    ins_feats = build_installment_features(ins)
    del ins

    print("[6/8] Building POS CASH features...")
    pos = pd.read_csv(DATA / "POS_CASH_balance.csv")
    pos_feats = build_pos_cash_features(pos)
    del pos

    print("[7/8] Building credit card features...")
    cc = pd.read_csv(DATA / "credit_card_balance.csv")
    cc_feats = build_credit_card_features(cc)
    del cc

    print("[8/8] Joining all features...")
    for feats in [bureau_feats, prev_feats, ins_feats, pos_feats, cc_feats]:
        df = df.merge(feats, on="SK_ID_CURR", how="left")

    cat_cols = df.select_dtypes("object").columns.tolist()
    df = pd.get_dummies(df, columns=cat_cols, dummy_na=False)

    bool_cols = df.select_dtypes(bool).columns
    df[bool_cols] = df[bool_cols].astype(np.int8)

    out_path = DATA / "processed" / f"final_{mode}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Saved → {out_path}  shape={df.shape}")

    return df


if __name__ == "__main__":
    build_dataset("train")
