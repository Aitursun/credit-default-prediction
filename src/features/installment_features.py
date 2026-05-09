import numpy as np
import pandas as pd


def build_installment_features(ins: pd.DataFrame) -> pd.DataFrame:
    ins = ins.copy()
    ins["PAYMENT_DIFF"] = ins["AMT_INSTALMENT"] - ins["AMT_PAYMENT"]
    ins["DAYS_PAST_DUE"] = ins["DAYS_INSTALMENT"] - ins["DAYS_ENTRY_PAYMENT"]
    ins["IS_OVERDUE"] = (ins["DAYS_PAST_DUE"] > 0).astype(np.int8)
    ins["PAYMENT_RATIO"] = ins["AMT_PAYMENT"] / ins["AMT_INSTALMENT"].replace(0, np.nan)

    result = ins.groupby("SK_ID_CURR").agg(
        INS_COUNT=("SK_ID_PREV", "count"),
        INS_DPD_MEAN=("DAYS_PAST_DUE", "mean"),
        INS_DPD_MAX=("DAYS_PAST_DUE", "max"),
        INS_DPD_SUM=("DAYS_PAST_DUE", "sum"),
        INS_OVERDUE_RATIO=("IS_OVERDUE", "mean"),
        INS_PAYMENT_DIFF_MEAN=("PAYMENT_DIFF", "mean"),
        INS_PAYMENT_DIFF_MAX=("PAYMENT_DIFF", "max"),
        INS_PAYMENT_DIFF_SUM=("PAYMENT_DIFF", "sum"),
        INS_AMT_INSTALMENT_MEAN=("AMT_INSTALMENT", "mean"),
        INS_AMT_PAYMENT_MEAN=("AMT_PAYMENT", "mean"),
        INS_PAYMENT_RATIO_MEAN=("PAYMENT_RATIO", "mean"),
    ).reset_index()

    result["INS_DPD_MAX"] = result["INS_DPD_MAX"].clip(lower=0)
    result["INS_DPD_MEAN"] = result["INS_DPD_MEAN"].clip(lower=0)

    return result
