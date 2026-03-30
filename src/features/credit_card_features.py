import numpy as np
import pandas as pd


def build_credit_card_features(cc: pd.DataFrame) -> pd.DataFrame:
    cc = cc.copy()
    cc["AMT_BALANCE"] = cc["AMT_BALANCE"].replace(365243, np.nan)
    cc["AMT_CREDIT_LIMIT_ACTUAL"] = cc["AMT_CREDIT_LIMIT_ACTUAL"].replace(0, np.nan)
    cc["UTILIZATION_RATE"] = cc["AMT_BALANCE"] / cc["AMT_CREDIT_LIMIT_ACTUAL"]

    result = cc.groupby("SK_ID_CURR").agg(
        CC_COUNT=("SK_ID_PREV", "count"),
        CC_MONTHS_COUNT=("MONTHS_BALANCE", "count"),
        CC_AMT_BALANCE_MEAN=("AMT_BALANCE", "mean"),
        CC_AMT_BALANCE_MAX=("AMT_BALANCE", "max"),
        CC_AMT_CREDIT_LIMIT_MEAN=("AMT_CREDIT_LIMIT_ACTUAL", "mean"),
        CC_UTILIZATION_MEAN=("UTILIZATION_RATE", "mean"),
        CC_UTILIZATION_MAX=("UTILIZATION_RATE", "max"),
        CC_SK_DPD_MEAN=("SK_DPD", "mean"),
        CC_SK_DPD_MAX=("SK_DPD", "max"),
        CC_SK_DPD_DEF_MEAN=("SK_DPD_DEF", "mean"),
        CC_SK_DPD_DEF_MAX=("SK_DPD_DEF", "max"),
        CC_AMT_DRAWINGS_TOTAL_MEAN=("AMT_DRAWINGS_CURRENT", "mean"),
        CC_CNT_DRAWINGS_TOTAL_MEAN=("CNT_DRAWINGS_CURRENT", "mean"),
        CC_AMT_PAYMENT_TOTAL_MEAN=("AMT_PAYMENT_CURRENT", "mean"),
    ).reset_index()

    return result
