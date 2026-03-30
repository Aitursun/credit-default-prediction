import numpy as np
import pandas as pd


def build_pos_cash_features(pos: pd.DataFrame) -> pd.DataFrame:
    result = pos.groupby("SK_ID_CURR").agg(
        POS_COUNT=("SK_ID_PREV", "count"),
        POS_MONTHS_COUNT=("MONTHS_BALANCE", "count"),
        POS_SK_DPD_MEAN=("SK_DPD", "mean"),
        POS_SK_DPD_MAX=("SK_DPD", "max"),
        POS_SK_DPD_DEF_MEAN=("SK_DPD_DEF", "mean"),
        POS_SK_DPD_DEF_MAX=("SK_DPD_DEF", "max"),
        POS_CNT_INSTALMENT_MEAN=("CNT_INSTALMENT", "mean"),
        POS_CNT_INSTALMENT_FUTURE_MEAN=("CNT_INSTALMENT_FUTURE", "mean"),
        POS_COMPLETED_COUNT=("NAME_CONTRACT_STATUS", lambda x: (x == "Completed").sum()),
        POS_ACTIVE_COUNT=("NAME_CONTRACT_STATUS", lambda x: (x == "Active").sum()),
    ).reset_index()

    result["POS_COMPLETED_RATIO"] = result["POS_COMPLETED_COUNT"] / result["POS_COUNT"].replace(0, np.nan)

    return result
