import numpy as np
import pandas as pd


def build_bureau_features(bureau: pd.DataFrame, bureau_balance: pd.DataFrame) -> pd.DataFrame:
    bb_agg = bureau_balance.groupby("SK_ID_BUREAU").agg(
        BB_MONTHS_COUNT=("MONTHS_BALANCE", "count"),
        BB_STATUS_C_COUNT=("STATUS", lambda x: (x == "C").sum()),
        BB_STATUS_DPD_COUNT=("STATUS", lambda x: x.isin(["1", "2", "3", "4", "5"]).sum()),
        BB_MAX_DPD_STATUS=("STATUS", lambda x: x[x.isin(["1", "2", "3", "4", "5"])].max() if x.isin(["1", "2", "3", "4", "5"]).any() else "0"),
    ).reset_index()

    bureau = bureau.merge(bb_agg, on="SK_ID_BUREAU", how="left")

    agg_dict = {
        "SK_ID_BUREAU": "count",
        "CREDIT_ACTIVE": [lambda x: (x == "Active").sum(), lambda x: (x == "Closed").sum()],
        "AMT_CREDIT_SUM": ["sum", "mean", "max"],
        "AMT_CREDIT_SUM_DEBT": ["sum", "mean", "max"],
        "AMT_CREDIT_SUM_OVERDUE": ["sum", "mean", "max"],
        "DAYS_CREDIT": ["min", "max", "mean"],
        "DAYS_CREDIT_ENDDATE": ["min", "max", "mean"],
        "DAYS_CREDIT_UPDATE": "mean",
        "CNT_CREDIT_PROLONG": ["sum", "max"],
        "AMT_ANNUITY": ["sum", "mean"],
        "BB_MONTHS_COUNT": "sum",
        "BB_STATUS_C_COUNT": "sum",
        "BB_STATUS_DPD_COUNT": "sum",
    }

    result = bureau.groupby("SK_ID_CURR").agg(agg_dict)
    result.columns = ["_".join(c).upper().strip("_") for c in result.columns]
    result = result.rename(columns={
        "SK_ID_BUREAU_COUNT": "BUREAU_LOAN_COUNT",
        "CREDIT_ACTIVE_<LAMBDA_0>": "BUREAU_ACTIVE_COUNT",
        "CREDIT_ACTIVE_<LAMBDA_1>": "BUREAU_CLOSED_COUNT",
    })
    result = result.reset_index()

    active = bureau[bureau["CREDIT_ACTIVE"] == "Active"].groupby("SK_ID_CURR").agg(
        BUREAU_ACTIVE_DEBT_SUM=("AMT_CREDIT_SUM_DEBT", "sum"),
        BUREAU_ACTIVE_CREDIT_SUM=("AMT_CREDIT_SUM", "sum"),
        BUREAU_ACTIVE_OVERDUE_SUM=("AMT_CREDIT_SUM_OVERDUE", "sum"),
    ).reset_index()

    result = result.merge(active, on="SK_ID_CURR", how="left")

    result.columns = [c.replace("<LAMBDA_0>", "ACTIVE").replace("<LAMBDA_1>", "CLOSED") for c in result.columns]

    return result
