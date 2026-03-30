import numpy as np
import pandas as pd


def build_previous_features(prev: pd.DataFrame) -> pd.DataFrame:
    prev = prev.copy()
    prev["AMT_CREDIT"] = prev["AMT_CREDIT"].replace(365243, np.nan)
    prev["AMT_ANNUITY"] = prev["AMT_ANNUITY"].replace(365243, np.nan)
    prev["DAYS_FIRST_DRAWING"] = prev["DAYS_FIRST_DRAWING"].replace(365243, np.nan)
    prev["DAYS_FIRST_DUE"] = prev["DAYS_FIRST_DUE"].replace(365243, np.nan)
    prev["DAYS_LAST_DUE_1ST_VERSION"] = prev["DAYS_LAST_DUE_1ST_VERSION"].replace(365243, np.nan)
    prev["DAYS_LAST_DUE"] = prev["DAYS_LAST_DUE"].replace(365243, np.nan)
    prev["DAYS_TERMINATION"] = prev["DAYS_TERMINATION"].replace(365243, np.nan)

    prev["APP_CREDIT_RATIO"] = prev["AMT_APPLICATION"] / prev["AMT_CREDIT"].replace(0, np.nan)
    prev["CREDIT_DOWN_PAYMENT_RATIO"] = prev["AMT_DOWN_PAYMENT"] / prev["AMT_CREDIT"].replace(0, np.nan)

    total = prev.groupby("SK_ID_CURR").agg(
        PREV_LOAN_COUNT=("SK_ID_PREV", "count"),
        PREV_AMT_APPLICATION_MEAN=("AMT_APPLICATION", "mean"),
        PREV_AMT_APPLICATION_MAX=("AMT_APPLICATION", "max"),
        PREV_AMT_CREDIT_MEAN=("AMT_CREDIT", "mean"),
        PREV_AMT_CREDIT_SUM=("AMT_CREDIT", "sum"),
        PREV_AMT_ANNUITY_MEAN=("AMT_ANNUITY", "mean"),
        PREV_AMT_DOWN_PAYMENT_MEAN=("AMT_DOWN_PAYMENT", "mean"),
        PREV_RATE_DOWN_PAYMENT_MEAN=("RATE_DOWN_PAYMENT", "mean"),
        PREV_APP_CREDIT_RATIO_MEAN=("APP_CREDIT_RATIO", "mean"),
        PREV_DAYS_DECISION_MEAN=("DAYS_DECISION", "mean"),
        PREV_DAYS_DECISION_MIN=("DAYS_DECISION", "min"),
        PREV_CNT_PAYMENT_MEAN=("CNT_PAYMENT", "mean"),
        PREV_CNT_PAYMENT_SUM=("CNT_PAYMENT", "sum"),
    ).reset_index()

    approved = prev[prev["NAME_CONTRACT_STATUS"] == "Approved"].groupby("SK_ID_CURR").agg(
        PREV_APPROVED_COUNT=("SK_ID_PREV", "count"),
        PREV_APPROVED_AMT_CREDIT_MEAN=("AMT_CREDIT", "mean"),
        PREV_APPROVED_AMT_ANNUITY_MEAN=("AMT_ANNUITY", "mean"),
    ).reset_index()

    refused = prev[prev["NAME_CONTRACT_STATUS"] == "Refused"].groupby("SK_ID_CURR").agg(
        PREV_REFUSED_COUNT=("SK_ID_PREV", "count"),
    ).reset_index()

    result = total.merge(approved, on="SK_ID_CURR", how="left").merge(refused, on="SK_ID_CURR", how="left")
    result["PREV_APPROVED_RATIO"] = result["PREV_APPROVED_COUNT"] / result["PREV_LOAN_COUNT"].replace(0, np.nan)
    result["PREV_REFUSED_RATIO"] = result["PREV_REFUSED_COUNT"] / result["PREV_LOAN_COUNT"].replace(0, np.nan)

    return result
