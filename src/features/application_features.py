import numpy as np
import pandas as pd


def build_application_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["AGE_YEARS"] = -out["DAYS_BIRTH"] / 365

    out["IS_UNEMPLOYED"] = (out["DAYS_EMPLOYED"] == 365243).astype(np.int8)
    out["DAYS_EMPLOYED"] = out["DAYS_EMPLOYED"].replace(365243, np.nan)
    out["YEARS_EMPLOYED"] = -out["DAYS_EMPLOYED"] / 365

    out["CREDIT_INCOME_RATIO"] = out["AMT_CREDIT"] / out["AMT_INCOME_TOTAL"]
    out["ANNUITY_INCOME_RATIO"] = out["AMT_ANNUITY"] / out["AMT_INCOME_TOTAL"]
    out["CREDIT_GOODS_RATIO"] = out["AMT_CREDIT"] / out["AMT_GOODS_PRICE"].replace(0, np.nan)
    out["ANNUITY_CREDIT_RATIO"] = out["AMT_ANNUITY"] / out["AMT_CREDIT"].replace(0, np.nan)

    ext = out[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]]
    out["EXT_SOURCE_MEAN"] = ext.mean(axis=1)
    out["EXT_SOURCE_MIN"] = ext.min(axis=1)
    out["EXT_SOURCE_MAX"] = ext.max(axis=1)
    out["EXT_SOURCE_STD"] = ext.std(axis=1)

    out["INCOME_PER_PERSON"] = out["AMT_INCOME_TOTAL"] / (out["CNT_FAM_MEMBERS"].replace(0, np.nan) + 1)
    out["EMPLOYED_TO_AGE_RATIO"] = out["YEARS_EMPLOYED"] / out["AGE_YEARS"].replace(0, np.nan)

    out["DAYS_REGISTRATION_AGE_RATIO"] = out["DAYS_REGISTRATION"] / out["DAYS_BIRTH"].replace(0, np.nan)
    out["DAYS_ID_PUBLISH_AGE_RATIO"] = out["DAYS_ID_PUBLISH"] / out["DAYS_BIRTH"].replace(0, np.nan)

    out["PHONE_TO_EMPLOY_RATIO"] = out["DAYS_LAST_PHONE_CHANGE"] / out["DAYS_EMPLOYED"].replace(0, np.nan)
    out["PHONE_TO_BIRTH_RATIO"] = out["DAYS_LAST_PHONE_CHANGE"] / out["DAYS_BIRTH"].replace(0, np.nan)

    doc_cols = [c for c in out.columns if c.startswith("FLAG_DOCUMENT_")]
    out["DOCUMENT_COUNT"] = out[doc_cols].sum(axis=1)

    out["AMT_REQ_CREDIT_BUREAU_TOTAL"] = out[[
        "AMT_REQ_CREDIT_BUREAU_HOUR", "AMT_REQ_CREDIT_BUREAU_DAY",
        "AMT_REQ_CREDIT_BUREAU_WEEK", "AMT_REQ_CREDIT_BUREAU_MON",
        "AMT_REQ_CREDIT_BUREAU_QRT", "AMT_REQ_CREDIT_BUREAU_YEAR"
    ]].sum(axis=1)

    return out
