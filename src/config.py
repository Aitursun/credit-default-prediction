from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent

DATA_DIR      = ROOT_DIR / "data"
RAW_DATA_DIR  = DATA_DIR         
PROC_DATA_DIR = DATA_DIR / "processed"

RAW_FILES = {
    "train":        RAW_DATA_DIR / "application_train.csv",
    "test":         RAW_DATA_DIR / "application_test.csv",
    "bureau":       RAW_DATA_DIR / "bureau.csv",
    "bureau_bal":   RAW_DATA_DIR / "bureau_balance.csv",
    "prev_app":     RAW_DATA_DIR / "previous_application.csv",
    "pos_cash":     RAW_DATA_DIR / "POS_CASH_balance.csv",
    "installments": RAW_DATA_DIR / "installments_payments.csv",
    "credit_card":  RAW_DATA_DIR / "credit_card_balance.csv",
}

PROCESSED_FILES = {
    "final_train": PROC_DATA_DIR / "final_train.parquet",
    "final_test":  PROC_DATA_DIR / "final_test.parquet",
}

MODELS_DIR  = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"

TARGET_COL  = "TARGET"
ID_COL      = "SK_ID_CURR"
RANDOM_SEED = 42
TEST_SIZE   = 0.2

for folder in [PROC_DATA_DIR, MODELS_DIR, REPORTS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)