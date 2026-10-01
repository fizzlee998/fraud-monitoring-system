import numpy as np
import pandas as pd

SET_A = ["is_transfer", "log_amount", "hour_of_day", "is_night"]
SET_B = SET_A + ["dest_zero_before", "oldbalance_dest", "newbalance_dest", "dest_error"]
SET_C = SET_B + [
    "oldbalance_org",
    "newbalance_orig",
    "orig_error",
    "sender_emptied",
    "full_balance_moved",
    "amount_to_balance",
]

RENAME = {
    "oldbalanceOrg": "oldbalance_org",
    "newbalanceOrig": "newbalance_orig",
    "oldbalanceDest": "oldbalance_dest",
    "newbalanceDest": "newbalance_dest",
    "isFraud": "is_fraud",
    "isFlaggedFraud": "is_flagged_fraud",
}


def load_paysim(path):
    df = pd.read_csv(path)
    df = df[df["type"].isin(["TRANSFER", "CASH_OUT"])].copy()
    df = df.rename(columns=RENAME)
    df["hour_of_day"] = df["step"] % 24
    df["is_night"] = df["hour_of_day"].between(0, 6).astype(int)
    df["is_transfer"] = (df["type"] == "TRANSFER").astype(int)
    df["log_amount"] = np.log1p(df["amount"])
    df["dest_zero_before"] = (df["oldbalance_dest"] == 0).astype(int)
    df["dest_error"] = df["oldbalance_dest"] + df["amount"] - df["newbalance_dest"]
    df["orig_error"] = df["oldbalance_org"] - df["amount"] - df["newbalance_orig"]
    df["sender_emptied"] = ((df["newbalance_orig"] == 0) & (df["oldbalance_org"] > 0)).astype(int)
    df["full_balance_moved"] = ((df["amount"] == df["oldbalance_org"]) & (df["oldbalance_org"] > 0)).astype(int)
    df["amount_to_balance"] = df["amount"] / (df["oldbalance_org"] + 1)
    return df