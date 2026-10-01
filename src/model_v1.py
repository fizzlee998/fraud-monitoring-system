from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, precision_score, recall_score

path = next(Path("data").glob("*.csv"))
df = pd.read_csv(path)
df = df[df["type"].isin(["TRANSFER", "CASH_OUT"])].copy()

df["hour"] = df["step"] % 24
df["night"] = df["hour"].between(0, 6).astype(int)
df["is_transfer"] = (df["type"] == "TRANSFER").astype(int)
df["log_amount"] = np.log1p(df["amount"])
df["orig_error"] = df["oldbalanceOrg"] - df["amount"] - df["newbalanceOrig"]
df["dest_error"] = df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]
df["sender_emptied"] = ((df["newbalanceOrig"] == 0) & (df["oldbalanceOrg"] > 0)).astype(int)
df["dest_zero_before"] = (df["oldbalanceDest"] == 0).astype(int)
df["amount_to_balance"] = df["amount"] / (df["oldbalanceOrg"] + 1)
df["full_balance_moved"] = ((df["amount"] == df["oldbalanceOrg"]) & (df["oldbalanceOrg"] > 0)).astype(int)

base = [
    "is_transfer", "log_amount", "hour", "night", "sender_emptied", "dest_zero_before",
    "orig_error", "dest_error", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest",
]
leaky = ["full_balance_moved", "amount_to_balance"]

cutoff = df["step"].quantile(0.7)
train = df[df["step"] <= cutoff]
test = df[df["step"] > cutoff]
print("train rows:", len(train), "fraud:", train["isFraud"].sum())
print("test rows:", len(test), "fraud:", test["isFraud"].sum())

y_test = test["isFraud"].to_numpy()

print("rule baseline: precision", precision_score(y_test, test["isFlaggedFraud"]), "recall", recall_score(y_test, test["isFlaggedFraud"]))


def evaluate(name, features):
    model = HistGradientBoostingClassifier(max_iter=200, random_state=42)
    model.fit(train[features], train["isFraud"])
    scores = model.predict_proba(test[features])[:, 1]
    k = int(len(test) * 0.01)
    top = np.argsort(scores)[::-1][:k]
    hits = y_test[top].sum()
    print(
        f"{name}: PR-AUC {average_precision_score(y_test, scores):.4f} | "
        f"precision@1% {hits / k:.4f} | recall@1% {hits / y_test.sum():.4f}"
    )


evaluate("with leaky features", base + leaky)
evaluate("without leaky features", base)