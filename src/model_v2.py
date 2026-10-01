from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score

path = next(Path("data").glob("*.csv"))
df = pd.read_csv(path)
df = df[df["type"].isin(["TRANSFER", "CASH_OUT"])].copy()

df["hour"] = df["step"] % 24
df["night"] = df["hour"].between(0, 6).astype(int)
df["is_transfer"] = (df["type"] == "TRANSFER").astype(int)
df["log_amount"] = np.log1p(df["amount"])
df["dest_zero_before"] = (df["oldbalanceDest"] == 0).astype(int)
df["dest_error"] = df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]
df["orig_error"] = df["oldbalanceOrg"] - df["amount"] - df["newbalanceOrig"]
df["sender_emptied"] = ((df["newbalanceOrig"] == 0) & (df["oldbalanceOrg"] > 0)).astype(int)
df["full_balance_moved"] = ((df["amount"] == df["oldbalanceOrg"]) & (df["oldbalanceOrg"] > 0)).astype(int)
df["amount_to_balance"] = df["amount"] / (df["oldbalanceOrg"] + 1)

set_a = ["is_transfer", "log_amount", "hour", "night"]
set_b = set_a + ["dest_zero_before", "oldbalanceDest", "newbalanceDest", "dest_error"]
set_c = set_b + [
    "oldbalanceOrg", "newbalanceOrig", "orig_error",
    "sender_emptied", "full_balance_moved", "amount_to_balance",
]
sets = {
    "A: type, amount, time": set_a,
    "B: A + destination account": set_b,
    "C: B + sender balance signals": set_c,
}

cutoff = df["step"].quantile(0.7)
train = df[df["step"] <= cutoff]
test = df[df["step"] > cutoff]
y_test = test["isFraud"].to_numpy()
print("test fraud rate:", round(y_test.mean(), 5))

for name, features in sets.items():
    model = HistGradientBoostingClassifier(max_iter=200, random_state=42)
    model.fit(train[features], train["isFraud"])
    scores = model.predict_proba(test[features])[:, 1]
    order = np.argsort(scores)[::-1]
    row = f"{name}: PR-AUC {average_precision_score(y_test, scores):.4f}"
    for budget in (0.0025, 0.005, 0.01):
        k = int(len(test) * budget)
        row += f" | recall@{budget * 100:g}% {y_test[order[:k]].sum() / y_test.sum():.3f}"
    print(row)