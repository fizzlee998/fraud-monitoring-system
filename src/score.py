import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from src.features import load_paysim

load_dotenv()

BUDGET = 0.01

artifact = joblib.load("models/fraud_model_b.joblib")
model = artifact["model"]
features = artifact["features"]
cutoff = artifact["cutoff_step"]

df = load_paysim(next(Path("data").glob("*.csv")))
train = df[df["step"] <= cutoff]
test = df[df["step"] > cutoff].copy()
big_amount = train["amount"].quantile(0.99)

test["risk_score"] = model.predict_proba(test[features])[:, 1]
test["source_row"] = test.index
test["day_number"] = test["step"] // 24

k = int(len(test) * BUDGET)
alerts = test.nlargest(k, "risk_score").copy()
alerts["alert_rank"] = range(1, len(alerts) + 1)
test["is_alert"] = test.index.isin(alerts.index).astype(int)

reason_parts = pd.DataFrame(
    {
        "a": np.where(alerts["dest_zero_before"] == 1, "Destination account empty before transfer", ""),
        "b": np.where(alerts["is_night"] == 1, "Night-time activity", ""),
        "c": np.where(alerts["amount"] >= big_amount, "Unusually large amount", ""),
        "d": np.where(alerts["is_transfer"] == 1, "Account-to-account transfer", ""),
    },
    index=alerts.index,
)
alerts["reasons"] = reason_parts.apply(lambda r: "; ".join(v for v in r if v), axis=1)

alerts_out = alerts.rename(columns={"type": "txn_type", "is_flagged_fraud": "flagged_by_rule"})[
    [
        "alert_rank",
        "source_row",
        "step",
        "day_number",
        "hour_of_day",
        "txn_type",
        "amount",
        "nameOrig",
        "nameDest",
        "oldbalance_org",
        "newbalance_orig",
        "oldbalance_dest",
        "newbalance_dest",
        "risk_score",
        "reasons",
        "flagged_by_rule",
        "is_fraud",
    ]
].rename(columns={"nameOrig": "name_orig", "nameDest": "name_dest"})

daily = (
    test.assign(caught=lambda d: d["is_alert"] * d["is_fraud"])
    .groupby("day_number")
    .agg(
        transactions=("step", "size"),
        frauds=("is_fraud", "sum"),
        alerts=("is_alert", "sum"),
        frauds_caught=("caught", "sum"),
    )
    .reset_index()
)

total_fraud = int(test["is_fraud"].sum())
caught = int(alerts["is_fraud"].sum())
summary = pd.DataFrame(
    [
        {
            "test_transactions": int(len(test)),
            "test_frauds": total_fraud,
            "alert_budget_pct": BUDGET * 100,
            "alerts_raised": int(len(alerts)),
            "frauds_caught": caught,
            "precision": caught / len(alerts),
            "recall": caught / total_fraud,
            "rule_frauds_caught": int(test.loc[test["is_flagged_fraud"] == 1, "is_fraud"].sum()),
            "rule_alerts": int(test["is_flagged_fraud"].sum()),
        }
    ]
)

engine = create_engine(os.environ["DATABASE_URL"])
with engine.begin() as conn:
    conn.execute(text("create schema if not exists fraud_scored"))
    alerts_out.to_sql("alerts", conn, schema="fraud_scored", if_exists="replace", index=False, chunksize=2000, method="multi")
    daily.to_sql("daily_summary", conn, schema="fraud_scored", if_exists="replace", index=False)
    summary.to_sql("run_summary", conn, schema="fraud_scored", if_exists="replace", index=False)

print(summary.T)
print("alerts written:", len(alerts_out))