import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect, text

load_dotenv()

path = next(Path("data").glob("*.csv"))
df = pd.read_csv(path)
df = df[df["type"].isin(["TRANSFER", "CASH_OUT"])]

fraud = df[df["isFraud"] == 1]
normal = df[df["isFraud"] == 0].sample(n=300000, random_state=42)
sample = pd.concat([fraud, normal]).sort_values("step").reset_index(drop=True)
sample.insert(0, "txn_id", range(1, len(sample) + 1))

sample = sample.rename(
    columns={
        "nameOrig": "name_orig",
        "oldbalanceOrg": "oldbalance_org",
        "newbalanceOrig": "newbalance_orig",
        "nameDest": "name_dest",
        "oldbalanceDest": "oldbalance_dest",
        "newbalanceDest": "newbalance_dest",
        "isFraud": "is_fraud",
        "isFlaggedFraud": "is_flagged_fraud",
    }
)

engine = create_engine(os.environ["DATABASE_URL"])

with engine.begin() as conn:
    conn.execute(text("create schema if not exists fraud_raw"))
    if inspect(conn).has_table("transactions", schema="fraud_raw"):
        conn.execute(text("truncate table fraud_raw.transactions"))
    sample.to_sql(
        "transactions",
        conn,
        schema="fraud_raw",
        if_exists="append",
        index=False,
        chunksize=4000,
        method="multi",
    )

print("loaded fraud_raw.transactions:", len(sample), "rows, fraud:", int(sample["is_fraud"].sum()))