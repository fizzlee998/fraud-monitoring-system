from pathlib import Path

import pandas as pd

path = next(Path("data").glob("*.csv"))
df = pd.read_csv(path)
df = df[df["type"].isin(["TRANSFER", "CASH_OUT"])].copy()

print("rows in scope:", len(df), "fraud:", df["isFraud"].sum(), "rate:", df["isFraud"].mean())

active = df[df["oldbalanceOrg"] > 0].copy()
active["emptied"] = active["newbalanceOrig"] == 0
print("sender emptied, balance > 0 only:")
print(active.groupby("isFraud")["emptied"].mean())

df["full_balance_moved"] = (df["amount"] == df["oldbalanceOrg"]) & (df["oldbalanceOrg"] > 0)
print("amount equals full balance:")
print(df.groupby("isFraud")["full_balance_moved"].mean())

df["dest_zero_before"] = df["oldbalanceDest"] == 0
print("destination balance zero before:")
print(df.groupby("isFraud")["dest_zero_before"].mean())

df["hour"] = df["step"] % 24
print(df.groupby("hour")["isFraud"].agg(["count", "mean"]).round(5))

print("zero-amount rows:", (df["amount"] == 0).sum())
print("fraud among zero-amount rows:", df.loc[df["amount"] == 0, "isFraud"].sum())

flagged = df[df["isFlaggedFraud"] == 1]
print("flagged by rule:", len(flagged), "of which fraud:", flagged["isFraud"].sum())

print(df[df["isFraud"] == 1]["nameDest"].value_counts().head(5))