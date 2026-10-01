from pathlib import Path

import pandas as pd

files = list(Path("data").glob("*.csv"))
if not files:
    raise SystemExit("No CSV found in data/")

path = files[0]
print(path.name)

df = pd.read_csv(path)
print(df.shape)
print(df.dtypes)
print("missing values:", df.isna().sum().sum())

print(df["isFraud"].value_counts())
print("fraud rate:", df["isFraud"].mean())

print(df.groupby("type")["isFraud"].agg(["count", "sum", "mean"]))
print(df.groupby("isFraud")["amount"].describe())
print(df["isFlaggedFraud"].value_counts())

fraud = df[df["isFraud"] == 1]
normal = df[df["isFraud"] == 0]
print("fraud, origin balance emptied:", (fraud["newbalanceOrig"] == 0).mean())
print("normal, origin balance emptied:", (normal["newbalanceOrig"] == 0).mean())