import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, precision_score, recall_score

from src.features import SET_A, SET_B, SET_C, load_paysim

df = load_paysim(next(Path("data").glob("*.csv")))

cutoff = float(df["step"].quantile(0.7))
train = df[df["step"] <= cutoff]
test = df[df["step"] > cutoff]
y = test["is_fraud"].to_numpy()

results = {
    "cutoff_step": cutoff,
    "train_rows": int(len(train)),
    "test_rows": int(len(test)),
    "test_frauds": int(y.sum()),
    "test_fraud_rate": float(y.mean()),
    "rule_baseline": {
        "precision": float(precision_score(y, test["is_flagged_fraud"])),
        "recall": float(recall_score(y, test["is_flagged_fraud"])),
    },
    "models": {},
}

fitted = {}
for name, features in {"A": SET_A, "B": SET_B, "C": SET_C}.items():
    model = HistGradientBoostingClassifier(max_iter=200, random_state=42)
    model.fit(train[features], train["is_fraud"])
    scores = model.predict_proba(test[features])[:, 1]
    order = np.argsort(scores)[::-1]
    entry = {"pr_auc": float(average_precision_score(y, scores))}
    for budget in (0.0025, 0.005, 0.01):
        k = int(len(test) * budget)
        hits = int(y[order[:k]].sum())
        label = f"{budget * 100:g}pct"
        entry[f"recall_at_{label}"] = hits / int(y.sum())
        entry[f"precision_at_{label}"] = hits / k
    results["models"][name] = entry
    fitted[name] = model

Path("models").mkdir(exist_ok=True)
Path("reports").mkdir(exist_ok=True)
joblib.dump(
    {"model": fitted["B"], "features": SET_B, "cutoff_step": cutoff},
    "models/fraud_model_b.joblib",
)
Path("reports/metrics.json").write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))