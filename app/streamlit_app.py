import json
import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()

st.set_page_config(page_title="Fraud Alert Review", layout="wide")

ROOT = Path(__file__).resolve().parent.parent


@st.cache_resource
def get_engine():
    url = os.getenv("DATABASE_URL") or st.secrets["DATABASE_URL"]
    return create_engine(url)


@st.cache_data(ttl=3600)
def load_tables():
    engine = get_engine()
    alerts = pd.read_sql("select * from fraud_scored.alerts order by alert_rank", engine)
    daily = pd.read_sql("select * from fraud_scored.daily_summary order by day_number", engine)
    summary = pd.read_sql("select * from fraud_scored.run_summary", engine).iloc[0]
    return alerts, daily, summary


@st.cache_data
def load_metrics():
    return json.loads((ROOT / "reports" / "metrics.json").read_text())


alerts, daily, summary = load_tables()
metrics = load_metrics()

st.title("Fraud Alert Review")
st.caption(
    "Synthetic PaySim transactions. The model was trained on earlier activity and scored on a later, unseen period. "
    "Amounts are in the dataset's simulated currency."
)

rule_recall = summary["rule_frauds_caught"] / summary["test_frauds"]

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Transactions scored", f"{int(summary['test_transactions']):,}")
c2.metric("Alerts raised", f"{int(summary['alerts_raised']):,}", f"top {summary['alert_budget_pct']:g}% by risk", delta_color="off")
c3.metric("Fraud caught", f"{summary['recall']:.1%}", f"{int(summary['frauds_caught']):,} of {int(summary['test_frauds']):,}", delta_color="off")
c4.metric("Alert precision", f"{summary['precision']:.1%}")
c5.metric("Old rule caught", f"{rule_recall:.2%}")

tab1, tab2, tab3, tab4 = st.tabs(["Alert queue", "Model performance", "Daily trend", "Method and limits"])

with tab1:
    all_types = sorted(alerts["txn_type"].unique())
    left, mid, right = st.columns(3)
    min_score = left.slider("Minimum risk score", 0.0, 1.0, 0.5, 0.05)
    types = mid.multiselect("Transaction type", all_types, default=all_types)
    top_n = int(right.number_input("Rows to show", 10, 2000, 200, 10))

    view = alerts[(alerts["risk_score"] >= min_score) & (alerts["txn_type"].isin(types))]
    st.write(f"{len(view):,} alerts match. Showing the top {min(len(view), top_n):,}.")

    shown = view.head(top_n)[
        ["alert_rank", "risk_score", "txn_type", "amount", "name_orig", "name_dest", "hour_of_day", "reasons", "is_fraud"]
    ].copy()
    shown["risk_score"] = shown["risk_score"].round(3)
    shown["is_fraud"] = shown["is_fraud"].map({1: "Fraud", 0: "Not fraud"})
    shown = shown.rename(columns={"is_fraud": "outcome (simulated label)"})
    st.dataframe(shown, hide_index=True)
    st.download_button("Download matching alerts (CSV)", view.to_csv(index=False), "alerts.csv", "text/csv")

with tab2:
    budgets = [("0.25pct", "0.25%"), ("0.5pct", "0.5%"), ("1pct", "1%")]
    labels = {
        "A": "A: type, amount, time",
        "B": "B: adds destination account (deployed)",
        "C": "C: adds sender balance (simulation artifact)",
    }
    fig = go.Figure()
    for key, label in labels.items():
        fig.add_trace(
            go.Bar(
                name=label,
                x=[b[1] for b in budgets],
                y=[metrics["models"][key][f"recall_at_{b[0]}"] for b in budgets],
            )
        )
    fig.update_layout(
        barmode="group",
        title="Share of fraud caught by review budget",
        xaxis_title="Share of transactions reviewed",
        yaxis_title="Fraud caught (recall)",
        yaxis_tickformat=".0%",
        height=420,
    )
    st.plotly_chart(fig)

    rows = []
    for key, label in labels.items():
        m = metrics["models"][key]
        rows.append(
            {
                "Feature set": label,
                "PR-AUC": round(m["pr_auc"], 3),
                "Recall at 1%": f"{m['recall_at_1pct']:.1%}",
                "Precision at 1%": f"{m['precision_at_1pct']:.1%}",
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.write(
        f"The existing rule catches {metrics['rule_baseline']['recall']:.2%} of fraud "
        f"at {metrics['rule_baseline']['precision']:.0%} precision."
    )

with tab3:
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=daily["day_number"], y=daily["frauds"], name="Frauds that occurred"))
    fig2.add_trace(go.Bar(x=daily["day_number"], y=daily["frauds_caught"], name="Frauds caught in alerts"))
    fig2.update_layout(barmode="overlay", title="Fraud per day in the test period", xaxis_title="Simulation day", yaxis_title="Transactions", height=420)
    st.plotly_chart(fig2)

with tab4:
    st.markdown(
        """
**Data.** PaySim, a public synthetic dataset. Only transfer and cash-out transactions are in scope, because all fraud sits there.

**Split.** The model trains on the earliest 70% of the timeline and is scored on the later 30%, so it never sees the future.

**Alert budget.** The top 1% of transactions by risk score become alerts, which stands in for analyst review capacity.

**Trade-off.** At this budget, roughly six in ten alerts are false alarms. A smaller budget raises precision but catches less fraud.

**Limits.** Real fraud data is confidential. The sender-balance features give near-perfect scores only because of how PaySim was generated, so the deployed model leaves them out. Results on live transactions would be lower.
"""
    )