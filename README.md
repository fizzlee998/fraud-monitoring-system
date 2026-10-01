# Fraud Monitoring System

A fraud detection and alert-ranking system for fintech transactions. It scores transactions with a machine learning model, turns the highest-risk ones into a ranked review queue, and measures honestly how much fraud that queue catches.

## The problem

Fraud teams at banks and fintechs face two problems at once. Fraud is rare, so the alert queue is mostly noise, and analysts have limited time to review it. Rule-based systems are easy to explain but catch only the patterns someone already thought of.

This project asks a practical question: if a team can only review a small slice of transactions, how much fraud can a model catch, and how many false alarms come with it?

## Results

Evaluated on a later, unseen period of the timeline. The test set has 818,514 transactions and 4,570 frauds (0.56%).

| | Alerts raised | Fraud caught | Alert precision |
|---|---|---|---|
| Existing rule (`isFlaggedFraud`) | 13 | 0.3% | 100% |
| Model, top 0.25% by risk | about 2,046 | 43.0% | 95.9% |
| Model, top 1% by risk | 8,185 | 66.7% | 37.2% |

The model catches about 67% of fraud while reviewing 1% of transactions. The cost is that roughly six in ten alerts at that budget are false alarms. A smaller budget raises precision and lowers coverage, and the dashboard lets you see that trade-off.

### What each group of signals contributes

| Feature set | PR-AUC | Fraud caught at 1% |
|---|---|---|
| A: transaction type, amount, time of day | 0.31 | 40.7% |
| B: A plus destination account signals (deployed) | 0.64 | 66.8% |
| C: B plus sender balance signals | 0.99 | 99.9% |

Set C scores almost perfectly because of how PaySim was generated: simulated fraud always drains the sender's full balance. Real transactions do not hand you a signal like that, so the deployed model uses set B and set C is reported only as an artifact ceiling.

## Architecture

```mermaid
flowchart LR
    A[PaySim CSV] --> B[Python: sample and load]
    B --> C[(Supabase Postgres<br/>fraud_raw)]
    C --> D[dbt staging view]
    D --> E[dbt mart: transaction features]
    A --> F[Python: features and training]
    F --> G[Gradient boosting model]
    G --> H[Scoring and alert ranking]
    H --> I[(fraud_scored)]
    I --> J[Streamlit review dashboard]
```

## How it works

1. **Data.** PaySim, a public synthetic dataset of 6.36 million mobile money transactions. All fraud sits in transfers and cash-outs, so the project scopes to those two types: 2.77 million transactions, 8,213 frauds (0.30%).
2. **Warehouse sample.** All fraud cases plus 300,000 random normal transactions are loaded into Supabase (308,213 rows), keeping storage within the free tier.
3. **dbt.** Staging and mart models build one feature row per transaction. Tests check unique IDs, valid transaction types, valid fraud labels, and non-negative amounts.
4. **Model.** A gradient boosting classifier trained on the earliest 70% of the timeline. The split is by time, never random, so the model cannot learn from the future.
5. **Alert queue.** Test-period transactions are scored, and the top 1% become alerts with plain-language reasons, such as an empty destination account or night-time activity.
6. **Dashboard.** Streamlit shows the alert queue with filters, model performance by feature set, a daily trend, and a method and limits page.

## Findings from exploring the data

- Fraud is concentrated in two transaction types: transfers (0.77% fraud) and cash-outs (0.18%). Payments, cash-ins and debits contain none.
- Fraud is far more common at night. The rate rises from under 0.2% during the day to above 50% around 5 am, when normal traffic is almost silent.
- Fraudulent transfers often go to destination accounts with an empty balance: 65% of fraud versus 14% of normal transactions.
- The existing rule flags only 16 transactions in the whole dataset, out of 8,213 frauds.

## Limitations

- The data is synthetic. Absolute scores will not transfer to live transactions, and the amounts are in a simulated currency, not naira.
- Precision and recall come from the full test period. The warehouse sample over-represents fraud (2.7% versus 0.3%), so figures computed from it would look better than they are.
- The model is trained once on a single time split. It has no hyperparameter tuning, probability calibration, or live monitoring.
- Reasons shown on alerts are rule-based explanations of common patterns, not model attributions.

## Next steps

- Retraining on a schedule and monitoring score drift.
- Per-alert explanations from the model itself.
- A feedback loop where analyst decisions on alerts become new training labels.

## Tech stack

Python, pandas, scikit-learn, dbt, Supabase (Postgres), SQLAlchemy, Streamlit, Plotly

## Project structure

```
src/                 exploration, features, training, scoring and loading scripts
dbt_project/         staging and mart models, tests
app/                 Streamlit review dashboard
reports/             saved evaluation metrics
```

## Run it yourself

1. Clone the repo and create a virtual environment:

```bash
git clone https://github.com/fizzlee998/fraud-monitoring-system.git
cd fraud-monitoring-system
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2. Download the PaySim dataset from Kaggle and place the CSV in `data/`.

3. Create a free Supabase project and a `.env` file:

```
DATABASE_URL=postgresql+psycopg://USER:PASSWORD@HOST:5432/postgres
DBT_HOST=your_host
DBT_USER=your_user
DBT_PASSWORD=your_password
```

4. Load the sample, build and test the dbt models:

```bash
python src/load_sample.py
cd dbt_project
dotenv -f ../.env run -- dbt run --profiles-dir .
dotenv -f ../.env run -- dbt test --profiles-dir .
cd ..
```

5. Train the model, score the test period, and start the dashboard:

```bash
python -m src.train
python -m src.score
streamlit run app/streamlit_app.py
```

## Author

Abdulafeez Oyewola

[GitHub](https://github.com/fizzlee998) | [LinkedIn](https://linkedin.com/in/oyewola-abdulafeez-19782a2b) | [Portfolio](https://oyewolaabdulafeez.netlify.app)
