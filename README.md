# ml-feature-serving

Point-in-time correct features shared by training and serving, a scoring API with a prediction log, drift checks, and bandit traffic allocation. Measured on public credit data.

Design note: `docs/design.md`. Raw results: `results/serving-results.json`.

- Data: UCI "Default of Credit Card Clients" (30,000 clients, six months of payment history).
- Target: the client is delinquent in the month after the as-of month.
- Split: train on as-of months 3 and 4, test on as-of month 5.
- Models: logistic regression, gradient boosting, and a weak baseline that sees only limit and age.

## Results

| Check | Result |
|---|---|
| Offline against online features, 3,000 clients, 3 as-of months | maximum absolute difference 0.0 |
| AUC with point-in-time features (logistic) | 0.719 |
| AUC when features use month 6 data (leaky) | 0.939 |
| Test AUC: logistic, boosted, weak baseline | 0.719, 0.780, 0.619 |
| PSI on real month-5 data against training months | every feature below 0.02, no alert |
| PSI on `mean_bill_window` after an injected shift, factor 1.5 / 2.0 / 3.0 | 0.099 / 0.230 / 0.494 (alert threshold 0.2) |
| API, 300 requests through the in-process test client | 300 logged rows, none without model version, latency p50 9.0 ms and p95 12.2 ms |
| API, unknown client | HTTP 404 |

Routing, 5,000 replayed requests over three models, five seeds. Arm accuracy at a fixed review rate: 0.888, 0.891 and 0.732.

| Policy | Mean cumulative regret | Per seed |
|---|---|---|
| Thompson sampling | 18.1 | 27, 27, 13, 23, 2 |
| Fixed even split | 259.7 | 281, 259, 233, 274, 253 |

What the results say:
- The leakage check shows why the as-of rule matters: using future months lifts AUC from 0.72 to 0.94, which would never hold in production.
- The monitor does not alert on a 1.5 times shift of one feature (PSI 0.099). It alerts at 2 times. A threshold of 0.2 is blunt for moderate shifts.
- Thompson sampling cuts regret mostly by moving traffic away from the weak model. The two strong models are within 0.3 points of each other, so the bandit cannot separate them in 5,000 requests.

## Limits

- The data has monthly granularity and no event timestamps, so "point in time" means month boundaries.
- Offline against online agreement is 0.0 because both paths call the same function. The check guards against future changes that break that. It does not show agreement between two independent implementations.
- The online store is SQLite, not a managed key-value store. Only 3,000 sampled clients are materialized.
- API latency is measured in-process with the test client. It excludes the network and a real server.
- Routing is replayed on stored labels with a fixed review rate. It is not live traffic and does not model delayed labels.
- Real month-5 drift is tiny because it is a single public snapshot. Drift detection was exercised only with injected shifts.
- No cloud services (SageMaker, DynamoDB, Lambda, CDK) were used, so nothing here shows experience with them.

## Run it

```sh
uv venv .venv && uv pip install --python .venv/bin/python -e . pytest openpyxl
curl -L -o data/credit.xls "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
.venv/bin/python -c "import pandas as pd; pd.read_excel('data/credit.xls', header=1).to_csv('data/credit.csv', index=False)"
.venv/bin/python -m pytest
.venv/bin/python -m feature_serving.experiments data/credit.csv results/serving-results.json .
```

The download is a zip. Unzip it and use the `.xls` inside.
