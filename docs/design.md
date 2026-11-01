# Design note

Target: Facio, Senior Machine Learning Engineer. The posting asks for features that match in training and production, standard model serving with traceable predictions, drift monitoring, and traffic allocation across several models (A/B tests and bandits).

## Data

UCI "Default of Credit Card Clients": 30,000 clients, six months of payment status, bill amount and payment amount, plus static fields. It is public. Months are numbered 1 (April) to 6 (September).

## Design

1. One feature function (`features.compute_features`) with an `as_of_month` argument. It reads only events with month at or before `as_of_month`. Offline training and online serving call the same function.
2. Offline path: the function runs over all clients. Online path: a materializer calls the same function per client and writes rows to a SQLite table with the feature version and a snapshot hash.
3. Target: the client is delinquent (payment status of 1 or more) in the month after `as_of_month`.
4. Temporal split: train on as-of months 3 and 4, test on as-of month 5.
5. Serving: a FastAPI endpoint reads online features, picks a model with a router, scores, and writes a prediction log row with model, model version, feature version and snapshot hash.
6. Monitoring: population stability index per feature between training and test months, with an alert at 0.2.
7. Routing: Thompson sampling over three models, compared with a fixed even split.

## Checks to run

- Skew: maximum absolute difference between offline and online features.
- Leakage: AUC with point-in-time features against AUC with features that use month 6 data.
- Drift: PSI on real data and on an injected shift.
- Routing: cumulative regret over 5,000 replayed requests, five seeds.
- API: logged rows against requests, and latency of the in-process test client.

## Scope limits

The data has monthly granularity and no real timestamps. The online store is SQLite, not a managed key-value store. Routing is replayed on stored labels, not run on live traffic.
