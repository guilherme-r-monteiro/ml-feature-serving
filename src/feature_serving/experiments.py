import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from feature_serving.api import create_app
from feature_serving.data import load_clients, load_events
from feature_serving.drift import PSI_ALERT, alerts, psi_table
from feature_serving.features import FEATURE_NAMES, compute_features, leaky_features, target
from feature_serving.modeling import NamedModel, auc, fit_models
from feature_serving.router import ThompsonRouter, simulate
from feature_serving.stores import materialize, online_frame, open_database

TRAIN_MONTHS = (3, 4)
TEST_MONTH = 5
SAMPLE_CLIENTS = 3000
REQUESTS = 5000
SEEDS = (0, 1, 2, 3, 4)
API_REQUESTS = 300
SHIFT_FACTORS = (1.5, 2.0, 3.0)


def stacked(events: pd.DataFrame, clients: pd.DataFrame, months: tuple[int, ...]) -> tuple[pd.DataFrame, pd.Series]:
    parts = [(compute_features(events, clients, m), target(events, m)) for m in months]
    features = pd.concat([f.loc[f.index.intersection(t.index)] for f, t in parts], ignore_index=True)
    labels = pd.concat([t.loc[f.index.intersection(t.index)] for f, t in parts], ignore_index=True)
    return features, labels


def skew_report(events: pd.DataFrame, clients: pd.DataFrame, sample: list[int], path: Path) -> dict[str, float]:
    connection = open_database(path)
    sample_events = events[events["client_id"].isin(sample)]
    sample_clients = clients.loc[sample]
    for month in (*TRAIN_MONTHS, TEST_MONTH):
        materialize(connection, sample_events, sample_clients, month)
    differences = []
    for month in (*TRAIN_MONTHS, TEST_MONTH):
        offline = compute_features(sample_events, sample_clients, month).sort_index()
        online = online_frame(connection, month)
        differences.append(float((offline - online.loc[offline.index]).abs().to_numpy().max()))
    connection.close()
    return {"clients": len(sample), "max_abs_difference": max(differences)}


def leakage_report(events: pd.DataFrame, clients: pd.DataFrame) -> dict[str, float]:
    features, labels = stacked(events, clients, TRAIN_MONTHS)
    test_labels = target(events, TEST_MONTH)
    honest_test = compute_features(events, clients, TEST_MONTH).loc[test_labels.index]
    honest = fit_models(features, labels)[0]
    leaky_train = leaky_features(events, clients)
    leaky_labels = pd.concat([target(events, m) for m in TRAIN_MONTHS]).groupby(level=0).first()
    leaky_model = fit_models(leaky_train.loc[leaky_labels.index], leaky_labels)[0]
    return {
        "auc_point_in_time": auc(honest, honest_test, test_labels),
        "auc_with_future_data": auc(leaky_model, leaky_train.loc[test_labels.index], test_labels),
    }


def reward_matrix(models: list[NamedModel], features: pd.DataFrame, labels: pd.Series, rate: float) -> np.ndarray:
    columns = []
    for model in models:
        scores = model.score(features)
        threshold = np.quantile(scores, 1 - rate)
        columns.append(((scores >= threshold).astype(int) == labels.to_numpy()).astype(int))
    return np.stack(columns, axis=1)


def routing_report(rewards: np.ndarray) -> dict[str, object]:
    adaptive = [simulate(rewards, REQUESTS, seed, True) for seed in SEEDS]
    uniform = [simulate(rewards, REQUESTS, seed, False) for seed in SEEDS]
    return {
        "arm_accuracy": rewards.mean(axis=0).tolist(),
        "regret_thompson_mean": statistics.mean(adaptive),
        "regret_uniform_mean": statistics.mean(uniform),
        "regret_thompson": adaptive,
        "regret_uniform": uniform,
    }


def api_report(connection, models: list[NamedModel], sample: list[int]) -> dict[str, float]:
    app = create_app(connection, models, ThompsonRouter.create(len(models), 0))
    client = TestClient(app)
    latencies = []
    for client_id in sample[:API_REQUESTS]:
        started = time.perf_counter()
        response = client.post("/score", json={"client_id": client_id, "as_of": TEST_MONTH})
        latencies.append((time.perf_counter() - started) * 1000)
        response.raise_for_status()
    rows = connection.execute("SELECT COUNT(*), COUNT(DISTINCT snapshot), SUM(model_version IS NULL) FROM prediction_log").fetchone()
    missing = client.post("/score", json={"client_id": -1, "as_of": TEST_MONTH}).status_code
    return {
        "requests": API_REQUESTS,
        "logged_rows": rows[0],
        "rows_without_model_version": rows[2],
        "unknown_client_status": missing,
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
    }


def main(csv_path: Path, output: Path, work_dir: Path) -> None:
    events, clients = load_events(csv_path), load_clients(csv_path)
    rng = np.random.default_rng(0)
    sample = sorted(int(c) for c in rng.choice(clients.index.to_numpy(), SAMPLE_CLIENTS, replace=False))
    report: dict[str, object] = {"skew": skew_report(events, clients, sample, work_dir / "online.db")}
    report["leakage"] = leakage_report(events, clients)
    features, labels = stacked(events, clients, TRAIN_MONTHS)
    models = fit_models(features, labels)
    test_labels = target(events, TEST_MONTH)
    test_features = compute_features(events, clients, TEST_MONTH).loc[test_labels.index]
    report["model_auc"] = {m.name: auc(m, test_features, test_labels) for m in models}
    table = psi_table(features, test_features)
    injected = {
        str(factor): psi_table(features, test_features.assign(mean_bill_window=test_features["mean_bill_window"] * factor))
        for factor in SHIFT_FACTORS
    }
    report["drift"] = {
        "psi_real": table,
        "alerts_real": alerts(table),
        "psi_injected_mean_bill": {factor: values["mean_bill_window"] for factor, values in injected.items()},
        "alerts_injected": {factor: alerts(values) for factor, values in injected.items()},
        "threshold": PSI_ALERT,
    }
    report["routing"] = routing_report(reward_matrix(models, test_features, test_labels, float(labels.mean())))
    report["api"] = api_report(open_database(work_dir / "online.db"), models, sample)
    output.write_text(json.dumps(report, indent=2))
    sys.stdout.write(json.dumps({k: report[k] for k in ("skew", "leakage", "model_auc")}) + "\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
