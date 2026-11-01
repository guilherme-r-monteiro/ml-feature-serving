import pandas as pd
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

from feature_serving.api import create_app
from feature_serving.features import FEATURE_NAMES, compute_features
from feature_serving.modeling import NamedModel
from feature_serving.router import ThompsonRouter
from feature_serving.stores import materialize, online_frame, open_database
from tests.test_features import sample_clients, sample_events


def test_online_and_offline_features_agree() -> None:
    connection = open_database(":memory:")
    materialize(connection, sample_events(), sample_clients(), 4)
    offline = compute_features(sample_events(), sample_clients(), 4)
    assert (offline - online_frame(connection, 4)).abs().to_numpy().max() == 0.0


def build_client() -> tuple[TestClient, object]:
    connection = open_database(":memory:")
    materialize(connection, sample_events(), sample_clients(), 4)
    frame = pd.DataFrame([[0.0] * len(FEATURE_NAMES), [1.0] * len(FEATURE_NAMES)], columns=list(FEATURE_NAMES))
    estimator = LogisticRegression().fit(frame, [0, 1])
    model = NamedModel("logistic", "v1", tuple(FEATURE_NAMES), estimator)
    return TestClient(create_app(connection, [model], ThompsonRouter.create(1, 0))), connection


def test_score_is_logged_with_model_and_feature_snapshot() -> None:
    client, connection = build_client()
    body = client.post("/score", json={"client_id": 1, "as_of": 4}).json()
    row = connection.execute("SELECT model, model_version, feature_version, snapshot FROM prediction_log").fetchone()
    assert row == (body["model"], body["model_version"], body["feature_version"], body["snapshot"])


def test_unknown_client_returns_404() -> None:
    client, _ = build_client()
    assert client.post("/score", json={"client_id": 99, "as_of": 4}).status_code == 404
