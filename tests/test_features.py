import pandas as pd

from feature_serving.drift import PSI_ALERT, psi
from feature_serving.features import compute_features, target
from feature_serving.router import ThompsonRouter

import numpy as np


def sample_events() -> pd.DataFrame:
    rows = []
    for month in range(1, 7):
        rows.append({"client_id": 1, "month": month, "pay_status": month, "bill_amount": 100 * month, "payment_amount": 50})
    return pd.DataFrame(rows)


def sample_clients() -> pd.DataFrame:
    return pd.DataFrame({"LIMIT_BAL": [1000], "SEX": [1], "EDUCATION": [1], "MARRIAGE": [1], "AGE": [30]}, index=pd.Index([1], name="client_id"))


def test_features_ignore_events_after_as_of() -> None:
    early = compute_features(sample_events(), sample_clients(), 3)
    changed = sample_events()
    changed.loc[changed["month"] > 3, "bill_amount"] = 999999
    assert early.equals(compute_features(changed, sample_clients(), 3))


def test_features_use_window_of_three_months() -> None:
    features = compute_features(sample_events(), sample_clients(), 4)
    assert features.loc[1, "mean_bill_window"] == 300 and features.loc[1, "last_pay_status"] == 4


def test_target_reads_next_month() -> None:
    assert target(sample_events(), 3).loc[1] == 1


def test_psi_is_small_for_same_and_large_for_shifted() -> None:
    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, 5000)
    assert psi(base, rng.normal(0, 1, 5000)) < 0.05
    assert psi(base, rng.normal(1.5, 1, 5000)) > PSI_ALERT


def test_thompson_router_prefers_better_arm() -> None:
    router = ThompsonRouter.create(2, 0)
    for _ in range(500):
        arm = router.choose()
        router.update(arm, 1 if arm == 1 else 0)
    assert router.successes[1] > router.successes[0]
