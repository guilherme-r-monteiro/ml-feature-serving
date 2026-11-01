import hashlib
import json

import numpy as np
import pandas as pd

WINDOW_MONTHS = 3
FEATURE_NAMES = (
    "limit_balance",
    "age",
    "last_pay_status",
    "max_pay_status_window",
    "mean_bill_window",
    "payment_ratio_window",
    "utilization_last",
    "bill_trend_window",
)
STATIC_FEATURES = {"limit_balance": "LIMIT_BAL", "age": "AGE"}


def feature_version() -> str:
    payload = json.dumps({"names": FEATURE_NAMES, "window": WINDOW_MONTHS}).encode()
    return hashlib.sha256(payload).hexdigest()[:10]


def window_events(events: pd.DataFrame, as_of_month: int) -> pd.DataFrame:
    return events[(events["month"] <= as_of_month) & (events["month"] > as_of_month - WINDOW_MONTHS)]


def aggregate_window(window: pd.DataFrame) -> pd.DataFrame:
    grouped = window.sort_values("month").groupby("client_id")
    bills = grouped["bill_amount"]
    total_bill = bills.sum().replace(0, np.nan)
    return pd.DataFrame(
        {
            "last_pay_status": grouped["pay_status"].last(),
            "max_pay_status_window": grouped["pay_status"].max(),
            "mean_bill_window": bills.mean(),
            "payment_ratio_window": (grouped["payment_amount"].sum() / total_bill).fillna(1.0).clip(0, 5),
            "utilization_last": bills.last(),
            "bill_trend_window": bills.last() - bills.first(),
        }
    )


def compute_features(events: pd.DataFrame, clients: pd.DataFrame, as_of_month: int) -> pd.DataFrame:
    aggregated = aggregate_window(window_events(events, as_of_month))
    joined = aggregated.join(clients, how="inner")
    joined["limit_balance"] = joined["LIMIT_BAL"]
    joined["age"] = joined["AGE"]
    joined["utilization_last"] = joined["utilization_last"] / joined["LIMIT_BAL"]
    return joined[list(FEATURE_NAMES)].astype(float)


def snapshot_hash(values: dict[str, float]) -> str:
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()[:12]


def leaky_features(events: pd.DataFrame, clients: pd.DataFrame) -> pd.DataFrame:
    return compute_features(events, clients, as_of_month=int(events["month"].max()))


def target(events: pd.DataFrame, as_of_month: int) -> pd.Series:
    nxt = events[events["month"] == as_of_month + 1].set_index("client_id")["pay_status"]
    return (nxt >= 1).astype(int).rename("delinquent_next_month")
