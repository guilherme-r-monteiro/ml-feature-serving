import numpy as np
import pandas as pd

BINS = 10
PSI_ALERT = 0.2
EPSILON = 1e-4


def psi(reference: np.ndarray, current: np.ndarray) -> float:
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, BINS + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_share = np.histogram(reference, edges)[0] / len(reference) + EPSILON
    cur_share = np.histogram(current, edges)[0] / len(current) + EPSILON
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def psi_table(reference: pd.DataFrame, current: pd.DataFrame) -> dict[str, float]:
    return {column: psi(reference[column].to_numpy(), current[column].to_numpy()) for column in reference.columns}


def alerts(table: dict[str, float]) -> list[str]:
    return [column for column, value in table.items() if value >= PSI_ALERT]
