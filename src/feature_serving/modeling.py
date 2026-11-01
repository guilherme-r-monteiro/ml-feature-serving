from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 0
WEAK_FEATURES = ("limit_balance", "age")


@dataclass(frozen=True)
class NamedModel:
    name: str
    version: str
    columns: tuple[str, ...]
    estimator: object

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict_proba(frame[list(self.columns)])[:, 1]


def fit_models(features: pd.DataFrame, labels: pd.Series) -> list[NamedModel]:
    columns = tuple(features.columns)
    logistic = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(features, labels)
    boosted = HistGradientBoostingClassifier(random_state=SEED, max_iter=150).fit(features, labels)
    weak = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(features[list(WEAK_FEATURES)], labels)
    return [
        NamedModel("logistic", "v1", columns, logistic),
        NamedModel("boosted", "v1", columns, boosted),
        NamedModel("weak_baseline", "v1", WEAK_FEATURES, weak),
    ]


def auc(model: NamedModel, features: pd.DataFrame, labels: pd.Series) -> float:
    return float(roc_auc_score(labels, model.score(features)))
