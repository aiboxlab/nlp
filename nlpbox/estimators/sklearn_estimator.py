"""
"""
from __future__ import annotations

import numpy as np

from nlpbox.core.estimator import Estimator


class SklearnEstimator(Estimator):
    def __init__(self, estimator) -> None:
        self._estimator = estimator

    def predict(self, X) -> np.ndarray:
        preds = self._estimator.predict(X)
        return np.array(preds)

    def fit(self, X, y):
        self._estimator.fit(X, y)

    @property
    def hyperparameters(self) -> dict:
        return self.params

    @property
    def params(self) -> dict:
        return self._estimator.get_params()
