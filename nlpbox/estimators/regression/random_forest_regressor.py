"""Esse módulo contém a implementação
do Random Forest.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor as _RFRegressor

from nlpbox.core import Estimator


class RandomForestRegressor(Estimator):
    def __init__(self,
                 n_estimators: int = 100,
                 criterion: str = 'squared_error',
                 max_features: str | None = 'sqrt',
                 bootstrap: bool = False,
                 random_state: int = None):
        self._hyperparams = dict(n_estimators=n_estimators,
                                 criterion=criterion,
                                 max_features=max_features,
                                 bootstrap=bootstrap,
                                 random_state=random_state)

        self._rf = _RFRegressor(verbose=0,
                                warm_start=False,
                                **self._hyperparams)

    def predict(self, X) -> np.ndarray:
        preds = self._rf.predict(X)
        return np.array(preds)

    def fit(self, X, y):
        self._rf.fit(X, y)

    @property
    def hyperparameters(self) -> dict:
        return self._hyperparams

    @property
    def params(self) -> dict:
        params = self._rf.get_params()

        return {k: v for k, v in params.items()
                if k not in self.hyperparameters}