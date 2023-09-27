"""Esse módulo contém um regressor
do catboost.
"""
from __future__ import annotations

import numpy as np
from catboost import CatBoostRegressor as _CatBoostRegressor
from numpy.typing import ArrayLike

from aibox.nlp.core.estimator import Estimator


class CatBoostRegressor(Estimator):
    def __init__(self,
                 n_estimators: int = 100,
                 learning_rate: float = 0.01,
                 random_state: int | None = None):
        super().__init__(random_state=random_state)
        self._hyperparams = dict(n_estimators=n_estimators,
                                 learning_rate=learning_rate,
                                 random_state=self.random_state,
                                 loss_function='RMSE')
        self._catboost_regressor = _CatBoostRegressor(**self._hyperparams)

    def predict(self, X: ArrayLike, **kwargs) -> np.ndarray:
        del kwargs

        preds = self._catboost_regressor.predict(X)
        return np.array(preds)

    def fit(self, X: ArrayLike, y: ArrayLike, **kwargs):
        del kwargs

        self._catboost_regressor.fit(X, y, silent=True)

    @property
    def hyperparameters(self) -> dict:
        return self.hyperparameters

    @property
    def params(self) -> dict:
        params = self._catboost_regressor.get_all_params()
        return {k: v for k, v in params.items()
                if k not in self.hyperparameters}
