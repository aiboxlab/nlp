"""Esse módulo contém a implementação
de um classificador XGBoost.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from xgboost import XGBRegressor as _XGBRegressor

from nlpbox.core import Estimator


class XGBoostRegressor(Estimator):
    def __init__(self,
                 n_estimators: int = 100,
                 learning_rate: float = None,
                 grow_policy: int = None,
                 booster: str = None,
                 tree_method: str = None,
                 random_state: int | None = None):
        super().__init__(random_state=random_state)
        self._hyperparams = dict(n_estimators=n_estimators,
                                 learning_rate=learning_rate,
                                 grow_policy=grow_policy,
                                 booster=booster,
                                 three_method=tree_method,
                                 random_state=self.random_state)

        self._xgb = _XGBRegressor(verbosity=0,
                                  warm_start=False,
                                  **self._hyperparams)

    def predict(self, X: ArrayLike, **kwargs) -> np.ndarray:
        del kwargs

        preds = self._xgb.predict(X)
        return np.array(preds)

    def fit(self, X: ArrayLike, y: ArrayLike, **kwargs):
        del kwargs

        self._xgb.fit(X, y)

    @property
    def hyperparameters(self) -> dict:
        return self._hyperparams

    @property
    def params(self) -> dict:
        params = self._xgb.get_params()

        return {k: v for k, v in params.items()
                if k not in self.hyperparameters}
