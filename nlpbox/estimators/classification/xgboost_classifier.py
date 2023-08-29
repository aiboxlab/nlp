"""Esse módulo contém a implementação
de um classificador XGBoost.
"""
from __future__ import annotations

import numpy as np
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier as _XGBClassifier

from nlpbox.core import Estimator


class XGBoostClassifier(Estimator):
    def __init__(self,
                 n_estimators: int = 100,
                 learning_rate: float = None,
                 grow_policy: int = None,
                 booster: str = None,
                 three_method: str = None,
                 random_state: int = None):
        self._hyperparams = dict(n_estimators=n_estimators,
                                 learning_rate=learning_rate,
                                 grow_policy=grow_policy,
                                 booster=booster,
                                 three_method=three_method,
                                 random_state=random_state)

        self._xgb = _XGBClassifier(verbosity=0,
                                   warm_start=False,
                                   **self._hyperparams)
        self._encoder = LabelEncoder()

        for k, p in self._xgb.get_params().items():
            if k in self._hyperparams:
                self._hyperparams[k] = p

    def predict(self, X) -> np.ndarray:
        preds = self._xgb.predict(X)
        preds = self._encoder.inverse_transform(preds)
        return np.array(preds)

    def fit(self, X, y):
        self._encoder.fit(np.unique(y))
        y_ = self._encoder.transform(y)
        self._xgb.fit(X, y_)

    @property
    def hyperparameters(self) -> dict:
        return self._hyperparams

    @property
    def params(self) -> dict:
        params = self._xgb.get_params()

        return {k: v for k, v in params.items()
                if k not in self.hyperparameters}
