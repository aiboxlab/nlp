"""Esse módulo contém a implementação
de um classificador LightGBM.
"""
from __future__ import annotations

import numpy as np
from lightgbm import LGBMClassifier as _LGBMClassifier
from numpy.typing import ArrayLike

from aibox.nlp.core import Estimator


class LGBMClassifier(Estimator):
    def __init__(self,
                 n_estimators: int = 100,
                 learning_rate: float = 0.1,
                 boosting_type: str = 'gbdt',
                 importance_type: str = 'split',
                 class_weight: str | dict = None,
                 random_state: int | None = None):
        super().__init__(random_state=random_state)
        self._hyperparams = dict(n_estimators=n_estimators,
                                 learning_rate=learning_rate,
                                 boosting_type=boosting_type,
                                 importance_type=importance_type,
                                 class_weight=class_weight,
                                 random_state=self.random_state)

        self._lgbm = None

    def predict(self, X: ArrayLike, **kwargs) -> np.ndarray:
        del kwargs

        preds = self._lgbm.predict(X)
        return np.array(preds)

    def fit(self, X: ArrayLike, y: ArrayLike, **kwargs):
        del kwargs

        n = np.unique(y).size
        objective = 'multiclass' if n > 2 else 'binary'
        self._lgbm = _LGBMClassifier(verbose=-1,
                                     objective=objective,
                                     **self._hyperparams)
        self._lgbm.fit(X, y)

    @property
    def hyperparameters(self) -> dict:
        return self._hyperparams

    @property
    def params(self) -> dict:
        params = self._lgbm.get_params()

        return {k: v for k, v in params.items()
                if k not in self.hyperparameters}
