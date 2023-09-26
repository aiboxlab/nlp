"""Esse módulo contém um Wrapper
para estimadores do scikit-learn.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from nlpbox.core.estimator import Estimator


class SklearnEstimator(Estimator):
    """Wrapper para estimadores do scikit-learn.
    """

    def __init__(self, estimator) -> None:
        """Construtor. Recebe o estimador
        do scikit a ser envelopado.

        Args:
            estimator: estimador do scikit que possui
                os métodos `fit` e `predict`.
        """
        self._estimator = estimator

    def predict(self, X: ArrayLike, **kwargs) -> np.ndarray:
        del kwargs

        preds = self._estimator.predict(X)
        return np.array(preds)

    def fit(self, X: ArrayLike, y: ArrayLike, **kwargs):
        del kwargs

        self._estimator.fit(X, y)

    @property
    def hyperparameters(self) -> dict:
        return self.params

    @property
    def params(self) -> dict:
        return self._estimator.get_params()
