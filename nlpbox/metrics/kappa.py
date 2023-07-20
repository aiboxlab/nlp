"""Esse módulo contém a implementação
de métricas relacionadas com o Kappa.
"""
from __future__ import annotations

import numpy as np
import sklearn.metrics

from nlpbox.core import Metric


class CohensKappaScore(Metric):

    def __init__(self, weights: str = None) -> None:
        self._w = weights

    def compute(self,
                y_true: np.ndarray,
                y_pred: np.ndarray) -> np.ndarray[np.float32]:
        return _get_kappa_score(y_true,
                                y_pred,
                                weights=self._w)


class NeighborCohensKappaScore(Metric):
    def __init__(self,
                 neighbor_limit: int = 1,
                 weights: str = None) -> None:
        assert neighbor_limit >= 1
        self._w = weights
        self._neighbor_limit = neighbor_limit

    def compute(self,
                y_true: np.ndarray,
                y_pred: np.ndarray) -> np.ndarray[np.float32]:
        vectorized = np.vectorize(self._get_target_if_neighbor)
        y_pred_neighbor = vectorized(y_true,
                                     y_pred)
        return _get_kappa_score(y_true,
                                y_pred_neighbor,
                                weights=self._w)

    def _get_target_if_neighbor(self,
                                target: int,
                                value: int) -> int:
        if abs(value - target) <= self._neighbor_limit:
            return target

        return value


def _get_kappa_score(y_true: np.ndarray,
                     y_pred: np.ndarray,
                     weights) -> np.ndarray[np.float32]:

    # Condições para cálculo do Kappa
    assert y_true.shape == y_pred.shape
    assert y_true.ndim == 1
    assert np.issubdtype(y_true.dtype, np.integer)
    assert np.issubdtype(y_pred.dtype, np.integer)

    # Implementação atual usa o scikit
    return sklearn.metrics.cohen_kappa_score(y_true,
                                             y_pred,
                                             weights=weights)
