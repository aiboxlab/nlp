"""Esse módulo contém a implementação
de métricas de erro (RMSE, MAE, R2,
MSE, etc).
"""
from __future__ import annotations

import numpy as np

from nlpbox.core import Metric

from . import utils


class MAE(Metric):
    """Mean Absolute Error (MAE).
    """

    @utils.to_float32_array
    def compute(self,
                y_true: np.ndarray,
                y_pred: np.ndarray) -> np.ndarray[np.float32]:
        return np.mean(np.abs(y_pred - y_true))


class MSE(Metric):
    """Mean Squared Error (MAE).
    """

    @utils.to_float32_array
    def compute(self,
                y_true: np.ndarray,
                y_pred: np.ndarray) -> np.ndarray[np.float32]:
        return np.mean((y_true - y_pred) ** 2)


class RMSE(Metric):
    """Root Mean Squared Error (MAE).
    """

    @utils.to_float32_array
    def compute(self,
                y_true: np.ndarray,
                y_pred: np.ndarray) -> np.ndarray[np.float32]:
        return np.sqrt(MSE().compute(y_true, y_pred))


class R2(Metric):
    """R-squared Error (R2).
    """

    @utils.to_float32_array
    def compute(self,
                y_true: np.ndarray,
                y_pred: np.ndarray) -> np.ndarray[np.float32]:
        y_bar = y_true.mean()
        ss_tot = ((y_true - y_bar) ** 2).sum()
        ss_res = ((y_true - y_pred) ** 2).sum()

        return 1 - (ss_res / ss_tot)
