"""Esse módulo contém a definição
de uma interface básica para o cálculo
de métricas.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np


class Metric(Protocol):
    """Essa é a interface para uma métrica.

    Toda métrica recebe os valores reais e os
    preditos por algum estimador e retorna
    uma numpy array com os resultados.
    """

    def compute(self,
                y_true: np.ndarray,
                y_pred: np.ndarray) -> np.ndarray[np.float32]:
        """Computa o valor dessa métrica para as
        entradas recebidas.

        Args:
            y_true: valores reais.
            y_pred: valores preditos por algum estimator.
            **kwargs: argumentos extras que podem ser utilizados
                por algumas métricas para controlar seu
                comportamento.

        Returns:
            metrics: NumPy array com os valores da métrica.
        """
        pass
