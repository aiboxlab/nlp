"""Esse módulo contém a implementação
da métrica f1-score.
"""
from __future__ import annotations

import numpy as np
import sklearn.metrics

from aibox.nlp.core import Metric

from . import utils


class F1Score(Metric):
    """Métrica para cálculo do F1-Score.
    """

    def __init__(self,
                 labels: list[int] = None,
                 average: str = None,
                 zero_division: float = 0.0) -> None:
        """Construtor. Essa métrica produz resultados diferentes
        dependendo das configurações.

        Caso labels não seja passado, inferimos quais são as labels
        de acordo com `y_true`.

        Já para average, temos as seguintes regras:
            - Caso None, temos os valores por classe;
            - Caso 'micro', calculamos a métrica global contando
                o número total de TP, FN e FP.
            - Caso 'macro', calculamos a métrica para cada classe
                e retornamos a média não ponderada.
            - Caso 'weighted', realizamos o mesmo processo que 'macro'
                mas retornamos a média ponderada onde os pesos são
                a quantidade de instâncias para aquela classe.

        Args:
            labels (list[int], opcional): lista com labels utilizadas,
                caso não seja passado é inferido das entradas.
            average (str, opcional): nenhum (None), micro, macro,
                weighted.
            zero_division (float): valor caso ocorra divisão por zero.
        """
        self._avg = average
        self._labels = labels
        self._zero_div = zero_division

    @utils.to_float32_array
    def compute(self,
                y_true: np.ndarray,
                y_pred: np.ndarray) -> np.ndarray[np.float32]:
        labels = self._labels

        if labels is None:
            # Inferindo as classes
            #   de acordo com y_true.
            labels = np.unique(y_true)

        return sklearn.metrics.f1_score(y_true=y_true,
                                        y_pred=y_pred,
                                        average=self._avg,
                                        labels=labels,
                                        zero_division=self._zero_div)

    def name(self) -> str:
        prefix = 'Class '

        if self._avg is not None:
            prefix = self._avg + ' '

        return prefix.capitalize() + 'F1-score'
