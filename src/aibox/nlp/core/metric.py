"""Esse módulo contém a definição
de uma interface básica para o cálculo
de métricas.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Metric(ABC):
    """Essa é a interface para uma métrica.

    Toda métrica recebe os valores reais e os
    preditos por algum estimador e retorna
    uma numpy array com os resultados.
    """

    @abstractmethod
    def name(self) -> str:
        """Nome dessa métrica, toda
        métrica possui um nome único.

        Se dois instâncias de uma métrica
        possuem o mesmo nome, o valor
        do método `compute(...)` é
        o mesmo para ambas instâncias.

        Returns:
            str: nome e identificador dessa métrica.
        """

    @abstractmethod
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray[np.float32]:
        """Computa o valor dessa métrica para as
        entradas recebidas.

        Args:
            y_true: valores reais.
            y_pred: valores preditos por algum estimator.

        Returns:
            metrics: NumPy array com os valores da métrica.
        """

    def __repr__(self):
        """Representação de uma
        métrica como string.

        Returns:
            str: nome da classe seguido pelo
                nome da métrica.
        """
        return f"{self.__class__.__name__}: {self.name()}"

    def __eq__(self, other):
        """Função de igualdade.
        Duas métricas são iguais se possuem
        o mesmo nome.

        Args:
            other (object): outro objeto.

        Returns:
            bool: se o outro objeto é o mesmo.
        """
        if isinstance(other, Metric):
            return self.name() == other.name()

        return False
