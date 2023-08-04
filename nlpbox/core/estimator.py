"""Esse módulo contém a interface
para estimadores.
"""
from __future__ import annotations

from pathlib import Path
from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import ArrayLike


class Estimator(ABC):
    """Essa é a interface básica para um
    estimador.
    """

    @abstractmethod
    def predict(self, X: ArrayLike) -> np.ndarray:
        """Realiza a predição utilizando os parâmetros
        atuais do modelo.

        Args:
            X: array-like com formato (n_samples, n_features).

        Returns:
            NumPy array com as predições para cada amostra.
        """

    @abstractmethod
    def fit(self, X: ArrayLike, y: ArrayLike) -> None:
        """Realiza o treinamento do estimador
        utilizando as features X com os targets
        y.

        Args:
            X: array-like com formato (n_samples, n_features).
            y: array-like com formato (n_samples,).
        """

    @property
    @abstractmethod
    def hyperparameters(self) -> dict:
        """Retorna um dicionário descrevendo
        os hiper-parâmetros do estimador.

        Returns:
            Hiper-parâmetros do estimador.
        """

    @property
    @abstractmethod
    def params(self) -> dict:
        """Retorna um dicionário com os parâmetros
        para esse estimador. Os parâmetros retornados
        descrevem totalmente o estado do modelo (e,g.
        pesos de uma rede, superfícies de decisão, estrutura
        da árvore de decisão, etc).

        Returns:
            Parâmetros do estimador.
        """
