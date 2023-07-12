"""Esse módulo contém a definição
da interface básica para extratores
de características.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np


class Features(Protocol):
    """Essa classe representa a interface
    para um conjunto de features númericas.
    """

    def names(self) -> set[str]:
        """Retorna o nome das características
        presentes nessa instância.
        """

    def as_dict(self) -> dict[str, float]:
        """Retorna os pares <feature, valor> (ordenados
        de acordo com o nome da característica) para
        as características presentes nessa instância.
        """

    def as_numpy(self) -> np.ndarray[np.float32]:
        """Retorna os valores das características
        como uma NumPy array. As features são ordenadas
        na ordem lexicográfica.
        """


class FeatureExtractor(Protocol):
    """Essa classe representa a interface de um
    extrator de características.
    """

    def extract(self, text: str, **kwargs) -> Features:
        pass
