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
        pass

    def as_dict(self) -> dict[str, float]:
        pass

    def as_numpy(self) -> np.ndarray[np.float32]:
        pass


class FeatureExtractor(Protocol):
    """Essa classe representa a interface de um
    extrator de características.
    """

    def extract(self, text: str, **kwargs) -> Features:
        pass
