"""Esse módulo contém as entidades relativas
à extração de características textuais.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch


class FeatureSet(ABC):
    """Representa um conjunto de características
    para um texto. Todas as características
    possuem um nome e todos os resultados são
    ordenados seguindo a ordem lexicográfica.
    """

    @abstractmethod
    def as_dict(self) -> dict[str, float]:
        """Retorna os valores das características
        desse conjunto para um dado texto.

        Returns:
            Dicionário (str -> float) com as
                características.
        """

    def as_numpy(self) -> np.ndarray[np.float32]:
        """Retorna as características como uma
        NumPy array. Os valores de cada índice são
        correspondentes às características na ordem
        de `names()`.

        Returns:
            NumPy array de np.float32 representando
                os valores das características.
        """
        return np.array(list(self.as_dict().values()),
                        dtype=np.float32)

    def as_tensor(self,
                  kind: str = 'torch',
                  device: str | None = None) -> torch.Tensor:
        """Retorna as características como um
        tensor. Os valores de cada índice são
        correspondentes às características na ordem
        de `names()`.

        Args:
            kind (str): tipo do tensor ('torch').
            device (str, opcional): dispositivo de armazenamento.

        Returns:
            Tensor do torch representado os valores das
                características.
        """
        tensor = torch.from_numpy(self.as_numpy())

        if device is not None:
            tensor = tensor.to(device)

        return tensor

    def names(self) -> list[str]:
        """Retorna os nomes das características
        em ordem lexicográfica. Todos os outros
        métodos apresentam os valores conforme
        essa ordem.

        Returns:
            Lista de str com o nome das características
                desse conjunto.
        """
        return list(self.as_dict())


class FeatureExtractor(ABC):
    """Representa um extrator de características,
    que possibilita extrair um conjunto de características
    de um texto passado como entrada.
    """

    @abstractmethod
    def extract(self, text: str) -> FeatureSet:
        """

        Args:
            text (str): texto para extração de características.

        Returns:
            Instância de um FeatureSet, representando
                as características e valores para
                esse texto.
        """
