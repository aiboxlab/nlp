"""Módulo com utilidades para
o cacheamento na vetorização de
textos.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import ArrayLike

from aibox.nlp.core import TrainableVectorizer, Vectorizer


class VectorizerCache(ABC):
    """Interface básica para um cacheador
    de vetorizador.
    """

    @abstractmethod
    def get(self, text: str) -> np.ndarray | None:
        """Obtém a representação numérica para
        esse texto caso exista no cache,
        ou retorna None.

        Args:
            text (str): texto.

        Returns:
            representação numérica ou None.
        """

    @abstractmethod
    def save(self, text: str, data: np.ndarray, overwrite: bool = False) -> bool:
        """Adiciona uma entrada no cache.

        Args:
            text (str): texto.
            data (np.ndarray | torch.Tensor): representação
                numérica.
            overwrite (bool): se devemos sobrescrever caso
                o texto já esteja no cache (default=False).

        Returns:
            bool: indica se foi realizado o salvamento ou
                não.
        """

    def as_dict(self) -> dict[str, np.ndarray]:
        """Retorna esse cache como
        um dicionário de textos
        para NumPy arrays.

        Returns:
            dict: dicionário com os textos cacheados.
        """


class DictVectorizerCache(VectorizerCache):
    def __init__(self, max_limit: int = -1):
        """Construtor. Pode ter um tamanho
        máximo para a quantidade de entradas
        armazenadas no cache.

        Args:
            max_limit (int, opcional): quantidade máxima
                de entradas para armazenar. Se menor ou
                igual a 0, não são aplicados limites
                (default=-1).
        """
        self._max = max_limit
        self._cache: dict[str, np.ndarray] = dict()

    def get(self, text: str) -> np.ndarray | None:
        return self._cache.get(text, None)

    def save(self, text: str, data: np.ndarray, overwrite: bool = False) -> bool:
        if not overwrite and text in self._cache:
            return False

        self._cache[text] = data
        self._prune_to_limit()
        return True

    def as_dict(self) -> dict[str, np.ndarray]:
        return self._cache.copy()

    def _prune_to_limit(self):
        mem_size = len(self._cache)

        if self._max <= 0 or mem_size <= self._max:
            return

        keys = list(self._cache)
        diff = mem_size - self._max

        for k in keys[0:diff]:
            del self._cache[k]


class CachedVectorizer(Vectorizer):
    def __init__(self, vectorizer: Vectorizer, memory: VectorizerCache | None = None):
        self._vectorizer = vectorizer
        self._memory = memory

    @property
    def memory(self) -> VectorizerCache:
        return self._memory

    @property
    def vectorizer(self) -> Vectorizer:
        return self._vectorizer

    def _vectorize(self, text: str, **kwargs) -> ArrayLike:
        arr = self._memory.get(text)

        if arr is None:
            # Garantindo que não kwargs não contém
            #   chave duplicada
            kwargs.pop("vector_type", None)

            # Realizando vetorização
            arr = self._vectorizer.vectorize(text, vector_type="numpy", **kwargs)

            # Salvando na memória
            _ = self._memory.save(text, arr)

        return arr


class TrainableCachedVectorizer(TrainableVectorizer):
    def __init__(
        self, vectorizer: TrainableVectorizer, memory: VectorizerCache | None = None
    ) -> None:
        self._cache = CachedVectorizer(vectorizer=vectorizer, memory=memory)
        self._trained = False

    @property
    def memory(self) -> VectorizerCache:
        return self._cache.memory

    @property
    def vectorizer(self) -> Vectorizer:
        return self._cache.vectorizer

    def fit(self, X: ArrayLike, y: ArrayLike | None = None, **kwargs) -> None:
        if not self._trained:
            self._cache.vectorizer.fit(X, y, **kwargs)
            self._trained = True

    def _vectorize(self, text: str, **kwargs) -> ArrayLike:
        return self._cache._vectorize(text, **kwargs)
