"""Módulo com utilidades para
o cacheamento na extração de
características.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from aibox.nlp.core import FeatureExtractor, FeatureSet


class FeatureCache(ABC):
    """Interface básica para um cacheador
    de características.
    """

    @abstractmethod
    def get(self, text: str) -> FeatureSet | None:
        """Obtém o feature set para esse texto,
        caso exista no cache, ou retorna None.

        Args:
            text (str): texto.

        Returns:
            FeatureSet: conjunto de características
                armazenadas ou None.
        """

    @abstractmethod
    def save(self,
             text: str,
             data: FeatureSet,
             overwrite: bool = False) -> bool:
        """Adiciona uma entrada no cache.

        Args:
            text (str): texto.
            data (FeatureSet): conjunto de características.
            overwrite (bool): se devemos sobrescrever caso
                o texto já esteja no cache (default=False).

        Returns:
            bool: indica se foi realizado o salvamento ou
                não.
        """

    def as_dict(self) -> dict[str, FeatureSet]:
        """Retorna esse cache como
        um dicionário de textos para FeatureSet.

        Returns:
            dict: dicionário com os textos cacheados.
        """


class DictFeatureCache(FeatureCache):
    def __init__(self,
                 max_limit: int = -1):
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
        self._cache: dict[str, FeatureSet] = dict()

    def get(self, text: str) -> FeatureSet | None:
        return self._cache.get(text, None)

    def save(self,
             text: str,
             data: FeatureSet,
             overwrite: bool = False) -> bool:
        if not overwrite and text in self._cache:
            return False

        self._cache[text] = data
        self._prune_to_limit()
        return True

    def as_dict(self) -> dict[str, FeatureSet]:
        return self._cache.copy()

    def _prune_to_limit(self):
        mem_size = len(self._cache)

        if self._max <= 0 or mem_size <= self._max:
            return

        keys = list(self._cache)
        diff = mem_size - self._max

        for k in keys[0:diff]:
            del self._cache[k]


class CachedExtractor(FeatureExtractor):
    def __init__(self,
                 extractor: FeatureExtractor,
                 memory: FeatureCache | None = None):
        self._extractor = extractor
        self._memory = memory

    @property
    def memory(self) -> FeatureCache:
        return self._memory

    @property
    def extractor(self) -> FeatureExtractor:
        return self._extractor

    def extract(self, text: str) -> FeatureSet:
        features = self._memory.get(text)

        if features is None:
            features = self._extractor.extract(text)
            _ = self._memory.save(text, features)

        return features
