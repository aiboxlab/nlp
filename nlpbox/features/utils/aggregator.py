"""Módulo com utilidades para agregação
de múltiplas features/extratores.
"""
from __future__ import annotations

from multiprocessing import Pool
from typing import Iterable

from nlpbox.core import FeatureExtractor, FeatureSet


class AggregatedFeatures(FeatureSet):
    def __init__(self, *features: FeatureSet):
        self._features = features

    def as_dict(self) -> dict[str, float]:
        combined_dict = {k: v
                         for fs in self._features
                         for k, v in fs.as_dict().items()}
        sorted_dict = dict(sorted(combined_dict.items(),
                                  key=lambda x: x[0]))
        return sorted_dict

    @property
    def features_sets(self) -> Iterable[FeatureSet]:
        return self._features


class AggregatedFeatureExtractor(FeatureExtractor):
    def __init__(self, *extractors,
                 n_parallel: int = None) -> None:
        self._extractors = extractors

        assert (n_parallel is None) or (n_parallel > 0)
        self._n_parallel = None

    @property
    def extractors(self) -> list[FeatureExtractor]:
        return self._extractors

    def extract(self, text: str) -> AggregatedFeatures:
        if self._n_parallel is None:
            features = [e.extract(text) for e in self._extractors]
        else:
            with Pool(self._n_parallel) as pool:
                features = pool.map(_apply_extraction,
                                    [(e, text) for e in self._extractors])

        return AggregatedFeatures(*features)


def _apply_extraction(data) -> FeatureSet:
    e, t = data
    return e.extract(t)
