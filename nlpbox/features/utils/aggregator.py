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
    def __init__(self, *extractors) -> None:
        self._extractors = extractors

    @property
    def extractors(self) -> list[FeatureExtractor]:
        return self._extractors

    def extract(self, text: str, **kwargs) -> AggregatedFeatures:
        del kwargs

        features = [e.extract(text) for e in self._extractors]
        return AggregatedFeatures(*features)
