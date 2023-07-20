"""
"""
from __future__ import annotations

from nlpbox.core import FeatureExtractor, Vectorizer


class FeatureExtractorVectorizer(Vectorizer):
    def __init__(self, extractor: FeatureExtractor) -> None:
        self._extractor = extractor

    def _vectorize(self, text: str):
        feature_set = self._extractor.extract(text)
        return feature_set.as_numpy()
