"""Esse módulo contém características
de similaridade entre textos baseadas
no FuzzySearch.
"""
from __future__ import annotations

from dataclasses import dataclass

import fuzzysearch
import polyfuzz
import spacy
from fuzzywuzzy import fuzz

from aibox.nlp import resources
from aibox.nlp.core import FeatureExtractor

from .utils import DataclassFeatureSet


@dataclass(frozen=True)
class FuzzySearchSimilarityFeatures(DataclassFeatureSet):
    fuzz_ratio: float
    fuzz_partial_ratio: float
    fuzz_token_sort_ratio: float
    fuzz_token_set_ratio: float
    fuzz_partial_token_set_ratio: float
    fuzz_partial_token_sort_ratio: float
    fuzzysearch_near_matches: float
    fuzz_wratio: float


class FuzzySearchSimilarityExtractor(FeatureExtractor):
    def __init__(self, reference_text: str):

        def _n_near_matches(t, p) -> float:
            near = fuzzysearch.find_near_matches(t,
                                                 p,
                                                 max_l_dist=10)
            return float(len(near))

        self._ref_text = reference_text
        self._features = {
            'fuzz_ratio': fuzz.ratio,
            'fuzz_partial_ratio': fuzz.partial_ratio,
            'fuzz_token_sort_ratio': fuzz.token_sort_ratio,
            'fuzz_token_set_ratio': fuzz.token_set_ratio,
            'fuzz_partial_token_set_ratio': fuzz.partial_token_set_ratio,
            'fuzz_partial_token_sort_ratio': fuzz.partial_token_sort_ratio,
            'fuzzysearch_near_matches': _n_near_matches,
            'fuzz_wratio': fuzz.WRatio
        }

    @property
    def reference_text(self) -> str:
        return self._ref_text

    @reference_text.setter
    def reference_text(self, value: str):
        self._ref_text = value

    def extract(self, text: str, **kwargs) -> FuzzySearchSimilarityFeatures:
        del kwargs

        return FuzzySearchSimilarityFeatures(**{
            k: float(f(text, self._ref_text))
            for k, f in self._features.items()
        })
