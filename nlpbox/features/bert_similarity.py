"""Esse módulo contém características
de similaridade entre textos baseadas
no BERT.
"""
from __future__ import annotations

from dataclasses import dataclass

import spacy
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

from nlpbox import resources
from nlpbox.core import FeatureExtractor

from .utils import DataclassFeatureSet


@dataclass(frozen=True)
class BERTSimilarityFeatures(DataclassFeatureSet):
    bert_similarity_cosine: float


class BERTSimilarityExtractor(FeatureExtractor):
    def __init__(self,
                 reference_text: str,
                 bert_model: SentenceTransformer = None,
                 device: str = 'cpu') -> None:
        if bert_model is None:
            model_name = 'neuralmind/bert-base-portuguese-cased'
            bert_model = SentenceTransformer(model_name,
                                             device=device)
        self._model = bert_model
        self._ref_text = reference_text
        self._ref_embdedings = self._model.encode([reference_text.lower()],
                                                  convert_to_tensor=True)

    @property
    def reference_text(self) -> str:
        return self._ref_text

    @reference_text.setter
    def reference_text(self, value: str) -> str:
        self._ref_text = value
        self._ref_embdedings = self._model.encode([self._ref_text.lower()],
                                                  convert_to_tensor=True)

    def extract(self, text: str) -> BERTSimilarityFeatures:
        embeddings = self._model.encode([text.lower()], convert_to_tensor=True)
        similarity = cos_sim(embeddings, self._ref_embdedings).item()
        return BERTSimilarityFeatures(bert_similarity_cosine=similarity)
