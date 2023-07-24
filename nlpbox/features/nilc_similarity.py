"""Esse módulo contém características
clássicas de similaridade entre textos.
"""
from __future__ import annotations

from dataclasses import dataclass

import spacy
from gensim.models import KeyedVectors

from nlpbox import resources
from nlpbox.core import FeatureExtractor
from nlpbox.factory import register

from .utils import DataclassFeatureSet


@dataclass(frozen=True)
class NILCSimilarityFeatures(DataclassFeatureSet):
    similarity_jaccard: float
    similarity_dice: float
    similarity_cosine_cbow: float
    similarity_word_movers_cbow: float


@register('features.nilc_similarityBR')
class NILCSimilarityExtractor(FeatureExtractor):
    def __init__(self,
                 reference_text: str,
                 nlp: spacy.Language = None):
        if nlp is None:
            nlp = spacy.load('pt_core_news_md')

        root_dir = resources.path('external/nilc-word2vec50.v1')
        word2vec_path = root_dir.joinpath('cbow_s50.bin')
        self._wv = KeyedVectors.load_word2vec_format(word2vec_path,
                                                     binary=False)
        self._ref_doc = nlp(reference_text)
        self._ref_tokens = [t.text for t in self._ref_doc]
        self._nlp = nlp

    @property
    def reference_text(self) -> str:
        return self._ref_doc.text

    @reference_text.setter
    def reference_text(self, value: str) -> None:
        self._ref_doc = self._nlp(value)
        self._ref_tokens = [t.text for t in self._ref_doc]

    def extract(self, text: str) -> NILCSimilarityFeatures:
        doc = self._nlp(text)
        text_tokens = [t.text for t in doc]

        jaccard = self.jaccard(text_tokens, self._ref_tokens)
        dice = self.dice(text_tokens, self._ref_tokens)
        cos_cbow = 0.0
        word_movers_cbow = 0.0
        x = self._validate(text_tokens)
        y = self._validate(self._ref_tokens)

        try:
            cos_cbow = self._cosine_similarity(x, y)
        except Exception:
            pass

        try:
            word_movers_cbow = self._word_movers_distance(x, y)
        except Exception:
            pass

        return NILCSimilarityFeatures(
            similarity_jaccard=jaccard,
            similarity_dice=dice,
            similarity_cosine_cbow=cos_cbow,
            similarity_word_movers_cbow=word_movers_cbow)

    @staticmethod
    def jaccard(tokens_x: list[str], tokens_y: list[str]) -> float:
        """ Similaridade de jaccard.

        Args:
            tokens_x (list[str]): lista de sentenças.
            tokens_y (list[str]): lista de sentenças.

        Returns:
            float: similaridade.
        """
        x, y = set(tokens_x), set(tokens_y)
        if not x and not y:
            return 0.0
        return len(x & y) / len(x | y)

    @staticmethod
    def dice(tokens_x: list[str], tokens_y: list[str]) -> float:
        """ Similaridade de dice.

        Args:
            tokens_x (list[str]): lista de sentenças.
            tokens_y (list[str]): lista de sentenças.

        Returns:
            float: similaridade.
        """
        x, y = set(tokens_x), set(tokens_y)
        if not x and not y:
            return 0.0
        return (2 * len(x & y)) / (len(x) + len(y))

    def _validate(self, sent: list[str]) -> list[str]:
        return [t for t in sent if t in self._wv]

    def _cosine_similarity(self,
                           x: list[str],
                           y: list[str]) -> float:
        return self._wv.n_similarity(x, y)

    def _word_movers_distance(self,
                              x: list[str],
                              y: list[str]) -> float:
        x = ' '.join(x)
        y = ' '.join(y)
        return self._wv.wmdistance(x, y)
