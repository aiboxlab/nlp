"""Esse módulo contém características
relacionadas à densidade de padrões
sintáticos.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import spacy

from nlpbox.core import FeatureExtractor

from .utils import DataclassFeatureSet


@dataclass(frozen=True)
class SyntacticPatternsDensityFeatures(DataclassFeatureSet):
    gerund_verbs: float
    max_noun_phrase: float
    min_noun_phrase: float
    mean_noun_phrase: float


class SyntacticPatternsDensityExtractor(FeatureExtractor):
    def __init__(self, nlp: spacy.Language | None = None):
        if nlp is None:
            nlp = spacy.load('pt_core_news_md')

        self._nlp = nlp

    def extract(self,  text: str, **kwargs) -> SyntacticPatternsDensityFeatures:
        del kwargs

        doc = self._nlp(text)
        sentences = [sent.text for sent in doc.sents]
        features = {
            'gerund_verbs': 0.0,
            'max_noun_phrase': 0.0,
            'min_noun_phrase': 0.0,
            'mean_noun_phrase': 0.0
        }

        if len(sentences) > 1:
            gerund_verbs = self.compute_gerund_verbs(doc)
            minimum, maximum, mean = self.compute_min_max_mean_noun_phrases(
                doc)
            features['gerund_verbs'] = gerund_verbs
            features['max_noun_phrase'] = maximum
            features['min_noun_phrase'] = minimum
            features['mean_noun_phrase'] = mean

        return SyntacticPatternsDensityFeatures(**features)

    def compute_gerund_verbs(self, doc) -> float:
        """Método que computa a proporção de verbos
        no gerúndio em relação a todos os verbos do texto.
        """
        verbs = [token.text for token in doc if token.pos_ ==
                 'VERB' or token.pos_ == 'AUX']
        if len(verbs) == 0:
            return 0
        verbs_gerund = [token.text for token in doc
                        if (token.pos_ == 'VERB' or token.pos_ == 'AUX')
                        and 'Ger' in token.morph.get('VerbForm')]
        return len(verbs_gerund) / len(verbs)

    def compute_min_max_mean_noun_phrases(self, doc) -> [float, float, float]:
        """Método que computa o tamanho do menor,
        maior e média dos sintagmas nominais do texto.
        """
        noun_phrases_sizes = []
        noun_phrases = [noun_phrase for noun_phrase in doc.noun_chunks]
        if len(noun_phrases) == 0:
            return 0.0, 0.0, 0.0

        for noun_phrase in noun_phrases:
            tokens = [t for t in noun_phrase]
            noun_phrases_sizes.append(len(tokens))

        minimum = float(min(noun_phrases_sizes))
        maximum = float(max(noun_phrases_sizes))
        mean = np.mean(noun_phrases_sizes)

        return minimum, maximum, mean
