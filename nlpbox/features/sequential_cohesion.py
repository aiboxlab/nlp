"""Esse módulo contém as características
relacionadas com Coesão Sequencial
"""
from __future__ import annotations

from dataclasses import dataclass

import spacy
from spacy.tokens.doc import Doc
from TRUNAJOD.entity_grid import EntityGrid, get_local_coherence

from nlpbox.core import FeatureExtractor

from .utils import DataclassFeatureSet


@dataclass(frozen=True)
class SequentialCohesionFeatures(DataclassFeatureSet):
    local_coh_pu: float
    local_coh_pw: float
    local_coh_pacc: float
    local_coh_pu_dist: float
    local_coh_pw_dist: float
    local_coh_pacc_dist: float
    jaccard_adj_sentences: float


class SequentialCohesionExtractor(FeatureExtractor):
    def __init__(self, nlp: spacy.Language = None):
        if nlp is None:
            nlp = spacy.load('pt_core_news_md')

        self._nlp = nlp

    def extract(self, text: str, **kwargs) -> SequentialCohesionFeatures:
        del kwargs

        doc = self._nlp(text)
        local_coherence = self._compute_local_coherence(doc)
        jaccard_adj_sentences = self._compute_jaccard_adj_sentences(doc)

        return SequentialCohesionFeatures(
            local_coh_pu=local_coherence[0],
            local_coh_pw=local_coherence[1],
            local_coh_pacc=local_coherence[2],
            local_coh_pu_dist=local_coherence[3],
            local_coh_pw_dist=local_coherence[4],
            local_coh_pacc_dist=local_coherence[5],
            jaccard_adj_sentences=jaccard_adj_sentences)

    def _compute_local_coherence(self, doc: Doc) -> list:
        """Método que computa as medidas de coerência local
        baseada no método de grade de entidades.
        """
        sentences = [sent for sent in doc.sents]
        local_coherence = [0, 0, 0, 0, 0, 0]
        if len(sentences) >= 2:
            try:
                egrid = EntityGrid(doc)
                local_coherence = get_local_coherence(egrid)
            except ZeroDivisionError:
                local_coherence = [0, 0, 0, 0, 0, 0]

        return local_coherence

    def _compute_jaccard_adj_sentences(self, doc: Doc) -> float:
        """Método que computa a sobreposição de unigramas usando
        a medida de Jaccard entre frases adjacentes.
        """
        sentences = [sent for sent in doc.sents]
        total_sentences = len(sentences)

        if total_sentences <= 1:
            return 0.0
        all_lemmatized_tokens = []

        for sentence in sentences:
            tokens_sentences = [t.lemma_.lower()
                                for t in sentence
                                if not t.is_stop and t.pos_ != 'PUNCT' and
                                t.pos_ != 'SYM' and t.text != '\n']
            all_lemmatized_tokens.append(tokens_sentences)

        mean_jaccard = 0.0
        for i in range(total_sentences - 1):
            set_i = set(all_lemmatized_tokens[i])
            set_next = set(all_lemmatized_tokens[i + 1])
            set_union = set_i.union(set_next)
            set_intersection = set_i.intersection(set_next)
            mean_jaccard += len(set_intersection) / \
                len(set_union) if len(set_union) > 0 else 0

        mean_jaccard /= total_sentences
        return mean_jaccard
