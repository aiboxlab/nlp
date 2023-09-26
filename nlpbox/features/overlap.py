"""Esse módulo contém características
de sobreposição.
"""
from __future__ import annotations

from dataclasses import dataclass

import spacy
from rouge import Rouge
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from spacy.tokens import Doc

from nlpbox.core import FeatureExtractor

from .utils import DataclassFeatureSet, sentencizers


@dataclass(frozen=True)
class OverlapFeatures(DataclassFeatureSet):
    overlap_unigrams_sents: float
    cosine_sim_tfids_sents: float


class OverlapExtractor(FeatureExtractor):
    def __init__(self, nlp: spacy.Language = None):
        if nlp is None:
            nlp = spacy.load('pt_core_news_md')

        self._nlp = nlp

    def extract(self, text: str, **kwargs) -> OverlapFeatures:
        del kwargs

        sentences = sentencizers.spacy_sentencizer(text, self._nlp)
        sentences = list(map(lambda s: s.text, sentences))

        overlap_unigrams_sents = 0
        cosine_sim_tfids_sents = 0

        if len(sentences) > 1:
            overlap_unigrams_sents = self._adjacent_sents_rouge(sentences)
            cosine_sim_tfids_sents = self._adjacent_sents_cos_sim(sentences)

        return OverlapFeatures(overlap_unigrams_sents=overlap_unigrams_sents,
                               cosine_sim_tfids_sents=cosine_sim_tfids_sents)

    @staticmethod
    def _adjacent_sents_rouge(sentences: list[str]) -> float:
        """ Método que computa a sobreposição de unigramas usando a
        medida do ROUGE-1 entre frases adjacentes.

        Args:
            doc (Doc): Doc com o texto.
            sentences (list): sentenças que devem ser utilizadas.

        Returns:
            dict: dicionário com as características.
        """

        sentences_size = len(sentences)
        mean_r1 = 0.0

        if sentences_size > 0:
            if sentences_size >= 2:
                sentences_size -= 1

            evaluator = Rouge(metrics=['rouge-n', 'rouge-l'],
                              max_n=2,
                              limit_length=True,
                              length_limit=300,
                              length_limit_type='words',
                              apply_avg=True,
                              apply_best=False,
                              alpha=0.5,
                              weight_factor=1.2,
                              stemming=False)
            mean_r1 = 0

            try:
                for i in range(sentences_size):
                    rouge_scores = evaluator.get_scores(sentences[i],
                                                        sentences[i+1])
                    mean_r1 += rouge_scores['rouge-1']['r']
                mean_r1 /= sentences_size
            except ZeroDivisionError:
                pass

        return mean_r1

    @staticmethod
    def _adjacent_sents_cos_sim(sentences: list[str]) -> float:
        """Método que computa a similaridade do cosseno usando a representação
        TF-IDF entre frases adjacentes.

        Args:
            sentences (list): sentenças.

        Returns:
            float: similaridade cosseno através do TF-IDF.
        """

        sentences_size = len(sentences)
        mean_sim_tfidf = 0.0

        if sentences_size > 0:
            if sentences_size >= 2:
                sentences_size -= 1

            tfidf_vect = TfidfVectorizer()
            sents_vect_tfidf = tfidf_vect.fit_transform(sentences).toarray()
            mean_sim_tfidf = 0

            for i in range(sentences_size):
                v1 = sents_vect_tfidf[i]
                v2 = sents_vect_tfidf[i+1]
                mean_sim_tfidf += cosine_similarity([v1], [v2])[0][0]

            mean_sim_tfidf /= sentences_size

        return mean_sim_tfidf
