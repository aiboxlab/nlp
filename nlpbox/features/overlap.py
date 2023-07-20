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
from TRUNAJOD.entity_grid import EntityGrid, get_local_coherence


@dataclass(frozen=True)
class OverlapFeatures:
    local_coh_pu: float
    local_coh_pw: float
    local_coh_pacc: float
    local_coh_pu_dist: float
    local_coh_pw_dist: float
    local_coh_pacc_dist: float
    overlap_unigrams_sents: float
    cosine_sim_tfids_sents: float

    @classmethod
    def extract(cls,
                text: str,
                doc: spacy.tokens.Doc | None = None,
                spacy_pipeline: spacy.Language | None = None,
                sentences: list | None = None,
                **kwargs) -> OverlapFeatures:

        if doc is None:
            if spacy_pipeline is None:
                spacy_pipeline = spacy.load("pt_core_news_md")

            doc = spacy_pipeline(text)

        if sentences is None:
            sentences = [s.text for s in doc.sents if len(s.text.strip()) > 0]

        local_coherence = [0 for _ in range(6)]

        if len(sentences) >= 2:
            try:
                egrid = EntityGrid(doc)
                local_coherence = get_local_coherence(egrid)
            except ZeroDivisionError:
                pass

        overlap_unigrams_sents = 0
        cosine_sim_tfids_sents = 0

        if len(sentences) > 1:
            overlap_unigrams_sents = cls._adjacent_sents_rouge(sentences)
            cosine_sim_tfids_sents = cls._adjacent_sents_cos_sim(sentences)

        return OverlapFeatures(local_coh_pu=local_coherence[0],
                               local_coh_pw=local_coherence[1],
                               local_coh_pacc=local_coherence[2],
                               local_coh_pu_dist=local_coherence[3],
                               local_coh_pw_dist=local_coherence[4],
                               local_coh_pacc_dist=local_coherence[5],
                               overlap_unigrams_sents=overlap_unigrams_sents,
                               cosine_sim_tfids_sents=cosine_sim_tfids_sents)

    @staticmethod
    def _adjacent_sents_rouge(sentences: list) -> float:
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
    def _adjacent_sents_cos_sim(sentences: list) -> float:
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
