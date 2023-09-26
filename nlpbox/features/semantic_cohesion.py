"""Esse módulo contém características
relacionadas a coesão semântica
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass

import kenlm
import numpy as np
import spacy
from gensim.matutils import cossim, full2sparse, sparse2full
from scipy.linalg import pinv

from nlpbox import resources
from nlpbox.core import FeatureExtractor

from .utils import DataclassFeatureSet


@dataclass(frozen=True)
class SemanticFeatures(DataclassFeatureSet):
    lsa_adj_mean: float
    lsa_adj_std: float
    lsa_all_mean: float
    lsa_all_std: float
    lsa_givenness_mean: float
    lsa_givenness_std: float
    lsa_paragraph_mean: float
    lsa_paragraph_std: float
    lsa_span_mean: float
    lsa_span_std: float
    cross_entropy: float


class SemanticExtractor(FeatureExtractor):
    def __init__(self, nlp: spacy.Language | None = None):
        if nlp is None:
            nlp = spacy.load('pt_core_news_md')

        root_dir = resources.path('feature_models/semantic_cohesion.v1')
        lsa_model_path = root_dir.joinpath('brwac_full_lsa_word_dict.pkl')
        kenlm_model_path = root_dir.joinpath('corpus_3gram.binary')

        with lsa_model_path.open('rb') as lsa_file:
            self._lsa_model = pickle.load(lsa_file)

        self._num_topics = 300
        self._kenlm_model = kenlm.Model(str(kenlm_model_path))
        self._nlp = nlp

    def extract(self, text: str, **kwargs) -> SemanticFeatures:
        del kwargs

        doc = self._nlp(text)
        sentences = [sent for sent in doc.sents]
        sentences_tokens = []
        for sentence in sentences:
            tokens = [token.text.lower()
                      for token in sentence if not token.is_punct]
            sentences_tokens.append(tokens)
        lsa_adj_sentences = self._compute_lsa_adj_sentences(sentences_tokens)
        lsa_all_sentences = self._compute_lsa_all_sentences(sentences_tokens)
        lsa_givenness = self._compute_lsa_givenness(sentences_tokens)
        lsa_paragraph = self._compute_lsa_paragraphs(text)
        lsa_span = self._compute_lsa_span(sentences_tokens)

        features = {
            'cross_entropy': self._compute_cross_entropy(doc)
        }

        features.update(lsa_adj_sentences)
        features.update(lsa_all_sentences)
        features.update(lsa_givenness)
        features.update(lsa_paragraph)
        features.update(lsa_span)

        return SemanticFeatures(**features)

    def _compute_lsa_adj_sentences(self,
                                   sentences_tokens: list[list[str]]) -> dict[str,
                                                                              float]:
        """ Computa similaridade entre sentenças adjacentes no texto

        Args:
            sentences_tokens: lista com todas os tokens de cada frase

        Returns:
            Dicionário com a média e o desvio padrão
                das similaridades de sentenças adjacentes
        """
        features = {
            'lsa_adj_mean': 0,
            'lsa_adj_std': 0
        }
        if len(sentences_tokens) <= 1:
            return features

        similarities = []
        for i in range(len(sentences_tokens) - 1):
            similarity = SemanticExtractor.compute_similarity(
                sentences_tokens[i], sentences_tokens[i + 1], self._lsa_model)
            similarities.append(similarity)

        features['lsa_adj_mean'] = np.mean(similarities)
        features['lsa_adj_std'] = np.std(similarities)
        return features

    def _compute_lsa_all_sentences(self,
                                   sentences_tokens: list[list[str]]) -> dict[str,
                                                                              float]:
        """ Computa similaridade entre todos os
        pares de sentença possíveis no texto.

        Args:
            sentences_tokens: lista com todas os tokens de cada frase

        Returns:
            Dicionário com a média e o desvio padrão das
                similaridades entre os pares de sentenças do texto
        """
        features = {
            'lsa_all_mean': 0,
            'lsa_all_std': 0
        }
        if len(sentences_tokens) <= 1:
            return features

        similarities = []
        for i in range(len(sentences_tokens)):
            for j in range(i + 1, len(sentences_tokens)):
                similarity = SemanticExtractor.compute_similarity(
                    sentences_tokens[i], sentences_tokens[j], self._lsa_model)
                similarities.append(similarity)

        features['lsa_all_mean'] = np.mean(similarities)
        features['lsa_all_std'] = np.std(similarities)
        return features

    def _compute_lsa_givenness(self,
                               sentences_tokens: list[list[str]]) -> dict[str,
                                                                          float]:
        """ Computa a média e o desvio padrão do givenness do texto.
        Givenness é a similaridade entre
        uma sentença e todo texto que a precede

        Args:
            sentences_tokens: lista com todas os tokens de cada frase

        Returns:
            Dicionário com a média e o desvio padrão do
                givenness de todas as sentenças do texto
        """
        features = {
            'lsa_givenness_mean': 0,
            'lsa_givenness_std': 0
        }
        if len(sentences_tokens) <= 1:
            return features

        similarities = []
        for i in range(1, len(sentences_tokens)):
            previous_sentences_tokens = []

            for j in range(i - 1, -1, -1):
                previous_sentences_tokens.extend(sentences_tokens[j])

            similarity = SemanticExtractor.compute_similarity(sentences_tokens[i],
                                                              previous_sentences_tokens,
                                                              self._lsa_model)
            similarities.append(similarity)

        features['lsa_givenness_mean'] = np.mean(similarities)
        features['lsa_givenness_std'] = np.std(similarities)
        return features

    def _compute_lsa_paragraphs(self, text: str) -> dict[str, float]:
        """ Computa similaridade entre paragrafos adjacentes no texto

        Args:
            text (str): texto com mais de um paragrafo

        Returns:
            Dicionário com a média e o desvio padrão das
                similaridades de paragrafos adjacentes
        """
        features = {
            'lsa_paragraph_mean': 0,
            'lsa_paragraph_std': 0
        }
        paragraphs = text.split('\n')
        if len(paragraphs) <= 1:
            return features

        list_tokens_paragraphs = []
        for i in range(0, len(paragraphs)):
            tokens_paragraph = SemanticExtractor.withdraw_characters(
                paragraphs[i]).split()
            list_tokens_paragraphs.append(tokens_paragraph)

        similarities = []
        for i in range(len(paragraphs) - 1):
            similarity = SemanticExtractor.compute_similarity(list_tokens_paragraphs[i],
                                                              list_tokens_paragraphs[i + 1],
                                                              self._lsa_model)
            similarities.append(similarity)

        features['lsa_paragraph_mean'] = np.mean(similarities)
        features['lsa_paragraph_std'] = np.std(similarities)
        return features

    def _compute_lsa_span(self,
                          sentences_tokens: list[list[str]]) -> dict[str,
                                                                     float]:
        """ Computa média e o desvio padrão do span de
        cada sentença do texto a partir da segunda.

        O método utiliza k sentenças anteriores como
        base para um sub-espaço vetorial, decompondo a
        sentença atual em duas componentes: uma pertencente
        ao sub-espaço das sentenças anteriores (informação dada)
        e outra perpendicular (informação nova).

        Args:
            sentences_tokens: lista com todas os tokens de cada frase

        Returns:
            Dicionário com a média e o desvio padrão do span de cada sentença
        """
        features = {
            'lsa_span_mean': 0,
            'lsa_span_std': 0
        }

        if len(sentences_tokens) <= 1:
            return features

        spans = np.zeros(len(sentences_tokens) - 1)

        for i in range(1, len(sentences_tokens)):
            past_sentences = sentences_tokens[:i]
            span_dimensions = len(past_sentences)
            if span_dimensions > self._num_topics - 1:
                beginning = past_sentences[0:span_dimensions -
                                           self._num_topics]
                past_sentences[0] = beginning

            # Computa vetor das sentenças anteriores
            past_sentences_vectors = [
                sparse2full(SemanticExtractor.compute_doc2vec(
                    sentence, self._lsa_model), self._num_topics)
                for sentence in past_sentences
            ]

            # Computa vetor da sentença atual
            sentence_i_vec = SemanticExtractor.compute_doc2vec(
                sentences_tokens[i], self._lsa_model)
            current_sentence_vector = sparse2full(
                sentence_i_vec, self._num_topics)
            current_sentence_array = np.array(
                current_sentence_vector).reshape(self._num_topics, 1)
            past_sentences_vectors_trans = np.array(past_sentences_vectors).T

            projection_matrix = np.dot(
                np.dot(
                    past_sentences_vectors_trans,
                    pinv(np.dot(past_sentences_vectors_trans.T,
                         past_sentences_vectors_trans))
                ),
                past_sentences_vectors_trans.T
            )

            projection = np.dot(projection_matrix,
                                current_sentence_array).ravel()
            spans[i - 1] = cossim(
                full2sparse(current_sentence_vector),
                full2sparse(projection)
            )

        features['lsa_span_mean'] = np.mean(spans)
        features['lsa_span_std'] = np.std(spans)
        return features

    def _compute_cross_entropy(self, doc: spacy.Doc) -> float:
        """Computa a entropia cruzada do texto usando um modelo 3-gram

        Args:
            doc (spacy.Doc): Objeto Doc contendo o texto

        Returns:
            Valor da entropia cruzada do texto
        """
        sentences = [sent.text.lower() for sent in doc.sents]
        scores = [-1 / len(sentence) * self._kenlm_model.score(sentence)
                  for sentence in sentences]
        return sum(scores) / len(scores) if scores else 0

    @staticmethod
    def withdraw_characters(s: str) -> str:
        """Remove caracteres especiais de uma string

        Args:
            s (str): string original

        Returns: nova string sem os caracteres especiais
        """
        chars = '.,!()/_:;[]{}%@?"'
        res = s.translate(str.maketrans('', '', chars))
        return res.lower()

    @staticmethod
    def compute_word2vec(word: str,
                         model: dict[str, list[float]]) -> np.ndarray:
        """Computa embeddings de um token

        Args:
            word: token
            model: Modelo gerador de embeddings

        Returns: Vetor dos embeddings da respectiva palavra
        """
        lower = word.lower()
        if lower in model:
            return model[lower]

        return np.full(300, 0.001)

    @staticmethod
    def compute_doc2vec(tokens: list[str],
                        model: dict[str, list[float]]) -> list[tuple[int,
                                                                     float]]:
        """Computa o vetor de um documento (lista de tokens)

        Args:
            tokens: Lista de tokens
            model: Modelo gerador de embeddings

        Returns: Embeddings do documento
        """
        word_vec = []
        for word in tokens:
            if len(word) > 2:
                vec = SemanticExtractor.compute_word2vec(word, model)
                word_vec.append(vec)

        if len(word_vec) < 1:
            return []

        doc_vec = np.average(word_vec, axis=0)
        ids = list(range(0, 300))
        zdoc_vec = list(zip(ids, doc_vec))
        return zdoc_vec

    @staticmethod
    def compute_similarity(doc1: list[str], doc2: list[str],
                           model: dict[str, list[float]]) -> float:
        """Computa a simlaridade do cosseno entre dois documentos

        Args:
            doc1: Lista de tokens
            doc2: Lista de tokens
            model: Modelo gerador de embeddings

        Returns: Similaridade do cosseno entre
            os embeddings dos dois documentos
        """
        doc1_emb = SemanticExtractor.compute_doc2vec(doc1, model)
        doc2_emb = SemanticExtractor.compute_doc2vec(doc2, model)
        if doc1 is None or doc2 is None:
            return 0
        return cossim(doc1_emb, doc2_emb)
