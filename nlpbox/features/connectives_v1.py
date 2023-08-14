"""Esse módulo contém características
relacionadas com o uso de conectivos.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import spacy

from nlpbox import resources
from nlpbox.core import FeatureExtractor

from .utils import DataclassFeatureSet, patterns, sentencizers


@dataclass(frozen=True)
class ConnectivesFeaturesV1(DataclassFeatureSet):
    sent_conn_adicao: float
    sent_conn_conclusao: float
    sent_conn_correcao: float
    sent_conn_restricao: float
    sent_conn_inclusao: float
    sent_conn_condicao: float
    sent_conn_resumo: float
    sent_conn_certeza: float
    sent_conn_justificativa: float
    sent_conn_comparacao: float
    sent_conn_conformidade: float
    sent_conn_concessao: float
    sent_conn_consequencia: float
    sent_conn_oposicao: float
    sent_conn_tempo: float
    sent_conn_exemplificacao: float
    sent_conn_prioridade: float
    sent_conn_comprovacao: float
    sent_conn_exclusao: float
    sent_conn_mediacao: float
    sent_conn_alternancia: float
    sent_conn_finalidade: float
    sent_conn_minimidade: float
    sent_conn_duvida: float
    sent_conn_proporcao: float
    sent_conn_marcacao: float
    sent_conn_explicacao: float
    sent_conn_reafirmacao: float
    sent_conn_opiniao: float
    sent_conn_esclarecimento: float
    sent_conn_deducao: float
    sent_conn_chamar_atencao: float
    sent_all_conn: float


class ConnectivesExtractorV1(FeatureExtractor):
    def __init__(self, nlp: spacy.Language | None = None):
        self._load_connectives()
        self._nlp = nlp

    def extract(self, text: str) -> ConnectivesFeaturesV1:
        sentences = sentencizers.spacy_sentencizer(text, self._nlp)

        overall_freq = 0.0
        connectives_scores = {k: 0.0 for k in self._connectives}
        connectives_scores['sent_all_conn'] = 0.0
        sentences_size = len(sentences)

        if sentences_size > 0:
            for feature, connectives in self._connectives.items():
                hits, _ = patterns.count_connectives_in_sentences(
                    connectives,
                    sentences)

                overall_freq += hits
                frequency = hits / sentences_size
                connectives_scores[feature] = frequency

            overall_freq /= sentences_size
            connectives_scores['sent_all_conn'] = overall_freq

        return ConnectivesFeaturesV1(**connectives_scores)

    def _load_connectives(self):
        def _class_to_feature_name(name: str) -> str:
            feature_name = name.lower()
            feature_name = feature_name.replace(' ', '_')
            feature_name = unidecode(feature_name)
            feature_name = f'sent_conn_{feature_name.strip()}'
            return feature_name

        resource_dir = resources.path('dictionary/connectives-list.v1')
        resource = resource_dir.joinpath('connectives.json')

        with resource.open('r', encoding='utf-8') as f:
            connectives = json.load(f)

        self._connectives = {
            _class_to_feature_name(k): v
            for k, v in connectives.items()
        }
