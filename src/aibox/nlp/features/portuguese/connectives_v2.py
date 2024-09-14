"""Esse módulo contém características
relacionadas com o uso de conectivos.
"""

from __future__ import annotations

from dataclasses import dataclass

import spacy

from aibox.nlp.core import FeatureExtractor
from aibox.nlp.features.utils import DataclassFeatureSet, patterns, sentencizers


@dataclass(frozen=True)
class ConnectivesFeaturesV2(DataclassFeatureSet):
    additive_neg_ratio: float
    additive_pos_ratio: float
    cause_neg_ratio: float
    cause_pos_ratio: float
    log_neg_ratio: float
    log_pos_ratio: float
    and_ratio: float
    if_ratio: float
    logic_operators_ratio: float
    negation_ratio: float
    or_ratio: float
    all_conn_ratio: float


class ConnectivesExtractorV2(FeatureExtractor):
    def __init__(self, nlp: spacy.Language | None = None):
        self._connectives = {
            "additive_neg_ratio": ["entretanto", "mas", "porem", "antes", "todavia"],
            "additive_pos_ratio": ["e", "como", "bem como", "alem disso"],
            "cause_neg_ratio": [
                "mesmo",
                "embora",
                "contudo",
                "no entanto",
                "apesar de",
                "apesar disso",
                "apesar disto",
                "a menos que",
            ],
            "cause_pos_ratio": ["habilita", "para", "se", "somente se", "assim"],
            "log_neg_ratio": ["pelo contrario", "ainda", "cada vez que", "embora"],
            "log_pos_ratio": [
                "similarmente",
                "por outro lado",
                "de novo",
                "somente se",
                "assim",
                "para este fim",
                "desde que",
            ],
            "and_ratio": ["e"],
            "if_ratio": ["se"],
            "logic_operators_ratio": ["e", "ou", "nao", "nenhum", "nenhuma", "se"],
            "negation_ratio": ["nao", "nem", "nunca", "jamais", "tampouco"],
            "or_ratio": ["ou"],
        }
        self._nlp = nlp

    def extract(self, text: str, **kwargs) -> ConnectivesFeaturesV2:
        del kwargs

        sentences = sentencizers.spacy_sentencizer(text, self._nlp)
        connectives_metrics = {k: 0.0 for k in self._connectives}
        connectives_metrics["all_conn_ratio"] = 0.0
        overall_hits = 0
        total_words = 0

        if len(sentences) > 0:
            for feature, connectives in self._connectives.items():
                hits, total_words = patterns.count_connectives_in_sentences(
                    connectives, sentences
                )

                if total_words > 0:
                    connectives_metrics[feature] = hits / total_words
                    overall_hits += hits

            if total_words > 0:
                overall_frequency = overall_hits / total_words
                connectives_metrics["all_conn_ratio"] = overall_frequency

        return ConnectivesFeaturesV2(**connectives_metrics)
