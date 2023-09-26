"""Esse módulo contém características
relacionadas com regência verbal
e nominal.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import spacy

from nlpbox import resources
from nlpbox.core import FeatureExtractor

from .utils import DataclassFeatureSet


@dataclass(frozen=True)
class RegencyFeatures(DataclassFeatureSet):
    verb_regency_score: float
    nominal_regency_score: float


class RegencyExtractor(FeatureExtractor):
    def __init__(self,
                 nlp: spacy.Language = None):
        if nlp is None:
            nlp = spacy.load('pt_core_news_md')

        self._nlp = nlp

        root_dir = resources.path('dictionary/morph-checker.v1')
        verbs_path = root_dir.joinpath('verb_pattern.txt')
        regencies_path = root_dir.joinpath('vregence_dict.json')

        with verbs_path.open() as f1, regencies_path.open() as f2:
            self._verbs = set(map(lambda line: str(line).replace('\n', ''),
                                  f1.readlines()))
            self._verb_regencies = {key: set(value)
                                    for key, value in json.load(f2).items()}

        root_dir = resources.path('dictionary/nominal-regency.v1')
        regencies_path = root_dir.joinpath('nominal_regency_dict.json')

        with regencies_path.open() as f:
            self._name_regencies = {key: set(value)
                                    for key, value in json.load(f).items()}
            self._names = set(self._name_regencies.keys())

    def extract(self, text: str) -> RegencyFeatures:
        doc = self._nlp(text)
        score_verb = self._score(*self._check_regency(doc,
                                                      self._verbs,
                                                      self._verb_regencies))
        score_nominal = self._score(*self._check_regency(doc,
                                                         self._names,
                                                         self._name_regencies))
        return RegencyFeatures(verb_regency_score=score_verb,
                               nominal_regency_score=score_nominal)

    @staticmethod
    def _score(hits, errors) -> float:
        total = hits + errors
        return hits / total if total else 1.0

    @staticmethod
    def _check_regency(doc: spacy.tokens.Doc,
                       word_set,
                       regencies) -> tuple[int, int]:
        errors = 0
        matches = 0

        for token in doc[:-1]:
            # Check whether the token is a verb and has regency
            if not RegencyExtractor._has_regency(token, word_set):
                # if not, continue
                continue

            # Convert both lemma and next token to lower case
            lemma = token.lemma_.lower()
            next_token = doc[token.i + 1].text.lower()

            if next_token in regencies[lemma]:
                # If the following token is in the list of possible
                #   pronouns, it is a hit.
                matches += 1
            else:
                # Otherwise, it's an error/miss
                errors += 1

        return errors, matches

    @staticmethod
    def _has_regency(t: spacy.tokens.Token, target) -> bool:
        return t.lemma_.lower() in target
