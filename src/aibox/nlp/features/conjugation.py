"""Esse módulo contém características
relacionadas com conjugação verbal.
"""
from __future__ import annotations

import enum
import json
from dataclasses import dataclass
from typing import Set, Tuple

import spacy
from spacy import matcher

from aibox.nlp import resources
from aibox.nlp.core import FeatureExtractor

from .utils import DataclassFeatureSet


@dataclass(frozen=True)
class ConjugationFeatures(DataclassFeatureSet):
    conjugation_score: float
    conjugation_first_ratio: float
    conjugation_second_ratio: float
    conjugation_third_ratio: float
    conjugation_irregular_ratio: float


class Conjugation(enum.Enum):
    """Classe utilitária, armazena as 3 conjugações
    verbais da língua portuguesa.
    """
    FIRST = 'ar'
    SECOND = 'er'
    THIRD = 'ir'

    @staticmethod
    def from_verb(regular_verb_root: str) -> Conjugation:
        """ Obtém a conjugação de um verbo regular a partir
        de sua raiz (infinitivo).

        Args:
            regular_verb_root (str): verbo em sua forma
                normal (e.g., cantar, ler).

        Returns:
            Conjugação desse verbo regular (1ª, 2ª ou 3ª).
        """
        if regular_verb_root.lower() == 'pôr':
            # Caso específico na língua portuguesa
            return Conjugation.SECOND

        # Supondo que seja a forma no infinitivo, a terminação
        #   é dada pela duas últimas letras
        ending = regular_verb_root[-2:].lower()

        return Conjugation.from_desinencia(ending)

    @staticmethod
    def from_desinencia(desinencia: str) -> Conjugation:
        """ Obtém a conjugação de acordo com a desinência
        do verbo.

        Args:
            desinencia (str): desinência do verbo (e.g., ER, AR, IR).

        Returns:
            Conjugation: Conjugação para essa desinência (1ª, 2ª ou 3ª).
        """
        try:
            return Conjugation(desinencia)
        except ValueError:
            return None


class ConjugationExtractor(FeatureExtractor):
    def __init__(self, nlp: spacy.language.Language | None = None):

        if nlp is None:
            nlp = spacy.load("pt_core_news_md")

        self._nlp = nlp
        self._matcher = matcher.Matcher(self._nlp.vocab)
        self._matcher.add("verb", [[{"POS": "VERB"}]])

        root_dir = resources.path('dictionary/verb-conjugation.v1')
        irregular_path = root_dir.joinpath('conjugation_irregular.json')
        paradigms_path = root_dir.joinpath('conjugation_desinencias.json')

        with irregular_path.open('r') as f1, paradigms_path.open('r') as f2:
            self._paradigms = {key: set(value)
                               for key, value in json.load(f2).items()}
            self._irregular = set(json.load(f1))

    def extract(self, text: str, **kwargs) -> ConjugationFeatures:
        del kwargs

        errors, hits, frequency = self._check(text)
        total_verbs = hits + errors
        score = 0.0
        ratios = {f'conjugation_{k}_ratio': float(v)
                  for k, v in frequency.items()}

        if total_verbs > 0:
            score = hits / total_verbs
            ratios = {k: v / total_verbs for k, v in ratios.items()}

        return ConjugationFeatures(conjugation_score=score,
                                   **ratios)

    def _check(self, text: str) -> Tuple[int, int, dict[str, int]]:
        errors = 0
        hits = 0
        frequency = {k.name.lower(): 0 for k in Conjugation}
        frequency.update({'irregular': 0})

        doc = self._nlp(text)

        for span in self._matcher(doc, as_spans=True):
            token = span[0]
            text = token.text.lower()

            if text in self._irregular:
                # Caso o texto esteja na lista de verbos
                #   irregulares conhecidos, temos um acerto
                hits += 1

                # E aumentamos a frequência de verbos
                #   irregulares.
                frequency['irregular'] += 1
                continue

            lemma = token.lemma_.lower()
            conjugation = Conjugation.from_verb(lemma)

            if not conjugation:
                # Caso não tenha sido encontrada
                #   o paradigma de conjugação, consideramos
                #   que tenha ocorrido um erro.
                errors += 1
                continue

            # Incrementamos a frequência de verbos
            #   dessa conjugação.
            frequency[conjugation.name.lower()] += 1

            # As desinências dos verbos paradigmas podem possuir
            #   de 2 a 7 caracteres.
            desinencia = set([text[-2:], text[-3:], text[-4:],
                             text[-5:], text[-6:], text[-7:]])

            # Obtemos a lista de desinências para essa conjugação
            desinencias = self._get_desinencias(conjugation)

            # Caso os conjunto sejam disjuntos, temos um erro
            if desinencia.isdisjoint(desinencias):
                errors += 1
            else:
                hits += 1

        return errors, hits, frequency

    def _get_desinencias(self, conjugation: Conjugation) -> Set[str]:
        return self._paradigms[conjugation.name]
