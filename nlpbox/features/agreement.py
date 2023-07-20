"""Esse módulo contém características
relacionadas com concordância verbal
e nominal.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import cogroo4py.cogroo
import language_tool_python as langtool
import spacy

from nlpbox.core import FeatureExtractor

from .utils import DataclassFeatureSet


@dataclass(frozen=True)
class AgreementFeatures(DataclassFeatureSet):
    verb_agreement_score: float
    nominal_agreement_score: float


class AgreementExtractor(FeatureExtractor):
    def __init__(self,
                 nlp: spacy.Language = None,
                 cogroo: cogroo4py.cogroo.Cogroo = None,
                 tool: langtool.LanguageTool = None):
        if nlp is None:
            nlp = spacy.load('pt_core_news_md')

        if cogroo is None:
            cogroo = cogroo4py.cogroo.Cogroo()

        if tool is None:
            tool = langtool.LanguageTool("pt-BR")

        self._nlp = nlp
        self._cogroo = cogroo
        self._tool = tool
        self._va_matcher = spacy.matcher.Matcher(self._nlp.vocab)
        self._va_matcher.add("verb",
                             [[{"POS": {"IN": ["PRON", "NOUN", "PROPN"]}},
                               {"POS": "VERB"}]])
        self._va_clauses = ['Number', 'Person']
        self._cogroo_rules = {'xml:17',
                              'xml:21',
                              'xml:25',
                              'xml:38',
                              'xml:40',
                              'xml:92',
                              'xml:95',
                              'xml:103',
                              'xml:104',
                              'xml:105',
                              'xml:114',
                              'xml:115',
                              'xml:124'}
        self._langtool_rules = {'TODOS_NUMBER_AGREEMENT',
                                'CUJA_CUJO_MASCULINO_FEMININO',
                                'GENERAL_NUMBER_AGREEMENT_ERRORS'}

    def extract(self, text: str) -> AgreementFeatures:
        va_score = 0.0
        na_score = 0.0

        # Calculando nota de concordância verbal
        h, e = self._va_check(text)
        t = (h + e)

        if t > 0:
            va_score = h / t

        # Calculando nota de concordância nominal
        n_erros, n_rules = self._na_check(text)
        na_score = 1.0 - (n_erros / n_rules)

        return AgreementFeatures(verb_agreement_score=va_score,
                                 nominal_agreement_score=na_score)

    def _va_check(self, text: str) -> tuple[int, int]:
        doc = self._nlp(text)
        errors = 0
        matches = 0

        for span in self._va_matcher(doc, as_spans=True):
            subj = span[0].morph.to_dict()
            verb = span[1].morph.to_dict()

            # Verb agreement must occur in person and number
            for k in self._va_clauses:
                if k not in subj or k not in verb:
                    continue

                if subj[k] != verb[k]:
                    # If subject and verb doesn't match, increment error count
                    errors += 1
                else:
                    # Otherwise, increment hits/matches count
                    matches += 1

        return errors, matches

    def _na_check(self, text: str) -> tuple[int, int]:
        def _get_n_mistakes(fn, rules):
            all_mistakes = fn()
            mistakes = filter(lambda m: m in rules,
                              map(lambda m: m.ruleId,
                                  all_mistakes))
            mistakes = set(mistakes)
            return len(mistakes)

        n_cogroo_mistakes = _get_n_mistakes(
            lambda: self._cogroo.grammar_check(text).mistakes,
            self._cogroo_rules)
        n_langtool_mistakes = _get_n_mistakes(
            lambda: self._tool.check(text),
            self._langtool_rules)

        total_mistakes = n_langtool_mistakes + n_cogroo_mistakes
        total_rules = len(self._cogroo_rules) + len(self._langtool_rules)

        return total_mistakes, total_rules
