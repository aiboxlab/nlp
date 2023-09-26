"""Esse módulo contém características
de complexidade sintática.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import spacy
from spacy.tokens.doc import Doc

from nlpbox.core import FeatureExtractor

from .utils import DataclassFeatureSet
from .utils.clauses_parser import extract_clauses_by_verbs


@dataclass(frozen=True)
class SyntacticComplexityFeatures(DataclassFeatureSet):
    adverbs_before_main_verb_ratio: float
    infinite_subordinate_clauses: float
    words_before_main_verb: float
    clauses_per_sentence: float
    coordinate_conjunctions_per_clauses: float
    passive_ratio: float
    coord_conj_ratio: float
    subord_conj_ratio: float
    sentences_with_1_clauses: float
    sentences_with_2_clauses: float
    sentences_with_3_clauses: float
    sentences_with_4_clauses: float
    sentences_with_5_clauses: float
    sentences_with_6_clauses: float
    sentences_with_7_clauses: float
    std_noun_phrase: float


class SyntacticComplexityExtractor(FeatureExtractor):
    def __init__(self, nlp: spacy.Language = None):
        if nlp is None:
            nlp = spacy.load('pt_core_news_md')

        self._nlp = nlp

    def extract(self, text: str, **kwargs) -> dict:
        del kwargs

        doc = self._nlp(text)
        adverbs_before_main_verb_ratio = self._adverbs_before_main_verb_ratio(
            doc)
        infinite_subordinate_clauses = self._infinite_subordinate_clauses(
            doc)
        words_before_main_verb = self._words_before_main_verb(doc)
        clauses_per_sentence = self._clauses_per_sentence(doc)
        coord_conj_per_clauses = self._coord_conj_per_clauses(
            doc)
        passive_ratio = self._passive_ratio(doc)
        coord_conj_ratio, subord_conj_ratio = self._coord_subord_conj_ratio(
            doc)
        relative_clauses = self._relative_clauses(doc)
        sentences_with_n_clauses = self._sentences_with_n_clauses(doc)
        std_noun_phrase = self._std_noun_phrase(doc)

        return SyntacticComplexityFeatures(
            adverbs_before_main_verb_ratio=adverbs_before_main_verb_ratio,
            infinite_subordinate_clauses=infinite_subordinate_clauses,
            words_before_main_verb=words_before_main_verb,
            clauses_per_sentence=clauses_per_sentence,
            coordinate_conjunctions_per_clauses=coord_conj_per_clauses,
            passive_ratio=passive_ratio,
            coord_conj_ratio=coord_conj_ratio,
            subord_conj_ratio=subord_conj_ratio,
            std_noun_phrase=std_noun_phrase,
            **sentences_with_n_clauses)

    def _adverbs_before_main_verb_ratio(self, doc: Doc) -> float:
        """Método que computa a Proporção de orações
        com advérbio antes do verbo principal em relação à quantidade
        de orações do texto.
        """
        sentences = [sent for sent in doc.sents]
        if len(sentences) == 0:
            return 0

        all_clauses = []
        all_adverbs = []
        all_verbs = []
        for sentence in sentences:
            adverbs_sent = [adv for adv in sentence
                            if adv.pos_ == 'ADV']
            all_adverbs.extend(adverbs_sent)
            verbs_sent = [verb for verb in sentence
                          if verb.pos_ == 'VERB' and
                          verb.dep_ != 'xcomp']
            all_verbs.extend(verbs_sent)
            clauses_sent = extract_clauses_by_verbs(sentence)
            if clauses_sent is not None:
                all_clauses.extend(clauses_sent)
        counter_adverbs = 0

        for i in range(0, len(all_clauses)):
            for j in range(0, len(all_verbs)):
                verb_index = all_clauses[i].find(str(all_verbs[j]))
                if verb_index != -1:
                    for k in range(0, len(all_adverbs)):
                        if all_clauses[i].find(str(all_adverbs[k]),
                                               0,
                                               verb_index) != -1:
                            counter_adverbs += 1
                            break

        return counter_adverbs / len(all_clauses)

    def _infinite_subordinate_clauses(self, doc: Doc) -> float:
        """Método que computa a proporção de orações
        subordinadas reduzidas em relação à quantidade de
        orações do texto.
        """
        infinite_forms = [['Ger'], ['Inf'], ['Part']]
        verbs_tags = ['VERB', 'AUX']
        verbs = [token.text for token in doc if token.pos_ in verbs_tags]
        verbs_infinite = [token.text for token in doc
                          if token.pos_ == 'VERB' and
                          token.morph.get('VerbForm') in infinite_forms]

        return len(verbs_infinite) / len(verbs) if len(verbs) != 0 else 0

    def _words_before_main_verb(self, doc: Doc) -> float:
        """Método que computa a quantidade média de
        palavras antes dos verbos principais das
        orações principais das sentenças.
        """
        number_of_words = 0

        for token in doc:
            if not token.is_punct:
                if token.dep_ != 'ROOT':
                    number_of_words += 1
                else:
                    break

        return number_of_words

    def _clauses_per_sentence(self, doc: Doc) -> float:
        """Método que computa a média de
        orações por sentença.
        """
        assert doc is not None, 'Error DOC is None'
        sentences = [sent for sent in doc.sents]
        if len(sentences) == 0:
            return 0
        total_clauses = 0
        for sentence in sentences:
            clauses_sent = extract_clauses_by_verbs(sentence)
            if clauses_sent is not None:
                total_clauses += len(clauses_sent)
        return total_clauses / len(sentences)

    def _coord_conj_per_clauses(self, doc: Doc) -> float:
        """Método que computa a proporção de
        conjunções coordenativas em relação ao
        total de orações do texto.
        """
        sentences = [sent for sent in doc.sents]

        if len(sentences) == 0:
            return 0

        tag_conj = ['CCONJ']
        all_conjunctions = []
        all_clauses = []

        for sentence in sentences:
            conjunctions_sent = [
                token for token in sentence if token.pos_ in tag_conj]
            all_conjunctions.extend(conjunctions_sent)
            clauses_sent = extract_clauses_by_verbs(sentence)
            if clauses_sent is not None:
                all_clauses.extend(clauses_sent)

        ratio = 0.0
        n = len(all_clauses)
        if n > 0:
            ratio = len(all_conjunctions) / n

        return ratio

    def _passive_ratio(self, doc: Doc) -> float:
        """Método que computa a proporção de
        orações na voz passiva analítica em
        relação à quantidade de orações do texto.
        """
        sentences = [sent for sent in doc.sents]
        if len(sentences) == 0:
            return 0

        passives_verbs = [token.text for token in doc
                          if token.morph.get('Voice') == ['Pass']]
        all_clauses = []
        for sentence in sentences:
            clauses_sent = extract_clauses_by_verbs(sentence)
            if clauses_sent is not None:
                all_clauses.extend(clauses_sent)

        ratio = 0.0
        n = len(all_clauses)
        if n > 0:
            ratio = len(passives_verbs) / n

        return ratio

    def _coord_subord_conj_ratio(self, doc: Doc) -> tuple:
        """Método que computa a proporção de conjunções
        coordenativas e subordinativas em relação ao total de
        conjunções do texto.
        """
        tag_cconj = ['CCONJ']
        tag_sconj = ['SCONJ']
        cconj = []
        sconj = []

        for token in doc:
            if not token.is_punct:
                if token.pos_ in tag_cconj:
                    cconj.append(token.text)
                elif token.pos_ in tag_sconj:
                    sconj.append(token.text)

        total_conj = len(cconj) + len(sconj)
        if total_conj == 0:
            return 0, 0

        return len(cconj) / total_conj, len(sconj) / total_conj

    def _relative_clauses(self, doc: Doc) -> float:
        """Método que computa a proporção de orações
        relativas em relação ao total de orações do texto.
        """
        sentences = [sent for sent in doc.sents]
        if len(sentences) == 0:
            return 0
        relative_clauses = [token.text for token in doc
                            if token.morph.get('PronType') == ['Rel']]
        all_clauses = []
        for sentence in sentences:
            clauses_sent = extract_clauses_by_verbs(sentence)
            if clauses_sent is not None:
                all_clauses.extend(clauses_sent)

        ratio = 0.0
        n = len(all_clauses)
        if n > 0:
            ratio = len(relative_clauses) / n

        return ratio

    def _sentences_with_n_clauses(self, doc: Doc) -> dict:
        """Método que computa a Proporção de sentenças que
        contenham n (1, 2, 3, 4, 5, 6 e 7 ou mais) orações.
        """
        sentences = [sent for sent in doc.sents]
        sentences_with_n_clauses = {
            f'sentences_with_{i}_clauses': 0
            for i in range(1, 8)
        }

        if len(sentences) == 0:
            return sentences_with_n_clauses

        all_clauses = []
        for sentence in sentences:
            clauses_sent = extract_clauses_by_verbs(sentence)
            if clauses_sent is not None:
                all_clauses.append(len(clauses_sent))

        for i in range(1, 8):
            key = f'sentences_with_{i}_clauses'

            if i >= 7:
                def cond(c): return c >= i
            else:
                def cond(c): return c == i

            n = len(tuple(filter(cond, all_clauses)))
            sentences_with_n_clauses[key] = n

        return sentences_with_n_clauses

    def _std_noun_phrase(self, doc: Doc) -> float:
        """Método que computa o desvio padrão do
        tamanho dos sintagmas nominais do texto.
        """
        noun_phases = [noun_phase for noun_phase in doc.noun_chunks]

        if len(noun_phases) == 0:
            return 0

        noun_phases_sizes = []

        for noun_phase in noun_phases:
            tokens = [t for t in noun_phase]
            noun_phases_sizes.append(len(tokens))

        return np.std(noun_phases_sizes)
