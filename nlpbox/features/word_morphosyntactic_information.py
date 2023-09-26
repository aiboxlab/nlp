"""Esse módulo contém características
relacionadas as informações morfosintáticas
das palavras
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import spacy
from spacy.tokens import Doc

from nlpbox.core import FeatureExtractor
from nlpbox.features.utils.clauses_parser import extract_clauses_by_verbs

from .utils import DataclassFeatureSet


@dataclass(frozen=True)
class WordMorphosyntacticInformationFeatures(DataclassFeatureSet):
    adjective_ratio: float
    adjectives_max: float
    adjectives_min: float
    adjectives_std: float
    adverbs: float
    adverbs_diversity_ratio: float
    adverbs_max: float
    adverbs_min: float
    adverbs_std: float
    nouns_ratio: float
    nouns_max: float
    nouns_min: float
    nouns_std: float
    verbs_ratio: float
    verbs_max: float
    verbs_min: float
    verbs_std: float
    infinitive_verbs: float
    inflected_verbs: float
    non_inflected_verbs: float
    indicative_condition_ratio: float
    indicative_future_ratio: float
    prepositions_per_clause: float
    prepositions_per_sentence: float
    content_words: float
    function_words: float
    punctuation_ratio: float
    ratio_function_to_content_words: float
    pronoun_ratio: float
    personal_pronouns: float
    oblique_pronouns_ratio: float
    indefinite_pronoun_ratio: float
    relative_pronouns_ratio: float
    first_person_pronouns: float
    second_person_pronouns: float
    third_person_pronouns: float
    pronouns_min: float
    pronouns_max: float
    pronouns_std: float


class WordMorphosyntacticInformationExtractor(FeatureExtractor):
    def __init__(self, nlp: spacy.Language = None):
        if nlp is None:
            nlp = spacy.load('pt_core_news_md')
        self._nlp = nlp

    def extract(self, text: str, **kwargs) -> WordMorphosyntacticInformationFeatures:
        del kwargs

        doc = self._nlp(text)
        features = {}

        features.update(self._compute_morphosyntactic_adjectives(doc))
        features.update(self._compute_morphosyntactic_adverbs(doc))
        features.update(self._compute_morphosyntactic_nouns(doc))
        features.update(self._compute_morphosyntactic_verbs(doc))
        features.update(self._compute_morphosyntactic_nouns(doc))
        features.update(self._compute_prepositions_per_clause_sentence(doc))
        features.update(self._compute_morphosyntactic_other_metrics(doc))
        features.update(self._compute_morphosyntactic_pronouns(doc))

        return WordMorphosyntacticInformationFeatures(**features)

    def _compute_morphosyntactic_adjectives(self, doc: Doc) -> dict[str,
                                                                    float]:
        sentences = [sent for sent in doc.sents]

        adjective_features = {
            'adjective_ratio': 0.0,
            'adjectives_max': 0.0,
            'adjectives_min': 0.0,
            'adjectives_std': 0.0
        }

        if len(sentences) == 0:
            return adjective_features

        all_tokens = [token.text for token in doc if not token.is_punct]
        all_adjectives = [token.text for token in doc if token.pos_ == 'ADJ']

        adjective_features['adjective_ratio'] = len(
            all_adjectives) / len(all_tokens) if len(all_tokens) != 0 else 0

        adjectives_ratio_sentences = []
        for sentence in sentences:
            tokens_sentence = [
                token.text for token in sentence if not token.is_punct]
            if len(tokens_sentence) > 0:
                adjectives_sentence = [
                    token.text for token in sentence if token.pos_ == 'ADJ']
                adjectives_ratio_sentences.append(
                    len(adjectives_sentence) / len(tokens_sentence))

        if len(adjectives_ratio_sentences) > 0:
            adjective_features['adjectives_max'] = max(
                adjectives_ratio_sentences)
            adjective_features['adjectives_min'] = min(
                adjectives_ratio_sentences)
            adjective_features['adjectives_std'] = np.std(
                adjectives_ratio_sentences)

        return adjective_features

    def _compute_morphosyntactic_adverbs(self, doc: Doc):
        sentences = [sent for sent in doc.sents]

        adverbs_features = {
            'adverbs': 0.0,
            'adverbs_diversity_ratio': 0.0,
            'adverbs_max': 0.0,
            'adverbs_min': 0.0,
            'adverbs_std': 0.0
        }

        if len(sentences) == 0:
            return adverbs_features

        all_tokens = [token.text.lower()
                      for token in doc if not token.is_punct]
        n_tokens = len(all_tokens)
        all_adverbs = [token.text.lower()
                       for token in doc if token.pos_ == 'ADV']
        n_adverbs = len(all_adverbs)

        if n_tokens > 0:
            adverbs_features['adverbs'] = n_adverbs / n_tokens

        if n_adverbs > 0:
            n = len(set(all_adverbs))
            adverbs_features['adverbs_diversity_ratio'] = n / n_adverbs

        adverbs_ratio_sentences = []
        for sentence in sentences:
            tokens_sentences = [token.text
                                for token in sentence
                                if not token.is_punct]
            adverbs_sentence = [token.text
                                for token in sentence
                                if token.pos_ == 'ADV']

            if len(tokens_sentences) > 0:
                adverbs_ratio_sentences.append(
                    len(adverbs_sentence) / len(tokens_sentences))

        if len(adverbs_ratio_sentences) > 0:
            adverbs_features['adverbs_max'] = max(adverbs_ratio_sentences)
            adverbs_features['adverbs_min'] = min(adverbs_ratio_sentences)
            adverbs_features['adverbs_std'] = np.std(adverbs_ratio_sentences)

        return adverbs_features

    def _compute_morphosyntactic_nouns(self, doc: Doc) -> dict[str, float]:
        sentences = [sent for sent in doc.sents]

        nouns_features = {
            'nouns_ratio': 0.0,
            'nouns_max': 0.0,
            'nouns_min': 0.0,
            'nouns_std': 0.0
        }

        if len(sentences) == 0:
            return nouns_features

        pos_tags = {'NOUN', 'PROPN'}

        all_tokens = [token.text.lower()
                      for token in doc if not token.is_punct]
        all_nouns = [token.text.lower()
                     for token in doc if token.pos_ in pos_tags]

        nouns_features['nouns_ratio'] = len(
            all_nouns) / len(all_tokens) if len(all_tokens) != 0 else 0

        nouns_sentences_ratios = []
        for sentence in sentences:
            tokens_sentence = [
                token.text for token in sentence if not token.is_punct]
            nouns_sentence = [
                token.text for token in sentence if token.pos_ in pos_tags]
            if len(tokens_sentence) > 0 and len(nouns_sentence) > 0:
                nouns_sentences_ratios.append(
                    len(nouns_sentence) / len(tokens_sentence))

        if len(nouns_sentences_ratios) > 0:
            nouns_features['nouns_max'] = max(nouns_sentences_ratios)
            nouns_features['nouns_min'] = min(nouns_sentences_ratios)
            nouns_features['nouns_std'] = np.std(nouns_sentences_ratios)

        return nouns_features

    def _compute_morphosyntactic_verbs(self, doc: Doc) -> dict[str, float]:
        sentences = [sent for sent in doc.sents]

        verbs_features = {
            'verbs_ratio': 0.0,
            'verbs_max': 0.0,
            'verbs_min': 0.0,
            'verbs_std': 0.0,
            'infinitive_verbs': 0.0,
            'inflected_verbs': 0.0,
            'non_inflected_verbs': 0.0,
            'indicative_condition_ratio': 0.0,
            'indicative_future_ratio': 0.0
        }

        if len(sentences) == 0:
            return verbs_features

        verbs_pos_tags = {'VERB', 'AUX'}

        all_tokens = [token.text for token in doc if not token.is_punct]
        all_verbs = [token for token in doc if token.pos_ in verbs_pos_tags]

        if len(all_tokens) == 0 or len(all_verbs) == 0:
            return verbs_features

        infinitive_verbs = [token.text
                            for token in all_verbs
                            if token.morph.get('VerbForm') == ['Inf']]
        inflected_verbs = [token.text
                           for token in all_verbs
                           if token.morph.get('VerbForm') == ['Fin']]
        non_inflected_verbs = [token.text
                               for token in all_verbs
                               if token.morph.get('VerbForm') != ['Fin']]
        indicative_condition = [token.text
                                for token in all_verbs
                                if token.morph.get('Mood') == ['Cnd']]
        indicative_future = [token.text
                             for token in all_verbs
                             if token.morph.get('Tense') == ['Fut']]

        verbs_features['verbs_ratio'] = len(all_verbs) / len(all_tokens)
        verbs_features['infinitive_verbs'] = len(
            infinitive_verbs) / len(all_verbs)
        verbs_features['inflected_verbs'] = len(
            inflected_verbs) / len(all_verbs)
        verbs_features['non_inflected_verbs'] = len(
            non_inflected_verbs) / len(all_verbs)

        if len(inflected_verbs) > 0:
            verbs_features['indicative_condition_ratio'] = len(
                indicative_condition) / len(inflected_verbs)
            verbs_features['indicative_future_ratio'] = len(
                indicative_future) / len(inflected_verbs)

        verbs_sentences_ratios = []
        for sentence in sentences:
            tokens_sentences = [token.text
                                for token in sentence
                                if not token.is_punct]
            verbs_sentence = [token.text
                              for token in sentence
                              if token.pos_ in verbs_pos_tags]

            if len(tokens_sentences) > 0:
                verbs_sentences_ratios.append(
                    len(verbs_sentence) / len(tokens_sentences))

        if len(verbs_sentences_ratios) > 0:
            verbs_features['verbs_max'] = max(verbs_sentences_ratios)
            verbs_features['verbs_min'] = min(verbs_sentences_ratios)
            verbs_features['verbs_std'] = np.std(verbs_sentences_ratios)

        return verbs_features

    def _compute_prepositions_per_clause_sentence(self, doc: Doc) -> dict[str, float]:
        sentences = [sent for sent in doc.sents]
        features_preposition = {
            'prepositions_per_clause': 0.0,
            'prepositions_per_sentence': 0.0
        }

        if len(sentences) == 0:
            return features_preposition

        total_prepositions = len(
            [token.text for token in doc if token.pos_ == 'ADP'])

        if total_prepositions == 0:
            return features_preposition

        all_clauses = []
        for sentence in sentences:
            clauses_sentence = extract_clauses_by_verbs(sentence)
            if clauses_sentence is not None:
                all_clauses.extend(clauses_sentence)

        if len(all_clauses) > 0:
            features_preposition['prepositions_per_clause'] = total_prepositions / \
                len(all_clauses)
            features_preposition['prepositions_per_sentence'] = total_prepositions / len(
                sentences)

        return features_preposition

    def _compute_morphosyntactic_pronouns(self, doc: Doc) -> dict[str, float]:
        features_pronouns = {
            'pronoun_ratio': 0.0,
            'personal_pronouns': 0.0,
            'oblique_pronouns_ratio': 0.0,
            'indefinite_pronoun_ratio': 0.0,
            'relative_pronouns_ratio': 0.0,
            'first_person_pronouns': 0.0,
            'second_person_pronouns': 0.0,
            'third_person_pronouns': 0.0,
            'pronouns_min': 0.0,
            'pronouns_max': 0.0,
            'pronouns_std': 0.0
        }

        sentences = [sent for sent in doc.sents]

        if len(sentences) == 0:
            return features_pronouns

        pronouns_oblique_tags = [['Acc'], ['Dat']]
        pronouns_types_tags = [['Prs'], ['Rcp'], ['Int'], [
            'Rel'], ['Dem'], ['Emp'], ['Tot'], ['Neg'], ['Ind']]
        all_tokens = [token for token in doc if not token.is_punct]

        if len(all_tokens) == 0:
            return features_pronouns

        all_pronouns = [token.text for token in all_tokens if token.morph.get(
            'PronType') in pronouns_types_tags]

        if len(all_pronouns) == 0:
            return features_pronouns

        all_personal_pronouns = [token.text
                                 for token in all_tokens
                                 if token.morph.get('Case') == ['Nom']
                                 and token.morph.get('PronType') == ['Prs']]
        all_pronouns_oblique = [
            token.text
            for token in all_tokens
            if token.morph.get('Case') in pronouns_oblique_tags]
        all_pronouns_indefinite = [token.text for token in all_tokens
                                   if token.morph.get('PronType') == ['Ind']]
        all_pronouns_relative = [token.text
                                 for token in all_tokens
                                 if token.morph.get('PronType') == ['Rel']]
        all_pronons_first_person = [
            token.text
            for token in all_tokens
            if token.morph.get('Person') == ['1']
            and token.morph.get('PronType') in pronouns_types_tags]

        all_pronouns_second_person = [
            token.text
            for token in all_tokens
            if token.morph.get('Person') == ['2']
            and token.morph.get('PronType') in pronouns_types_tags]

        all_pronouns_third_person = [
            token.text
            for token in all_tokens
            if token.morph.get('Person') == ['3']
            and token.morph.get('PronType') in pronouns_types_tags]

        features_pronouns['pronoun_ratio'] = len(
            all_pronouns) / len(all_tokens)
        features_pronouns['personal_pronouns'] = len(
            all_personal_pronouns) / len(all_tokens)
        features_pronouns['oblique_pronouns_ratio'] = len(
            all_pronouns_oblique) / len(all_pronouns)
        features_pronouns['indefinite_pronoun_ratio'] = len(
            all_pronouns_indefinite) / len(all_pronouns)
        features_pronouns['relative_pronouns_ratio'] = len(
            all_pronouns_relative) / len(all_pronouns)

        if len(all_personal_pronouns) > 0:
            features_pronouns['first_person_pronouns'] = len(
                all_pronons_first_person) / len(all_personal_pronouns)
            features_pronouns['second_person_pronouns'] = len(
                all_pronouns_second_person) / len(all_personal_pronouns)
            features_pronouns['third_person_pronouns'] = len(
                all_pronouns_third_person) / len(all_personal_pronouns)

        pronouns_ratio_sentences = []
        for sentence in sentences:
            tokens_sentences = [token
                                for token in sentence
                                if not token.is_punct]
            pronouns_sentence = [
                token.text
                for token in tokens_sentences
                if token.morph.get('PronType') in pronouns_types_tags]
            pronouns_ratio_sent = len(
                pronouns_sentence) / len(tokens_sentences) if len(pronouns_sentence) > 0 else 0
            pronouns_ratio_sentences.append(pronouns_ratio_sent)

        if len(pronouns_ratio_sentences) > 0:
            features_pronouns['pronouns_max'] = max(pronouns_ratio_sentences)
            features_pronouns['pronouns_min'] = min(pronouns_ratio_sentences)
            features_pronouns['pronouns_std'] = np.std(
                pronouns_ratio_sentences)

        return features_pronouns

    def _compute_morphosyntactic_other_metrics(self, doc: Doc) -> dict[str, float]:
        features = {
            'content_words': 0.0,
            'function_words': 0.0,
            'punctuation_ratio': 0.0,
            'ratio_function_to_content_words': 0.0
        }
        content_words_pos_tags = ['NOUN', 'PROPN', 'ADJ', 'VERB', 'AUX', 'ADV']
        total_content_words = len([token.text
                                   for token in doc
                                   if not token.is_punct
                                   and token.pos_ in content_words_pos_tags])
        total_functional_words = len([
            token.text
            for token in doc
            if not token.is_punct
            and token.pos_ not in content_words_pos_tags])
        total_punctuation = len(
            [token.text for token in doc if token.is_punct])
        total_words = total_content_words + total_functional_words

        if total_words > 0:
            features['content_words'] = total_content_words / total_words
            features['function_words'] = total_functional_words / total_words
            features['punctuation_ratio'] = total_punctuation / total_words

        if total_content_words > 0:
            features['ratio_function_to_content_words'] = total_functional_words / \
                total_content_words

        return features
