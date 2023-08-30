"""Esse módulo contém características
de simplicidade textual.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

import spacy
from spacy.tokens.doc import Doc

from nlpbox import resources
from nlpbox.core import FeatureExtractor

from .utils import DataclassFeatureSet


@dataclass(frozen=True)
class TextualSimplicityFeatures(DataclassFeatureSet):
    dialog_pron_ratio: float
    easy_conj_ratio: float
    hard_conj_ratio: float
    short_sentence_ratio: float
    medium_short_sentence_ratio: float
    medium_long_sentence_ratio: float
    long_sentence_ratio: float
    simple_word_ratio: float


class TextualSimplicityExtractor(FeatureExtractor):
    def __init__(self, nlp: spacy.Language = None):
        if nlp is None:
            nlp = spacy.load('pt_core_news_md')

        self._easy = [
            'como', 'se', 'mas', 'quando', 'ou',
            'que', 'porque', 'e', 'assim',
            'porém', 'por isso que', 'por isso',
            'por enquanto', 'enquanto isso',
            'enquanto', 'pois', 'além de', 'então',
            'daí', 'por exemplo', 'ou seja',
            'sem que', 'para que', 'cada vez que',
            'antes que', 'assim como', 'tanto quanto',
            'feito', 'que nem', 'toda vez que',
            'a não ser que', 'depois que', 'até que',
            'desde', 'nem bem', 'tanto que',
            'segundo', 'assim que', 'tanto que',
            'tão que', 'sem que', 'ora']
        self._easy = [f' {c}' for c in self._easy]

        self._hard = [
            'todavia', 'eis', 'a fim de',
            'ao passo que', 'conforme', 'tais',
            'contudo', 'bem como', 'logo',
            'à medida que', 'entretanto', 'desde que',
            'mesmo que', 'ainda que', 'de acordo com',
            'uma vez que', 'por sua vez', 'sobretudo',
            'até', 'ainda', 'caso', 'no entanto', 'nem',
            'quanto', 'já', 'já que', 'outrossim',
            'mas também', 'como também', 'não só',
            'mas ainda', 'tampouco', 'senão também',
            'bem assim', 'ademais', 'antes',
            'não obstante', 'sem embargo', 'ao passo que',
            'de outra forma', 'em todo caso', 'aliás',
            'de outro modo', 'por conseguinte',
            'em consequência de', 'por consequência',
            'consequentemente', 'conseguintemente',
            'isso posto', 'pelo que', 'de modo que',
            'de maneira que', 'de forma que',
            'em vista disso', 'por onde', 'porquanto',
            'posto que', 'isto é', 'ademais',
            'senão', 'dado que', 'visto como',
            'vez que', 'de vez que', 'pois que', 'agora',
            'na medida em que', 'sendo que', 'como que',
            'como quer que', 'eis que', 'sendo assim',
            'tal qual', 'ao invés de', 'conquanto',
            'por muito que', 'visto que', 'uma vez que',
            'quanto mais', 'quanto menos', 'se bem que',
            'apesar de que', 'suposto que',
            'ainda quando', 'quando mesmo',
            'a despeito de', 'conquanto que',
            'sem embargo de que', 'por outro lado',
            'em contrapartida', 'sem embargo',
            'muito embora', 'inclusive se',
            'por mais que', 'por menos que',
            'por pouco que', 'contanto que', 'salvo se',
            'com tal que', 'caso que', 'consoante',
            'tal que', 'de forma que', 'à proporção que',
            'ao passo que', 'mal', 'tão logo',
            'entretanto', 'sob esse aspecto',
            'sob esse prisma', 'sob esse ponto de vista',
            'sob esse enfoque', 'embora', 'portanto',
            'além disso']
        self._hard = [f' {c}' for c in self._hard]

        self._nlp = nlp

        concrete_words = resources.path('dictionary/'
                                        'biderman-concrete-words.v1')
        concrete_words = concrete_words.joinpath('words.txt')
        with concrete_words.open('r', encoding='utf-8') as file:
            list_concrete = file.read().split('\n')

        biderman_words = resources.path('dictionary/biderman-words.v1')
        biderman_words = biderman_words.joinpath('words.txt')
        with biderman_words.open('r', encoding='utf-8') as file:
            list_biderman = file.readlines()

        list_biderman = [line.split(',')[0] for line in list_biderman]
        list_concrete.extend(list_biderman)
        self._simple_words = set(list_biderman)

    def extract(self, text: str) -> dict:
        doc = self._nlp(text)
        dialog_pron_ratio = self._compute_dialog_pron_ratio(doc)
        easy_conj_ratio, hard_conj_ratio = self._compute_conj_ratio(doc)
        sentences_len_ratio_dict = self._compute_sentence_length_ratio(doc)
        simple_word_ratio = self._compute_simple_word_ratio(doc)

        return TextualSimplicityFeatures(dialog_pron_ratio=dialog_pron_ratio,
                                         easy_conj_ratio=easy_conj_ratio,
                                         hard_conj_ratio=hard_conj_ratio,
                                         simple_word_ratio=simple_word_ratio,
                                         **sentences_len_ratio_dict)

    def _compute_dialog_pron_ratio(self, doc: Doc) -> float:
        """Método que computa a proporção de pronomes
        pessoais que indicam uma conversa com o leitor ("eu",
        "tu", "você" e "vocês") em relação ao total de pronomes
        pessoais presentes no texto.
        """
        all_personal_pronouns = []
        sentences = [sent for sent in doc.sents]
        for sentence in sentences:
            personal_pronouns = [token.lower_
                                 for token in sentence
                                 if not token.is_punct and
                                 'Prs' in token.morph.get('PronType')]

            if len(personal_pronouns) != 0:
                all_personal_pronouns.extend(personal_pronouns)

        total_personal_pronouns = len(all_personal_pronouns)
        if total_personal_pronouns == 0:
            return 0

        # Pronomes que indicam uma conversação com o leitor
        all_personal_pronouns_reader = ['eu', 'tu', 'você',
                                        'voces', 'nós',
                                        'vós']
        pronouns_reader = [p for p in all_personal_pronouns
                           if p in all_personal_pronouns_reader]
        return len(pronouns_reader) / total_personal_pronouns

    def _count_patterns(self, list_terms: list, sentence) -> int:
        """Método que procura termos em uma frase.
        """
        sentence = f'{sentence} '
        count = 0
        for term in list_terms:
            matches = re.findall(f'{term.strip()} ', sentence)
            count += len(matches)
        return count

    def _compute_frequency(self, list_terms: list, sentences: list) -> float:
        """Método que computa a incidência dos termos da
        lista na lista de frases.
        """
        frequency = 0
        for sentence in sentences:
            frequency += self._count_patterns(list_terms, sentence)
        return frequency

    def _compute_conj_ratio(self, doc: Doc) -> [float, float]:
        """Função que computa a proporção de conjunções fáceis
        e difíceis em relação a todas as palavras do texto.
        """
        all_tokens = [token.text for token in doc
                      if not token.is_punct]
        total_tokens = len(all_tokens)
        if total_tokens == 0:
            return 0, 0

        sentences = [sent.text.lower() for sent in doc.sents]
        frequency_easy = self._compute_frequency(self._easy,
                                                 sentences)
        frequency_hard = self._compute_frequency(self._hard,
                                                 sentences)
        easy_conj_metric = frequency_easy / total_tokens
        hard_conj_metric = frequency_hard / total_tokens

        return easy_conj_metric, hard_conj_metric

    def _compute_sentence_length_ratio(self, doc: Doc) -> dict:
        """Método que computa a proporção de sentenças
        (longas, médias e curtas) em relação a todas as
        sentenças do texto.
        """
        sentences = [sent for sent in doc.sents]

        if len(sentences) == 0:
            return {
                'short_sentence_ratio': 0,
                'medium_short_sentence_ratio': 0,
                'medium_long_sentence_ratio': 0,
                'long_sentence_ratio': 0
            }

        total_short_sents = 0
        total_medium_short_sents = 0
        total_medium_long_sents = 0
        total_long_sents = 0

        for sentence in sentences:
            tokens = [token.text for token in sentence
                      if not token.is_punct]
            if len(tokens) <= 11:
                total_short_sents += 1
            elif 12 <= len(tokens) <= 13:
                total_medium_short_sents += 1
            elif 14 <= len(tokens) <= 15:
                total_medium_long_sents += 1
            else:
                total_long_sents += 1

        total_sentences = len(sentences)

        total_short_sents /= total_sentences
        total_medium_short_sents /= total_sentences
        total_medium_long_sents /= total_sentences
        total_long_sents /= total_sentences

        metrics = {
            'short_sentence_ratio': total_short_sents,
            'medium_short_sentence_ratio': total_medium_short_sents,
            'medium_long_sentence_ratio': total_medium_long_sents,
            'long_sentence_ratio': total_long_sents
        }

        return metrics

    def _compute_simple_word_ratio(self, doc: Doc) -> float:
        """Método que computa a proporção de palavras
        de conteúdo simples, sobre o total de palavras de
        conteúdo do texto.
        """
        sentences = [sent for sent in doc.sents]

        if len(sentences) == 0:
            return 0

        total_simple_words = 0
        total_content_words = 0

        content_words_pos = {'NOUN', 'PROPN', 'ADJ', 'ADV', 'VERB'}

        for sentence in sentences:
            tokens = []
            for token in sentence:
                if not token.is_punct and token.pos_ in content_words_pos:
                    if token.pos_ == 'VERB':
                        tokens.append(token.lemma_.lower())
                    else:
                        tokens.append(token.lower_)

            simple_words = [word for word in tokens
                            if word in self._simple_words]
            total_content_words += len(tokens)
            total_simple_words += len(simple_words)
            ratio = 0.0
            try:
                retio = total_simple_words / total_content_words
            except ZeroDivisionError:
                pass

        return ratio
