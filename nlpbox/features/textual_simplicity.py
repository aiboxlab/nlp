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
    pass


class TextualSimplicityExtractor:
    def __init__(self, concrete_words_file: str):
        self.easy_conjunctions = ['como', 'se', 'mas', 'quando', 'ou',
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

        self.hard_conjunctions = ['todavia', 'eis', 'a fim de',
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
                                  'apesar de que', 'suposto que', 'ainda quando',
                                  'quando mesmo', 'a despeito de', 'conquanto que',
                                  'sem embargo de que', 'por outro lado',
                                  'em contrapartida', 'sem embargo', 'muito embora',
                                  'inclusive se', 'por mais que', 'por menos que',
                                  'por pouco que', 'contanto que', 'salvo se',
                                  'com tal que', 'caso que', 'consoante', 'tal que', 'de forma que',
                                  'à proporção que', 'ao passo que', 'mal', 'tão logo', 'entretanto',
                                  'sob esse aspecto', 'sob esse prisma', 'sob esse ponto de vista',
                                  'sob esse enfoque', 'embora', 'portanto', 'além disso']

        self.list_simple_words = self.read_simple_words_files(
            concrete_words_file, biderman_words_file)

    def __call__(self, doc: Doc) -> dict:
        """
            Método que extrai as features relacionadas a simplicidade textual.
            :return um dicionário contendo as features relacionadas a simplicidade textual.
        """
        return self.compute_textual_simplicity_features(doc)

    def compute_textual_simplicity_features(self, doc: Doc) -> dict:
        """
            Método que computa as features do uso de simplicidade textual.
                1. dialog_pron_ratio
                2. easy_conj_ratio
                3. hard_conj_ratio
                4. short_sentence_ratio
                5. medium_short_sentence_ratio
                6. medium_long_sentence_ratio
                7. long_sentence_ratio
                8. simple_word_ratio
            :param doc:
            :return:
        """

        assert doc is not None, 'Error DOC is None'

        dialog_pron_ratio = self.compute_dialog_pron_ratio(doc)
        easy_conj_ratio, hard_conj_ratio = self.compute_conj_ratio(doc)
        sentences_len_ratio_dict = self.compute_sentence_length_ratio(doc)
        simple_word_ratio = self.compute_simple_word_ratio(doc)

        features = {
            'dialog_pron_ratio': dialog_pron_ratio,
            'easy_conj_ratio': easy_conj_ratio,
            'hard_conj_ratio': hard_conj_ratio,
            'simple_word_ratio': simple_word_ratio,
        }

        features.update(sentences_len_ratio_dict)

        return features

    @staticmethod
    def compute_dialog_pron_ratio(doc: Doc) -> float:
        """
            Método que computa a proporção de pronomes pessoais que indicam uma conversa com o leitor
            ("eu", "tu", "você" e "vocês") em relação ao total de pronomes pessoais presentes no texto.
            :param doc:
            :return:
        """
        assert doc is not None, 'Error DOC is None'
        all_personal_pronouns = []
        sentences = [sent for sent in doc.sents]
        for sentence in sentences:
            personal_pronouns = [token.text.lower() for token in sentence
                                 if not token.is_punct and 'Prs' in token.morph.get('PronType')]
            if len(personal_pronouns) != 0:
                all_personal_pronouns.extend(personal_pronouns)
        total_personal_pronouns = len(all_personal_pronouns)
        if total_personal_pronouns == 0:
            return 0
        # Pronomes que indicam uma conversação com o leitor
        all_personal_pronouns_reader = [
            'eu', 'tu', 'você', 'voces', 'nós', 'vós']
        pronouns_reader = [
            p for p in all_personal_pronouns if p in all_personal_pronouns_reader]
        return len(pronouns_reader) / total_personal_pronouns

    @staticmethod
    def count_patterns(list_terms: list, sentence) -> int:
        """
            Método que procura termos em uma frase.
            :param list_terms: lista de termos.
            :param sentence: frase.
            :return: frequência de ocorrência de cada termo da lista na frase.
        """
        sentence = f'{sentence} '
        count = 0
        for term in list_terms:
            matches = re.findall(f'{term.strip()} ', sentence)
            count += len(matches)
        return count

    def compute_frequency(self, list_terms: list, sentences: list) -> float:
        """
            Método que computa a incidência dos termos da lista na lista de frases.
            :param list_terms:
            :param sentences:
            :return:
        """
        frequency = 0
        for sentence in sentences:
            frequency += self.count_patterns(list_terms, sentence)
        return frequency

    def compute_conj_ratio(self, doc: Doc) -> [float, float]:
        """
            Função que computa a proporção de conjunções fáceis e difíceis em relação a todas as palavras do texto.
            :param doc:
            :return:
        """

        assert doc is not None, 'Error DOC is None'

        all_tokens = [token.text for token in doc if not token.is_punct]
        total_tokens = len(all_tokens)
        if total_tokens == 0:
            return 0, 0
        sentences = [sent.text.lower() for sent in doc.sents]
        frequency_easy = self.compute_frequency(
            self.easy_conjunctions, sentences)
        frequency_hard = self.compute_frequency(
            self.hard_conjunctions, sentences)
        easy_conj_metric = frequency_easy / total_tokens
        hard_conj_metric = frequency_hard / total_tokens
        return easy_conj_metric, hard_conj_metric

    @staticmethod
    def compute_sentence_length_ratio(doc: Doc) -> dict:
        """
            Método que computa a proporção de sentenças (longas, médias e curtas) em relação a todas as
            sentenças do texto.
            :param doc:
            :return:
        """

        assert doc is not None, 'Error DOC is None'

        sentences = [sent for sent in doc.sents]

        if len(sentences) == 0:
            return {
                'short_sentence_ratio': 0, 'medium_short_sentence_ratio': 0,
                'medium_long_sentence_ratio': 0, 'long_sentence_ratio': 0
            }

        total_short_sents = 0
        total_medium_short_sents = 0
        total_medium_long_sents = 0
        total_long_sents = 0

        for sentence in sentences:
            tokens = [token.text for token in sentence if not token.is_punct]
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
            'short_sentence_ratio': total_short_sents, 'medium_short_sentence_ratio': total_medium_short_sents,
            'medium_long_sentence_ratio': total_medium_long_sents, 'long_sentence_ratio': total_long_sents
        }

        return metrics

    @staticmethod
    def read_simple_words_files(concrete_words_file: str, biderman_words_file: str) -> list:
        with open(concrete_words_file, encoding='utf-8') as file:
            list_concrete = file.read().split('\n')
        with open(biderman_words_file, encoding='utf-8') as file:
            list_biderman = file.readlines()
        list_biderman = [line.split(',')[0] for line in list_biderman]
        list_concrete.extend(list_biderman)
        return list(set(list_biderman))

    def compute_simple_word_ratio(self, doc: Doc) -> float:
        """
            Método que computa a proporção de palavras de conteúdo simples, sobre o total de palavras de
            conteúdo do texto.
            :param doc:
            :return:
        """

        assert doc is not None, 'Error DOC is None'

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
                        tokens.append(token.text.lower())
            simple_words = [
                word for word in tokens if word in self.list_simple_words]
            total_content_words += len(tokens)
            total_simple_words += len(simple_words)

        return 0 if total_content_words == 0 else total_simple_words / total_content_words
