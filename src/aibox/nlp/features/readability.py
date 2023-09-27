"""Esse módulo extrai as features
relacionadas à leiturabilidade.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import pyphen
import spacy

from aibox.nlp import resources
from aibox.nlp.core import FeatureExtractor

from .utils import DataclassFeatureSet


@dataclass(frozen=True)
class ReadabilityFeatures(DataclassFeatureSet):
    brunet_indice: float
    adapted_dalechall: float
    flesch_indice: float
    gunning_fox_indice: float
    honore_statistics: float
    readibility_indice: float
    token_var_idx: float


class ReadabilityExtractor(FeatureExtractor):
    def __init__(self, nlp: spacy.Language | None = None):
        if nlp is None:
            nlp = spacy.load('pt_core_news_md')

        root_path = resources.path('dictionary/biderman-words.v1')
        words_file = root_path.joinpath('words.txt')
        self._biderman_words = self.read_words_file(words_file)
        self._nlp = nlp
        self._dict = pyphen.Pyphen(lang='pt-BR')

    def extract(self, text: str, **kwargs) -> ReadabilityFeatures:
        del kwargs

        doc = self._nlp(text)
        sentences = [sent.text for sent in doc.sents]

        features = {
            'brunet_indice': 0.0,
            'adapted_dalechall': 0.0,
            'flesch_indice': 0.0,
            'gunning_fox_indice': 0.0,
            'honore_statistics': 0.0,
            'readibility_indice': 0.0,
            'token_var_idx': 0.0
        }

        if len(sentences) > 1:
            features['brunet_indice'] = self.compute_brunet_indice(doc)
            features['adapted_dalechall'] = self.compute_adapted_dalechall(
                doc)
            features['flesch_indice'] = self.compute_flesch_indice(doc)
            features['gunning_fox_indice'] = self.compute_gunning_fox_indice(
                doc)
            features['honore_statistics'] = self.compute_honore_statistics(doc)
            features['readibility_indice'] = self.compute_readibility_indice(
                doc)
            features['token_var_idx'] = self.compute_token_var_idx(doc)

        return ReadabilityFeatures(**features)

    def read_words_file(self, words_file: Path) -> set[str]:
        with words_file.open('r', encoding='utf-8') as file:
            list_biderman = file.read().split('\n')

        list_biderman = [line.split(',')[0] for line in list_biderman]
        return set(list_biderman)

    def compute_brunet_indice(self, doc) -> float:
        """Método que computa a estatística de Brunet
        é uma forma de type/token ratio menos sensível ao
        tamanho do texto. Eleva-se o número de types à
        constante -0,165 e depois eleva-se o número de tokens a
        esse resultado.
        """
        tokens = [token.lower_ for token in doc if not token.is_punct]
        if len(tokens) == 0:
            return 0

        distinct_tokens = set(tokens)
        return len(tokens) ** (len(distinct_tokens) ** -0.165)

    def compute_adapted_dalechall(self, doc) -> float:
        """Método que computa a fórmula de leiturabilidade de Dalechall
        adaptada combina a quantidade de palavras não
        familiares com a quantidade média de palavras por sentença.
        Palavras não familiares" são aquelas que não
        constam do vocabulário básico conhecido por alunos do quarto ano.
        Para fins dessa métrica, foram utilizadas as entradas do Dicionário
        de Palavras Simples de Maria Tereza Biderman.
        """
        sentences = [sent for sent in doc.sents]
        if len(sentences) == 0:
            return 0

        adp = ['no', 'na', 'nas', 'do', 'da', 'das']
        tokens = [token.lemma_.lower()
                  if token.lower_ not in adp
                  else token.orth_ for token in doc
                  if not token.is_punct]

        if len(tokens) == 0:
            return 0

        unfamiliar_words = [token for token in tokens
                            if token not in self._biderman_words]
        words_by_sentence_ratio = len(tokens) / len(sentences)
        unfamiliar_ratio = len(unfamiliar_words) / len(tokens)
        adapted_dalechall = (0.1579 * unfamiliar_ratio) + \
            (0.0496 * words_by_sentence_ratio) + 3.6365

        return adapted_dalechall

    def compute_flesch_indice(self, doc) -> float:
        """Método que computa o índice de Leiturabilidade
        de Flesch busca uma correlação entre tamanhos médios de
        palavras e sentenças.
        """
        sentences = [sent for sent in doc.sents]
        qtde_sentences = len(sentences)

        if qtde_sentences == 0:
            return 0

        words = [word.lower_ for word in doc if not word.is_punct]
        qtde_words = len(words)

        if qtde_words == 0:
            return 0

        mean_words_by_sentence = qtde_words / qtde_sentences
        total_syllables = 0
        for word in words:
            syllables = self._dict.inserted(word).split('-')
            total_syllables += len(syllables)

        mean_syllables_words = total_syllables / qtde_words
        flesch = 248.835 - (1.015 * mean_words_by_sentence) - \
            (84.6 * mean_syllables_words)

        return flesch

    def compute_gunning_fox_indice(self, doc) -> float:
        """Método que computa o índice de leiturabilidade
        Gunning Fog (também conhecido como Gunning FoX) soma a
        quantidade média de palavras por sentença ao percentual
        de palavras difíceis no texto e multiplica tudo
        por 4. O resultado está diretamente ligado aos 12 níveis
        do ensino americano. Índices superiores a 12
        representam textos extremamente complexos.
        """
        sentences = [sent for sent in doc.sents]
        qtde_sentences = len(sentences)

        if qtde_sentences == 0:
            return 0

        words = [word.lower_ for word in doc if not word.is_punct]
        qtde_words = len(words)

        if qtde_words == 0:
            return 0

        mean_words_by_sentences = qtde_words / qtde_sentences
        total_difficult_words = 0

        for word in words:
            syllables = self._dict.inserted(word).split('-')
            if len(syllables) > 2:
                total_difficult_words += 1

        mean_difficult_words = total_difficult_words / qtde_words
        gunning_fox = (mean_words_by_sentences + mean_difficult_words) * 0.4

        return gunning_fox

    def compute_honore_statistics(self, doc) -> float:
        """Método que computa a estatística de Honoré
        que é um tipo de type/token ratio que leva em
        consideração, além da quantidade de types e tokens,
        a quantidade de hapax legomena.
        """
        words = [word.lower_ for word in doc if not word.is_punct]
        distinct_words = set(words)
        words_freq_1 = [word for word in words if words.count(word) == 1]
        qtde_words = len(words)
        qtde_words_freq_1 = len(words_freq_1)
        qtde_distinct_words = len(distinct_words)

        if qtde_words_freq_1 != qtde_distinct_words:
            honore_statistics = (100 * math.log10(qtde_words)) / \
                                (1 - (qtde_words_freq_1 / qtde_distinct_words))
        else:
            honore_statistics = 0

        return honore_statistics

    def compute_readibility_indice(self, doc) -> float:
        """Método que computa índice de leiturabilidade
        do texto que leva em consideração a média de palavras
        por frase e a média de caracteres por palavra.
        """
        sentences = [sent for sent in doc.sents]

        if len(sentences) == 0:
            return 0

        tokens = [token.lower_ for token in doc if not token.is_punct]
        total_tokens = len(tokens)
        total_characters = sum([len(token) for token in tokens])
        mean_tokens_sentences = total_tokens / len(sentences)
        mean_characters_tokens = total_characters / total_tokens
        readibility_idx = 0.5 * mean_tokens_sentences + \
            4.71 * mean_characters_tokens - 21.43

        return readibility_idx

    def compute_token_var_idx(self, doc) -> float:
        """Método que computa índice de variação
        da quantidade de palavras do texto.
        """
        tokens = [token.lower_ for token in doc if not token.is_punct]
        total_tokens = len(tokens)

        if total_tokens == 0:
            return 0

        log_tokens = math.log(total_tokens)
        factor = math.log(len(set(tokens))) / \
            log_tokens if log_tokens > 0 else 0
        factor = math.log(2 - factor)
        token_var_idx = log_tokens / factor if factor > 0 else 0

        return token_var_idx
