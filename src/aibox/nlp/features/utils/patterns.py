"""Módulo com funções utilitárias para
buscar/contar padrões no texto.
"""

from __future__ import annotations

import re

import unidecode


def count_connectives_in_sentences(
    connectives: list, sentences: list
) -> tuple[int, int]:
    """
    Método que busca a ocorrência de conectivas nas sentenças do texto.
    :param connectives:
    :param sentences:
    :return:
    """
    total_hits = 0
    total_words = 0

    for sentence in sentences:
        tokens_sentences = [
            unidecode.unidecode(token.text.lower())
            for token in sentence
            if not token.is_punct
        ]
        sentence_text = " ".join(tokens_sentences)
        total_words += len(tokens_sentences)
        total_hits += count_connectives_in_sentence(connectives, sentence_text)

    return total_hits, total_words


def count_connectives_in_sentence(list_connectives: list, sentence: str) -> int:
    """
    Método que procura os conectivos em uma frase.
    :param list_connectives: lista de conectivos.
    :param sentence: frase.
    :return: frequência de ocorrência de cada conectivo da lista na frase.
    """
    sentence = f" {sentence} "
    count = 0

    for connective in list_connectives:
        matches = re.findall(f" {connective.strip()} ", sentence)
        count += len(matches)

    return count
