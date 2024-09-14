"""Esse módulo contém características
relacionadas com segmentação de palavras.
"""

from __future__ import annotations

import functools
import operator
from dataclasses import dataclass
from typing import Iterable, Literal

import spacy
from spacy import tokens
from spellchecker import SpellChecker

from aibox.nlp import resources
from aibox.nlp.core import FeatureExtractor
from aibox.nlp.features.utils import DataclassFeatureSet


@dataclass(frozen=True)
class WordSegmentationFeatures(DataclassFeatureSet):
    hypo_segmentation_score: float
    hyper_segmentation_score: float


class NorvigHypoSegmentaton:
    """Norvig's hiposegmentation algorithm
    adapted from https://norvig.com/ngrams/index.html
    """

    def __init__(self, dictionary: SpellChecker) -> None:
        self._dict = dictionary

    def __call__(self, text: str) -> list[str]:
        return self.segment(text)

    @functools.lru_cache(100)
    def segment(self, text: str) -> list[str]:
        "Return a list of words that is the best segmentation of text."
        if not text:
            return []
        candidates = ([first] + self.segment(rem) for first, rem in self.splits(text))
        return max(candidates, key=self.Pwords)

    def splits(self, text: str, max_len: int = 30) -> list[tuple[str, str]]:
        "Return a list of all possible (first, rem) pairs, len(first)<=L."
        return [(text[: i + 1], text[i + 1 :]) for i in range(min(len(text), max_len))]

    def Pwords(self, words: list[str]) -> float:
        "The Naive Bayes probability of a sequence of words."
        return functools.reduce(
            operator.mul, (self._dict.word_usage_frequency(w) for w in words)
        )


class UspSpellChecker(SpellChecker):
    def __init__(self, distance: int = 2, case_sensitive: bool = False) -> None:
        path = resources.path("dictionary/usp-spell-wordfreq.v1")
        path = path.joinpath("usp-spell-wordfreq.gz")
        super().__init__(
            local_dictionary=str(path), distance=distance, case_sensitive=case_sensitive
        )


class WordSegmentationExtractor(FeatureExtractor):
    def __init__(self, nlp: spacy.Language = None):
        if nlp is None:
            nlp = spacy.load("pt_core_news_md")

        self._dict = UspSpellChecker(distance=1, case_sensitive=True)
        self._nlp = nlp
        self._hypo_seg = NorvigHypoSegmentaton(self._dict)

    def extract(self, text: str, **kwargs) -> WordSegmentationFeatures:
        del kwargs

        doc = self._nlp(text)
        score_hyper = 0.0
        score_hypo = 0.0
        doc_size = len(doc)

        if doc_size > 0:
            # Calculando nota para hipersegmentação
            errors_hyper = list(self._search_hyper(doc))
            score_hyper = 1.0 - (len(errors_hyper) / doc_size)

            # Calculando nota para hiposegmentação
            errors_hypo = errors = sum(
                map(lambda tok: len(self._fix_hypo(tok)) > 1, doc)
            )
            score_hypo = 1.0 - (errors_hypo / doc_size)

        return WordSegmentationFeatures(
            hypo_segmentation_score=score_hypo, hyper_segmentation_score=score_hyper
        )

    def _detect_hyper(self, span: tokens.Span) -> Literal[False] | str:
        """Detects and corrects a span of tokens in case of hypersegmentation

        Return: correction if hypersegmented else False
        """
        # se todos os tokens estiverem no dicionário,
        # então não tem nada de estranho
        if all(tok.lower_ in self._dict.word_frequency for tok in span):
            return False

        word = "".join(tok.lower_ for tok in span)

        # se a concatenação estiver no vocabulário
        if word in self._dict:
            return word

        # se a concatenação precisava de uma pequena correção (distâcia 1)
        fixed = self._dict.correction(word)
        if fixed != word:
            return fixed

        # se mesmo concatenando e corrigindo, a palavra não fizer sentido,
        # então era apenas um erro e não uma hipersegmentação
        return False

    def _search_hyper(self, doc: tokens.Doc) -> Iterable[tuple[tokens.Span, str]]:
        """Search over the doc an find hypersegmentation of words

        returns: tuples with span and correction
        """
        for span in self._ngrams(doc):
            if not all(tok.is_alpha for tok in span):
                continue
            correction = self._detect_hyper(span)
            if correction:
                yield span, correction

    def _fix_hypo(self, token: tokens.Token) -> tokens.Doc:
        """Detects if a token is a hiposegmentation
        if true, return the correction (list of splited words)

        Args:
            token (tokens.Token): The token to detect

        Returns:
            tokens.Doc: if false, returns a new doc the token.
            if true, returns a doc with the splited words
        """

        if token.lower_ in self._dict or not token.is_alpha or not token.is_oov:
            return self._token_as_doc(token)

        words = self._hypo_seg(token.lower_)

        if len(words) <= 1:
            return self._token_as_doc(token)

        if token.is_title:
            words[0] = words[0].capitalize()

        spaces = [True] * (len(words) - 1) + [token.whitespace_]
        return tokens.Doc(token.vocab, words=words, spaces=spaces)

    @staticmethod
    def _ngrams(
        doc: tokens.Doc | list[str], size: int = 2, step: int = 1
    ) -> Iterable[tokens.Span | list[str]]:
        """Iterate over the doc returning a window of tokens each time
        similar to nltk.ngrams

        >>> tuple(ngrams('abcde')) -> ('ab', 'bc', 'cd', 'de')
        >>> tuple(ngrams('abcde', 3)) -> ('abc', 'bcd', 'cde')
        """
        for begin in range(0, len(doc) - size + 1, step):
            yield doc[begin : begin + size]

    @staticmethod
    def _words_spaces(tokens: list[tokens.Token]) -> list[tuple[str, bool]]:
        """Return the Doc constructable part of a sequence of tokens

        ex: words_and_spaces(doc1)
        ex: words_and_spaces(span_from_doc2)
        """
        return [(tok.text, tok.whitespace_) for tok in tokens]

    @staticmethod
    def _token_as_doc(token: tokens.Token) -> tokens.Doc:
        return tokens.Doc(token.vocab, words=[token.orth], spaces=[token.whitespace_])
