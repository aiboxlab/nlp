"""Esse módulo contém características do
LIWC.
"""
from __future__ import annotations

from dataclasses import dataclass

import spacy

from aibox.nlp import resources
from aibox.nlp.core import FeatureExtractor
from aibox.nlp.features.utils import DataclassFeatureSet


@dataclass(frozen=True)
class LiwcFeatures(DataclassFeatureSet):
    funct: float
    pronoun: float
    ppron: float
    i: float
    we: float
    you: float
    shehe: float
    they: float
    ipron: float
    article: float
    verb: float
    auxverb: float
    past: float
    present: float
    future: float
    adverb: float
    preps: float
    conj: float
    negate: float
    quant: float
    number: float
    swear: float
    social: float
    family: float
    friend: float
    humans: float
    affect: float
    posemo: float
    negemo: float
    anx: float
    anger: float
    sad: float
    cogmech: float
    insight: float
    cause: float
    discrep: float
    tentat: float
    certain: float
    inhib: float
    incl: float
    excl: float
    percept: float
    see: float
    hear: float
    feel: float
    bio: float
    body: float
    health: float
    sexual: float
    ingest: float
    relativ: float
    motion: float
    space: float
    time: float
    work: float
    achieve: float
    leisure: float
    home: float
    money: float
    relig: float
    death: float
    assent: float
    nonfl: float
    filler: float


class LiwcExtractor(FeatureExtractor):
    def __init__(self, nlp: spacy.Language = None):
        if nlp is None:
            nlp = spacy.load('pt_core_news_md')

        self._nlp = nlp
        self._indice_to_name = {
            1: 'funct', 2: 'pronoun', 3: 'ppron', 4: 'i',
            5: 'we', 6: 'you', 7: 'shehe', 8: 'they', 9: 'ipron',
            10: 'article', 11: 'verb', 12: 'auxverb',
            13: 'past', 14: 'present', 15: 'future', 16: 'adverb',
            17: 'preps', 18: 'conj', 19: 'negate', 20: 'quant',
            21: 'number', 22: 'swear', 121: 'social', 122: 'family',
            123: 'friend', 124: 'humans', 125: 'affect', 126: 'posemo',
            127: 'negemo', 128: 'anx', 129: 'anger', 130: 'sad',
            131: 'cogmech', 132: 'insight', 133: 'cause', 134: 'discrep',
            135: 'tentat', 136: 'certain', 137: 'inhib', 138: 'incl',
            139: 'excl', 140: 'percept', 141: 'see', 142: 'hear',
            143: 'feel', 146: 'bio', 147: 'body', 148: 'health',
            149: 'sexual', 150: 'ingest', 250: 'relativ', 251: 'motion',
            252: 'space', 253: 'time', 354: 'work', 355: 'achieve',
            356: 'leisure', 357: 'home', 358: 'money', 359: 'relig',
            360: 'death', 462: 'assent', 463: 'nonfl', 464: 'filler'
        }

        self._words = []
        root_dir = resources.path('dictionary/liwc-dictionary.v1')
        words_path = root_dir.joinpath('LIWC2007_Portugues_win.dic.txt')
        with words_path.open(mode='r',
                             encoding='latin1',
                             errors='ignore') as file:
            lines = file.read().split('\n')
            for line in lines:
                words = line.split('\t')
                self._words.append(words)

        indices_path = root_dir.joinpath('indices.txt')
        with indices_path.open(mode='r',
                               encoding='utf-8',
                               errors='ignore') as file:
            self._indices = file.read().split('\n')

    def extract(self, text: str, **kwargs) -> LiwcFeatures:
        del kwargs

        doc = self._nlp(text)
        tokens = [t.text for t in doc
                  if t.pos_ not in {'PUNCT', 'SYM'}]
        liwc_dict = {v: 0 for v in self._indice_to_name.values()}

        for token in tokens:
            position = self._search(token)
            if position:
                tam = len(self._words[position[0]])
                for i in range(tam):
                    id_ = self._words[position[0]][i]
                    if id_ in self._indices:
                        feature_name = self._indice_to_name[int(id_)]
                        liwc_dict[feature_name] = liwc_dict[feature_name] + 1

        return LiwcFeatures(**{k: float(v) for k, v in liwc_dict.items()})

    def _search(self, token: str) -> list:
        """Função que busca a ocorrência de um
        token na lista de tokens.

        Args:
            token (str): token a ser buscado.

        Returns:
            list: lista de ocorrências.
        """
        return [self._words.index(x)
                for x in self._words
                if token in x]
