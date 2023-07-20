"""Esse módulo contém características do
LIWC.
"""
from __future__ import annotations

import typing
from dataclasses import dataclass

from apa_nlp._resources import get_resource


@dataclass(frozen=True)
class LiwcFeatures:
    _words: typing.ClassVar[list[str]] = None
    _indices: typing.ClassVar[list] = None
    _indice_to_name: typing.ClassVar[dict] = None
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

    @classmethod
    def extract(cls,
                text: str,
                **kwargs) -> LiwcFeatures:
        if cls._words is None:
            cls._load_liwc()

        liwc_dict = {v: 0 for k, v in cls._indice_to_name.items()}

        for token in tokens:
            position = cls._search(token)
            if position:
                tam = len(cls._words[position[0]])
                for i in range(tam):
                    id_ = cls._words[position[0]][i]
                    if id_ in cls._indices:
                        feature_name = cls._indice_to_name[int(id_)]
                        liwc_dict[feature_name] = liwc_dict[feature_name] + 1

        return LiwcFeatures(**liwc_dict)

    @classmethod
    def _load_liwc(cls):
        cls._indice_to_name = {
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

        cls._words = []
        words_path = get_resource('dictionary/liwc_words.txt')
        with words_path.open(mode='r',
                             encoding='latin1',
                             errors='ignore') as file:
            lines = file.read().split('\n')
            for line in lines:
                words = line.split('\t')
                self.liwc_words.append(words)

        with indices_path.open(mode='r',
                               encoding='utf-8',
                               errors='ignore') as file:
            cls._indices = file.read().split('\n')

    @classmethod
    def _search(cls, token: str) -> list:
        """Função que busca a ocorrência de um
        token na lista de tokens.

        Args:
            token (str): token a ser buscado.

        Returns:
            list: lista de ocorrências.
        """
        return [cls._words.index(x)
                for x in cls._words
                if token in x]
