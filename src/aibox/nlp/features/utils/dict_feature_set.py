"""Módulo utilitário que define
um FeatureSet a partir de um
dicionário qualquer.
"""
from __future__ import annotations

from aibox.nlp.core import FeatureSet


class DictFeatureSet(FeatureSet):
    def __init__(self, data: dict[str, float]):
        self._d = data

    def as_dict(self) -> dict[str, float]:
        lexical_sorted_dict = dict(sorted(self._d.items(),
                                          key=lambda x: x[0]))
        return lexical_sorted_dict

