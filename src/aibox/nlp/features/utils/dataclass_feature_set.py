from __future__ import annotations

import dataclasses

from aibox.nlp.core import FeatureSet


class DataclassFeatureSet(FeatureSet):
    """Implementação de um FeatureSet que supõe
    que a classe base é um dataclass (i.e., possui
    um método `asdict()`).
    """

    def as_dict(self) -> dict[str, float]:
        unordered_dict = dataclasses.asdict(self)
        lexical_sorted_dict = dict(sorted(unordered_dict.items(),
                                          key=lambda x: x[0]))
        return lexical_sorted_dict
