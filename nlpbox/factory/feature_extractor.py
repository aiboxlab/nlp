"""Esse método contém funções utilitárias
para construção e obtenção de extratores
de características através de nomes.
"""
from __future__ import annotations

from nlpbox.core import FeatureExtractor
from nlpbox.features.utils.aggregator import AggregatedFeatureExtractor

from .class_registry import get_class


def get_extractor(features: list[str],
                  configs: list[dict] = None) -> FeatureExtractor:
    """Obtém um extrator de características para
    todas as características na lista `features`.

    Args:
        features (list[str]): lista com as características.
        configs (list[dict], opcional): parâmetros para serem passados
            aos construtores dos extratores de características.

    Returns:
        Extrator de características.
    """
    assert isinstance(features, list)

    if configs is None:
        configs = [dict() for _ in features]

    features = list(features)
    configs = list(configs)
    assert len(configs) == len(features)

    if len(features) == 1:
        return get_class(features[0])(**configs[0])

    return AggregatedFeatureExtractor(*[get_class(f)(**c)
                                        for f, c in zip(features, configs)])
