"""
"""
from typing import Iterable

from nlpbox.core import FeatureExtractor
from nlpbox.features.utils.aggregator import AggregatedFeatureExtractor

from .class_registry import get_class


def get_extractor(features: Iterable[str],
                  configs: Iterable[dict] = None) -> FeatureExtractor:
    if configs is None:
        configs = [dict() for _ in features]

    features = list(features)
    configs = list(configs)
    assert len(configs) == len(features)

    if len(features) == 1:
        return get_class(features[0])(**configs[0])

    return AggregatedFeatureExtractor(*[get_class(f)(**c)
                                        for f, c in zip(features, configs)])
