"""
"""
from nlpbox.core import Estimator, Pipeline, Vectorizer

from .class_registry import get_class


def get_pipeline(pipeline: str,
                 pipeline_config: dict = dict()) -> Pipeline:
    pipeline = get_class(pipeline)(**pipeline_config)
    assert isinstance(pipeline, Pipeline)

    return pipeline


def make_pipeline(vectorizer: str,
                  estimator: str,
                  vectorizer_config: dict = dict(),
                  estimator_config: dict = dict()) -> Pipeline:
    vectorizer = get_class(vectorizer)(**vectorizer_config)
    estimator = get_class(estimator)(**estimator_config)

    assert isinstance(vectorizer, Vectorizer)
    assert isinstance(estimator, Estimator)

    return Pipeline(vectorizer,
                    estimator)
