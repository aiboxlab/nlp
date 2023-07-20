"""
"""
from __future__ import annotations

from sklearn.ensemble import ExtraTreesClassifier

from nlpbox.core.pipeline import Pipeline
from nlpbox.estimators.sklearn_estimator import SklearnEstimator
from nlpbox.factory import register
from nlpbox.features.cohmetrix import CohMetrixExtractor


@register('cohmetrix_extratrees_classification')
class CohMetrixExtraTreesClassification(Pipeline):
    def __init__(self, **kwargs):
        super().__init__(
            vectorizer=CohMetrixExtractor(),
            estimator=SklearnEstimator(ExtraTreesClassifier(**kwargs)))
