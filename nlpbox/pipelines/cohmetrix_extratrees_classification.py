"""Esse módulo contém uma pipeline
de classificação utilizando o
CohMetrix com um ensemble de Extremely
Randomized Trees.
"""
from __future__ import annotations

from sklearn.ensemble import ExtraTreesClassifier

from nlpbox.core.pipeline import Pipeline
from nlpbox.estimators.sklearn_estimator import SklearnEstimator
from nlpbox.features.cohmetrix import CohMetrixExtractor


class CohMetrixExtraTreesClassification(Pipeline):
    def __init__(self, **kwargs):
        super().__init__(
            vectorizer=CohMetrixExtractor(),
            estimator=SklearnEstimator(ExtraTreesClassifier(**kwargs)))
