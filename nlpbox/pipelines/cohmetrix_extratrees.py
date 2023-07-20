"""
"""
from __future__ import annotations

from sklearn.ensemble import ExtraTreesClassifier

from nlpbox.core.pipeline import Pipeline
from nlpbox.estimators.sklearn_estimator import SklearnEstimator
from nlpbox.features.cohmetrix import CohMetrixExtractor
from nlpbox.vectorizers import FeatureExtractorVectorizer


class CohMetrixExtraTreesClassification(Pipeline):
    def __init__(self, **kwargs):
        super().__init__(
            vectorizer=FeatureExtractorVectorizer(CohMetrixExtractor()),
            estimator=SklearnEstimator(ExtraTreesClassifier(**kwargs)))
