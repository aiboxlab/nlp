"""
"""
from __future__ import annotations

from sklearn.ensemble import ExtraTreesClassifier

from nlpbox.core.pipeline import Pipeline
from nlpbox.estimators.sklearn_estimator import SklearnEstimator
from nlpbox.factory import register
from nlpbox.vectorizers import TFIDFVectorizer


@register('tfidf_extratrees_classification')
class TFIDFExtraTreesClassification(Pipeline):
    def __init__(self, **kwargs):
        super().__init__(
            vectorizer=TFIDFVectorizer(),
            estimator=SklearnEstimator(ExtraTreesClassifier(**kwargs)))
