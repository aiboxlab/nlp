"""
"""
from __future__ import annotations

from sklearn.ensemble import ExtraTreesClassifier as SklearnEtree

from nlpbox.core.pipeline import Pipeline
from nlpbox.vectorizer.tfidf_vectorizer import TFIDFVectorizer
from nlpbox.estimators.sklearn_estimator import SklearnEstimator


class ExtraTreesClassifier(Pipeline):
    def __init__(self, **kwargs):
        super().__init__(
            vectorizer=TFIDFVectorizer(),
            estimator=SklearnEstimator(SklearnEtree(**kwargs)))
