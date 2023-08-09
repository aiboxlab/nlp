"""Esse módulo contém uma pipeline
de classificação utilizando o
um vetorizador TF-IDF com um ensemble
de Extremely Randomized Trees.
"""
from __future__ import annotations

from sklearn.ensemble import ExtraTreesClassifier

from nlpbox.core.pipeline import Pipeline
from nlpbox.estimators.sklearn_estimator import SklearnEstimator
from nlpbox.vectorizers import TFIDFVectorizer


class TFIDFExtraTreesClassification(Pipeline):
    def __init__(self, **kwargs):
        super().__init__(
            vectorizer=TFIDFVectorizer(),
            estimator=SklearnEstimator(ExtraTreesClassifier(**kwargs)))
