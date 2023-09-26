"""Esse módulo contém uma pipeline
de classificação utilizando o
CohMetrix com um ensemble de Extremely
Randomized Trees.
"""
from __future__ import annotations

from nlpbox.core.pipeline import Pipeline
from nlpbox.estimators.classification.extra_trees_classifier import \
    ExtraTreesClassifier
from nlpbox.features.cohmetrix import CohMetrixExtractor


class CohMetrixExtraTreesClassification(Pipeline):
    def __init__(self,
                 random_state: int | None = None,
                 etree_config: dict() = None):
        super().__init__(
            vectorizer=CohMetrixExtractor(),
            estimator=ExtraTreesClassifier(**etree_config,
                                           random_state=random_state),
            name=f'cohmetrix_etree_clf_seed{random_state}')
