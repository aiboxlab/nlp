"""Esse módulo contém funções utilitárias
para a construção e obtenção de pipelines.
"""

from __future__ import annotations

from aibox.nlp.core import Estimator, Pipeline, Vectorizer

from .class_registry import get_class


def get_pipeline(pipeline: str, pipeline_config: dict = dict()) -> Pipeline:
    """Carrega uma pipeline com o nome passado.

    Args:
        pipeline (str): nome da pipeline.
        pipeline_config (dict, opcional): configuração da pipeline.

    Returns:
        Pipeline.
    """
    pipeline = get_class(pipeline)(**pipeline_config)
    assert isinstance(pipeline, Pipeline)

    return pipeline


def make_pipeline(
    vectorizer: str,
    estimator: str,
    vectorizer_config: dict = dict(),
    estimator_config: dict = dict(),
) -> Pipeline:
    """Constrói uma pipeline dado o vetorizador
    e estimador a serem utilizados.

    Args:
        vectorizer (str): nome do vetorizador.
        estimator (str): nome do estimador.
        vectorizer_config (dict, opcional): configurações do
            vetorizador (passadas ao construtor).
        estimator_config (dict, opcional): configurações do
            estimador (passadas ao construtor).

    Returns:
        Pipeline com os parâmetros selecionados.
    """
    vectorizer = get_class(vectorizer)(**vectorizer_config)
    estimator = get_class(estimator)(**estimator_config)

    assert isinstance(vectorizer, Vectorizer)
    assert isinstance(estimator, Estimator)

    return Pipeline(vectorizer, estimator)
