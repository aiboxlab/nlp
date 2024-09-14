"""Módulo para serialização de pipelines.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import joblib

from aibox.nlp.core import Pipeline


def save_pipeline(
    pipeline: Pipeline, save_path: str | Path, method: Literal["joblib"] = "joblib"
):
    """Realiza o salvamento de uma pipeline
    para o disco.

    Args:
        pipeline (Pipeline): instância da pipeline.
        save_path (str | Path): caminho de salvamento.
        method ('joblib', optional): método de serialização,
            atualmente apenas o joblib é
            suportado. (default='joblib')
    """
    del method
    joblib.dump(pipeline, save_path)


def load_pipeline(path: str | Path) -> Pipeline:
    """Realiza o carregamento de uma pipeline.

    Args:
        path (str | Path): caminho para pipeline.

    Returns:
        Pipeline: pipeline carregada.
    """
    obj = joblib.load(path)
    assert isinstance(obj, Pipeline)

    return obj
