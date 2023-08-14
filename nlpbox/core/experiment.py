"""Esse módulo contém a definição
da interface para um experimento.
"""
from __future__ import annotations

import typing
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .dataset import Dataset
from .metric import Metric
from .pipeline import Pipeline


@dataclass(frozen=True)
class ExperimentResult:
    best_pipeline: Pipeline
    best_pipeline_name: str
    best_metrics: dict[str, np.ndarray]
    best_pipeline_test_predictions: np.ndarray
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    metrics_history: dict[str, dict[str, np.ndarray]]
    pipeline_history: dict[str, Pipeline | None]
    extras: object | None = None


@dataclass(frozen=True)
class ExperimentConfiguration:
    dataset: Dataset
    metrics: list[str]
    best_criteria: str
    extras: object | None = None


class Experiment(ABC):
    @abstractmethod
    def run(self) -> ExperimentResult:
        """Executa o experimento e retorna
        os resultados.

        Returns:
            ExperimentResult: resultados do
                experimento.
        """

    @abstractmethod
    def config(self) -> ExperimentConfiguration:
        """Retorna as configurações desse experimento.

        Returns:
            ExperimentConfiguration: configuração do
                experimento.
        """
