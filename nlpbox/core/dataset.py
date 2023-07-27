"""Esse módulo contém a definição
de uma interface básica para datasets.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class Dataset(ABC):
    """Essa classe representa um Dataset
    para classificação ou regressão.
    """

    @abstractmethod
    def to_frame(self) -> pd.DataFrame:
        """Converte esse dataset para
        um DataFrame com as colunas:
            1. text: textos;
            2. target: label;

        Returns:
            Retorna uma representação desse dataset
                como um DataFrame.
        """

    @abstractmethod
    def cv_splits(self,
                  k: int,
                  stratified: bool,
                  seed: int) -> list[pd.DataFrame]:
        """Retorna splits para serem utilizados. Esse método
        particiona o dataset em `k` partes aleatórias de tamanho
        similar.

        Args:
            k (int): quantidade de splits.
            stratified: se cada split deve ser estratificado.
            seed: seed randômica para geração dos splits.

        Returns:
            Lista com `k` DataFrames.
        """

    @abstractmethod
    def train_test_split(self,
                         frac_train: float,
                         seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Obtém os conjuntos de treino e teste desse Dataset como
        DataFrames.

        Args:
            frac_train (float): fração de amostras para treinamento.
            seed (int): seed randômica para geração dos cojuntos.

        Returns:
            (df_train, df_test): tupla com os conjuntos de treino
                e teste para esse dataset.

        """
