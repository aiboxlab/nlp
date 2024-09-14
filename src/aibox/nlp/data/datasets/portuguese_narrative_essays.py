"""Esse módulo contém o dataset
considerando as redações do Ensino Fundamental
do projeto do MEC.
"""

from __future__ import annotations

import pandas as pd

from aibox.nlp import resources
from aibox.nlp.core import Dataset

from . import utils


class DatasetPortugueseNarrativeEssays(Dataset):
    def __init__(self, target_competence: str):
        """Construtor. Permite selecionar qual
        competência deve ser utilizada pelo dataset.

        A versão utilizada aqui é a unificação de todos splits
        presentes em:
            - https://www.kaggle.com/datasets/moesiof/portuguese-narrative-essays

        Args:
            target_competence (str): competência ('cohesion',
                'thematic_coherence', 'formal_register', 'text_typology').
        """
        # Carregamento do Dataset
        root_dir = resources.path("datasets/portuguese-narrative-essays.v1")
        self._target = target_competence
        self._df = pd.concat([pd.read_csv(p) for p in root_dir.rglob("*.csv")])

        # Adicionando e renomeando colunas
        self._df = self._df.rename(columns=dict(essay="text"))
        self._df["target"] = self._df[self._target]

        # Reorganizando a ordem do DataFrame: text, target
        #   vem primeiro e depois as colunas faltantes.
        cols = list(self._df.columns)
        cols.remove("text")
        cols.remove("target")
        self._df = self._df[["text", "target"] + cols]

    @property
    def competence(self) -> str:
        return self._target

    def to_frame(self):
        return self._df.copy()

    def cv_splits(self, k: int, stratified: bool, seed: int) -> list[pd.DataFrame]:
        """Obtém os splits para validação cruzada. Todos
        os splits são estratificados.

        Args:
            k (int): quantidade de splits/folds.
            stratified (bool): desconsiderado, sempre estratificado.
            seed (int): seed randômica para geração dos folds.

        Returns:
            list[pd.DataFrame]
        """
        del stratified
        return utils.stratified_splits_clf(df=self._df, k=k, seed=seed)

    def train_test_split(
        self, frac_train: float, stratified: bool, seed: int
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Obtém os conjuntos de treino e teste desse Dataset como
        DataFrames.

        Args:
            frac_train (float): fração de amostras para treinamento.
            stratified (bool): desconsiderado, sempre estratificado.
            seed (int): seed randômica para geração dos folds.

        Returns:
            (df_train, df_test): tupla com os conjuntos de treino
                e teste para esse dataset.
        """
        del stratified
        return utils.train_test_clf(df=self._df, frac_train=frac_train, seed=seed)
