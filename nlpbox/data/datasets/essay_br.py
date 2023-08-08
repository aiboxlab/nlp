"""Esse módulo contém o dataset
Essay-BR (versão original e estendida)
com redações do Ensino Médio.
"""
from __future__ import annotations

import json
from enum import Enum
from typing import ClassVar

import pandas as pd

from nlpbox import resources
from nlpbox.core import Dataset

from . import utils


class DatasetEssayBR(Dataset):
    def __init__(self,
                 extended: bool,
                 target_competence: str):
        """Construtor. Permite selecionar qual
        competência deve ser utilizada pelo dataset
        e qual versão deve ser utilizada (original
        ou estendida).

        As versões utilizadas pela biblioteca se encontram
        disponíveis nos repositórios originais do GitHub:
            - https://github.com/rafaelanchieta/essay/tree/master/essay-br
                - Commit: da35364a0e213310ce83e55a613fbaa58d134bd3
            - https://github.com/lplnufpi/essay-br/tree/main/extended-corpus
                - Commit: fb6391a79cbb12dff877eb442c2a31caa7f00c77

        São aplicados alguns pós-processamentos visto que os dados originais
        possuem redações duplicadas e/ou faltantes.

        Args:
            extender (bool): se devemos utilizar a versão estendida.
            target_competence (str): competência ('C1', 'C2', 'C3',
                'C4', 'C5' ou 'score').
        """
        target_resource = 'essay-br-extended' if extended else 'essay-br'
        root_dir = resources.path(f'datasets/{target_resource}.v1')
        self._target = target_competence
        self._df = pd.read_csv(root_dir.joinpath('dataset.csv'))

        # Garantindo que o DataFrame possui os dados
        #   necessários.
        assert self._target in self._df.columns
        assert 'text' in self._df.columns

        # Pós-processamentos
        # 1. Remoação de vazios (NaN, Nulls, etc)
        self._df.dropna(ignore_index=True,
                        inplace=True)

        # 2. Remoção de redações duplicadas
        self._df.drop_duplicates(subset='text',
                                 ignore_index=True,
                                 inplace=True)

        # Adicionando nova coluna com o target
        self._df['target'] = self._df[self._target]

        # Reorganizando a ordem do DataFrame: text, target
        #   vem primeiro e depois as colunas faltantes.
        cols = list(self._df.columns)
        cols.remove('text')
        cols.remove('target')
        self._df = self._df[['text', 'target'] + cols]

    @property
    def competence(self) -> str:
        return self._target

    def to_frame(self):
        return self._df.copy()

    def cv_splits(self, k: int,
                  stratified: bool,
                  seed: int) -> list[pd.DataFrame]:
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
        return utils.stratified_splits_clf(df=self._df,
                                           k=k,
                                           seed=seed)

    def train_test_split(self,
                         frac_train: float,
                         seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        return utils.train_test_clf(df=self._df,
                                    frac_train=frac_train,
                                    seed=seed)
