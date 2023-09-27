"""Esse módulo contém uma classe adapter
para representar DataFrames como
Datasets.
"""
from __future__ import annotations

import pandas as pd
from pandas.api import types

from aibox.nlp.core import Dataset

from . import utils


class DatasetDF(Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 text_column: str,
                 target_column: str,
                 copy: bool = True,
                 drop_others: bool = False):
        """Construtor. Recebe o DataFrame as informações
        sobre as colunas e realiza uma mudança no nome
        das colunas de texto e target.

        Args:
            df (pd.DataFrame): dataframe com os dados.
            text_column (str): coluna que possui os textos.
            target_column (str): coluna com os valores target.
            copy (bool): se devemos armazenar uma cópia do
                DataFrame (default=True).
            drop_others (bool): se devemos remover outras
                colunas que não sejam as de texto e
                target (default=False).
        """
        assert text_column in df.columns, 'Coluna não encontrada.'
        assert target_column in df.columns, 'Coluna não encontrada.'
        assert len(df) > 0, 'DataFrame não pode ser vazio.'

        if copy:
            df = df.copy()

        self._df = df.rename(columns={text_column: 'text',
                                      target_column: 'target'})

        if drop_others:
            columns = set(self._df.columns.tolist())
            columns.remove('text')
            columns.remove('target')
            self._df.drop(columns=columns,
                          inplace=True)

        has_duplicates = self._df.text.duplicated().any()
        has_na_text = self._df.text.isnull().any()
        is_numeric = types.is_numeric_dtype(self._df.target.dtype)
        has_na_target = self._df.target.isnull().any()
        assert not has_na_text, 'Não devem existir textos NULL.'
        assert not has_duplicates, 'Não devem existir textos duplicados.'
        assert not has_na_target, 'Não devem existir targets NULL.'
        assert is_numeric, 'Coluna "target" deve ser numérica.'

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    def to_frame(self):
        return self._df.copy()

    def cv_splits(self, k: int,
                  stratified: bool,
                  seed: int) -> list[pd.DataFrame]:
        if stratified and self._is_classification():
            return utils.stratified_splits_clf(df=self._df,
                                               k=k,
                                               seed=seed)

        return utils.splits(df=self._df,
                            k=k,
                            seed=seed)

    def train_test_split(self,
                         frac_train: float,
                         stratified: bool,
                         seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        if stratified and self._is_classification():
            return utils.train_test_clf(df=self._df,
                                        frac_train=frac_train,
                                        seed=seed)

        return utils.train_test(df=self._df,
                                frac_train=frac_train,
                                seed=seed)

    def _is_classification(self) -> bool:
        return types.is_integer_dtype(self._df.target.dtype)
