"""Esse módulo contém interfaces
relativas aos datasets e corpus
disponibilizados.
"""
from __future__ import annotations

from typing import Protocol

import pandas as pd


class Dataset(Protocol):
    """Representa a interface básica
    de um dataset.

    Todo dataset possui um conjunto de textos
    e seus respectivos targets que podem ser
    utilizados para classificação ou regressão.
    """

    def as_dataframe(self) -> pd.DataFrame:
        """Retorna uma representação desse
        dataset como um DataFrame. O DataFrame
        sempre vai possuir 2 colunas:
            - `text`: com o texto;
            - `target`: com o valor esperado para
                aquele texto;

        Returns:
            Representação do dataset como um
                DataFrame do Pandas.
        """
