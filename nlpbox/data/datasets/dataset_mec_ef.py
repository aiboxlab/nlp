"""Esse módulo contém o dataset
considerando as redações do Ensino Fundamental
do projeto do MEC.
"""
from __future__ import annotations

import json
from enum import Enum
from typing import ClassVar

import pandas as pd

from nlpbox import resources
from nlpbox.core import Dataset

from . import utils


class DatasetMecEf(Dataset):
    _COMPETENCES: ClassVar[set[str]] = {
        'cohesion',
        'thematic_coherence',
        'formal_register',
        'text_typology'
    }
    _KEY_MOTIV_SITUATION: ClassVar[str] = 'motivating_situation'
    _KEY_TEXT: ClassVar[str] = 'text'
    _KEY_COMPETENCES: ClassVar[str] = 'consolidated_competences'

    def __init__(self,
                 target_competence: str):
        """Construtor. Permite selecionar qual
        competência deve ser utilizada pelo dataset.

        Args:
            target_competence (str): competência.
        """
        root_dir = resources.path('datasets/corpus-mec-ef.v1')
        json_path = root_dir.joinpath('dataset.json')
        with json_path.open('r', encoding='utf-8') as f:
            json_data = json.load(f)

        data = {
            self._KEY_TEXT: [],
            'target': [],
            self._KEY_MOTIV_SITUATION: []
        }

        data.update({k: [] for k in self._COMPETENCES})

        for entry in json_data:
            data[self._KEY_TEXT].append(entry[self._KEY_TEXT])
            data[self._KEY_MOTIV_SITUATION].append(
                entry[self._KEY_MOTIV_SITUATION])
            competences = entry[self._KEY_COMPETENCES]

            for c in self._COMPETENCES:
                data[c].append(competences[c])

            data['target'].append(competences[target_competence])

        self._target = target_competence
        self._df = pd.DataFrame(data)

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
