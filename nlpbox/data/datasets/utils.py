"""Esse módulo contém funções e classes
utilitárias para criação de Datasets.
"""
from __future__ import annotations

import pandas as pd

import numpy as np


def _train_test_clf(df: pd.DataFrame,
                    frac_train: float,
                    seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Criação de um gerador auxiliar
    rng = np.random.default_rng(seed)

    def _sample(d):
        # Selecionamos uma seed randômica para obter o sample
        random_state = seed_generator.integers(0,
                                               int(1e10),
                                               size=1).item()
        # Escolhemos uma parcela aleatória do DataFrame d
        d_ = d.sample(frac=frac_train,
                      random_state=seed_generator.integers(0,))
        return d_

    # Agrupar os dados de acordo com os níveis
    groupby = df.groupby('target',
                         group_keys=False)

    # Obtemos (100*frac_train)% de amostras de cada grupo
    df_train = groupby.apply(sample)

    # O que sobrou, faz parte do conjunto de testes
    df_test = df[~df['text'].isin(df_train['text'])]

    # Retornamos train e test
    return df_train, df_test


def _stratified_splits_clf(df: pd.DataFrame,
                           k: int,
                           seed: int) -> list[pd.DataFrame]:
    # Criação de um gerador auxiliar
    rng = np.random.default_rng(seed)

    # Lista para armazenar os folds
    folds = []

    # Fração de amostras por fold
    frac_per_fold = 1 / k

    # Auxiliares
    col_text = 'text'
    col_classes = 'target'
    groupby = df.groupby(col_classes,
                         group_keys=False)
    df_ = df

    # Contabilizando a quantidade de amostras por classe
    count = groupby[col_text].count()

    # Calculando a quantidade de amostras que devemos obter
    #   para cada classe.
    samples_classes = count * frac_per_fold
    samples_classes_int = samples_classes.apply(int)
    samples_per_class = samples_classes_int.to_dict()

    # Quantidade de que vão "sobrar" para cada classe
    # A quantidade de faltantes por classe sempre vai ser < k
    samples_leftovers = (count - (samples_level_int * n_folds)).to_dict()

    def _sample(d: pd.DataFrame, rand: int) -> pd.DataFrame:
        nonlocal samples_leftovers

        level = d[col_classes].unique().item()
        leftover = 0

        # Caso existam "leftovers", adicionamos nesse fold um deles
        if samples_leftovers[level] > 0:
            samples_leftovers[level] -= 1
            leftover = 1

        return d.sample(n=samples_per_level[level] + leftover,
                        random_state=rand)

    # Criação dos folds (exceto último)
    for i in range(n_folds - 1):
        # Selecionamos uma seed randômica para obter o sample
        rand = seed_generator.integers(0,
                                       int(1e10),
                                       size=1).item()
        fold = groupby.apply(lambda d: _sample(d, rand))
        folds.append(fold)

        # Remoção das amostras desse fold do DataFrame original
        df_ = df_[~df_[col_text].isin(fold[col_text])]
        groupby = df_.groupby(col_classes,
                              group_keys=False)

    # O que sobrou do DataFrame faz parte do último fold
    folds.append(df_)

    return folds
