"""Esse módulo contém funções e classes
utilitárias para criação de Datasets.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def train_test_clf(df: pd.DataFrame,
                   frac_train: float,
                   seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retorna uma tupla com os DataFrames de treino
    e teste para o dataset de classificação recebido. Ambos splits
    são estratificados com relação a distribuição das classes.

    Args:
        df (pd.DataFrame): dataset.
        frac_train (float): porcentagem de amostras para treino.
        seed (int): seed randômica para geração dos splits.

    Returns:
        df_train, df_test
    """
    # Garantindo pré-condições
    _assert_preconditions(df)

    # Criação de um gerador auxiliar
    rng = np.random.default_rng(seed)

    def _sample(d):
        # Retorna um sub-dataframe
        # Selecionamos uma seed randômica para obter o sample
        random_state = rng.integers(0,
                                    int(1e6),
                                    size=1).item()
        # Escolhemos uma parcela aleatória do DataFrame d
        d_ = d.sample(frac=frac_train,
                      random_state=random_state)
        return d_

    # Agrupar os dados de acordo com os níveis
    groupby = df.groupby('target',
                         group_keys=False)

    # Obtemos (100*frac_train)% de amostras de cada grupo
    df_train = groupby.apply(_sample)

    # O que sobrou, faz parte do conjunto de testes
    df_test = df[~df.text.isin(df_train.text)]

    # Garantindo que o conjunto de textos
    #   se manteve.
    _assert_same_texts([df_train, df_test], df)

    # Retornamos train e test
    return df_train, df_test


def train_test(df: pd.DataFrame,
               frac_train: float,
               seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retorna uma tupla com os DataFrames de treino
    e teste para o dataset de classificação recebido.

    Args:
        df (pd.DataFrame): dataset.
        frac_train (float): porcentagem de amostras para treino.
        seed (int): seed randômica para geração dos splits.

    Returns:
        df_train, df_test
    """
    # Garantindo pré-condições
    _assert_preconditions(df)

    # Obtemos (100*frac_train)% de amostras de cada grupo
    df_train = df.sample(frac=frac_train,
                         replace=False,
                         random_state=seed)

    # O que sobrou, faz parte do conjunto de testes
    df_test = df[~df.text.isin(df_train.text)]

    # Garantindo que o conjunto de textos
    #   se manteve.
    _assert_same_texts([df_train, df_test], df)

    # Retornamos train e test
    return df_train, df_test


def stratified_splits_clf(df: pd.DataFrame,
                          k: int,
                          seed: int) -> list[pd.DataFrame]:
    """Retorna `k` splits estratitifcados para o dataset
    recebido.

    Args:
        df (pd.DataFrame): dataset.
        k (int): quantidade de splits.
        seed (int): seed randômica.

    Returns:
        list[pd.DataFrame]
    """
    # Garantindo pré-condições
    _assert_preconditions(df)

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
    samples_leftovers = (count - (samples_classes_int * k)).to_dict()

    def _sample(d: pd.DataFrame, rand: int) -> pd.DataFrame:
        nonlocal samples_leftovers

        level = d[col_classes].unique().item()
        leftover = 0

        # Caso existam "leftovers", adicionamos nesse fold um deles
        if samples_leftovers[level] > 0:
            samples_leftovers[level] -= 1
            leftover = 1

        return d.sample(n=samples_per_class[level] + leftover,
                        random_state=rand)

    # Criação dos folds (exceto último)
    for i in range(k - 1):
        # Selecionamos uma seed randômica para obter o sample
        rand = rng.integers(0,
                            int(1e6),
                            size=1).item()
        fold = groupby.apply(lambda d: _sample(d, rand))
        folds.append(fold)

        # Remoção das amostras desse fold do DataFrame original
        df_ = df_[~df_[col_text].isin(fold[col_text])]
        groupby = df_.groupby(col_classes,
                              group_keys=False)

    # O que sobrou do DataFrame faz parte do último fold
    folds.append(df_)

    # Garantindo que o conjunto de textos
    #   se manteve.
    _assert_same_texts(folds, df)

    return folds


def splits(df: pd.DataFrame,
           k: int,
           seed: int) -> list[pd.DataFrame]:
    """Retorna `k` splits para o dataset
    recebido.

    Args:
        df (pd.DataFrame): dataset.
        k (int): quantidade de splits.
        seed (int): seed randômica.

    Returns:
        list[pd.DataFrame]
    """
    # Garantindo pré-condições
    _assert_preconditions(df)

    # Criação de um gerador auxiliar
    rng = np.random.default_rng(seed)

    # Lista para armazenar os folds
    folds = []

    # Fração de amostras por fold
    frac_per_fold = 1 / k

    # Auxiliares
    col_text = 'text'
    df_ = df

    # Criação dos folds (exceto último)
    for _ in range(k - 1):
        # Selecionamos uma seed randômica para obter o sample
        rand = rng.integers(0,
                            int(1e6),
                            size=1).item()
        fold = df_.sample(frac=frac_per_fold,
                          replace=False,
                          random_state=rand)
        folds.append(fold)

        # Remoção das amostras desse fold do DataFrame original
        df_ = df_[~df_[col_text].isin(fold[col_text])]

    # O que sobrou do DataFrame faz parte do último fold
    folds.append(df_)

    # Garantindo que o conjunto de textos
    #   se manteve.
    _assert_same_texts(folds, df)

    return folds


def _assert_preconditions(df: pd.DataFrame):
    assert not df.text.duplicated().any()
    assert not df.text.isnull().any()
    assert not df.target.isnull().any()


def _assert_same_texts(objs: list[pd.DataFrame],
                       original_df: pd.DataFrame):
    # Garantindo que o somatório de todos folds
    #   é igual ao original
    assert sum(map(len, objs)) == len(original_df)

    # Garantindo que o conjunto de textos
    #   é igual ao original
    concat = pd.concat(objs)
    concat = concat.sort_values(by='text')
    df_ = original_df.sort_values(by='text')
    assert (concat.text == df_.text).all()
