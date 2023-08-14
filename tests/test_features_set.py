"""Esse é um teste para garantir
que os identificadores das features são
únicos em toda a biblioteca.
"""
import inspect
from dataclasses import fields

import nlpbox.features
from nlpbox.core import FeatureSet
from nlpbox.features.utils import DataclassFeatureSet

_UTILITY_FEATURE_SETS = {DataclassFeatureSet}
_SKIP_MODULES = {'utils'}


def test_no_duplicated_features():
    """Realiza um teste para garantir que
    todos os identificadores de características
    são únicos.
    """
    global_ids = []

    for name, module in inspect.getmembers(nlpbox.features,
                                           predicate=inspect.ismodule):
        if name in _SKIP_MODULES:
            # Alguns módulos não devem ser checados
            continue

        # Colentado todas as classes presentes nesse módulo
        classes = [m for _, m in inspect.getmembers(module,
                                                    predicate=inspect.isclass)]

        # Coletando todas as classes que implementam um FeatureSet
        # OBS:. excluímos FeatureSet's utilitários que podem ter
        #   sido importados
        feature_sets = [c for c in classes
                        if (c not in _UTILITY_FEATURE_SETS)
                        and issubclass(c, FeatureSet)]

        # Condição da biblioteca: 1 módulo só pode ter 1 feature set
        assert len(feature_sets) == 1, f'{name}, {feature_sets}'

        # Obtemos o feature set
        fs = feature_sets[0]

        # Salvamos os nomes de features desse feature set na lista
        #   de identificadores global.
        global_ids.extend([field.name for field in fields(fs)])

    # Garantimos que não existem IDs duplicados
    assert len(global_ids) == len(set(global_ids))
