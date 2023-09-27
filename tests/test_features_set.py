"""Esse é um teste para garantir
que os identificadores das features são
únicos em toda a biblioteca.
"""
import importlib
import inspect
import pkgutil
from dataclasses import fields

import aibox.nlp.features
from aibox.nlp.core import FeatureSet
from aibox.nlp.features.utils import DataclassFeatureSet

_UTILITY_FEATURE_SETS = {DataclassFeatureSet}
_SKIP_MODULES = {'utils'}  


def test_no_duplicated_features():
    """Realiza um teste para garantir que
    todos os identificadores de características
    são únicos.
    """
    global_ids = []

    # Supõem que a estrutura do pacote `nlpbox.features` é:
    # __init__.py
    # feature1.py
    # feature2.py
    # feature3.py
    # subpackage
    #   non_feature.py
    # feature4.py
    # ....
    modules_info = [m for m in pkgutil.walk_packages(
        aibox.nlp.features.__path__)]

    for module_info in modules_info:
        name = module_info.name
        if name in _SKIP_MODULES:
            # Alguns módulos não devem ser checados
            continue

        # Importando módulo
        module = importlib.import_module(f'aibox.nlp.features.{name}')

        # Coletando todas as classes presentes nesse módulo
        classes = [m for _, m in inspect.getmembers(module,
                                                    predicate=inspect.isclass)]

        # Caso esse módulo não possua classe,
        #   não precisamos testar.
        if len(classes) < 1:
            continue

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

        # Obtemos os identificadores de features
        ids = [field.name for field in fields(fs)]

        # Garantimos que não existe duplicada
        assert len(ids) == len(set(ids))

        # Salvamos os nomes de features desse feature set na lista
        #   de identificadores global.
        global_ids.extend(ids)

    # Garantimos que não existem IDs duplicados
    assert len(global_ids) == len(set(global_ids))
