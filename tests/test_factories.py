"""Teste para o registro de classes
utilizadas pelo pacote factory.

Esse é um teste white-box, só
precisamos garantir que toda chave
no _registry vira uma classe e que não
existem identificadores duplicados.
"""
from __future__ import annotations

import pytest

from nlpbox.core import (Dataset, Estimator, FeatureExtractor, Metric,
                         Vectorizer)
from nlpbox.factory import class_registry

_REGISTRIES = [class_registry._registry_datasets,
               class_registry._registry_estimators,
               class_registry._registry_features,
               class_registry._registry_metrics,
               class_registry._registry_vectorizers]

_TARGET_CLS = [Dataset,
               Estimator,
               FeatureExtractor,
               Metric,
               Vectorizer]


@pytest.mark.parametrize("registry,parent_class",
                         [e for e in zip(_REGISTRIES, _TARGET_CLS)])
def test_names_for_factories(registry: dict[str, str],
                             parent_class: type):
    """Realiza o teste de todas chaves
    cadastradas em um dado registro de classes.

    Args:
        registry (dict[str, str]): registro de classes, que
            mapeia um nome único para um caminho de classes.
        parent_class (type): classe pai das classes retornadas
            nesse registro.
    """
    collected = []

    for key in registry:
        # Tentamos obter a classe com esse nome
        cls = class_registry.get_class(key)

        # Garantimos que uma classe foi retornada
        assert cls is not None

        # Garantimos que essa classe é sub-classe de
        #   uma classe pai.
        assert issubclass(cls, parent_class)

        # Guardamos essa classe em uma lista
        collected.append(cls)

    # Ao final de tudo, não podemos ter encontrado
    #   uma classe repetida (um registro precisa ter
    #   relação de 1:1)
    assert len(collected) == len(set(collected))


def test_no_duplicated_keys():
    """Realiza um teste para garantir que não
    existem chaves duplicadas em alguma das variáveis
    de registro de classes.
    """
    # Coletando o total de chaves presente
    #   no registro final
    complete_registry = class_registry._registry
    total_keys = len(complete_registry)

    # Coletando o total de chaves presentes
    #   nos registros de classes individuais
    total_individual_keys = sum(map(len, _REGISTRIES))

    # O somatório de chaves individuais tem que
    #   ser igual ao total.
    assert total_keys == total_individual_keys