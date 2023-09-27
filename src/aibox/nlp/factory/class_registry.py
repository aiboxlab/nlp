"""Esse módulo contém a implementação
de um Class Registry, permitindo que classes
sejam armazenados como uma string.
"""
from __future__ import annotations

import importlib
import json

try:
    from importlib.resources import files
except ImportError:
    # Python < 3.9 doesn't have the 
    #   same files(...) method.
    # Instead, we use the one provided
    # by the importlib_resources library
    from importlib_resources import files

class _Registry:
    def __init__(self) -> None:
        p = files('aibox.nlp.factory').joinpath("registry.json")
        with p.open('r', encoding='utf-8') as f:
            self._reg = json.load(f)

        self._reg['global'] = dict()
        global_prefix = 'aibox.nlp.'
        for key, prefix in zip(['features_br',
                                'vectorizers',
                                'estimators',
                                'metrics',
                                'datasets'],
                               ['features.portuguese.{0}',
                                'vectorizers.{0}',
                                'estimators.{0}',
                                'metrics.{0}',
                                'data.datasets.{0}']):
            # k é o ID da classe
            # v é o caminho relativo
            for k, v in self._reg[key].items():
                # full_v contém o caminho completo desde
                #   a raiz do pacote até a classe.
                full_v = global_prefix + prefix.format(v)

                # Garantindo que essa chave é única
                assert k not in self._reg['global']

                # Adicionamos esse novo caminho na chave global
                #   do registro.
                self._reg['global'][k] = full_v

    def get_registry_for(self, kind: str) -> dict[str, str]:
        """Retorna as entradas do registro para
        um dado tipo.

        Args:
            kind (str): tipo das entradas.

        Returns:
            dict[str, str]: registro para esse tipo.
        """
        return self._reg[kind]

    def get_path(self, identifier: str) -> str:
        """Dado um identificador para uma classe,
        retorna o caminho até essa classe seguindo
        o padrão de import de Python (e.g., 
        package_a.package_b.module_c.ClassD)

        Args:
            identifier (str): identificador.

        Returns:
            caminho até a classe.
        """
        return self._reg['global'][identifier]


_registry = _Registry()


def get_class(key: str) -> type:
    """Retorna a classe do identificador
    recebido como argumento.

    Args:
        key (str): identificador.

    Returns:
        Classe.
    """
    # Obtendo nome do pacote e classe
    class_path = _registry.get_path(key)
    splits = class_path.rsplit('.', 1)
    module_name = splits[0]
    class_name = splits[1]

    # Carregando módulo
    module = importlib.import_module(module_name)

    # Obtendo classe dentro desse módulo
    cls = getattr(module, class_name)

    return cls
