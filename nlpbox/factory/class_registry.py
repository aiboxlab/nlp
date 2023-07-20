"""Esse módulo contém a implementação
de um Class Registry como um decorador,
permitindo que classes sejam armazenados
como uma string (relação de 1:1).
"""
from __future__ import annotations

import typing


_registry = dict()


def register(key: str) -> typing.Callable[[type], type]:
    """Decorador que permite registrar uma classe
    com um identificador único.

    Args:
        key (str): identificador da classe.

    Raises:
        ValueError: caso o identificador já tinha
            sido registrador por outra classe.

    Returns:
        Callable que recebe uma classe, armazena
            essa classe no registro, e retorna a
            própria classe.
    """
    def _register_return_cls(t):
        if key in _registry:
            raise ValueError('Já foi registrada uma classe com '
                             f'nome "{key}".')

        _registry[key] = t
        return t

    return _register_return_cls


def get_class(key: str) -> type:
    """Retorna a classe do identificador
    recebido como argumento.

    Args:
        key (str): identificador.

    Returns:
        Classe.
    """
    return _registry[key]
