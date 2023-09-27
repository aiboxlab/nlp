"""Módulo com funções utilitárias
para o cálculo das métricas.
"""
from __future__ import annotations

from typing import Callable

import numpy as np


def to_float32_array(fn: Callable) -> Callable:
    """Wrapper que recebe um Callable
    e retorna um Callabel que converte
    as saídas para NumPy Array de floats.

    Args:
        fn: função.

    Returns:
        Nova função que converte a saída para
            NumPy array de float32.
    """

    def wrapper(*args, **kwargs):
        out = fn(*args, **kwargs)

        if not isinstance(out, np.ndarray):
            out = np.array(out, dtype=np.float32)

        return out.astype(np.float32)

    return wrapper
