"""Esse módulo implementa
um Lazy Loader para módulos.
"""

import importlib.util
import sys


def lazy_import(name: str):
    """Realiza o lazy import de um módulo.

    Args:
        name (str): nome do módulo.

    Returns:
        Lazy module.
    """
    spec = importlib.util.find_spec(name)
    loader = importlib.util.LazyLoader(spec.loader)
    spec.loader = loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    loader.exec_module(module)
    return module
