"""Módulo com funcionalidade para obtenção
de modelos e artefatos.
"""
from __future__ import annotations

from pathlib import Path

from nlpbox.lazy_loading import lazy_import

apart = lazy_import('apart')
_manager = None


def path(artifact: str) -> Path:
    """Retorna o caminho local para o artefato
    representado pela string passada. Caso o artefato
    não esteja disponível localmente, realiza o download.

    Args:
        artifact (str): identificador do artefato.

    Returns:
        str: caminho local desse artefato.
    """
    if _manager is None:
        _manager = apart.GoogleCloudArtifactManager(
            bucket='aibox-nlpbox')

    # Retorna o caminho local desse artefato e
    #   realiza o download caso necessário.
    return Path(manager.get(artifact=artifact))
