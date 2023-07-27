"""Módulo com funcionalidade para obtenção
de modelos e artefatos.
"""
from __future__ import annotations

from pathlib import Path

import apart

manager = apart.GoogleCloudArtifactManager(
    bucket='aibox-nlpbox'
)


def path(artifact: str) -> Path:
    """Retorna o caminho local para o artefato
    representado pela string passada. Caso o artefato
    não esteja disponível localmente, realiza o download.

    Args:
        artifact (str): identificador do artefato.

    Returns:
        str: caminho local desse artefato.
    """
    # Retorna o caminho local desse artefato e
    #   realiza o download caso necessário.
    return Path(manager.get(artifact=artifact))
