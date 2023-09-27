"""Módulo com funcionalidade para obtenção
de modelos e artefatos.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from zipfile import ZipFile

import requests
from tqdm import tqdm

from aibox.nlp import DATA_DIR


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
    return _resource_path(artifact)


def _get_url(artifact: str):
    """Retorna a URL pública para esse
    artefato no bucket do GCP.
    """
    prefix = 'https://storage.googleapis.com'
    bucket = 'aibox-nlpbox'
    return f'{prefix}/{bucket}/{artifact}.zip'


def _download_to_filesystem(artifact: str,
                            url: str,
                            target: Path):
    """Realiza o download do zip dessa URL
    para o diretório indicado.
    """
    size = int(requests.head(url).headers['Content-Length'])
    chunk_size = int(5e7)  # ~50MB

    if size < chunk_size:
        chunk_size = max(int(1e6),
                         size // 10)

    # Realizando download
    with requests.get(url,
                      stream=True,
                      allow_redirects=True) as r:
        r.raise_for_status()

        # Criando arquivo temporário para o zip
        with tempfile.NamedTemporaryFile(suffix='.zip') as temp:
            pbar = tqdm(desc=f'nlpbox: download {artifact}',
                        unit_divisor=1024,
                        unit='B',
                        unit_scale=True,
                        total=size)

            # Fazendo o download de cada chunk
            for chunk in r.iter_content(chunk_size=chunk_size):
                temp.write(chunk)
                pbar.update(len(chunk))

            # Salvando no sistema de arquivos
            with ZipFile(temp, 'r') as zip:
                zip.extractall(str(target))


def _resource_path(artifact: str) -> Path:
    # Obtendo diretório target para esse
    #   recurso.
    parts = artifact.split('/')
    target_dir = DATA_DIR.joinpath(*parts)

    if not target_dir.exists():
        # Criando diretório
        target_dir.mkdir(exist_ok=True,
                         parents=True)

        # Obtendo URL pública
        url = _get_url(artifact)

        # Realizando download para
        # o sistema de arquivos
        try:
            _download_to_filesystem(artifact,
                                    url,
                                    target_dir)
        except Exception as e:
            shutil.rmtree(target_dir)
            raise e

    return target_dir
