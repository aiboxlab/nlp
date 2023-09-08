"""Arquivo de inicialização.
"""
from platformdirs import user_data_dir
import os
from pathlib import Path
import logging

# Configurando logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# Obtendo diretório de dados da biblioteca
DATA_DIR = os.environ.get('NLPBOX_DATA',
                          user_data_dir('aibox-nlpbox',
                                        'aibox'))
DATA_DIR = Path(DATA_DIR).resolve().absolute()

# Limpando variáveis auxiliares do namespace
del os
del Path
del user_data_dir
del logger
