"""Arquivo de inicialização.
"""
import logging
import logging.config
import os
from pathlib import Path

from platformdirs import user_data_dir

LOGGING = {
    'version': 1,
    'formatters': {
        'brief': {
            'format': '[{levelname}] [{name}] {message}',
            'style': '{',
        }
    },
    'handlers': {
        'console': {
            'level': 'INFO',
            'class': 'logging.StreamHandler',
            'formatter': 'brief',
        },
    },
    'loggers': {
        'aibox.nlp': {
            'handlers': ['console'],
            'level': 'INFO',
            'propagate': False,
        }
    }
}

# Configurando logging
logging.config.dictConfig(LOGGING)


# Obtendo diretório de dados da biblioteca
DATA_DIR = os.environ.get('NLPBOX_DATA',
                          user_data_dir('aibox-nlpbox',
                                        'aibox'))
DATA_DIR = Path(DATA_DIR).resolve().absolute()

# Limpando variáveis auxiliares do namespace
del os
del Path
del user_data_dir
del LOGGING
