"""Arquivo de inicialização.
"""
import logging
import logging.config
import os
from pathlib import Path

import spacy
import spacy.cli
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

# Garantindo que as pipelines do spaCy estão
#   disponíveis
spacy_model = 'pt_core_news_md'
try:
    spacy.load(spacy_model)
except Exception:
    spacy.cli.download(spacy_model)

# Limpando variáveis auxiliares do namespace
del logging
del LOGGING
del os
del Path
del spacy
del spacy_model
del user_data_dir
