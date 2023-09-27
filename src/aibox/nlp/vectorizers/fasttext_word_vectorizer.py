"""Esse módulo contém um
vetorizador de palavras baseado
nos modelos do Fasttext.
"""
from __future__ import annotations

import re

import fasttext
import numpy as np
import spacy

from aibox.nlp.core import Vectorizer
from aibox.nlp import resources


class FasttextWordVectorizer(Vectorizer):
    def __init__(self,
                 language: str = 'pt',
                 dims: int = 50):
        """Construtor de um word2vec
        utilizando os modelos pré-treinados
        do FastText.

        Args:
            language (str): linguagem do modelo.
            dims (int): dimensões do embedding.
        """
        assert language in {'pt'}
        assert dims in {50}

        # Obtendo caminho para o modelo
        root = resources.path('embeddings/fasttext-cc-50.v1')
        model_path = root.joinpath('cc.pt.50.bin').absolute()

        # Carregando o modelo
        self._ft = fasttext.load_model(str(model_path))

    def _vectorize(self, text: str):
        words = self._tokenize(text)
        word_vectors = [self._ft.get_word_vector(w)
                        for w in words]
        return np.array(word_vectors)

    def _tokenize(self, text: str) -> list[str]:
        return re.sub(r'\s+', ' ', text).split()
