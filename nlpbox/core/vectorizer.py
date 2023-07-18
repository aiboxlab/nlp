from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch
from numpy.typing import ArrayLike


class Vectorizer(ABC):
    """Um vetorizador consegue converter
    textos (str) para uma representação númerica
    de um vetor/tensor.
    """

    def vectorize(self,
                  text: str,
                  vector_type: str = 'numpy',
                  device: str | None = None) -> np.ndarray | torch.Tensor:
        """Método para vetorização de um texto.

        Args:
            text (str): texto de entrada.
            vector_type (str, optional): tipo do vetor de saída ('numpy'
                ou 'torch').
            device: dispositvo para armazenamento do tensor Torch.

        Returns:

        """
        # Obtendo representação vetorial
        text_vector = self._vectorize(text)
        is_np = isinstance(text_vector, np.ndarray)
        is_torch = isinstance(text_vector, torch.Tensor)

        if not is_np and not is_torch:
            # Por padrão, convertemos para NumPy
            text_vector = np.array(text_vector, dtype=np.float32)

        # Caso seja necessário um tensor, convertemos
        if (vector_type == 'torch') and not is_torch:
            text_vector = torch.from_numpy(text_vector)

            if device is not None:
                text_vector = text_vector.to(device)

        # Caso seja necessário uma ndarray, convertemos
        if (vector_type == 'numpy') and is_torch:
            text_vector = text_vector.numpy()

        return text_vector

    @abstractmethod
    def _vectorize(self, text: str) -> ArrayLike:
        """Método privado para vetorização do texto
        e retorno de um array-like qualquer (e.g., lista,
        tupla, np.ndarray, torch.Tensor, etc).

        Args:
            text (str): texto que deve ser vetorizado.

        Returns:
            Array-like da representação númerica do texto.
        """


class TrainableVectorizer(Vectorizer):
    def fit(self, X, y=None) -> None:
        pass
