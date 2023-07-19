"""Esse módulo contém a definição da
interface básica para uma pipeline.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from .estimator import Estimator
from .vectorizer import Vectorizer, TrainableVectorizer


class Pipeline:
    """Essa é a interface básica para uma
    pipeline. Todas as etapas de uma pipeline
    são sequenciais, isto é, a saída de uma
    etapa é entrada para a próxima.

    Notes:
        Toda pipeline é composta por 3 componentes:
            1. Vetorizador
            2. Estimador
            3. Pós-processamento

        Quando o método `fit(X, y)` é invocado em uma pipeline,
        o seguinte processo ocorre para cada componente treinável `T`:
            1. Treinamenos `T` fazendo `T.fit(X, y)`;
            2. Calculamos o novo valor de `X = T.predict(X)`;
            3. Passamos o novo `X` e o mesmo `y`
                para a próxima etapa treinável;
        """

    def __init__(self,
                 vectorizer: Vectorizer,
                 estimator: Estimator,
                 postprocessing: Callable[[np.ndarray], np.ndarray] = None):
        self._vectorizer = vectorizer
        self._estimator = estimator

        if postprocessing is None:
            def postprocessing(x):
                return x

        self.postprocessing = postprocessing

    def predict(self, X) -> np.ndarray:
        """Realiza a predição utilizando os parâmetros
        atuais da pipeline.

        Args:
            X: array-like de strings com formato (n_samples,).

        Returns:
            NumPy array com as predições para cada amostra.
        """
        # Obtemos a representação vetorial para cada um dos
        #   textos
        X_ = self._batch_vectorize(X)

        # Calculamos as predições do estimador
        preds = self.estimator.predict(X_)

        # Aplicamos o pós processamento
        preds = self._postprocessing(preds)

        return preds

    def fit(self, X, y) -> None:
        """Realiza o treinamento da pipeline
        utilizando as entradas X com os targets
        y.

        Args:
            X: array-like de strings com formato (n_samples,).
            y: array-like com formato (n_samples,).
        """
        # Caso o vetorizador seja treinável
        if isinstance(self.vectorizer, TrainableVectorizer):
            self.vectorizer.fit(X, y)

        # Obtemos a representação vetorial para todos textos
        X_ = self._batch_vectorize(X)

        # Treinamos o estimador utilizando os vetores
        self.estimator.fit(X_, y)

    @property
    def vectorizer(self) -> Vectorizer:
        """Retorna o vetorizador dessa pipeline.

        Returns:
            Vetorizador.
        """
        return self._vectorizer

    @property
    def estimator(self) -> Estimator:
        """Retorna o estimador utilizado
        nessa pipeline.

        Returns:
            Estimador.
        """
        return self._estimator

    def postprocessing(self, y: np.ndarray) -> np.ndarray:
        return self._postprocessing(y)

    def _batch_vectorize(self, X):
        return np.array([self.vectorizer.vectorize(x) for x in X])
