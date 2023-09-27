"""Esse módulo contém a definição da
interface básica para uma pipeline.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import ArrayLike
from tqdm.auto import tqdm

from .estimator import Estimator
from .vectorizer import TrainableVectorizer, Vectorizer


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
            1. Treinamos `T` fazendo `T.fit(X, y)`;
            2. Calculamos o novo valor de `X = T.predict(X)`;
            3. Passamos o novo `X` e o mesmo `y`
                para a próxima etapa treinável;
        """

    def __init__(self,
                 vectorizer: Vectorizer,
                 estimator: Estimator,
                 postprocessing: Callable[[np.ndarray], np.ndarray] = None,
                 name: str | None = None):
        """Construtor.

        Args:
            vectorizer (Vectorizer): vetorizador.
            estimator (Estimator): estimador.
            postprocessing (Callable[[np.ndarray], 
                                      np.ndarray],
                            optional): pós-processamento (default=None).
            name (str | None, optional): Nome da pipeline. Por padrão,
                gera um nome aleatório.
        """
        if postprocessing is None:
            def postprocessing(x):
                return x

        if name is None:
            name = self._generate_name(vectorizer, estimator)

        self._vectorizer = vectorizer
        self._estimator = estimator
        self._postprocessing = postprocessing
        self._name = name

    def predict(self, X: ArrayLike, **kwargs) -> np.ndarray:
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
        preds = self.estimator.predict(X_, **kwargs)

        # Aplicamos o pós processamento
        preds = self._postprocessing(preds)

        return preds

    def fit(self, X: ArrayLike, y: ArrayLike, **kwargs) -> None:
        """Realiza o treinamento da pipeline
        utilizando as entradas X com os targets
        y.

        Args:
            X: array-like de strings com formato (n_samples,).
            y: array-like com formato (n_samples,).
        """
        # Caso o vetorizador seja treinável
        if isinstance(self.vectorizer, TrainableVectorizer):
            self.vectorizer.fit(X, y, **kwargs)

        # Obtemos a representação vetorial para todos textos
        X_ = self._batch_vectorize(X)

        # Treinamos o estimador utilizando os vetores
        self.estimator.fit(X_, y, **kwargs)

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

    @property
    def name(self) -> str:
        """Retorna o nome dessa pipeline.

        Returns:
            Nome da pipeline.
        """
        return self._name

    def postprocessing(self, y: np.ndarray) -> np.ndarray:
        return self._postprocessing(y)

    def _batch_vectorize(self, X):
        return [self.vectorizer.vectorize(x)
                for x in tqdm(X,
                              ascii=False,
                              desc='Vetorização',
                              leave=False)]

    @staticmethod
    def _generate_name(vectorizer: Vectorizer, estimator: Estimator) -> str:
        # Obtendo nome da classe do estimador
        estimator_name = estimator.__class__.__name__

        # Se for um agregado de features, obtemos o nome
        #   individual de cada uma
        extractors = getattr(vectorizer,
                             'extractors',
                             None)
        if extractors:
            vectorizer_name = '_'.join(v.__class__.__name__
                                       for v in extractors)
        else:
            vectorizer_name = vectorizer.__class__.__name__

        # Obtemos os parâmetros do estimador
        estimator_params = '_'.join(str(v) for v in
                                    estimator.hyperparameters.values()
                                    if not isinstance(v, dict))

        # Construímos o nome final da pipeline
        name = '_'.join([vectorizer_name,
                         estimator_name,
                         estimator_params,
                         f'seed_{estimator.random_state}'])
        return name
