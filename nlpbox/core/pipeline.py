"""Esse módulo contém a definição da
interface básica para uma pipeline.
"""
from __future__ import annotations

from typing import Protocol

from .estimator import Estimator


class Pipeline(Protocol):
    """Essa é a interface básica para uma
    pipeline. Todas as etapas de uma pipeline
    são sequenciais, isto é, a saída de uma
    etapa é entrada para a próxima.

    Notes
    -----

    Toda pipeline é composta por 4 componentes:
        1. Pré-processamentos
        2. Extração de características
        3. Preditores/Estimadores
        4. Pós-processamentos

    Cada componente é composto por uma ou mais etapas.

    É possível que qualquer uma das etapas seja uma
    identidade. Na prática, isso permite que alguma das
    etapas seja "pulada".

    Quando o método `fit(X, y)` é invocado em uma pipeline,
    o seguinte processo ocorre para cada etapa treinável `T`:
        1. Treinamenos `T` fazendo `T.fit(X, y)`;
        2. Calculamos o novo valor de `X`
            fazendo `X = T.predict(X)`;
        3. Passamos o novo `X` e o mesmo `y`
            para a próxima etapa treinável;
    """

    def predict(self, X) -> np.ndarray:
        """Realiza a predição utilizando os parâmetros
        atuais da pipeline.

        Args:
            X: array-like de strings com formato (n_samples,).

        Returns:
            NumPy array com as predições para cada amostra.
        """

    def fit(self, X, y) -> None:
        """Realiza o treinamento da pipeline
        utilizando as entradas X com os targets
        y.

        Args:
            X: array-like de strings com formato (n_samples,).
            y: array-like com formato (n_samples,).
        """

    @property
    def steps(self) -> dict:
        """Retorna um dicionário descrevendo
        as etapas dessa pipeline e os hiper-parâmetros
        (se aplicáveis) de cada uma.

        Returns:
            Etapas da pipeline.
        """

    @property
    def models(self) -> list[Estimator]:
        """Retorna os estimadores utilizados
        nessa pipeline.

        Returns:
            Lista de estimadores.
        """

    def save(self, save_path: Path | str, **kwargs) -> None:
        """Serializa e salva essa pipeline em um arquivo
        respeitando seus parâmetros e configurações atuais.

        Argumentos passados como kwargs podem ser utilizados
        para controle na serialização de algumas pipelines.

        Args:
            save_path: Path-like, define o caminho de
                salvamento da pipeline.
            **kwargs: argumentos extras que podem ser utilizados
                por algumas pipelines para customização
                da serialização.
        """
