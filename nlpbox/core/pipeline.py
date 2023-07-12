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

    def predict(self, X):
        pass

    def fit(self, X, y):
        pass

    @property
    def steps(self) -> dict:
        """Retorna um dicionário descrevendo
        as etapas dessa pipeline e os hiper-parâmetros
        (se aplicáveis) de cada uma.

        Returns:
            Etapas da pipeline.
        """

    @property
    def model(self) -> Estimator:
        pass

    def save(self, save_path: Path | str, **kwargs) -> None:
        """Serializa e salva esse estimador em um arquivo
        respeitando seus parâmetros e configurações atuais.

        Argumentos passados como kwargs podem ser utilizados
        para controle na serialização de alguns estimadores.

        Args:
            save_path: Path-like, define o caminho de
                salvamento do modelo.
            **kwargs: argumentos extras que podem ser utilizados
                por alguns estimadores para customização
                da serialização.
        """
