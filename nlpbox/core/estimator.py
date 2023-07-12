"""Esse módulo contém a definição da
interface básica para um estimador/preditor.
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol


class Estimator(Protocol):
    """Essa é a interface básica para um
    estimador.
    """

    def predict(self, X):
        pass

    def fit(self, X, y) -> None:
        pass

    @property
    def hyperparameters(self) -> dict:
        """Retorna um dicionário descrevendo
        os hiper-parâmetros do estimador.

        Returns:
            Hiper-parâmetros do estimador.
        """
    
    @property
    def params(self) -> dict:
        """Retorna um dicionário com os parâmetros
        para esse estimador. Os parâmetros retornados
        descrevem totalmente o estado do modelo (e,g.
        pesos de uma rede, superfícies de decisão, estrutura
        da árvore de decisão, etc).
        
        Returns:
            Parâmetros do estimador.
        """

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
