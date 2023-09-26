"""Esse módulo contém
um classificar baseado em
LSTMs.
"""
from __future__ import annotations

import typing

from nlpbox.estimators.generic.lstm import LSTMEstimator


class LSTMClassifier(LSTMEstimator):
    def __init__(self,
                 hidden_size: int = 20,
                 num_layers: int = 2,
                 epochs: int = 10,
                 bias: bool = True,
                 dropout_prob: float = 0,
                 bidirectional: bool = False,
                 proj_size: int = 0,
                 learning_rate: float = 1e-3,
                 optim_params: dict = dict(),
                 optim: typing.Literal['adam',
                                       'adamw',
                                       'rmsprop',
                                       'adagrad',
                                       'sgd'] = 'adamw',
                 random_state: int | None = None,
                 device: str = None):
        super().__init__(hidden_size=hidden_size,
                         num_layers=num_layers,
                         kind='classifier',
                         epochs=epochs,
                         bias=bias,
                         dropout_prob=dropout_prob,
                         bidirectional=bidirectional,
                         proj_size=proj_size,
                         learning_rate=learning_rate,
                         optim_params=optim_params,
                         optim=optim,
                         regression_ensure_bounds=False,
                         random_state=random_state,
                         device=device)
