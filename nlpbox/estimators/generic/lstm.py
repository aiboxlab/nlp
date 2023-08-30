"""Esse módulo contém um
estimador baseado em LSTM.
"""
from __future__ import annotations

import typing

import numpy as np
import torch
import torch.optim as optim
from numpy.typing import ArrayLike
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from nlpbox.core import Estimator


class LSTMEstimator(Estimator):
    _OPTIMIZER: typing.ClassVar[dict] = {
        'adam': optim.Adam,
        'adamw': optim.AdamW,
        'rmsprop': optim.RMSprop,
        'adagrad': optim.Adagrad,
        'sgd': optim.SGD
    }

    def __init__(self,
                 hidden_size: int,
                 num_layers: int,
                 kind: typing.Literal['classifier',
                                      'regressor'],
                 epochs: int = 100,
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
                 regression_ensure_bounds: bool = False,
                 train_batch_size: int = 256,
                 device: str = None):
        # Armazenando dispositivo
        self._device = device

        # Armazenando hiperparâmetros
        self._hyperparams = {
            'lstm_hidden_size': hidden_size,
            'lstm_layers': num_layers,
            'lstm_bias': bias,
            'lstm_dropout': dropout_prob,
            'lstm_bidirectional': bidirectional,
            'lstm_proj_size': proj_size,
            'last_layer': 'linear',
            'train_batch_size': train_batch_size,
            'epochs': epochs,
            'optimizer': optim,
            'learning_rate': learning_rate,
        }

        self._hyperparams.update({
            f'optimizer_{k}': v
            for k, v in optim_params.items()
        })

        # Inicializando o otimizador
        self._optim = self._OPTIMIZER[optim]
        self._lr = learning_rate
        self._epochs = epochs
        self._optim_params = optim_params

        # Inicializando o critério de otimização
        if kind == 'classifier':
            self._criterion = nn.CrossEntropyLoss()
        else:
            self._criterion = nn.MSELoss()

        # Inicializando o modelo
        self._model = None
        self._kind = kind
        self._encoder = None
        self._batch_size = train_batch_size

    def fit(self, X: ArrayLike, y: ArrayLike) -> None:
        # Atualizando formato do batch para
        #   (batch_size, max_seq_len, n_features)
        X = self._maybe_padded_sequence(X)

        # Extraindo o valor da última dimensão
        feature_size = X.size(dim=2)
        n_samples = X.size(dim=0)

        # Talvez converter "y" para um intervalo [0, N]
        y = self._maybe_convert_label(y,
                                      direction='from')
        y = torch.tensor(y, device=self._device)

        assert len(X.size()) == 3, ('Vetor tem que ser '
                                    '(batch, sequence, features)')

        # Criando o modelo
        self._create_model(input_size=feature_size,
                           y=y)

        # Criando dataset
        ds = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(ds,
                                             batch_size=256)
        del X, y

        # Criando otimizador
        self._optim = self._optim(self._model.parameters(),
                                  lr=self._lr,
                                  **self._optim_params)

        # Treinamento do modelo
        for _ in range(self._epochs):
            for data in loader:
                inputs, targets = data

                # Resetando o acúmulo de gradientes
                self._optim.zero_grad()

                # Obtendo a saída do modelo
                outputs = self._model(inputs)

                # Calculando o forward e backward-pass
                #   da função objetiva
                loss = self._criterion(outputs, targets)
                loss.backward()

                # Atualizar pesos com base na saída
                self._optim.step()

                # Removendo do escopo do for
                del outputs, loss

    def predict(self, X: ArrayLike) -> None:
        # Convertendo as características para um tensor
        X = self._maybe_padded_sequence(X)

        # Obtendo saídas do modelo
        preds = self._model(X)
        preds = self._maybe_update_preds(preds)

        # Convertendo para NumPy
        preds = preds.cpu().numpy()

        # Possivelmente converter para os labels
        #   vistos durante treino
        return self._maybe_convert_label(preds, direction='to')

    def _maybe_update_preds(self, preds: torch.Tensor) -> torch.Tensor:
        # Se for classificação, devemos obter
        #   o máximo da saída linear.
        if self._kind == 'classifier':
            _, preds = torch.max(preds, dim=1)

        return preds

    def _maybe_padded_sequence(self, X: ArrayLike) -> torch.Tensor:
        # Convertendo as características para um tensor
        maybe_variable_length = [torch.tensor(x, device=self._device)
                                 for x in X]

        # Novo shape: (batch_size, max_seq, features)
        X = pad_sequence(maybe_variable_length,
                         batch_first=True,
                         padding_value=0.0)

        # Retornando entradas como tensor
        return X

    def _create_model(self, input_size, y):
        if self._kind == 'classifier':
            out = torch.unique(y, dim=0).size(dim=0)
        else:
            out = 1

        h = self._hyperparams['lstm_hidden_size']
        self._model = nn.Sequential(
            nn.LSTM(input_size,
                    h,
                    self._hyperparams['lstm_layers'],
                    self._hyperparams['lstm_bias'],
                    True,
                    self._hyperparams['lstm_dropout'],
                    self._hyperparams['lstm_bidirectional'],
                    self._hyperparams['lstm_proj_size']),
            _LSTMTensorExtractor(),
            nn.Linear(h, out))

    def _maybe_convert_label(self,
                             y: ArrayLike,
                             direction: str) -> ArrayLike:
        if self._kind == 'regressor':
            # Não precisamos realizar conversão
            #   se for regressão.
            return y

        if direction == 'from':
            self._encoder = LabelEncoder()
            self._encoder.fit(y)
            return self._encoder.transform(y)

        return self._encoder.inverse_transform(y)

    @property
    def hyperparameters(self) -> dict:
        return self._hyperparams

    @property
    def params(self) -> dict:
        return self._hyperparams


class _LSTMTensorExtractor(nn.Module):
    def forward(self, x):
        # Output shape (batch, features, hidden)
        tensor, _ = x

        # Reshape shape (batch, hidden)
        return tensor[:, -1, :]
