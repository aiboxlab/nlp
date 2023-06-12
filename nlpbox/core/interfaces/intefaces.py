from typing import Any, Literal, Protocol, Iterable, Self, overload
import dataclasses as dc
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_pinball_loss
import threading


class FeatureFunc(Protocol):
    __name__: str

    @overload
    def __call__(self, text: str) -> float | int: ...

    @overload
    def __call__(self, text: Iterable[str]) -> np.ndarray[Literal['2,4'], np.dtype[np.float32]]: ...


class MetricFunc(Protocol):
    def __call__(self, y_true, y_pred, /, **kwargs) -> float: ...


class Predictor(Protocol):
    def predict(self, data: np.ndarray) -> np.ndarray: ...
    def fit(self, data: np.ndarray) -> Self: ...


class Experiment(Protocol):

    class Result(Protocol):
        scores: dict[str, float]

    def __init__(self, *models): ...

    def __call__(self, *models, metrics: list[MetricFunc | str]) -> Any: ...



from sklearn.base import TransformerMixin

class Features(TransformerMixin):
    def __init__(self, *features: FeatureFunc) -> None:
        self.features = features

    def transform(self, X): ...
    
    def fit(self, X, y): ...
    
    @property
    def names(self) -> list:
        return [feature.__name__ for feature in self.features]


