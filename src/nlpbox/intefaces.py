from typing import Protocol, Iterable, Self, overload
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
import dataclasses as dc


class FeatureFunction(Protocol):
    __name__: str

    @overload
    def __call__(self, text: str) -> float | int: ...

    @overload
    def __call__(self, text: Iterable[str]) -> list[float | int]: ...


class Extractor(Protocol):
    features: list[FeatureFunction]
    def fit(self, texts: Iterable[str]) -> pd.DataFrame: ...


@dc.dataclass
class ClassificationExperiment:
    extractor: Extractor
    pipeline: Pipeline

    def fit(self, dataset: Iterable[str], y) -> Self:
        features = self.extractor.fit(dataset)
        self.pipeline.fit(features)
        return self
