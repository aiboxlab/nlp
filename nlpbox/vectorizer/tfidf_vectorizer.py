"""
"""
from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer as SklearnTFIDF

from nlpbox.core.vectorizer import TrainableVectorizer


class TFIDFVectorizer(TrainableVectorizer):
    def __init__(self) -> None:
        self._tfidf = SklearnTFIDF()

    def _vectorize(self, text: str):
        sparse_matrix = self._tfidf.transform([text])
        arr = sparse_matrix.toarray()
        return arr.squeeze()

    def fit(self, X, y=None) -> None:
        self._tfidf.fit(X)
