"""
"""
from __future__ import annotations

from sklearn.svm import SVC as _SVC

from nlpbox.core import Estimator
from nlpbox.factory import register


@register('svm')
class SVM(Estimator):
    def __init__(self, C=1.0, kernel='rbf', degree=3,
                 gamma='scale', coef0=0.0, shrinking=True,
                 probability=False, tol=0.001, cache_size=200,
                 class_weight=None, max_iter=-1,
                 decision_function_shape='ovr', break_ties=False,
                 random_state=None):
        self._hyperparams = dict(
            C=C,
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            shrinking=shrinking,
            probability=probability,
            tol=tol,
            cache_size=cache_size,
            class_weight=class_weight,
            verbose=False,
            max_iter=max_iter,
            decision_function_shape=decision_function_shape,
            break_ties=break_ties,
            random_state=random_state)

        self._svc = _SVC(**self._hyperparams)

    def predict(self, X) -> np.ndarray:
        preds = self._svc.predict(X)
        return np.array(preds)

    def fit(self, X, y):
        self._svc.fit(X, y)

    @property
    def hyperparameters(self) -> dict:
        return self._hyperparams

    @property
    def params(self) -> dict:
        params = self._svc.get_params()

        return {k: v for k, v in params.items()
                if k not in self.hyperparameters}
