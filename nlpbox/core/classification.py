from sklearn.ensemble import ExtraTreesClassifier as _ExtraTreesClassifier
from catboost import CatBoostClassifier as _CatBoostClassifier

class ExtraTreesClassifier:
    def __init__(self, **kwargs) -> None:
        self.instance = _ExtraTreesClassifier(**kwargs)

    def predict(self, X):
        self.instance.predict(X)
    
    def fit(self, X, y):
        self.instance.fit(X, y)


class CatBoostClassifier:
    def __init__(self, **kwargs) -> None:
        self.instance = _CatBoostClassifier(**kwargs)

    def predict(self, X):
        self.instance.predict(X)
    
    def fit(self, X, y):
        self.instance.fit(X, y)

