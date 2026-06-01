from __future__ import annotations
from typing import Protocol
import numpy as np
import joblib
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


class ClassicalModel(Protocol):
    def fit(self, X: csr_matrix, y: np.ndarray) -> None: ...
    def predict(self, X: csr_matrix) -> np.ndarray: ...
    def predict_proba(self, X: csr_matrix) -> np.ndarray: ...
    def save(self, path: str) -> None: ...
    @classmethod
    def load(cls, path: str) -> ClassicalModel: ...


class LRModel:
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000)

    def fit(self, X: csr_matrix, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict(self, X: csr_matrix) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: csr_matrix) -> np.ndarray:
        return self.model.predict_proba(X)

    def save(self, path: str) -> None:
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: str) -> LRModel:
        instance = cls()
        instance.model = joblib.load(path)
        return instance


class SVCModel:
    def __init__(self):
        self.model = CalibratedClassifierCV(LinearSVC(max_iter=1000))

    def fit(self, X: csr_matrix, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict(self, X: csr_matrix) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: csr_matrix) -> np.ndarray:
        return self.model.predict_proba(X)

    def save(self, path: str) -> None:
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: str) -> SVCModel:
        instance = cls()
        instance.model = joblib.load(path)
        return instance


class RFModel:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, n_jobs=2)

    def fit(self, X: csr_matrix, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict(self, X: csr_matrix) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: csr_matrix) -> np.ndarray:
        return self.model.predict_proba(X)

    def save(self, path: str) -> None:
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: str) -> RFModel:
        instance = cls()
        instance.model = joblib.load(path)
        return instance


class XGBModel:
    def __init__(self):
        self.model = XGBClassifier(n_estimators=300, eval_metric='logloss', tree_method='hist')

    def fit(self, X: csr_matrix, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict(self, X: csr_matrix) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: csr_matrix) -> np.ndarray:
        return self.model.predict_proba(X)

    def save(self, path: str) -> None:
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: str) -> XGBModel:
        instance = cls()
        instance.model = joblib.load(path)
        return instance


MODEL_REGISTRY = {
    'lr': LRModel,
    'svc': SVCModel,
    'rf': RFModel,
    'xgb': XGBModel,
}
