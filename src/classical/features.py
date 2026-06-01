import os
import joblib
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from ..utils.config import CACHE_PATH

try:
    _STOP_WORDS = list(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords', quiet=True)
    _STOP_WORDS = list(stopwords.words('english'))


class TFIDFFeaturizer:
    def __init__(self, max_features: int = 50000, ngram_range: tuple = (1, 2)):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vectorizer = TfidfVectorizer(
            stop_words=_STOP_WORDS,
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,
            min_df=2,
        )

    def _clean(self, texts: list) -> list:
        return [t if isinstance(t, str) else '' for t in texts]

    def fit(self, texts: list) -> "TFIDFFeaturizer":
        self.vectorizer.fit(self._clean(texts))
        return self

    def transform(self, texts: list):
        return self.vectorizer.transform(self._clean(texts))

    def fit_transform(self, texts: list):
        return self.vectorizer.fit_transform(self._clean(texts))

    def save(self, dataset: str) -> None:
        os.makedirs(CACHE_PATH, exist_ok=True)
        joblib.dump(self.vectorizer, os.path.join(CACHE_PATH, f"tfidf_{dataset}.joblib"))

    @classmethod
    def load(cls, dataset: str) -> "TFIDFFeaturizer":
        path = os.path.join(CACHE_PATH, f"tfidf_{dataset}.joblib")
        instance = cls()
        instance.vectorizer = joblib.load(path)
        return instance
