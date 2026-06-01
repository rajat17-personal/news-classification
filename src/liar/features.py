"""
LIAR metadata feature builder.

Combines TF-IDF text features with structured metadata:
  - 5 speaker credit-history counts (numeric, log-scaled)
  - Party affiliation one-hot (top 8 + "other")
  - Subject multi-hot (top 20 subjects)
  - Context type grouped one-hot (tv_ad | speech | interview | press_release | tweet | other)

Speaker (2910 unique) and job_title (1183 unique) are too high-cardinality for one-hot;
they are captured indirectly via the credit-history counts.
"""
import os
import numpy as np
import pandas as pd
import joblib
from scipy.sparse import hstack, csr_matrix
from sklearn.preprocessing import MultiLabelBinarizer

from ..utils.config import CACHE_PATH

_CREDIT_COLS = [
    "barely_true_counts", "false_counts", "half_true_counts",
    "mostly_true_counts", "pants_on_fire_counts",
]

_TOP_PARTIES = ["republican", "democrat", "none", "organization",
                "independent", "newsmaker", "libertarian", "activist"]

_TOP_SUBJECTS = [
    "economy", "health-care", "taxes", "federal-budget", "education",
    "jobs", "state-budget", "candidates-biography", "elections",
    "immigration", "foreign-policy", "crime", "environment",
    "guns", "abortion", "energy", "history", "transportation",
    "medicare", "social-security",
]

_CONTEXT_MAP = {
    "tv":       ["a tv ad", "a television ad", "a campaign ad", "a tv interview",
                 "television", "tv", "fox news", "cnn", "msnbc", "nbc", "cbs", "abc"],
    "speech":   ["a speech", "a floor speech", "a debate", "a rally", "a campaign speech",
                 "a commencement speech", "a town hall meeting"],
    "interview":["an interview", "an interview on", "a radio interview", "a press conference",
                 "a news conference"],
    "release":  ["a news release", "a press release", "a statement", "a blog post",
                 "a blog posting", "a website", "an op-ed", "a column"],
    "tweet":    ["a tweet", "twitter", "facebook", "social media", "a facebook post"],
}


def _map_context(ctx: str) -> str:
    if not isinstance(ctx, str):
        return "other"
    ctx_lower = ctx.lower().strip()
    for group, keywords in _CONTEXT_MAP.items():
        if any(kw in ctx_lower for kw in keywords):
            return group
    return "other"


def _parse_subjects(subject_str) -> list:
    if not isinstance(subject_str, str):
        return []
    return [s.strip() for s in subject_str.split(",") if s.strip() in _TOP_SUBJECTS]


class LIARMetaFeaturizer:
    """
    Builds a combined feature matrix:
      [TF-IDF text | credit history | party one-hot | subject multi-hot | context one-hot]
    """

    def __init__(self, text_max_features: int = 30000):
        from ..classical.features import TFIDFFeaturizer
        self.tfidf = TFIDFFeaturizer(max_features=text_max_features, ngram_range=(1, 2))
        self.mlb = MultiLabelBinarizer(classes=_TOP_SUBJECTS)
        self.mlb.fit([[s] for s in _TOP_SUBJECTS])
        self._fitted = False

    # ------------------------------------------------------------------
    # Raw TSV loading — loads full columns, not the preprocessed CSV
    # ------------------------------------------------------------------

    @staticmethod
    def load_raw_tsv(split: str) -> pd.DataFrame:
        from ..utils.config import RAW_DATA_PATH
        raw_dir = os.path.join(RAW_DATA_PATH, "liar")
        fname = {"train": "train.tsv", "val": "valid.tsv", "test": "test.tsv"}[split]
        df = pd.read_csv(
            os.path.join(raw_dir, fname),
            sep="\t", header=None,
            names=["id", "label", "statement", "subject", "speaker", "job_title",
                   "state_info", "party_affiliation",
                   "barely_true_counts", "false_counts", "half_true_counts",
                   "mostly_true_counts", "pants_on_fire_counts", "context"],
        )
        return df

    # ------------------------------------------------------------------
    # Metadata feature builders
    # ------------------------------------------------------------------

    def _credit_features(self, df: pd.DataFrame) -> np.ndarray:
        """5 numeric credit-history columns, log1p scaled."""
        vals = df[_CREDIT_COLS].fillna(0).values.astype(np.float32)
        return np.log1p(vals)

    def _party_features(self, df: pd.DataFrame) -> np.ndarray:
        party = df["party_affiliation"].fillna("other").str.lower().str.strip()
        mat = np.zeros((len(df), len(_TOP_PARTIES) + 1), dtype=np.float32)
        for i, p in enumerate(party):
            if p in _TOP_PARTIES:
                mat[i, _TOP_PARTIES.index(p)] = 1.0
            else:
                mat[i, -1] = 1.0  # "other" bucket
        return mat

    def _subject_features(self, df: pd.DataFrame) -> np.ndarray:
        subjects = df["subject"].apply(_parse_subjects).tolist()
        return self.mlb.transform(subjects).astype(np.float32)

    def _context_features(self, df: pd.DataFrame) -> np.ndarray:
        groups = ["tv", "speech", "interview", "release", "tweet", "other"]
        ctx_labels = df["context"].apply(_map_context)
        mat = np.zeros((len(df), len(groups)), dtype=np.float32)
        for i, g in enumerate(ctx_labels):
            mat[i, groups.index(g)] = 1.0
        return mat

    def _meta_features(self, df: pd.DataFrame) -> np.ndarray:
        return np.hstack([
            self._credit_features(df),
            self._party_features(df),
            self._subject_features(df),
            self._context_features(df),
        ])

    # ------------------------------------------------------------------
    # Fit / transform
    # ------------------------------------------------------------------

    def fit_transform(self, df: pd.DataFrame, texts: list) -> csr_matrix:
        X_text = self.tfidf.fit_transform(texts)
        X_meta = csr_matrix(self._meta_features(df))
        self._fitted = True
        return hstack([X_text, X_meta])

    def transform(self, df: pd.DataFrame, texts: list) -> csr_matrix:
        X_text = self.tfidf.transform(texts)
        X_meta = csr_matrix(self._meta_features(df))
        return hstack([X_text, X_meta])

    def save(self, tag: str = "liar_meta") -> None:
        os.makedirs(CACHE_PATH, exist_ok=True)
        joblib.dump(self, os.path.join(CACHE_PATH, f"liar_meta_featurizer_{tag}.joblib"))

    @classmethod
    def load(cls, tag: str = "liar_meta") -> "LIARMetaFeaturizer":
        path = os.path.join(CACHE_PATH, f"liar_meta_featurizer_{tag}.joblib")
        return joblib.load(path)

    @property
    def feature_names(self) -> list[str]:
        text_names = list(self.tfidf.vectorizer.get_feature_names_out())
        credit_names = [f"credit_{c}" for c in _CREDIT_COLS]
        party_names  = [f"party_{p}" for p in _TOP_PARTIES] + ["party_other"]
        subject_names = [f"subject_{s}" for s in _TOP_SUBJECTS]
        context_names = ["ctx_tv", "ctx_speech", "ctx_interview",
                         "ctx_release", "ctx_tweet", "ctx_other"]
        return text_names + credit_names + party_names + subject_names + context_names
