import argparse
import os
import time

import numpy as np

from ..utils.config import CHECKPOINTS_PATH, SEED
from ..utils.seeds import set_seed
from ..classical.models import MODEL_REGISTRY
from .dataset import load_liar_splits, LIAR_LABEL_NAMES
from .features import LIARMetaFeaturizer
from .train import _evaluate_liar, _concat

NUM_CLASSES = 6


def run_classical_meta(model_keys: list[str]):
    set_seed(SEED)
    train_df_text, val_df_text, test_df_text = load_liar_splits()

    # Load raw TSV (to get metadata columns)
    raw_train = LIARMetaFeaturizer.load_raw_tsv("train")
    raw_val   = LIARMetaFeaturizer.load_raw_tsv("val")
    raw_test  = LIARMetaFeaturizer.load_raw_tsv("test")

    # The processed CSV has cleaned text; align on statement position using label
    from .dataset import LIAR_LABELS
    raw_train = raw_train[raw_train["label"].isin(LIAR_LABELS)].reset_index(drop=True)
    raw_val   = raw_val[raw_val["label"].isin(LIAR_LABELS)].reset_index(drop=True)
    raw_test  = raw_test[raw_test["label"].isin(LIAR_LABELS)].reset_index(drop=True)

    # Combine train+val for fitting (same strategy as text-only LIAR)
    raw_fit   = _concat_raw(raw_train, raw_val)
    fit_texts = train_df_text["text"].tolist() + val_df_text["text"].tolist()
    test_texts = test_df_text["text"].tolist()
    y_fit  = (raw_fit["label"].map(LIAR_LABELS)).values
    y_test = test_df_text["label"].values

    featurizer = LIARMetaFeaturizer(text_max_features=30000)
    X_fit  = featurizer.fit_transform(raw_fit, fit_texts)
    X_test = featurizer.transform(raw_test, test_texts)
    featurizer.save("liar_meta")
    n_text = len(featurizer.tfidf.vectorizer.vocabulary_)
    print(f"Feature matrix: {X_fit.shape}  "
        f"(text={n_text} + meta={X_fit.shape[1] - n_text})")

    for model_key in model_keys:
        print(f"\n=== {model_key.upper()} on LIAR (6-class + metadata) ===")
        cls = MODEL_REGISTRY[model_key]
        model = cls()
        if model_key == "svc":
            from sklearn.svm import LinearSVC
            from sklearn.calibration import CalibratedClassifierCV
            model.model = CalibratedClassifierCV(LinearSVC(max_iter=5000))

        t0 = time.perf_counter()
        model.fit(X_fit, y_fit)
        train_time = time.perf_counter() - t0

        ckpt_dir = os.path.join(CHECKPOINTS_PATH, "classical", "liar_meta")
        os.makedirs(ckpt_dir, exist_ok=True)
        import joblib
        joblib.dump(model, os.path.join(ckpt_dir, f"{model_key}.joblib"))

        def predict_fn(texts, _feat=featurizer, _m=model, _raw=raw_test):
            X = _feat.transform(_raw, texts)
            return _m.predict(X), _m.predict_proba(X)

        _evaluate_liar(
            predict_fn, test_df_text,
            model_name=f"{model_key}_liar_meta",
            train_time_sec=train_time,
            hyperparams={"model": model_key, "features": "tfidf+metadata",
                        "text_max_features": 30000, "ngram_range": "(1,2)"},
        )


def _concat_raw(*dfs):
    import pandas as pd
    return pd.concat(list(dfs), ignore_index=True)


def main():
    parser = argparse.ArgumentParser(
        description="Train classical models on LIAR with TF-IDF + metadata features"
    )
    parser.add_argument("--model", nargs="+", default=["all"],
                        choices=["lr", "svc", "rf", "xgb", "all"])
    args = parser.parse_args()

    model_keys = ["lr", "svc", "rf", "xgb"] if "all" in args.model else args.model
    run_classical_meta(model_keys)


if __name__ == "__main__":
    main()
