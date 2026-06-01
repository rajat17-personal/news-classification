"""
Ablation: remove source bylines (reuters, century wire, etc.) from text, then
retrain classical models and compare against baseline.

Tests the hypothesis: if models rely on outlet fingerprints rather than content,
stripping those tokens should cause a significant accuracy drop on within-dataset
evaluation while potentially *improving* cross-probe generalization.

Usage:
  python -m src.classical.ablation_strip_sources
  python -m src.classical.ablation_strip_sources --dataset isot welfake --model lr xgb
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

from ..utils.config import (
    ISOT_CONFIG, WELFAKE_CONFIG, RAW_DATA_PATH, PROCESSED_DATA_PATH, SEED,
)
from ..utils.preprocessing import clean_text, input_text
from ..utils.seeds import set_seed
from ..utils.combined_loader import COMBINED_CONFIG
from ..classical.features import TFIDFFeaturizer
from ..classical.models import MODEL_REGISTRY
from ..utils.data_loader import load_splits

DATASET_CONFIGS = {
    "isot":     ISOT_CONFIG,
    "welfake":  WELFAKE_CONFIG,
    "combined": COMBINED_CONFIG,
}


def _load_stripped(config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the standard processed splits but re-apply clean_text with strip_sources=True."""
    train_df, val_df, test_df = load_splits(config)
    for df in (train_df, val_df, test_df):
        df["text"] = df["text"].apply(lambda t: clean_text(t, strip_sources=True))
    return train_df, val_df, test_df


def _run_model(model_key: str, X_train, y_train, X_test, y_test, label: str):
    from sklearn.metrics import accuracy_score, f1_score
    model = MODEL_REGISTRY[model_key]()
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_sec = time.perf_counter() - t0
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="macro")
    print(f"  [{label}] {model_key.upper():6s}  acc={acc:.4f}  macro_f1={f1:.4f}  "
          f"train={train_sec:.1f}s")
    return model, acc, f1


def run_ablation(datasets: list[str], model_keys: list[str]):
    set_seed(SEED)
    results = {}

    for ds in datasets:
        config = DATASET_CONFIGS[ds]
        print(f"\n{'='*60}")
        print(f"Dataset: {ds.upper()}")
        print(f"{'='*60}")

        # --- Baseline (standard processed splits) ---
        train_base, _, test_base = load_splits(config)
        feat_base = TFIDFFeaturizer(max_features=50000)
        X_train_b = feat_base.fit_transform(train_base["text"].tolist())
        X_test_b  = feat_base.transform(test_base["text"].tolist())
        y_train = train_base["label"].values
        y_test  = test_base["label"].values

        print(f"\n  --- Baseline (with source tokens) ---")
        base_scores = {}
        for mk in model_keys:
            _, acc, f1 = _run_model(mk, X_train_b, y_train, X_test_b, y_test, "baseline")
            base_scores[mk] = (acc, f1)

        # --- Stripped (source bylines removed) ---
        train_str, _, test_str = _load_stripped(config)
        feat_str = TFIDFFeaturizer(max_features=50000)
        X_train_s = feat_str.fit_transform(train_str["text"].tolist())
        X_test_s  = feat_str.transform(test_str["text"].tolist())

        print(f"\n  --- Stripped (reuters / century wire removed) ---")
        strip_scores = {}
        stripped_models = {}
        for mk in model_keys:
            model, acc, f1 = _run_model(mk, X_train_s, y_train, X_test_s, y_test, "stripped")
            strip_scores[mk] = (acc, f1)
            stripped_models[mk] = (model, feat_str)

        # --- Delta summary ---
        print(f"\n  --- Delta (stripped − baseline) ---")
        print(f"  {'Model':<8}  {'ΔAcc':>8}  {'ΔF1':>8}  {'Verdict'}")
        print(f"  {'-'*50}")
        for mk in model_keys:
            da = strip_scores[mk][0] - base_scores[mk][0]
            df = strip_scores[mk][1] - base_scores[mk][1]
            verdict = ("✓ small drop — some content signal" if abs(df) < 0.01
                       else "↓ large drop — model relied heavily on source tokens" if df < 0
                       else "↑ improved — source tokens were noise")
            print(f"  {mk.upper():<8}  {da:>+.4f}   {df:>+.4f}   {verdict}")

        results[ds] = {
            "baseline": base_scores,
            "stripped": strip_scores,
            "stripped_models": stripped_models,
        }

        # --- Cross-probe: stripped ISOT model → stripped WELFake test ---
        if ds == "isot" and "welfake" in datasets:
            print(f"\n  --- Cross-probe: stripped-ISOT → stripped-WELFake ---")
            _, _, test_wf_str = _load_stripped(WELFAKE_CONFIG)
            # reindex WELFake text through ISOT's stripped vectorizer
            for mk in model_keys:
                model, feat = stripped_models[mk]
                X_wf = feat.transform(test_wf_str["text"].tolist())
                y_wf = test_wf_str["label"].values
                from sklearn.metrics import f1_score
                y_pred = model.predict(X_wf)
                f1 = f1_score(y_wf, y_pred, average="macro")
                print(f"  {mk.upper():<8} ISOT→WELFake (stripped) macro_f1={f1:.4f}")

    # --- Final comparison table ---
    print(f"\n{'='*60}")
    print("SUMMARY TABLE")
    print(f"{'='*60}")
    print(f"  {'Dataset':<10} {'Model':<8} {'Base F1':>8} {'Strip F1':>9} {'Δ F1':>7}")
    print(f"  {'-'*48}")
    for ds, r in results.items():
        for mk in model_keys:
            bf = r["baseline"][mk][1]
            sf = r["stripped"][mk][1]
            print(f"  {ds:<10} {mk.upper():<8} {bf:>8.4f} {sf:>9.4f} {sf-bf:>+7.4f}")

    # Save results JSON
    import json
    from datetime import datetime
    from ..utils.config import RESULTS_PATH
    out = {
        ds: {
            "baseline":  {mk: {"f1": v[1], "acc": v[0]} for mk, v in r["baseline"].items()},
            "stripped":  {mk: {"f1": v[1], "acc": v[0]} for mk, v in r["stripped"].items()},
        }
        for ds, r in results.items()
    }
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(RESULTS_PATH, f"ablation_strip_sources_{ts}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {path}")


def main():
    parser = argparse.ArgumentParser(description="Source-token strip ablation")
    parser.add_argument("--dataset", nargs="+", default=["isot", "welfake"],
                        choices=["isot", "welfake", "combined"])
    parser.add_argument("--model", nargs="+", default=["lr", "xgb"],
                        choices=["lr", "svc", "rf", "xgb", "all"])
    args = parser.parse_args()

    model_keys = ["lr", "svc", "rf", "xgb"] if "all" in args.model else args.model
    run_ablation(args.dataset, model_keys)


if __name__ == "__main__":
    main()
