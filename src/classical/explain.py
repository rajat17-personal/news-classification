"""
SHAP explainability for TF-IDF classical models.

Generates:
  - Top-N SHAP feature importance bar charts (global, per dataset)
  - Per-class SHAP summary plots (for multi-class / LIAR)
  - Waterfall plot for a single example (local explanation)
  - Saves all figures to results/shap/<dataset>_<model>/

Usage:
  python -m src.classical.explain --dataset isot --model lr
  python -m src.classical.explain --dataset welfake --model xgb
  python -m src.classical.explain --dataset all --model all
  python -m src.classical.explain --dataset isot --model lr --n-top 30 --sample 500
"""
import argparse
import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import shap

from ..utils.config import (
    CACHE_PATH, CHECKPOINTS_PATH, RESULTS_PATH,
    ISOT_CONFIG, WELFAKE_CONFIG,
)
from ..utils.data_loader import load_splits
from ..utils.combined_loader import COMBINED_CONFIG
from .features import TFIDFFeaturizer

DATASET_CONFIGS = {
    "isot":     ISOT_CONFIG,
    "welfake":  WELFAKE_CONFIG,
    "combined": COMBINED_CONFIG,
}

BINARY_LABEL_NAMES = ["Fake (0)", "Real (1)"]

SHAP_OUT_DIR = os.path.join(RESULTS_PATH, "shap")


def _load_model(dataset: str, model_key: str):
    """Load best checkpoint for a classical model."""
    path = os.path.join(CHECKPOINTS_PATH, "classical", dataset, f"{model_key}_best.joblib")
    if not os.path.exists(path):
        # fallback to non-best
        path = os.path.join(CHECKPOINTS_PATH, "classical", dataset, f"{model_key}.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No checkpoint found for {model_key} on {dataset} at {path}")
    return joblib.load(path)


def _load_featurizer(dataset: str, model_key: str) -> TFIDFFeaturizer:
    # try best-labelled vectoriser first, then fallback
    for tag in (f"best_{model_key}_{dataset}", f"{dataset}_{model_key}"):
        path = os.path.join(CACHE_PATH, f"tfidf_{tag}.joblib")
        if os.path.exists(path):
            feat = TFIDFFeaturizer()
            feat.vectorizer = joblib.load(path)
            return feat
    raise FileNotFoundError(f"No TF-IDF vectorizer found for {model_key} on {dataset}")


def _get_base_model(model, model_key: str):
    """Unwrap CalibratedClassifierCV for SVC, return raw sklearn estimator."""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.svm import LinearSVC
    if model_key == "svc":
        # shap.LinearExplainer works with the calibrated wrapper directly
        return model
    return model


def run_shap(dataset: str, model_key: str, n_top: int, n_sample: int):
    print(f"\n=== SHAP: {model_key.upper()} on {dataset} ===")
    config = DATASET_CONFIGS[dataset]
    _, _, test_df = load_splits(config)

    featurizer = _load_featurizer(dataset, model_key)
    model = _load_model(dataset, model_key)
    feature_names = featurizer.vectorizer.get_feature_names_out()

    # Sample test set for speed (SHAP on full 50k-feature space is slow for tree models)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(test_df), size=min(n_sample, len(test_df)), replace=False)
    sample_df = test_df.iloc[idx].reset_index(drop=True)
    X_sample = featurizer.transform(sample_df["text"].tolist())

    out_dir = os.path.join(SHAP_OUT_DIR, f"{dataset}_{model_key}")
    os.makedirs(out_dir, exist_ok=True)

    if model_key == "lr":
        _shap_linear(model, X_sample, feature_names, sample_df, dataset, model_key,
                     n_top, out_dir)
    elif model_key == "xgb":
        _shap_tree(model, X_sample, feature_names, sample_df, dataset, model_key,
                   n_top, out_dir)
    elif model_key in ("svc", "rf"):
        _shap_kernel(model, X_sample, feature_names, sample_df, dataset, model_key,
                     n_top, out_dir, n_bg=100)
    else:
        print(f"  Unsupported model key: {model_key}")


# ---------------------------------------------------------------------------
# LR — LinearExplainer (exact, fast on sparse TF-IDF)
# ---------------------------------------------------------------------------

def _shap_linear(model, X_sample, feature_names, sample_df, dataset, model_key,
                 n_top, out_dir):
    explainer = shap.LinearExplainer(model, X_sample, feature_perturbation="interventional")
    shap_values = explainer(X_sample)  # Explanation object

    # Attach feature names so beeswarm/waterfall show tokens not "Feature 35777"
    shap_values.feature_names = list(feature_names)

    # For binary LR, shap_values.values is shape (N, n_features) — positive class
    vals = shap_values.values
    if vals.ndim == 3:
        vals = vals[:, :, 1]

    _plot_global_importance(vals, feature_names, dataset, model_key, n_top, out_dir,
                            label="Real (positive class)")
    _plot_beeswarm(shap_values if vals.ndim == 2 else shap_values[:, :, 1],
                   feature_names, out_dir, dataset, model_key)
    _plot_waterfall_example(shap_values, sample_df, feature_names, out_dir, dataset, model_key)
    print(f"  SHAP plots saved → {out_dir}")


# ---------------------------------------------------------------------------
# XGBoost — TreeExplainer (exact, fast on dense)
# ---------------------------------------------------------------------------

def _shap_tree(model, X_sample, feature_names, sample_df, dataset, model_key,
               n_top, out_dir):
    # XGBoost needs dense; convert sparse
    X_dense = X_sample.toarray().astype(np.float32)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_dense)

    # Binary XGB: shap_values is (N, F) for the positive class
    if isinstance(shap_values, list):
        vals = shap_values[1]
    else:
        vals = shap_values

    _plot_global_importance(vals, feature_names, dataset, model_key, n_top, out_dir,
                            label="Real (positive class)")

    # Summary dot plot
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(vals[:, :n_top * 2],
                      X_dense[:, :n_top * 2],
                      feature_names=feature_names[:n_top * 2],
                      show=False, max_display=n_top)
    plt.title(f"SHAP Summary — {model_key.upper()} on {dataset}", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "summary_dot.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  SHAP plots saved → {out_dir}")


# ---------------------------------------------------------------------------
# RF / SVC — KernelExplainer (approximate, slower)
# ---------------------------------------------------------------------------

def _shap_kernel(model, X_sample, feature_names, sample_df, dataset, model_key,
                 n_top, out_dir, n_bg=100):
    print(f"  Using KernelExplainer (approximate) — n_bg={n_bg}, n_sample={X_sample.shape[0]}")
    # Use a small background dataset (k-means summary)
    X_bg = shap.kmeans(X_sample, n_bg)

    def predict_proba(X):
        from scipy.sparse import issparse, csr_matrix
        if not issparse(X):
            X = csr_matrix(X)
        return model.predict_proba(X)

    explainer = shap.KernelExplainer(predict_proba, X_bg)
    X_explain = X_sample[:min(200, X_sample.shape[0])]
    shap_values = explainer.shap_values(X_explain, nsamples=100)

    # Binary: take class-1 shap values
    vals = shap_values[1] if isinstance(shap_values, list) else shap_values

    _plot_global_importance(vals, feature_names, dataset, model_key, n_top, out_dir,
                            label="Real (positive class)")
    print(f"  SHAP plots saved → {out_dir}")


# ---------------------------------------------------------------------------
# Shared plot helpers
# ---------------------------------------------------------------------------

def _plot_global_importance(vals, feature_names, dataset, model_key, n_top, out_dir, label=""):
    mean_abs = np.abs(vals).mean(axis=0)
    top_idx = np.argsort(mean_abs)[-n_top:][::-1]
    top_names = [feature_names[i] for i in top_idx]
    top_vals = mean_abs[top_idx]

    fig, ax = plt.subplots(figsize=(10, max(6, n_top * 0.35)))
    colors = ["#e74c3c" if v > 0 else "#3498db" for v in top_vals]
    ax.barh(top_names[::-1], top_vals[::-1], color="#3498db")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(
        f"Top {n_top} TF-IDF Features by SHAP Importance\n"
        f"{model_key.upper()} on {dataset}  ({label})",
        fontsize=12
    )
    plt.tight_layout()
    path = os.path.join(out_dir, f"global_importance_top{n_top}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_beeswarm(shap_explanation, feature_names, out_dir, dataset, model_key):
    try:
        fig = plt.figure(figsize=(10, 8))
        shap.plots.beeswarm(shap_explanation, max_display=20, show=False)
        plt.title(f"SHAP Beeswarm — {model_key.upper()} on {dataset}", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "beeswarm.png"), dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"  Beeswarm plot skipped: {e}")


def _plot_waterfall_example(shap_explanation, sample_df, feature_names, out_dir, dataset, model_key):
    try:
        # Pick one fake and one real example
        for label_val, label_name in [(0, "fake"), (1, "real")]:
            mask = sample_df["label"].values == label_val
            if not mask.any():
                continue
            idx = np.where(mask)[0][0]
            fig = plt.figure(figsize=(12, 6))
            shap.plots.waterfall(shap_explanation[idx], max_display=15, show=False)
            plt.title(
                f"SHAP Waterfall — {model_key.upper()} on {dataset}\n"
                f"Example: {label_name} article",
                fontsize=11
            )
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"waterfall_{label_name}.png"),
                        dpi=150, bbox_inches="tight")
            plt.close()
    except Exception as e:
        print(f"  Waterfall plot skipped: {e}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SHAP explainability for classical TF-IDF models")
    parser.add_argument("--dataset", nargs="+", default=["isot"],
                        choices=["isot", "welfake", "combined", "all"])
    parser.add_argument("--model", nargs="+", default=["lr"],
                        choices=["lr", "svc", "rf", "xgb", "all"])
    parser.add_argument("--n-top", type=int, default=20,
                        help="Number of top features to display")
    parser.add_argument("--sample", type=int, default=500,
                        help="Number of test examples to explain (speed vs. accuracy tradeoff)")
    args = parser.parse_args()

    datasets = list(DATASET_CONFIGS.keys()) if "all" in args.dataset else args.dataset
    models   = ["lr", "svc", "rf", "xgb"]  if "all" in args.model   else args.model

    for ds in datasets:
        for mk in models:
            try:
                run_shap(ds, mk, args.n_top, args.sample)
            except FileNotFoundError as e:
                print(f"  Skipped ({e})")


if __name__ == "__main__":
    main()
