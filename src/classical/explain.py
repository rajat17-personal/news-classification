"""
SHAP explainability for TF-IDF classical models.

Usage:
  python -m src.classical.explain --dataset isot --model lr
"""
import argparse
import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import shap

from ..utils.config import (
    CACHE_PATH, CHECKPOINTS_PATH, RESULTS_PATH,
    ISOT_CONFIG, WELFAKE_CONFIG, NELA_DL_CONFIG,
)
from ..utils.data_loader import load_splits
from ..utils.combined_loader import COMBINED_CONFIG
from .features import TFIDFFeaturizer

DATASET_CONFIGS = {
    "isot":              ISOT_CONFIG,
    "welfake":           WELFAKE_CONFIG,
    "combined":          COMBINED_CONFIG,
    "nela_sampled_500k": NELA_DL_CONFIG,
}

BINARY_LABEL_NAMES  = ["Fake (0)", "Real (1)"]
NELA_LABEL_NAMES    = ["Reliable (0)", "Questionable (1)", "Conspiracy (2)"]

def _label_names_for(dataset: str) -> list[str]:
    if dataset.startswith("nela"):
        return NELA_LABEL_NAMES
    return BINARY_LABEL_NAMES

SHAP_OUT_DIR = os.path.join(RESULTS_PATH, "shap")


def _load_model(dataset: str, model_key: str):
    """Load best checkpoint for a classical model."""
    path = os.path.join(CHECKPOINTS_PATH, "classical", dataset, f"{model_key}_best.joblib")
    if not os.path.exists(path):
        path = os.path.join(CHECKPOINTS_PATH, "classical", dataset, f"{model_key}.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No checkpoint found for {model_key} on {dataset} at {path}")
    return joblib.load(path)

def _load_featurizer(dataset: str, model_key: str) -> TFIDFFeaturizer:
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
        return model
    return model


def run_shap(dataset: str, model_key: str, n_top: int, n_sample: int):
    print(f"\n=== SHAP: {model_key.upper()} on {dataset} ===")
    config = DATASET_CONFIGS[dataset]
    _, _, test_df = load_splits(config)

    featurizer = _load_featurizer(dataset, model_key)
    model = _load_model(dataset, model_key)
    feature_names = featurizer.vectorizer.get_feature_names_out()
    label_names = _label_names_for(dataset)

    rng = np.random.default_rng(42)
    idx = rng.choice(len(test_df), size=min(n_sample, len(test_df)), replace=False)
    sample_df = test_df.iloc[idx].reset_index(drop=True)
    X_sample = featurizer.transform(sample_df["text"].tolist())

    out_dir = os.path.join(SHAP_OUT_DIR, f"{dataset}_{model_key}")
    os.makedirs(out_dir, exist_ok=True)

    if model_key == "lr":
        _shap_linear(model, X_sample, feature_names, sample_df, dataset, model_key,
                    n_top, out_dir, label_names)
    elif model_key == "xgb":
        _shap_tree(model, X_sample, feature_names, sample_df, dataset, model_key,
                n_top, out_dir, label_names)
    elif model_key in ("svc", "rf"):
        _shap_kernel(model, X_sample, feature_names, sample_df, dataset, model_key,
                    n_top, out_dir, label_names, n_bg=100)
    else:
        print(f"  Unsupported model key: {model_key}")

def _shap_linear(model, X_sample, feature_names, sample_df, dataset, model_key,
                n_top, out_dir, label_names):
    explainer = shap.LinearExplainer(model, X_sample, feature_perturbation="interventional")
    shap_values = explainer(X_sample)
    shap_values.feature_names = list(feature_names)

    vals = shap_values.values  # (N, F) binary or (N, F, C) multiclass

    if vals.ndim == 3:
        # Multi-class: one plot per class
        for cls_idx, cls_name in enumerate(label_names):
            cls_vals = vals[:, :, cls_idx]
            _plot_global_importance(cls_vals, feature_names, dataset, model_key, n_top, out_dir,
                                    label=cls_name, cls_idx=cls_idx)
            sv_cls = shap_values[:, :, cls_idx]
            _plot_beeswarm(sv_cls, feature_names, out_dir, dataset, model_key, cls_idx=cls_idx)
            _plot_waterfall_example(sv_cls, sample_df, feature_names, out_dir, dataset,
                                    model_key, label_names, cls_idx=cls_idx)
    else:
        # Binary
        _plot_global_importance(vals, feature_names, dataset, model_key, n_top, out_dir,
                                label=label_names[-1])
        _plot_beeswarm(shap_values, feature_names, out_dir, dataset, model_key)
        _plot_waterfall_example(shap_values, sample_df, feature_names, out_dir, dataset,
                                model_key, label_names)

    print(f"  SHAP plots saved → {out_dir}")

# XGboost
def _shap_tree(model, X_sample, feature_names, sample_df, dataset, model_key,
               n_top, out_dir, label_names):
    X_dense = X_sample.toarray().astype(np.float32)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_dense)

    # Multi-class: list of (N, F) arrays, one per class
    # Binary: single (N, F) array or list of 2
    if isinstance(shap_values, list) and len(shap_values) > 2:
        for cls_idx, cls_name in enumerate(label_names):
            vals = shap_values[cls_idx]
            _plot_global_importance(vals, feature_names, dataset, model_key, n_top, out_dir,
                                    label=cls_name, cls_idx=cls_idx)
            shap.summary_plot(vals, X_dense, feature_names=list(feature_names),
                              show=False, max_display=n_top)
            plt.title(f"SHAP Summary — {model_key.upper()} on {dataset} ({cls_name})", fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"summary_dot_cls{cls_idx}.png"),
                        dpi=150, bbox_inches="tight")
            plt.close()
    else:
        vals = shap_values[1] if isinstance(shap_values, list) else shap_values
        _plot_global_importance(vals, feature_names, dataset, model_key, n_top, out_dir,
                                label=label_names[-1])
        shap.summary_plot(vals, X_dense, feature_names=list(feature_names),
                          show=False, max_display=n_top)
        plt.title(f"SHAP Summary — {model_key.upper()} on {dataset}", fontsize=13)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "summary_dot.png"), dpi=150, bbox_inches="tight")
        plt.close()

    print(f"  SHAP plots saved → {out_dir}")

#RF/SVC
def _shap_kernel(model, X_sample, feature_names, sample_df, dataset, model_key,
                 n_top, out_dir, label_names, n_bg=100):
    print(f"  Using KernelExplainer (approximate) — n_bg={n_bg}, n_sample={X_sample.shape[0]}")
    X_bg = shap.kmeans(X_sample, n_bg)

    def predict_proba(X):
        from scipy.sparse import issparse, csr_matrix
        if not issparse(X):
            X = csr_matrix(X)
        return model.predict_proba(X)

    explainer = shap.KernelExplainer(predict_proba, X_bg)
    X_explain = X_sample[:min(200, X_sample.shape[0])]
    shap_values = explainer.shap_values(X_explain, nsamples=100)

    if isinstance(shap_values, list) and len(shap_values) > 2:
        for cls_idx, cls_name in enumerate(label_names):
            _plot_global_importance(shap_values[cls_idx], feature_names, dataset, model_key,
                                    n_top, out_dir, label=cls_name, cls_idx=cls_idx)
    else:
        vals = shap_values[1] if isinstance(shap_values, list) else shap_values
        _plot_global_importance(vals, feature_names, dataset, model_key, n_top, out_dir,
                                label=label_names[-1])

    print(f"  SHAP plots saved → {out_dir}")

def _plot_global_importance(vals, feature_names, dataset, model_key, n_top, out_dir,
                            label="", cls_idx=None):
    mean_abs = np.abs(vals).mean(axis=0)
    top_idx = np.argsort(mean_abs)[-n_top:][::-1]
    top_names = [feature_names[i] for i in top_idx]
    top_vals = mean_abs[top_idx]

    fig, ax = plt.subplots(figsize=(10, max(6, n_top * 0.35)))
    ax.barh(top_names[::-1], top_vals[::-1], color="#3498db")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(
        f"Top {n_top} TF-IDF Features by SHAP Importance\n"
        f"{model_key.upper()} on {dataset}  ({label})",
        fontsize=12
    )
    plt.tight_layout()
    suffix = f"_cls{cls_idx}" if cls_idx is not None else ""
    path = os.path.join(out_dir, f"global_importance_top{n_top}{suffix}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_beeswarm(shap_explanation, feature_names, out_dir, dataset, model_key, cls_idx=None):
    try:
        fig = plt.figure(figsize=(10, 8))
        shap.plots.beeswarm(shap_explanation, max_display=20, show=False)
        suffix = f" — class {cls_idx}" if cls_idx is not None else ""
        plt.title(f"SHAP Beeswarm — {model_key.upper()} on {dataset}{suffix}", fontsize=12)
        plt.tight_layout()
        fname = f"beeswarm_cls{cls_idx}.png" if cls_idx is not None else "beeswarm.png"
        plt.savefig(os.path.join(out_dir, fname), dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"  Beeswarm plot skipped: {e}")


def _plot_waterfall_example(shap_explanation, sample_df, feature_names, out_dir, dataset,
                            model_key, label_names, cls_idx=None):
    try:
        for label_val, label_name in enumerate(label_names):
            mask = sample_df["label"].values == label_val
            if not mask.any():
                continue
            idx = np.where(mask)[0][0]
            fig = plt.figure(figsize=(12, 6))
            shap.plots.waterfall(shap_explanation[idx], max_display=15, show=False)
            cls_str = f"_cls{cls_idx}" if cls_idx is not None else ""
            plt.title(
                f"SHAP Waterfall — {model_key.upper()} on {dataset}\n"
                f"Example: {label_name}",
                fontsize=11
            )
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"waterfall_{label_val}{cls_str}.png"),
                        dpi=150, bbox_inches="tight")
            plt.close()
    except Exception as e:
        print(f"  Waterfall plot skipped: {e}")

def main():
    parser = argparse.ArgumentParser(description="SHAP explainability for classical TF-IDF models")
    parser.add_argument("--dataset", nargs="+", default=["isot"],
                        choices=["isot", "welfake", "combined", "nela_sampled_500k", "all"])
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
