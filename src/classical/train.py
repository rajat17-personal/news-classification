import argparse
import os
import time
from sklearn.model_selection import RandomizedSearchCV

from ..utils.config import (
    CHECKPOINTS_PATH, ISOT_CONFIG, WELFAKE_CONFIG,
    MAX_TFIDF_FEATURES, MAX_TFIDF_XGB, SEED,
    NELA_CONFIG, NELA_SAMPLED_CONFIG, NELA_DL_CONFIG,
)
from ..utils.data_loader import load_splits
from ..utils.metrics import evaluate
from ..utils.seeds import set_seed
from .features import TFIDFFeaturizer
from .models import MODEL_REGISTRY

from ..utils.combined_loader import COMBINED_CONFIG

DATASET_CONFIGS = {
    'isot': ISOT_CONFIG,
    'welfake': WELFAKE_CONFIG,
    'combined': COMBINED_CONFIG,
    'nela': NELA_CONFIG,
    'nela_sampled_100k': NELA_SAMPLED_CONFIG,
    'nela_sampled_500k': NELA_DL_CONFIG,
}

TUNE_GRIDS = {
    'lr':  {'C': [0.01, 0.1, 1, 10]},
    'svc': {'estimator__C': [0.01, 0.1, 1, 10]},
    'rf':  {'n_estimators': [100, 200, 300], 'max_depth': [None, 10, 20],
             'max_features': ['sqrt', 'log2', 0.1], 'min_samples_leaf': [1, 2, 5]},
    'xgb': {'n_estimators': [200, 300, 500], 'learning_rate': [0.05, 0.1, 0.3], 'max_depth': [3, 5, 7]},
}


def _max_features(model_key: str) -> int:
    return MAX_TFIDF_XGB if model_key == 'xgb' else MAX_TFIDF_FEATURES


def _tune(model, model_key: str, X_train, y_train) -> tuple[dict, list[dict]]:
    param_grid = TUNE_GRIDS[model_key]
    # cap parallel CV workers to avoid OOM on large datasets — RF and XGB are memory-heavy
    n_jobs = 2 if model_key in ('rf', 'xgb') else 4
    search = RandomizedSearchCV(
        model.model, param_grid, n_iter=10, cv=3, scoring='f1_macro',
        random_state=SEED, n_jobs=n_jobs,
    )
    search.fit(X_train, y_train)
    model.model = search.best_estimator_

    cv_results = [
        {"params": search.cv_results_["params"][i],
         "mean_cv_f1": round(float(search.cv_results_["mean_test_score"][i]), 4)}
        for i in range(len(search.cv_results_["params"]))
    ]
    cv_results.sort(key=lambda x: x["mean_cv_f1"], reverse=True)

    return search.best_params_, cv_results


def run(dataset_name: str, model_keys: list, tune: str):
    set_seed(SEED)
    config = DATASET_CONFIGS[dataset_name]
    train_df, _, test_df = load_splits(config)

    results = []
    tune_logs = {}  # model_key → list of cv candidate dicts
    for model_key in model_keys:
        print(f"\n=== {model_key.upper()} on {dataset_name} ===")

        featurizer = TFIDFFeaturizer(max_features=_max_features(model_key))
        X_train = featurizer.fit_transform(train_df['text'].tolist())
        featurizer.save(f"{dataset_name}_{model_key}")

        y_train = train_df['label'].values

        # For RF on large datasets (>200k rows): reduce trees and subsample
        # to stay within RAM. max_features='sqrt' is already set in RFModel.
        if model_key == 'rf' and len(train_df) > 200_000:
            model = MODEL_REGISTRY[model_key](n_estimators=50, max_samples=0.5)
            print(f"  Large dataset detected — RF: n_estimators=50, max_samples=0.5")
        else:
            model = MODEL_REGISTRY[model_key]()

        tuned_params = {}
        if tune != 'none':
            print(f"  Tuning {model_key}...")
            tuned_params, cv_results = _tune(model, model_key, X_train, y_train)
            tune_logs[model_key] = cv_results
            print(f"  Best params: {tuned_params}")

        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - t0

        ckpt_dir = os.path.join(CHECKPOINTS_PATH, 'classical', dataset_name)
        os.makedirs(ckpt_dir, exist_ok=True)
        model.save(os.path.join(ckpt_dir, f"{model_key}.joblib"))

        # full parameter record for JSON: tfidf settings + all model params
        # SVCModel wraps CalibratedClassifierCV — pull inner LinearSVC params to stay serializable
        raw_params = (
            {**model.model.get_params(deep=False),
             **{f"estimator__{k}": v for k, v in model.model.estimator.get_params().items()}}
            if model_key == 'svc'
            else model.model.get_params()
        )
        hyperparams = {
            "tfidf_max_features": _max_features(model_key),
            "tfidf_ngram_range": "(1, 2)",
            "tfidf_sublinear_tf": True,
            "tfidf_min_df": 2,
            **{k: v for k, v in raw_params.items() if k != 'estimator'},
            **tuned_params,
        }
        # only the keys that were actually searched — shown in MD summary
        display_params = tuned_params if tuned_params else {"tuning": "default"}

        def predict_fn(texts: list, _model=model, _feat=featurizer):
            X = _feat.transform(texts)
            return _model.predict(X), _model.predict_proba(X)

        bundle = evaluate(
            predict_fn=predict_fn,
            test_df=test_df,
            model_name=f"{model_key}_{dataset_name}",
            dataset_name=dataset_name,
            train_time_sec=train_time,
            hyperparams=hyperparams,
        )
        results.append((bundle, display_params))
        print(f"  acc={bundle.accuracy:.4f}  macro_f1={bundle.macro_f1:.4f}  auc={bundle.roc_auc:.4f}")

        _save_best_checkpoint(ckpt_dir, model_key, model, featurizer, hyperparams, bundle)

    # summary table — stdout
    print(f"\n{'Model':<12} {'Accuracy':<10} {'Macro F1':<10} {'ROC-AUC':<10} {'Train(s)':<10}")
    print("-" * 52)
    for b, _ in results:
        name = b.model_name.split('_')[0].upper()
        print(f"{name:<12} {b.accuracy:<10.4f} {b.macro_f1:<10.4f} {b.roc_auc:<10.4f} {b.train_time_sec:<10.1f}")

    _save_summary(results, tune_logs, dataset_name)


def _save_best_checkpoint(ckpt_dir: str, model_key: str, model, featurizer, hyperparams: dict, bundle) -> None:
    import json
    meta_path = os.path.join(ckpt_dir, f"{model_key}_best_meta.json")

    # load previous best macro_f1 if it exists
    prev_f1 = -1.0
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            prev_f1 = json.load(f).get("macro_f1", -1.0)

    if bundle.macro_f1 > prev_f1:
        model.save(os.path.join(ckpt_dir, f"{model_key}_best.joblib"))
        featurizer.save(f"best_{model_key}_{os.path.basename(ckpt_dir)}")
        meta = {
            "model_key": model_key,
            "dataset": os.path.basename(ckpt_dir),
            "macro_f1": round(bundle.macro_f1, 6),
            "accuracy": round(bundle.accuracy, 6),
            "roc_auc": round(bundle.roc_auc, 6),
            "hyperparams": hyperparams,
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"  Best checkpoint updated ({prev_f1:.4f} → {bundle.macro_f1:.4f}) → {model_key}_best.joblib")
    else:
        print(f"  No improvement ({bundle.macro_f1:.4f} ≤ {prev_f1:.4f}), best checkpoint unchanged")


def _save_summary(results: list[tuple], tune_logs: dict, dataset_name: str) -> None:
    from datetime import datetime
    from ..utils.config import RESULTS_PATH

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_dir = os.path.join(RESULTS_PATH, "summaries")
    os.makedirs(summary_dir, exist_ok=True)
    filename = f"classical_{dataset_name}_{timestamp}.md"

    md_lines = [
        f"# Classical Tier — {dataset_name} ({timestamp})\n",
        "## Results\n",
        "| Model | Accuracy | Macro F1 | Weighted F1 | ROC-AUC | Train(s) | Latency(ms) | Tuned Params |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for b, display_params in results:
        md_lines.append(
            f"| {b.model_name.split('_')[0].upper()} "
            f"| {b.accuracy:.4f} | {b.macro_f1:.4f} | {b.weighted_f1:.4f} "
            f"| {b.roc_auc:.4f} | {b.train_time_sec:.2f} | {b.inference_ms_per_sample:.3f} "
            f"| {display_params} |"
        )

    if tune_logs:
        md_lines += ["\n## Tuning Runs (all candidates, sorted by CV Macro F1)\n"]
        for model_key, candidates in tune_logs.items():
            md_lines.append(f"### {model_key.upper()}\n")
            param_keys = list(candidates[0]["params"].keys())
            header = "| " + " | ".join(param_keys) + " | CV Macro F1 |"
            sep    = "| " + " | ".join(["---"] * len(param_keys)) + " | --- |"
            md_lines += [header, sep]
            for c in candidates:
                vals = " | ".join(str(c["params"][k]) for k in param_keys)
                best_marker = " ✓" if c == candidates[0] else ""
                md_lines.append(f"| {vals} | {c['mean_cv_f1']}{best_marker} |")
            md_lines.append("")

    with open(os.path.join(summary_dir, filename), "w") as f:
        f.write("\n".join(md_lines) + "\n")

    print(f"\n  Summary saved → results/summaries/{filename}")


def main():
    parser = argparse.ArgumentParser(description="Train Tier-1 classical models")
    parser.add_argument('--dataset', required=True, choices=['isot', 'welfake', 'combined', 'nela', 'nela_sampled_100k', 'nela_sampled_500k'])
    parser.add_argument('--model', default=['all'], nargs='+', choices=['lr', 'svc', 'rf', 'xgb', 'all'])
    parser.add_argument('--tune', default='none', choices=['none', 'random'])
    args = parser.parse_args()

    model_keys = list(MODEL_REGISTRY.keys()) if 'all' in args.model else args.model
    run(args.dataset, model_keys, args.tune)


if __name__ == '__main__':
    main()
