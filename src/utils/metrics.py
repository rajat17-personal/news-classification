# evaluate(predict_fn, test_df, model_name, dataset_name, train_time_sec, hyperparams, split="test") -> MetricsBundle:

# Call predict_fn(test_df["text"].tolist()) → y_pred: np.ndarray
# Time a second CPU-only pass for latency (batch_size=1, 100-sample subset, report ms/sample)
# Compute: accuracy, macro_f1, weighted_f1, per-class P/R/F1, ROC-AUC, confusion matrix
# Build MetricsBundle
# Write JSON to RESULTS_RUNS / f"{model_name}_{dataset_name}_{split}_{timestamp}.json"
# Return bundle
# predict_fn contract:

# Input: list[str] of raw texts
# Output: np.ndarray of integer class predictions (shape [N])
# For AUC: the function may also return probas if the caller wraps predict_proba
import json
import os
import time
import numpy as np
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
from .config import RESULTS_PATH

LATENCY_SAMPLE_SIZE = 100


def _sanitise(obj):
    """Recursively replace float NaN/Inf with None so json.dump stays valid."""
    import math
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _sanitise(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitise(v) for v in obj]
    return obj  # number of individual samples used for CPU latency measurement

class MetricsBundle:
    def __init__(self, model_name, dataset_name, split, train_time_sec, inference_ms_per_sample,
                 hyperparams, accuracy, macro_f1, weighted_f1, per_class_metrics, roc_auc,
                 confusion_matrix, num_classes=2):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.split = split
        self.num_classes = num_classes
        self.train_time_sec = train_time_sec
        self.inference_ms_per_sample = inference_ms_per_sample
        self.hyperparams = hyperparams
        self.accuracy = accuracy
        self.macro_f1 = macro_f1
        self.weighted_f1 = weighted_f1
        self.per_class_metrics = per_class_metrics
        self.roc_auc = roc_auc
        self.confusion_matrix = confusion_matrix

    def to_dict(self):
        return {
            "model_name": self.model_name,
            "dataset_name": self.dataset_name,
            "split": self.split,
            "num_classes": self.num_classes,
            "train_time_sec": self.train_time_sec,
            "inference_ms_per_sample": self.inference_ms_per_sample,
            "hyperparams": self.hyperparams,
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "weighted_f1": self.weighted_f1,
            "per_class_metrics": self.per_class_metrics,
            "roc_auc": self.roc_auc,
            "confusion_matrix": self.confusion_matrix.tolist(),
        }

    def save(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.model_name}_{self.dataset_name}_{self.split}_{timestamp}.json"
        runs_dir = os.path.join(RESULTS_PATH, "runs")
        os.makedirs(runs_dir, exist_ok=True)
        with open(os.path.join(runs_dir, filename), "w") as f:
            json.dump(_sanitise(self.to_dict()), f, indent=2)


def _measure_latency(predict_fn, texts: list[str]) -> float:
    """CPU-only latency: run predict_fn one sample at a time, return mean ms/sample."""
    sample = texts[:LATENCY_SAMPLE_SIZE]
    times = []
    for text in sample:
        t0 = time.perf_counter()
        predict_fn([text])
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times))


def evaluate(predict_fn, test_df, model_name, dataset_name, train_time_sec, hyperparams, split="test"):
    y_true = test_df["label"].values
    texts = test_df["text"].tolist()

    y_pred, y_proba = predict_fn(texts)

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    per_class = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    conf_matrix = confusion_matrix(y_true, y_pred)

    n_classes = y_proba.shape[1]
    if n_classes == 2:
        roc_auc = roc_auc_score(y_true, y_proba[:, 1])
    else:
        roc_auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")

    inference_ms = _measure_latency(predict_fn, texts)

    bundle = MetricsBundle(
        model_name=model_name,
        dataset_name=dataset_name,
        split=split,
        num_classes=n_classes,
        train_time_sec=train_time_sec,
        inference_ms_per_sample=inference_ms,
        hyperparams=hyperparams,
        accuracy=accuracy,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        per_class_metrics={
            "precision": per_class[0].tolist(),
            "recall": per_class[1].tolist(),
            "f1": per_class[2].tolist(),
            "support": per_class[3].tolist(),
        },
        roc_auc=roc_auc,
        confusion_matrix=conf_matrix,
    )

    bundle.save()
    return bundle

