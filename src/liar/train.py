import argparse
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

from ..utils.config import (
    CHECKPOINTS_PATH, GLOVE_DIM, MAX_SEQ_LEN_DL, MAX_SEQ_LEN_TRANSFORMER, SEED,
)
from ..utils.metrics import MetricsBundle, _sanitise
from ..utils.seeds import set_seed
from .dataset import load_liar_splits, LIAR_LABELS, LIAR_LABEL_NAMES

NUM_CLASSES = 6
GLOVE_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "glove" / "glove.840B.300d.txt"


def _evaluate_liar(predict_fn, test_df, model_name: str, train_time_sec: float, hyperparams: dict):
    """evaluate() equivalent for 6-class: no ROC-AUC (multiclass needs macro-OvR)."""
    import json, time
    from datetime import datetime
    from sklearn.metrics import roc_auc_score
    from ..utils.config import RESULTS_PATH

    y_true = test_df["label"].values
    texts  = test_df["text"].tolist()
    y_pred, y_proba = predict_fn(texts)

    accuracy   = accuracy_score(y_true, y_pred)
    macro_f1   = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    per_class  = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    conf_mat   = confusion_matrix(y_true, y_pred)

    try:
        roc_auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
    except Exception:
        roc_auc = None

    result = {
        "model_name": model_name,
        "dataset_name": "liar",
        "split": "test",
        "num_classes": NUM_CLASSES,
        "label_names": LIAR_LABEL_NAMES,
        "train_time_sec": train_time_sec,
        "hyperparams": hyperparams,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "roc_auc": roc_auc,
        "per_class_metrics": {
            "precision": per_class[0].tolist(),
            "recall":    per_class[1].tolist(),
            "f1":        per_class[2].tolist(),
            "support":   per_class[3].tolist(),
        },
        "confusion_matrix": conf_mat.tolist(),
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    runs_dir = os.path.join(RESULTS_PATH, "runs")
    os.makedirs(runs_dir, exist_ok=True)
    fname = os.path.join(runs_dir, f"{model_name}_liar_test_{timestamp}.json")
    with open(fname, "w") as f:
        import json
        json.dump(_sanitise(result), f, indent=2)

    print(f"  acc={accuracy:.4f}  macro_f1={macro_f1:.4f}"
        + (f"  auc={roc_auc:.4f}" if roc_auc else ""))
    print(f"  Per-class F1: " +
        "  ".join(f"{LIAR_LABEL_NAMES[i]}={per_class[2][i]:.3f}"
                    for i in range(NUM_CLASSES) if i < len(per_class[2])))
    return result

def run_classical(models: list[str]):
    import time
    from ..classical.features import TFIDFFeaturizer
    from ..classical.models import MODEL_REGISTRY
    from sklearn.model_selection import cross_val_score

    train_df, val_df, test_df = load_liar_splits()
    # combine train+val for classical (small dataset — squeeze more signal)
    fit_df = _concat(train_df, val_df)

    featurizer = TFIDFFeaturizer(max_features=30000, ngram_range=(1, 2))
    X_train = featurizer.fit_transform(fit_df["text"].tolist())
    X_test  = featurizer.transform(test_df["text"].tolist())
    y_train = fit_df["label"].values
    y_test  = test_df["label"].values

    for model_key in models:
        print(f"\n=== {model_key.upper()} on LIAR (6-class) ===")
        cls = MODEL_REGISTRY[model_key]
        model = cls()
        # for multiclass we need to check SVC (CalibratedClassifierCV handles it automatically)
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - t0

        ckpt_dir = os.path.join(CHECKPOINTS_PATH, "classical", "liar")
        os.makedirs(ckpt_dir, exist_ok=True)
        model.save(os.path.join(ckpt_dir, f"{model_key}.joblib"))

        def predict_fn(texts, _feat=featurizer, _m=model):
            X = _feat.transform(texts)
            return _m.predict(X), _m.predict_proba(X)

        _evaluate_liar(predict_fn, test_df, f"{model_key}_liar", train_time,
                    {"model": model_key, "max_features": 30000, "ngram_range": "(1,2)"})

def run_deep_learning(archs: list[str], freeze_modes: list[str], epochs: int, batch_size: int):
    import time
    from ..deep_learning.dataset import VocabBuilder, build_dataloaders, TextDataset
    from ..deep_learning.embeddings import GloVeLoader
    from ..deep_learning.models import BiLSTM, TextCNN
    from ..deep_learning.trainer import Trainer, TrainerConfig

    train_df, val_df, test_df = load_liar_splits()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vocab_builder = VocabBuilder()
    vocab = vocab_builder.build(train_df["text"].tolist())
    glove = GloVeLoader(GLOVE_PATH, dim=GLOVE_DIM)

    for arch in archs:
        for freeze_mode in freeze_modes:
            freeze = freeze_mode == "frozen"
            name = f"{arch}_{freeze_mode}_liar"
            print(f"\n=== {name.upper()} ===")

            train_loader, val_loader, test_loader = build_dataloaders(
                train_df, val_df, test_df, vocab,
                max_len=MAX_SEQ_LEN_DL, batch_size=batch_size,
            )
            embedding = glove.build_matrix(vocab, freeze=freeze)
            model = (BiLSTM(embedding, num_classes=NUM_CLASSES)
                    if arch == "bilstm"
                    else TextCNN(embedding, num_classes=NUM_CLASSES))

            cfg = TrainerConfig(epochs=epochs, batch_size=batch_size)
            trainer = Trainer(model, cfg, name, "liar")
            history = trainer.train(train_loader, val_loader)

            ckpt_path = os.path.join(CHECKPOINTS_PATH, "deep_learning", "liar", f"{name}.pt")
            model.load_state_dict(torch.load(ckpt_path, map_location=cfg.device))
            model.to(cfg.device).eval()

            def predict_fn(texts, _m=model, _v=vocab, _dev=cfg.device):
                ds = TextDataset(texts, [0] * len(texts), _v, MAX_SEQ_LEN_DL)
                all_preds, all_probas = [], []
                with torch.no_grad():
                    for i in range(0, len(ds), batch_size):
                        ids = torch.stack(
                            [ds[j]["input_ids"] for j in range(i, min(i + batch_size, len(ds)))]
                        ).to(_dev)
                        logits = _m(ids)
                        probas = torch.softmax(logits, dim=-1).cpu().numpy()
                        all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
                        all_probas.append(probas)
                return np.array(all_preds), np.vstack(all_probas)

            _evaluate_liar(predict_fn, test_df, name, history.train_time_sec,
                        {"arch": arch, "freeze_mode": freeze_mode, "epochs": history.best_epoch})

def run_transformer(archs: list[str], freeze_modes: list[str], epochs: int, batch_size: int, fp16: bool):
    import time
    from transformers import AutoTokenizer
    from ..transformers_.models import TransformerClassifier
    from ..transformers_.dataset import build_dataloaders
    from ..transformers_.trainer import TransformerTrainer, TransformerTrainerConfig
    from ..transformers_.dataset import TransformerDataset

    HF_MODEL_NAMES = {"bert": "bert-base-uncased", "roberta": "roberta-base"}

    train_df, val_df, test_df = load_liar_splits()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for arch in archs:
        hf_name = HF_MODEL_NAMES[arch]
        tokenizer = AutoTokenizer.from_pretrained(hf_name)

        train_loader, val_loader, test_loader = build_dataloaders(
            train_df, val_df, test_df,
            tokenizer=tokenizer,
            max_len=MAX_SEQ_LEN_TRANSFORMER,
            batch_size=batch_size,
        )

        for freeze_mode in freeze_modes:
            freeze = freeze_mode == "frozen"
            name = f"{arch}_{freeze_mode}_liar"
            print(f"\n=== {name.upper()} ===")

            model = TransformerClassifier(hf_name, num_classes=NUM_CLASSES)
            if freeze:
                model.freeze_encoder()

            cfg = TransformerTrainerConfig(
                model_name=hf_name,
                dataset_name="liar",
                num_epochs=epochs,
                batch_size=batch_size,
                fp16=fp16,
                patience=2,
            )
            trainer = TransformerTrainer(cfg)
            history = trainer.train(model, train_loader, val_loader, variant_name=name)

            ckpt_path = cfg.save_dir / f"{name}.pt"
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            model.to(device).eval()

            def predict_fn(texts, _m=model, _tok=tokenizer, _dev=device):
                texts = [t if isinstance(t, str) else "" for t in texts]
                ds = TransformerDataset(texts, [0] * len(texts), _tok, MAX_SEQ_LEN_TRANSFORMER)
                all_preds, all_probas = [], []
                with torch.no_grad():
                    for i in range(0, len(ds), batch_size):
                        end = min(i + batch_size, len(ds))
                        items = [ds[j] for j in range(i, end)]
                        ids  = torch.stack([b["input_ids"]     for b in items]).to(_dev)
                        mask = torch.stack([b["attention_mask"] for b in items]).to(_dev)
                        ttids = None
                        if "token_type_ids" in items[0]:
                            ttids = torch.stack([b["token_type_ids"] for b in items]).to(_dev)
                        logits = _m(ids, mask, ttids)
                        probas = torch.softmax(logits, dim=-1).cpu().numpy()
                        all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
                        all_probas.append(probas)
                return np.array(all_preds), np.vstack(all_probas)

            _evaluate_liar(predict_fn, test_df, name, history.train_time_sec,
                        {"arch": arch, "hf_model": hf_name, "freeze_mode": freeze_mode,
                            "epochs": history.best_epoch, "fp16": fp16})

def _concat(*dfs):
    import pandas as pd
    return pd.concat(list(dfs), ignore_index=True)

def main():
    parser = argparse.ArgumentParser(description="P9.1 — Train on LIAR 6-class benchmark")
    parser.add_argument("--tier", required=True,
                        choices=["classical", "deep_learning", "transformer"])
    parser.add_argument("--model", nargs="+", default=["all"],
                        choices=["lr", "svc", "rf", "xgb",
                                "bilstm", "textcnn",
                                "bert", "roberta", "all"])
    parser.add_argument("--freeze", nargs="+", default=["finetuned"],
                        choices=["frozen", "finetuned", "both"])
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override epochs (defaults: classical=N/A, DL=20, transformer=5)")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--no-fp16", action="store_true")
    args = parser.parse_args()

    set_seed(SEED)
    fp16 = not args.no_fp16

    if args.tier == "classical":
        classical_models = ["lr", "svc", "rf", "xgb"] if "all" in args.model else args.model
        run_classical(classical_models)

    elif args.tier == "deep_learning":
        archs = ["bilstm", "textcnn"] if "all" in args.model else [
            m for m in args.model if m in ("bilstm", "textcnn")]
        freeze_modes = ["frozen", "finetuned"] if "both" in args.freeze else args.freeze
        epochs = args.epochs or 20
        batch_size = args.batch_size or 64
        run_deep_learning(archs, freeze_modes, epochs, batch_size)

    elif args.tier == "transformer":
        archs = ["bert", "roberta"] if "all" in args.model else [
            m for m in args.model if m in ("bert", "roberta")]
        freeze_modes = ["frozen", "finetuned"] if "both" in args.freeze else args.freeze
        epochs = args.epochs or 5
        batch_size = args.batch_size or 16
        run_transformer(archs, freeze_modes, epochs, batch_size, fp16)


if __name__ == "__main__":
    main()
