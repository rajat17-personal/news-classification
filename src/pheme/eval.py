"""
PHEME OOD evaluation runner.

Loads saved best checkpoints from each tier and evaluates on PHEME test split.
Results are saved to results/runs/ with split="ood_pheme" and picked up by
the results aggregator automatically.

Usage:
  # Download + preprocess first (once):
  python -m src.pheme.dataset --download --preprocess

  # Then evaluate all tiers:
  python -m src.pheme.eval --tier all
  python -m src.pheme.eval --tier classical
  python -m src.pheme.eval --tier deep_learning
  python -m src.pheme.eval --tier transformer
"""
import argparse
import os
from pathlib import Path

import numpy as np
import torch

from ..utils.config import (
    CHECKPOINTS_PATH, GLOVE_DIM, MAX_SEQ_LEN_DL, MAX_SEQ_LEN_TRANSFORMER,
)
from ..utils.metrics import evaluate
from .dataset import load_pheme_splits

GLOVE_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "glove" / "glove.840B.300d.txt"


# ---------------------------------------------------------------------------
# Classical
# ---------------------------------------------------------------------------

def _eval_classical(test_df, models: list[str], train_dataset: str):
    from ..classical.features import TFIDFFeaturizer
    from ..classical.models import MODEL_REGISTRY
    import json

    ckpt_dir = os.path.join(CHECKPOINTS_PATH, "classical", train_dataset)
    for model_key in models:
        ckpt_path = os.path.join(ckpt_dir, f"{model_key}_best.joblib")
        meta_path = os.path.join(ckpt_dir, f"{model_key}_best_meta.json")
        if not os.path.exists(ckpt_path):
            print(f"  SKIP {model_key} ({train_dataset}): checkpoint not found")
            continue

        featurizer = TFIDFFeaturizer.load(f"best_{model_key}_{train_dataset}")
        model = MODEL_REGISTRY[model_key].load(ckpt_path)
        with open(meta_path) as f:
            meta = json.load(f)

        def predict_fn(texts, _f=featurizer, _m=model):
            X = _f.transform(texts)
            return _m.predict(X), _m.predict_proba(X)

        name = f"{model_key}_{train_dataset}"
        print(f"  Evaluating {name} on PHEME ...", flush=True)
        bundle = evaluate(
            predict_fn=predict_fn,
            test_df=test_df,
            model_name=name,
            dataset_name="pheme",
            train_time_sec=0.0,
            hyperparams=meta.get("hyperparams", {}),
            split="ood_pheme",
        )
        print(f"    macro_f1={bundle.macro_f1:.4f}  acc={bundle.accuracy:.4f}  "
              f"auc={bundle.roc_auc:.4f}")


# ---------------------------------------------------------------------------
# Deep Learning
# ---------------------------------------------------------------------------

def _eval_deep_learning(test_df, variants: list[str], train_dataset: str):
    from ..deep_learning.dataset import TextDataset, VocabBuilder
    from ..deep_learning.embeddings import GloVeLoader
    from ..deep_learning.models import BiLSTM, TextCNN

    vocab = VocabBuilder.load(train_dataset)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 64
    ckpt_base  = os.path.join(CHECKPOINTS_PATH, "deep_learning", train_dataset)

    for variant in variants:
        ckpt_path = os.path.join(ckpt_base, f"{variant}.pt")
        if not os.path.exists(ckpt_path):
            print(f"  SKIP {variant} ({train_dataset}): checkpoint not found")
            continue

        arch   = "bilstm" if variant.startswith("bilstm") else "textcnn"
        freeze = variant.endswith("frozen")
        glove  = GloVeLoader(GLOVE_PATH, dim=GLOVE_DIM)
        emb    = glove.build_matrix(vocab, freeze=freeze)
        model  = BiLSTM(emb) if arch == "bilstm" else TextCNN(emb)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.to(device).eval()

        def predict_fn(texts, _m=model, _v=vocab, _dev=device):
            ds = TextDataset(texts, [0] * len(texts), _v, MAX_SEQ_LEN_DL)
            preds, probas = [], []
            with torch.no_grad():
                for i in range(0, len(ds), batch_size):
                    ids = torch.stack(
                        [ds[j]["input_ids"] for j in range(i, min(i + batch_size, len(ds)))]
                    ).to(_dev)
                    logits = _m(ids)
                    probas.append(torch.softmax(logits, dim=-1).cpu().numpy())
                    preds.extend(logits.argmax(dim=-1).cpu().tolist())
            return np.array(preds), np.vstack(probas)

        name = f"{variant}_{train_dataset}"
        print(f"  Evaluating {name} on PHEME ...", flush=True)
        bundle = evaluate(
            predict_fn=predict_fn,
            test_df=test_df,
            model_name=name,
            dataset_name="pheme",
            train_time_sec=0.0,
            hyperparams={"arch": arch, "freeze": freeze, "train_dataset": train_dataset},
            split="ood_pheme",
        )
        print(f"    macro_f1={bundle.macro_f1:.4f}  acc={bundle.accuracy:.4f}  "
              f"auc={bundle.roc_auc:.4f}")


# ---------------------------------------------------------------------------
# Transformer
# ---------------------------------------------------------------------------

def _eval_transformer(test_df, variants: list[str], train_dataset: str):
    from transformers import AutoTokenizer
    from ..transformers_.models import TransformerClassifier
    from ..transformers_.dataset import TransformerDataset

    HF_NAMES = {"bert": "bert-base-uncased", "roberta": "roberta-base"}
    device     = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 16
    ckpt_base  = os.path.join(CHECKPOINTS_PATH, "transformers_", train_dataset)

    for variant in variants:
        ckpt_path = os.path.join(ckpt_base, f"{variant}.pt")
        if not os.path.exists(ckpt_path):
            print(f"  SKIP {variant} ({train_dataset}): checkpoint not found")
            continue

        arch    = "bert" if variant.startswith("bert") else "roberta"
        hf_name = HF_NAMES[arch]
        tok     = AutoTokenizer.from_pretrained(hf_name)
        model   = TransformerClassifier(hf_name)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.to(device).eval()

        def predict_fn(texts, _m=model, _tok=tok, _dev=device):
            texts = [t if isinstance(t, str) else "" for t in texts]
            ds    = TransformerDataset(texts, [0] * len(texts), _tok, MAX_SEQ_LEN_TRANSFORMER)
            preds, probas = [], []
            with torch.no_grad():
                for i in range(0, len(ds), batch_size):
                    end   = min(i + batch_size, len(ds))
                    items = [ds[j] for j in range(i, end)]
                    ids   = torch.stack([b["input_ids"]      for b in items]).to(_dev)
                    mask  = torch.stack([b["attention_mask"]  for b in items]).to(_dev)
                    ttids = None
                    if "token_type_ids" in items[0]:
                        ttids = torch.stack([b["token_type_ids"] for b in items]).to(_dev)
                    logits = _m(ids, mask, ttids)
                    probas.append(torch.softmax(logits, dim=-1).cpu().numpy())
                    preds.extend(logits.argmax(dim=-1).cpu().tolist())
            return np.array(preds), np.vstack(probas)

        name = f"{variant}_{train_dataset}"
        print(f"  Evaluating {name} on PHEME ...", flush=True)
        bundle = evaluate(
            predict_fn=predict_fn,
            test_df=test_df,
            model_name=name,
            dataset_name="pheme",
            train_time_sec=0.0,
            hyperparams={"arch": arch, "hf_model": hf_name, "train_dataset": train_dataset},
            split="ood_pheme",
        )
        print(f"    macro_f1={bundle.macro_f1:.4f}  acc={bundle.accuracy:.4f}  "
              f"auc={bundle.roc_auc:.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_CLASSICAL_MODELS   = ["lr", "svc", "rf", "xgb"]
_DL_VARIANTS        = ["bilstm_frozen", "bilstm_finetuned", "textcnn_frozen", "textcnn_finetuned"]
_TRANSFORMER_VARIANTS = ["bert_frozen", "bert_finetuned", "roberta_frozen", "roberta_finetuned"]


def main():
    parser = argparse.ArgumentParser(description="PHEME OOD evaluation")
    parser.add_argument("--tier", default="all",
                        choices=["all", "classical", "deep_learning", "transformer"])
    parser.add_argument("--train-dataset", default="isot",
                        choices=["isot", "welfake", "combined"],
                        help="Which trained checkpoint to load (default: isot)")
    parser.add_argument("--model", nargs="+", default=["all"],
                        help="Specific model variants (default: all for the tier)")
    args = parser.parse_args()

    print("\n=== PHEME OOD Evaluation ===")
    print("NOTE: Expect lower scores — PHEME is tweet-length (~20 words) vs. "
          "article-length training data.\n")

    _, _, test_df = load_pheme_splits()
    test_df["text"] = test_df["text"].fillna("").astype(str)
    print(f"PHEME test: {len(test_df)} rows  "
          f"(false={( test_df['label']==0).sum()}  true={(test_df['label']==1).sum()})\n")

    run_classical    = args.tier in ("all", "classical")
    run_dl           = args.tier in ("all", "deep_learning")
    run_transformer  = args.tier in ("all", "transformer")

    if run_classical:
        models = _CLASSICAL_MODELS if "all" in args.model else args.model
        print(f"-- Classical ({args.train_dataset}) --")
        _eval_classical(test_df, models, args.train_dataset)

    if run_dl:
        variants = _DL_VARIANTS if "all" in args.model else args.model
        print(f"\n-- Deep Learning ({args.train_dataset}) --")
        _eval_deep_learning(test_df, variants, args.train_dataset)

    if run_transformer:
        variants = _TRANSFORMER_VARIANTS if "all" in args.model else args.model
        print(f"\n-- Transformer ({args.train_dataset}) --")
        _eval_transformer(test_df, variants, args.train_dataset)

    print("\nDone. Results saved to results/runs/ (split=ood_pheme).")
    print("Run: python -m src.utils.results_aggregator --all  to update results_table.md")


if __name__ == "__main__":
    main()
