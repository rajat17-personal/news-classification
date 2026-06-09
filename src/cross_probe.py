import argparse
import os

import numpy as np
import pandas as pd
import torch

from .utils.config import (
    CHECKPOINTS_PATH, GLOVE_DIM, ISOT_CONFIG, MAX_SEQ_LEN_DL,
    MAX_SEQ_LEN_TRANSFORMER, PROCESSED_DATA_PATH, WELFAKE_CONFIG,
)
from .utils.data_loader import load_splits
from .utils.metrics import evaluate

def _load_probe_df(source_dataset: str, target_dataset: str) -> pd.DataFrame:
    """Load target test split. For ISOT→WELFake use the deduplicated version."""
    processed_dir = os.path.join(PROCESSED_DATA_PATH, target_dataset)
    if source_dataset == "isot" and target_dataset == "welfake":
        deduped = os.path.join(processed_dir, "test_deduped.csv")
        if not os.path.exists(deduped):
            raise FileNotFoundError()
        df = pd.read_csv(deduped)
        print(f"  Using deduplicated WELFake test: {len(df)} rows")
    else:
        df = pd.read_csv(os.path.join(processed_dir, "test.csv"))
        print(f"  Using {target_dataset} test: {len(df)} rows")
    df["text"] = df["text"].fillna("").astype(str)
    return df

def _model_name_for_probe(arch: str, train_dataset: str, eval_dataset: str) -> str:
    return f"{arch}_{train_dataset}_to_{eval_dataset}_probe"

def _run_classical_probe(train_dataset: str, eval_dataset: str, models: list[str]):
    from .classical.features import TFIDFFeaturizer
    from .classical.models import MODEL_REGISTRY

    probe_df = _load_probe_df(train_dataset, eval_dataset)
    ckpt_dir = os.path.join(CHECKPOINTS_PATH, "classical", train_dataset)

    for model_key in models:
        meta_path = os.path.join(ckpt_dir, f"{model_key}_best_meta.json")
        ckpt_path = os.path.join(ckpt_dir, f"{model_key}_best.joblib")
        if not os.path.exists(ckpt_path):
            print(f"  SKIP {model_key}: checkpoint not found at {ckpt_path}")
            continue

        import json
        with open(meta_path) as f:
            meta = json.load(f)

        featurizer = TFIDFFeaturizer.load(f"best_{model_key}_{train_dataset}")
        cls = MODEL_REGISTRY[model_key]
        model = cls.load(ckpt_path)

        def predict_fn(texts, _feat=featurizer, _model=model):
            X = _feat.transform(texts)
            return _model.predict(X), _model.predict_proba(X)

        probe_name = _model_name_for_probe(f"{model_key}_{train_dataset}", train_dataset, eval_dataset)
        print(f"  Probing {model_key} ({train_dataset} → {eval_dataset}) ...", flush=True)
        bundle = evaluate(
            predict_fn=predict_fn,
            test_df=probe_df,
            model_name=probe_name,
            dataset_name=eval_dataset,
            train_time_sec=0.0,
            hyperparams=meta.get("hyperparams", {}),
            split="cross_probe",
        )
        print(f"    macro_f1={bundle.macro_f1:.4f}  acc={bundle.accuracy:.4f}  auc={bundle.roc_auc:.4f}")

def _run_dl_probe(train_dataset: str, eval_dataset: str, archs: list[str]):
    from .deep_learning.dataset import TextDataset, VocabBuilder
    from .deep_learning.models import BiLSTM, TextCNN
    from .deep_learning.embeddings import GloVeLoader
    from pathlib import Path

    GLOVE_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "glove" / "glove.840B.300d.txt"

    probe_df = _load_probe_df(train_dataset, eval_dataset)
    ckpt_base = os.path.join(CHECKPOINTS_PATH, "deep_learning", train_dataset)

    vocab = VocabBuilder.load(train_dataset)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 64

    for variant in archs:
        ckpt_path = os.path.join(ckpt_base, f"{variant}.pt")
        if not os.path.exists(ckpt_path):
            print(f"  SKIP {variant}: checkpoint not found at {ckpt_path}")
            continue

        arch = "bilstm" if variant.startswith("bilstm") else "textcnn"
        freeze = variant.endswith("frozen")

        glove = GloVeLoader(GLOVE_PATH, dim=GLOVE_DIM)
        embedding = glove.build_matrix(vocab, freeze=freeze)
        model = BiLSTM(embedding) if arch == "bilstm" else TextCNN(embedding)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.to(device).eval()

        def predict_fn(texts, _model=model, _vocab=vocab, _dev=device):
            ds = TextDataset(texts, [0] * len(texts), _vocab, MAX_SEQ_LEN_DL)
            all_preds, all_probas = [], []
            with torch.no_grad():
                for i in range(0, len(ds), batch_size):
                    ids = torch.stack(
                        [ds[j]["input_ids"] for j in range(i, min(i + batch_size, len(ds)))]
                    ).to(_dev)
                    logits = _model(ids)
                    probas = torch.softmax(logits, dim=-1).cpu().numpy()
                    all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
                    all_probas.append(probas)
            return np.array(all_preds), np.vstack(all_probas)

        probe_name = _model_name_for_probe(f"{variant}_{train_dataset}", train_dataset, eval_dataset)
        print(f"  Probing {variant} ({train_dataset} → {eval_dataset}) ...", flush=True)
        bundle = evaluate(
            predict_fn=predict_fn,
            test_df=probe_df,
            model_name=probe_name,
            dataset_name=eval_dataset,
            train_time_sec=0.0,
            hyperparams={"arch": arch, "freeze_mode": "frozen" if freeze else "finetuned",
                         "train_dataset": train_dataset},
            split="cross_probe",
        )
        print(f"    macro_f1={bundle.macro_f1:.4f}  acc={bundle.accuracy:.4f}  auc={bundle.roc_auc:.4f}")

def _run_transformer_probe(train_dataset: str, eval_dataset: str, archs: list[str]):
    from .transformers_.models import TransformerClassifier
    from .transformers_.dataset import TransformerDataset
    from transformers import AutoTokenizer

    HF_MODEL_NAMES = {"bert": "bert-base-uncased", "roberta": "roberta-base"}

    probe_df = _load_probe_df(train_dataset, eval_dataset)
    ckpt_base = os.path.join(CHECKPOINTS_PATH, "transformers_", train_dataset)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 16

    for variant in archs:
        ckpt_path = os.path.join(ckpt_base, f"{variant}.pt")
        if not os.path.exists(ckpt_path):
            print(f"  SKIP {variant}: checkpoint not found at {ckpt_path}")
            continue

        arch = "bert" if variant.startswith("bert") else "roberta"
        hf_name = HF_MODEL_NAMES[arch]

        tokenizer = AutoTokenizer.from_pretrained(hf_name)
        model = TransformerClassifier(hf_name)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.to(device).eval()

        def predict_fn(texts, _model=model, _tok=tokenizer, _dev=device):
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
                    logits = _model(ids, mask, ttids)
                    probas = torch.softmax(logits, dim=-1).cpu().numpy()
                    all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
                    all_probas.append(probas)
            return np.array(all_preds), np.vstack(all_probas)

        probe_name = _model_name_for_probe(f"{variant}_{train_dataset}", train_dataset, eval_dataset)
        print(f"  Probing {variant} ({train_dataset} → {eval_dataset}) ...", flush=True)
        bundle = evaluate(
            predict_fn=predict_fn,
            test_df=probe_df,
            model_name=probe_name,
            dataset_name=eval_dataset,
            train_time_sec=0.0,
            hyperparams={"arch": arch, "hf_model": hf_name, "train_dataset": train_dataset},
            split="cross_probe",
        )
        print(f"    macro_f1={bundle.macro_f1:.4f}  acc={bundle.accuracy:.4f}  auc={bundle.roc_auc:.4f}")

_DL_VARIANTS = ["bilstm_frozen", "bilstm_finetuned", "textcnn_frozen", "textcnn_finetuned"]
_TRANSFORMER_VARIANTS = ["bert_frozen", "bert_finetuned", "roberta_frozen", "roberta_finetuned"]
_CLASSICAL_MODELS = ["lr", "svc", "rf", "xgb"]

def main():
    parser = argparse.ArgumentParser(description="Cross-dataset probe runner")
    parser.add_argument("--tier", required=True, choices=["classical", "deep_learning", "transformer"])
    parser.add_argument(
        "--direction", required=True, choices=["a_to_b", "b_to_a", "combined_to_isot", "combined_to_welfake"],
        help="a_to_b=ISOT→WELFake  b_to_a=WELFake→ISOT  combined_to_*=combined→target",
    )
    parser.add_argument(
        "--model", nargs="+", default=["all"],
        help="Model variants to probe. Default: all for the given tier.",
    )
    args = parser.parse_args()

    direction_map = {
        "a_to_b":            ("isot",     "welfake"),
        "b_to_a":            ("welfake",  "isot"),
        "combined_to_isot":  ("combined", "isot"),
        "combined_to_welfake": ("combined", "welfake"),
    }
    train_ds, eval_ds = direction_map[args.direction]

    print(f"\n=== Cross-probe: {train_ds.upper()} → {eval_ds.upper()} [{args.tier}] ===")

    if args.tier == "classical":
        models = _CLASSICAL_MODELS if "all" in args.model else args.model
        _run_classical_probe(train_ds, eval_ds, models)

    elif args.tier == "deep_learning":
        variants = _DL_VARIANTS if "all" in args.model else args.model
        _run_dl_probe(train_ds, eval_ds, variants)

    elif args.tier == "transformer":
        variants = _TRANSFORMER_VARIANTS if "all" in args.model else args.model
        _run_transformer_probe(train_ds, eval_ds, variants)

if __name__ == "__main__":
    main()
