import argparse
import os
from pathlib import Path

import numpy as np
import torch

from ..utils.config import (
    CHECKPOINTS_PATH, GLOVE_DIM, ISOT_CONFIG, MAX_SEQ_LEN_DL,
    WELFAKE_CONFIG,
)
from ..utils.data_loader import load_splits
from ..utils.metrics import evaluate
from ..pheme.dataset import load_pheme_splits
from ..utils.combined_loader import COMBINED_CONFIG, load_combined_splits

GLOVE_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "glove" / "glove.840B.300d.txt"
PHEMEPLUS_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "phemeplus" / "PHEME_PLUS" / "all-rnr-annotated-threads"

_NELA_TO_BINARY = {0: 1, 1: 0, 2: 0}

EVAL_DATASETS = {
    "isot":     ISOT_CONFIG,
    "welfake":  WELFAKE_CONFIG,
    "combined": COMBINED_CONFIG,
}


def _load_phemeplus(threads_dir: Path) -> "pd.DataFrame":
    import glob, json, re
    import pandas as pd

    def _clean(text: str) -> str:
        text = re.sub(r"http\S+|www\S+", "", text)
        text = re.sub(r"@\w+", "@USER", text)
        text = re.sub(r"#(\w+)", r"\1", text)
        return re.sub(r"\s+", " ", text).strip()

    records = []
    for event_dir in sorted(os.listdir(threads_dir)):
        if event_dir.startswith("."):
            continue
        rumours_path = os.path.join(threads_dir, event_dir, "rumours")
        if not os.path.isdir(rumours_path):
            continue
        for thread_id in os.listdir(rumours_path):
            thread_path = os.path.join(rumours_path, thread_id)
            ann_path    = os.path.join(thread_path, "annotation.json")
            src_dir     = os.path.join(thread_path, "source-tweets")
            if not os.path.isfile(ann_path):
                continue
            src_files = [f for f in glob.glob(os.path.join(src_dir, "*.json"))
                         if not f.endswith("Zone.Identifier")]
            if not src_files:
                continue
            try:
                with open(ann_path) as f:
                    ann = json.load(f)
                with open(src_files[0]) as f:
                    tweet = json.load(f)
            except Exception:
                continue

            m = int(ann.get("misinformation", -1) if ann.get("misinformation", -1) != "" else -1)
            t = int(ann.get("true", -1) if ann.get("true", -1) != "" else -1)
            if m == 0 and t == 1:
                label = 1   # true rumour → Real
            elif m == 1 and t == 0:
                label = 0   # false rumour → Fake
            else:
                continue    # unverified

            text = tweet.get("full_text") or tweet.get("text", "")
            text = _clean(text)
            if not text:
                continue

            records.append({"text": text, "label": label,
                             "event": event_dir.replace("-all-rnr-threads", "")})

    df = pd.DataFrame(records)
    print(f"  PHEMEplus: {len(df)} verified rumour threads  "
          f"(fake={( df['label']==0).sum()}, real={(df['label']==1).sum()})")
    return df[["text", "label"]]


def _collapse_to_binary(
    preds_3class: np.ndarray,
    probas_3class: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Collapse 3-class NELA output to binary fake/real.
    Returns (binary_preds, binary_probas) where binary_probas has shape (N, 2):
      col 0 = P(Fake)  = P(Questionable) + P(Conspiracy)
      col 1 = P(Real)  = P(Reliable)
    """
    binary_preds = np.array([_NELA_TO_BINARY[p] for p in preds_3class])
    p_real = probas_3class[:, 0]                          # P(Reliable)
    p_fake = probas_3class[:, 1] + probas_3class[:, 2]   # P(Questionable + Conspiracy)
    binary_probas = np.column_stack([p_fake, p_real])
    return binary_preds, binary_probas


def _eval_classical(eval_dfs: dict, models: list[str], train_dataset: str):
    from ..classical.features import TFIDFFeaturizer
    from ..classical.models import MODEL_REGISTRY
    import json

    ckpt_dir = os.path.join(CHECKPOINTS_PATH, "classical", train_dataset)

    for model_key in models:
        ckpt_path = os.path.join(ckpt_dir, f"{model_key}_best.joblib")
        meta_path = os.path.join(ckpt_dir, f"{model_key}_best_meta.json")
        if not os.path.exists(ckpt_path):
            print(f"  SKIP {model_key}: checkpoint not found at {ckpt_path}")
            continue

        featurizer = TFIDFFeaturizer.load(f"best_{model_key}_{train_dataset}")
        model = MODEL_REGISTRY[model_key].load(ckpt_path)
        meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}

        for ds_name, test_df in eval_dfs.items():
            print(f"  {model_key} ({train_dataset}) → {ds_name} ...", flush=True)

            def predict_fn(texts, _f=featurizer, _m=model):
                X = _f.transform(texts)
                p3 = _m.predict(X)
                pr3 = _m.predict_proba(X)
                return _collapse_to_binary(p3, pr3)

            split_name = f"nela_to_{ds_name}"
            bundle = evaluate(
                predict_fn=predict_fn,
                test_df=test_df,
                model_name=f"{model_key}_{train_dataset}",
                dataset_name=ds_name,
                train_time_sec=0.0,
                hyperparams={**meta.get("hyperparams", {}), "eval_type": "nela_cross_eval"},
                split=split_name,
            )
            print(f"    macro_f1={bundle.macro_f1:.4f}  acc={bundle.accuracy:.4f}  "
                  f"auc={bundle.roc_auc:.4f}")

def _eval_deep_learning(eval_dfs: dict, variants: list[str], train_dataset: str):
    from ..deep_learning.dataset import TextDataset, VocabBuilder
    from ..deep_learning.embeddings import GloVeLoader
    from ..deep_learning.models import BiLSTM, TextCNN
    from ..utils.config import NELA_DL_CONFIG

    vocab = VocabBuilder.load(train_dataset)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 64
    ckpt_base = os.path.join(CHECKPOINTS_PATH, "deep_learning", train_dataset)
    num_classes = len(NELA_DL_CONFIG.label_map)

    glove = GloVeLoader(GLOVE_PATH, dim=GLOVE_DIM)

    for variant in variants:
        ckpt_path = os.path.join(ckpt_base, f"{variant}.pt")
        if not os.path.exists(ckpt_path):
            print(f"  SKIP {variant}: checkpoint not found at {ckpt_path}")
            continue

        arch  = "bilstm" if variant.startswith("bilstm") else "textcnn"
        freeze = variant.endswith("frozen")
        emb   = glove.build_matrix(vocab, freeze=freeze)

        if arch == "bilstm":
            model = BiLSTM(emb, num_classes=num_classes)
        else:
            model = TextCNN(emb, num_classes=num_classes)

        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.to(device).eval()

        for ds_name, test_df in eval_dfs.items():
            print(f"  {variant} ({train_dataset}) → {ds_name} ...", flush=True)

            def predict_fn(texts, _m=model, _v=vocab, _dev=device):
                ds = TextDataset(texts, [0] * len(texts), _v, MAX_SEQ_LEN_DL)
                preds3, probas3 = [], []
                with torch.no_grad():
                    for i in range(0, len(ds), batch_size):
                        ids = torch.stack(
                            [ds[j]["input_ids"] for j in range(i, min(i + batch_size, len(ds)))]
                        ).to(_dev)
                        logits = _m(ids)
                        probas3.append(torch.softmax(logits, dim=-1).cpu().numpy())
                        preds3.extend(logits.argmax(dim=-1).cpu().tolist())
                return _collapse_to_binary(np.array(preds3), np.vstack(probas3))

            split_name = f"nela_to_{ds_name}"
            bundle = evaluate(
                predict_fn=predict_fn,
                test_df=test_df,
                model_name=f"{variant}_{train_dataset}",
                dataset_name=ds_name,
                train_time_sec=0.0,
                hyperparams={"arch": arch, "freeze": freeze, "train_dataset": train_dataset,
                             "eval_type": "nela_cross_eval"},
                split=split_name,
            )
            print(f"    macro_f1={bundle.macro_f1:.4f}  acc={bundle.accuracy:.4f}  "
                  f"auc={bundle.roc_auc:.4f}")

def _eval_pheme_classical(pheme_df, models: list[str], train_dataset: str,
                           pheme_tag: str = "pheme"):
    from ..classical.features import TFIDFFeaturizer
    from ..classical.models import MODEL_REGISTRY
    import json

    ckpt_dir = os.path.join(CHECKPOINTS_PATH, "classical", train_dataset)

    for model_key in models:
        ckpt_path = os.path.join(ckpt_dir, f"{model_key}_best.joblib")
        if not os.path.exists(ckpt_path):
            print(f"  SKIP {model_key} ({pheme_tag}): checkpoint not found")
            continue

        featurizer = TFIDFFeaturizer.load(f"best_{model_key}_{train_dataset}")
        model = MODEL_REGISTRY[model_key].load(ckpt_path)
        meta_path = os.path.join(ckpt_dir, f"{model_key}_best_meta.json")
        meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}

        print(f"  {model_key} ({train_dataset}) → {pheme_tag} ...", flush=True)

        def predict_fn(texts, _f=featurizer, _m=model):
            X = _f.transform(texts)
            return _collapse_to_binary(_m.predict(X), _m.predict_proba(X))

        split_name = f"nela_to_{pheme_tag}" if pheme_tag != "pheme" else "ood_pheme"
        bundle = evaluate(
            predict_fn=predict_fn,
            test_df=pheme_df,
            model_name=f"{model_key}_{train_dataset}",
            dataset_name=pheme_tag,
            train_time_sec=0.0,
            hyperparams={**meta.get("hyperparams", {}), "eval_type": "nela_cross_eval"},
            split=split_name,
        )
        print(f"    macro_f1={bundle.macro_f1:.4f}  acc={bundle.accuracy:.4f}  "
              f"auc={bundle.roc_auc:.4f}")


def _eval_pheme_dl(pheme_df, variants: list[str], train_dataset: str,
                   pheme_tag: str = "pheme"):
    from ..deep_learning.dataset import TextDataset, VocabBuilder
    from ..deep_learning.embeddings import GloVeLoader
    from ..deep_learning.models import BiLSTM, TextCNN
    from ..utils.config import NELA_DL_CONFIG

    vocab = VocabBuilder.load(train_dataset)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 64
    ckpt_base = os.path.join(CHECKPOINTS_PATH, "deep_learning", train_dataset)
    num_classes = len(NELA_DL_CONFIG.label_map)
    glove = GloVeLoader(GLOVE_PATH, dim=GLOVE_DIM)

    for variant in variants:
        ckpt_path = os.path.join(ckpt_base, f"{variant}.pt")
        if not os.path.exists(ckpt_path):
            print(f"  SKIP {variant} (PHEME): checkpoint not found")
            continue

        arch = "bilstm" if variant.startswith("bilstm") else "textcnn"
        emb  = glove.build_matrix(vocab, freeze=variant.endswith("frozen"))
        model = BiLSTM(emb, num_classes=num_classes) if arch == "bilstm" else TextCNN(emb, num_classes=num_classes)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.to(device).eval()

        print(f"  {variant} ({train_dataset}) → {pheme_tag} ...", flush=True)

        def predict_fn(texts, _m=model, _v=vocab, _dev=device):
            ds = TextDataset(texts, [0] * len(texts), _v, MAX_SEQ_LEN_DL)
            preds3, probas3 = [], []
            with torch.no_grad():
                for i in range(0, len(ds), batch_size):
                    ids = torch.stack(
                        [ds[j]["input_ids"] for j in range(i, min(i + batch_size, len(ds)))]
                    ).to(_dev)
                    logits = _m(ids)
                    probas3.append(torch.softmax(logits, dim=-1).cpu().numpy())
                    preds3.extend(logits.argmax(dim=-1).cpu().tolist())
            return _collapse_to_binary(np.array(preds3), np.vstack(probas3))

        split_name = f"nela_to_{pheme_tag}" if pheme_tag != "pheme" else "ood_pheme"
        bundle = evaluate(
            predict_fn=predict_fn,
            test_df=pheme_df,
            model_name=f"{variant}_{train_dataset}",
            dataset_name=pheme_tag,
            train_time_sec=0.0,
            hyperparams={"arch": arch, "train_dataset": train_dataset,
                         "eval_type": "nela_cross_eval"},
            split=split_name,
        )
        print(f"    macro_f1={bundle.macro_f1:.4f}  acc={bundle.accuracy:.4f}  "
              f"auc={bundle.roc_auc:.4f}")

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate NELA-trained models on ISOT, WELFake, Combined, PHEME, PHEMEplus"
    )
    parser.add_argument("--tier", default="all",
                        choices=["all", "classical", "deep_learning"])
    parser.add_argument("--train-dataset", default="nela_sampled_500k",
                        choices=["nela", "nela_sampled_500k"],
                        help="Which NELA checkpoint set to load")
    parser.add_argument("--target", nargs="+",
                        choices=["isot", "welfake", "combined", "all"],
                        default=["all"],
                        help="Which article-level target datasets to eval (default: all 3)")
    parser.add_argument("--pheme", default="pheme",
                        choices=["pheme", "phemeplus", "both", "skip"],
                        help="Which PHEME variant to eval against (default: pheme)")
    args = parser.parse_args()

    train_ds = args.train_dataset

    # Resolve target datasets
    target_keys = list(EVAL_DATASETS.keys()) if "all" in args.target else args.target
    eval_dfs = {}
    print("[NELA eval] Loading article-level test sets...")
    for ds in target_keys:
        try:
            if ds == "combined":
                _, _, test_df = load_combined_splits()
            else:
                test_df = load_splits(EVAL_DATASETS[ds])[2]
            eval_dfs[ds] = test_df
            print(f"  {ds}: {len(test_df):,} rows")
        except Exception as e:
            print(f"  {ds}: SKIP ({e})")

    # Load PHEME variants
    pheme_dfs = {}
    if args.pheme in ("pheme", "both"):
        try:
            _, _, pheme_df = load_pheme_splits()
            pheme_dfs["pheme"] = pheme_df
            print(f"  pheme: {len(pheme_df):,} test tweets")
        except Exception as e:
            print(f"  PHEME unavailable ({e}) — skipping")
    if args.pheme in ("phemeplus", "both"):
        try:
            phemeplus_df = _load_phemeplus(PHEMEPLUS_DIR)
            pheme_dfs["phemeplus"] = phemeplus_df
        except Exception as e:
            print(f"  PHEMEplus unavailable ({e}) — skipping")

    classical_models = ["lr", "svc", "xgb", "rf"]
    dl_variants      = ["textcnn_finetuned", "bilstm_finetuned"]

    print(f"\n[NELA eval] Train dataset: {train_ds}")
    print(f"[NELA eval] Binary collapse: Reliable→Real, Questionable+Conspiracy→Fake\n")

    if args.tier in ("all", "classical"):
        print("=== Classical ===")
        if eval_dfs:
            _eval_classical(eval_dfs, classical_models, train_ds)
        for pheme_name, pheme_df in pheme_dfs.items():
            _eval_pheme_classical(pheme_df, classical_models, train_ds,
                                pheme_tag=pheme_name)

    if args.tier in ("all", "deep_learning"):
        print("\n=== Deep Learning ===")
        if eval_dfs:
            _eval_deep_learning(eval_dfs, dl_variants, train_ds)
        for pheme_name, pheme_df in pheme_dfs.items():
            _eval_pheme_dl(pheme_df, dl_variants, train_ds, pheme_tag=pheme_name)

    print("\n[NELA eval] Done. Run results aggregator to update results_table.md:")
    print("  python -m src.utils.results_aggregator --all")


if __name__ == "__main__":
    main()
