import argparse
import os
import time
from pathlib import Path

import torch

from ..utils.config import (
    CHECKPOINTS_PATH, GLOVE_DIM, ISOT_CONFIG, MAX_SEQ_LEN_DL, SEED, WELFAKE_CONFIG,
    NELA_CONFIG, NELA_SAMPLED_CONFIG, NELA_DL_CONFIG,
)
from ..utils.data_loader import load_splits
from ..utils.metrics import evaluate
from ..utils.seeds import set_seed
from .dataset import VocabBuilder, build_dataloaders
from .embeddings import GloVeLoader
from .models import BiLSTM, TextCNN
from .trainer import Trainer, TrainerConfig

from ..utils.combined_loader import COMBINED_CONFIG

DATASET_CONFIGS = {
    "isot": ISOT_CONFIG,
    "welfake": WELFAKE_CONFIG,
    "combined": COMBINED_CONFIG,
    "nela": NELA_CONFIG,
    "nela_sampled_100k": NELA_SAMPLED_CONFIG,
    "nela_sampled_500k": NELA_DL_CONFIG,
}

GLOVE_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "glove" / "glove.840B.300d.txt"


def _model_name(arch: str, freeze_mode: str) -> str:
    return f"{arch}_{freeze_mode}"


def run(
    dataset_name: str,
    archs: list[str],
    freeze_modes: list[str],
    input_strategy: str,
    epochs: int,
    batch_size: int,
):
    set_seed(SEED)
    config = DATASET_CONFIGS[dataset_name]
    train_df, val_df, test_df = load_splits(config)
    num_classes = len(config.label_map)

    # build / load vocab
    vocab_builder = VocabBuilder()
    vocab_cache = os.path.join(
        Path(__file__).resolve().parents[3] / "data" / "cache",
        f"vocab_{dataset_name}.json",
    )
    if os.path.exists(vocab_cache):
        print(f"Loading vocab from cache...")
        vocab = VocabBuilder.load(dataset_name)
    else:
        print(f"Building vocab...")
        vocab = vocab_builder.build(train_df["text"].tolist())
        vocab_builder.save(vocab, dataset_name)
    print(f"  Vocab size: {len(vocab):,}")

    # load GloVe once — shared across all variants
    glove = GloVeLoader(GLOVE_PATH, dim=GLOVE_DIM)

    results = []
    for arch in archs:
        for freeze_mode in freeze_modes:
            freeze = freeze_mode == "frozen"
            name = _model_name(arch, freeze_mode)
            print(f"\n=== {name.upper()} on {dataset_name} ===")

            train_loader, val_loader, test_loader = build_dataloaders(
                train_df, val_df, test_df, vocab,
                max_len=MAX_SEQ_LEN_DL,
                batch_size=batch_size,
            )

            embedding = glove.build_matrix(vocab, freeze=freeze)

            if arch == "bilstm":
                model = BiLSTM(embedding, num_classes=num_classes)
            else:
                model = TextCNN(embedding, num_classes=num_classes)

            trainer_cfg = TrainerConfig(
                epochs=epochs,
                batch_size=batch_size,
            )
            trainer = Trainer(model, trainer_cfg, name, dataset_name)
            history = trainer.train(train_loader, val_loader)

            # reload best checkpoint for evaluation
            ckpt_path = os.path.join(CHECKPOINTS_PATH, "deep_learning", dataset_name, f"{name}.pt")
            model.load_state_dict(torch.load(ckpt_path, map_location=trainer_cfg.device))
            model.to(trainer_cfg.device).eval()

            def predict_fn(texts: list, _model=model, _vocab=vocab, _dev=trainer_cfg.device):
                from .dataset import TextDataset
                import numpy as np
                ds = TextDataset(texts, [0] * len(texts), _vocab, MAX_SEQ_LEN_DL)
                all_preds, all_probas = [], []
                with torch.no_grad():
                    for i in range(0, len(ds), batch_size):
                        batch_ids = torch.stack(
                            [ds[j]["input_ids"] for j in range(i, min(i + batch_size, len(ds)))]
                        ).to(_dev)
                        logits = _model(batch_ids)
                        probas = torch.softmax(logits, dim=-1).cpu().numpy()
                        all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
                        all_probas.append(probas)
                return np.array(all_preds), np.vstack(all_probas)

            latency_ms = trainer.evaluate_latency(test_df["text"].tolist(), vocab, MAX_SEQ_LEN_DL)

            hyperparams = {
                "arch": arch,
                "freeze_mode": freeze_mode,
                "hidden_size": 256 if arch == "bilstm" else None,
                "num_filters": None if arch == "bilstm" else 128,
                "kernel_sizes": None if arch == "bilstm" else [2, 3, 4, 5],
                "num_layers": 2 if arch == "bilstm" else None,
                "dropout": 0.3,
                "epochs_trained": history.best_epoch,
                "learning_rate": trainer_cfg.learning_rate,
                "batch_size": batch_size,
                "max_seq_len": MAX_SEQ_LEN_DL,
                "glove_dim": GLOVE_DIM,
                "vocab_size": len(vocab),
                "input_strategy": input_strategy,
            }

            bundle = evaluate(
                predict_fn=predict_fn,
                test_df=test_df,
                model_name=name,
                dataset_name=dataset_name,
                train_time_sec=history.train_time_sec,
                hyperparams=hyperparams,
            )
            results.append(bundle)
            print(f"  acc={bundle.accuracy:.4f}  macro_f1={bundle.macro_f1:.4f}  "
                  f"auc={bundle.roc_auc:.4f}  latency={latency_ms:.2f}ms/sample")

    # summary table
    print(f"\n{'Model':<25} {'Accuracy':<10} {'Macro F1':<10} {'ROC-AUC':<10} {'Train(s)':<10}")
    print("-" * 65)
    for b in results:
        print(f"{b.model_name:<25} {b.accuracy:<10.4f} {b.macro_f1:<10.4f} "
              f"{b.roc_auc:<10.4f} {b.train_time_sec:<10.1f}")


def main():
    parser = argparse.ArgumentParser(description="Train Tier-2 deep learning models")
    parser.add_argument("--dataset", required=True, choices=["isot", "welfake", "combined", "nela", "nela_sampled_100k", "nela_sampled_500k"])
    parser.add_argument("--model", default=["all"], nargs="+",
                        choices=["bilstm", "textcnn", "all"])
    parser.add_argument("--freeze", default=["both"], nargs="+",
                        choices=["frozen", "finetuned", "both"])
    parser.add_argument("--input-strategy", default="full_body",
                        choices=["full_body", "headline_para", "headline"])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    archs = ["bilstm", "textcnn"] if "all" in args.model else args.model
    freeze_modes = ["frozen", "finetuned"] if "both" in args.freeze else args.freeze

    run(args.dataset, archs, freeze_modes, args.input_strategy, args.epochs, args.batch_size)


if __name__ == "__main__":
    main()
