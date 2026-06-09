import argparse
import os
import time

import torch
from transformers import AutoTokenizer

from ..utils.config import (
    CHECKPOINTS_PATH, ISOT_CONFIG, MAX_SEQ_LEN_TRANSFORMER, SEED, WELFAKE_CONFIG,
    NELA_CONFIG, NELA_SAMPLED_CONFIG,
)
from ..utils.data_loader import load_splits
from ..utils.metrics import evaluate
from ..utils.seeds import set_seed
from .dataset import build_dataloaders
from .models import TransformerClassifier
from .trainer import TransformerTrainer, TransformerTrainerConfig

from ..utils.combined_loader import COMBINED_CONFIG

DATASET_CONFIGS = {
    "isot": ISOT_CONFIG,
    "welfake": WELFAKE_CONFIG,
    "combined": COMBINED_CONFIG,
    "nela": NELA_CONFIG,
    "nela_sampled_100k": NELA_SAMPLED_CONFIG,
}

HF_MODEL_NAMES = {
    "bert": "bert-base-uncased",
    "roberta": "roberta-base",
}

def _variant_name(arch: str, freeze_mode: str) -> str:
    return f"{arch}_{freeze_mode}"

def run(
    dataset_name: str,
    archs: list[str],
    freeze_modes: list[str],
    epochs: int,
    batch_size: int,
    fp16: bool,
    gradient_checkpointing: bool,
):
    set_seed(SEED)
    config = DATASET_CONFIGS[dataset_name]
    train_df, val_df, test_df = load_splits(config)
    num_classes = len(config.label_map)

    results = []
    for arch in archs:
        hf_name = HF_MODEL_NAMES[arch]
        print(f"\nLoading tokenizer: {hf_name}")
        tokenizer = AutoTokenizer.from_pretrained(hf_name)

        train_loader, val_loader, test_loader = build_dataloaders(
            train_df, val_df, test_df,
            tokenizer=tokenizer,
            max_len=MAX_SEQ_LEN_TRANSFORMER,
            batch_size=batch_size,
        )

        for freeze_mode in freeze_modes:
            freeze = freeze_mode == "frozen"
            name = _variant_name(arch, freeze_mode)
            print(f"\n=== {name.upper()} on {dataset_name} ===")

            model = TransformerClassifier(hf_name, num_classes=num_classes)
            if freeze:
                model.freeze_encoder()
                trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(f"  Frozen encoder — trainable params: {trainable:,}")
            else:
                trainable = sum(p.numel() for p in model.parameters())
                print(f"  Full finetune — trainable params: {trainable:,}")

            trainer_cfg = TransformerTrainerConfig(
                model_name=hf_name,
                dataset_name=dataset_name,
                num_epochs=epochs,
                batch_size=batch_size,
                fp16=fp16,
                gradient_checkpointing=gradient_checkpointing,
                patience=2,
            )
            trainer = TransformerTrainer(trainer_cfg)
            history = trainer.train(model, train_loader, val_loader, variant_name=name)

            # reload best checkpoint
            ckpt_path = trainer_cfg.save_dir / f"{name}.pt"
            model.load_state_dict(torch.load(ckpt_path, map_location=trainer_cfg.device))
            model.to(trainer_cfg.device).eval()

            latency_ms = trainer.evaluate_latency(
                model, test_df["text"].tolist(), tokenizer, MAX_SEQ_LEN_TRANSFORMER
            )

            def predict_fn(texts: list, _model=model, _tok=tokenizer, _dev=trainer_cfg.device):
                import numpy as np
                from .dataset import TransformerDataset
                ds = TransformerDataset(texts, [0] * len(texts), _tok, MAX_SEQ_LEN_TRANSFORMER)
                all_preds, all_probas = [], []
                with torch.no_grad():
                    for i in range(0, len(ds), batch_size):
                        end = min(i + batch_size, len(ds))
                        batch_items = [ds[j] for j in range(i, end)]
                        ids  = torch.stack([b["input_ids"]      for b in batch_items]).to(_dev)
                        mask = torch.stack([b["attention_mask"]  for b in batch_items]).to(_dev)
                        ttids = None
                        if "token_type_ids" in batch_items[0]:
                            ttids = torch.stack([b["token_type_ids"] for b in batch_items]).to(_dev)
                        logits = _model(ids, mask, ttids)
                        probas = torch.softmax(logits, dim=-1).cpu().numpy()
                        all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
                        all_probas.append(probas)
                return np.array(all_preds), np.vstack(all_probas)

            hyperparams = {
                "arch": arch,
                "hf_model": hf_name,
                "freeze_mode": freeze_mode,
                "learning_rate": trainer_cfg.learning_rate,
                "weight_decay": trainer_cfg.weight_decay,
                "warmup_ratio": trainer_cfg.warmup_ratio,
                "batch_size": batch_size,
                "epochs_trained": history.best_epoch,
                "max_seq_len": MAX_SEQ_LEN_TRANSFORMER,
                "fp16": fp16,
            }

            bundle = evaluate(
                predict_fn=predict_fn,
                test_df=test_df,
                model_name=name,
                dataset_name=dataset_name,
                train_time_sec=history.train_time_sec,
                hyperparams=hyperparams,
            )
            results.append((bundle, latency_ms))
            print(f"  acc={bundle.accuracy:.4f}  macro_f1={bundle.macro_f1:.4f}  "
                  f"auc={bundle.roc_auc:.4f}  latency={latency_ms:.2f}ms/sample")

    print(f"\n{'Model':<30} {'Accuracy':<10} {'Macro F1':<10} {'ROC-AUC':<10} {'Train(s)':<10}")
    print("-" * 70)
    for b, _ in results:
        print(f"{b.model_name:<30} {b.accuracy:<10.4f} {b.macro_f1:<10.4f} "
              f"{b.roc_auc:<10.4f} {b.train_time_sec:<10.1f}")


def main():
    parser = argparse.ArgumentParser(description="Train Tier-3 transformer models")
    parser.add_argument("--dataset", required=True, choices=["isot", "welfake", "combined", "nela", "nela_sampled_100k"])
    parser.add_argument("--model", default=["all"], nargs="+",
                        choices=["bert", "roberta", "all"])
    parser.add_argument("--freeze", default=["both"], nargs="+",
                        choices=["frozen", "finetuned", "both"])
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--no-fp16", action="store_true",
                        help="Disable mixed precision (default: enabled on CUDA)")
    parser.add_argument("--gradient-checkpointing", action="store_true",
                        help="Enable gradient checkpointing to reduce VRAM usage")
    args = parser.parse_args()

    archs = ["bert", "roberta"] if "all" in args.model else args.model
    freeze_modes = ["frozen", "finetuned"] if "both" in args.freeze else args.freeze
    fp16 = not args.no_fp16

    run(args.dataset, archs, freeze_modes, args.epochs, args.batch_size,
        fp16=fp16, gradient_checkpointing=args.gradient_checkpointing)


if __name__ == "__main__":
    main()
