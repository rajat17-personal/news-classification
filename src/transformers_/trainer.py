import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score

from ..utils.config import CHECKPOINTS_PATH
from .models import TransformerClassifier


@dataclass
class TransformerTrainerConfig:
    model_name: str
    dataset_name: str
    num_epochs: int = 4
    batch_size: int = 16
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    fp16: bool = True
    gradient_checkpointing: bool = False
    patience: int = 2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir: Path = None

    def __post_init__(self):
        if self.save_dir is None:
            self.save_dir = Path(CHECKPOINTS_PATH) / "transformers_" / self.dataset_name
        os.makedirs(self.save_dir, exist_ok=True)
        # fp16 only makes sense on CUDA
        if self.device == "cpu":
            self.fp16 = False


@dataclass
class TransformerTrainingHistory:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    train_f1: list[float] = field(default_factory=list)
    val_f1: list[float] = field(default_factory=list)
    best_epoch: int = 0
    train_time_sec: float = 0.0


class TransformerTrainer:
    def __init__(self, config: TransformerTrainerConfig):
        self.config = config

    def train(
        self,
        model: TransformerClassifier,
        train_loader: DataLoader,
        val_loader: DataLoader,
        variant_name: str,
    ) -> TransformerTrainingHistory:
        cfg = self.config
        model = model.to(cfg.device)

        if cfg.gradient_checkpointing:
            model.encoder.gradient_checkpointing_enable()

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )

        total_steps = len(train_loader) * cfg.num_epochs
        warmup_steps = int(total_steps * cfg.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        scaler = GradScaler("cuda") if cfg.fp16 else None
        criterion = nn.CrossEntropyLoss()
        ckpt_path = cfg.save_dir / f"{variant_name}.pt"

        history = TransformerTrainingHistory()
        best_val_f1 = -1.0
        patience_counter = 0
        t0 = time.perf_counter()

        for epoch in range(1, cfg.num_epochs + 1):
            tr_loss, tr_f1 = self._run_epoch(model, train_loader, criterion, optimizer,
                                              scheduler, scaler, train=True)
            vl_loss, vl_f1 = self._run_epoch(model, val_loader, criterion, None,
                                              None, None, train=False)

            history.train_loss.append(tr_loss)
            history.val_loss.append(vl_loss)
            history.train_f1.append(tr_f1)
            history.val_f1.append(vl_f1)

            print(f"  Epoch {epoch:02d}/{cfg.num_epochs} — "
                  f"train loss={tr_loss:.4f} f1={tr_f1:.4f} | "
                  f"val loss={vl_loss:.4f} f1={vl_f1:.4f}", flush=True)

            if vl_f1 > best_val_f1:
                best_val_f1 = vl_f1
                history.best_epoch = epoch
                patience_counter = 0
                torch.save(model.state_dict(), ckpt_path)
            else:
                patience_counter += 1
                if patience_counter >= cfg.patience:
                    print(f"  Early stopping at epoch {epoch} (patience={cfg.patience})")
                    break

        history.train_time_sec = time.perf_counter() - t0
        print(f"  Best epoch: {history.best_epoch}  val_f1={best_val_f1:.4f}  "
              f"time={history.train_time_sec:.1f}s")
        return history

    def _run_epoch(self, model, loader, criterion, optimizer, scheduler, scaler, train: bool):
        model.train(train)
        total_loss, all_preds, all_labels = 0.0, [], []
        ctx = torch.enable_grad() if train else torch.no_grad()

        with ctx:
            for batch in loader:
                input_ids = batch["input_ids"].to(self.config.device)
                attention_mask = batch["attention_mask"].to(self.config.device)
                token_type_ids = batch.get("token_type_ids")
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(self.config.device)
                labels = batch["label"].to(self.config.device)

                if self.config.fp16 and train:
                    with autocast("cuda"):
                        logits = model(input_ids, attention_mask, token_type_ids)
                        loss = criterion(logits, labels)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()
                else:
                    logits = model(input_ids, attention_mask, token_type_ids)
                    loss = criterion(logits, labels)
                    if train:
                        optimizer.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)
                        optimizer.step()
                        scheduler.step()

                total_loss += loss.item() * len(labels)
                all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        avg_loss = total_loss / len(loader.dataset)
        macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        return avg_loss, macro_f1

    def evaluate_latency(
        self, model: TransformerClassifier, texts: list[str], tokenizer, max_len: int
    ) -> float:
        """CPU-only latency: 100 single-sample calls, returns mean ms/sample."""
        from .dataset import TransformerDataset
        model_device = next(model.parameters()).device
        model.to("cpu").eval()

        sample = texts[:100]
        ds = TransformerDataset(sample, [0] * len(sample), tokenizer, max_len)
        times = []
        with torch.no_grad():
            for i in range(len(ds)):
                item = ds[i]
                ids = item["input_ids"].unsqueeze(0)
                mask = item["attention_mask"].unsqueeze(0)
                ttids = item.get("token_type_ids")
                if ttids is not None:
                    ttids = ttids.unsqueeze(0)
                t = time.perf_counter()
                model(ids, mask, ttids)
                times.append((time.perf_counter() - t) * 1000)

        model.to(model_device)
        return float(np.mean(times))
