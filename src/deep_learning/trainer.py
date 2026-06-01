import os
import time
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from ..utils.config import CHECKPOINTS_PATH


@dataclass
class TrainerConfig:
    epochs: int = 20
    batch_size: int = 64
    learning_rate: float = 1e-3
    patience: int = 3
    grad_clip: float = 5.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class TrainingHistory:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    train_f1: list[float] = field(default_factory=list)
    val_f1: list[float] = field(default_factory=list)
    best_epoch: int = 0
    train_time_sec: float = 0.0


class Trainer:
    def __init__(self, model: nn.Module, config: TrainerConfig, model_name: str, dataset: str):
        self.model = model.to(config.device)
        self.config = config
        self.model_name = model_name
        self.dataset = dataset
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.learning_rate,
        )
        self.ckpt_dir = os.path.join(CHECKPOINTS_PATH, "deep_learning", dataset)
        os.makedirs(self.ckpt_dir, exist_ok=True)

    def _run_epoch(self, loader: DataLoader, train: bool) -> tuple[float, float]:
        self.model.train(train)
        total_loss, all_preds, all_labels = 0.0, [], []
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for batch in loader:
                input_ids = batch["input_ids"].to(self.config.device)
                labels = batch["label"].to(self.config.device)
                logits = self.model(input_ids)
                loss = self.criterion(logits, labels)
                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                    self.optimizer.step()
                total_loss += loss.item() * len(labels)
                all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
                all_labels.extend(labels.cpu().tolist())
        avg_loss = total_loss / len(loader.dataset)
        macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        return avg_loss, macro_f1

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> TrainingHistory:
        history = TrainingHistory()
        best_val_f1 = -1.0
        patience_counter = 0
        t0 = time.perf_counter()

        for epoch in range(1, self.config.epochs + 1):
            tr_loss, tr_f1 = self._run_epoch(train_loader, train=True)
            vl_loss, vl_f1 = self._run_epoch(val_loader, train=False)

            history.train_loss.append(tr_loss)
            history.val_loss.append(vl_loss)
            history.train_f1.append(tr_f1)
            history.val_f1.append(vl_f1)

            print(f"  Epoch {epoch:02d}/{self.config.epochs} — "
                  f"train loss={tr_loss:.4f} f1={tr_f1:.4f} | "
                  f"val loss={vl_loss:.4f} f1={vl_f1:.4f}", flush=True)

            if vl_f1 > best_val_f1:
                best_val_f1 = vl_f1
                history.best_epoch = epoch
                patience_counter = 0
                torch.save(self.model.state_dict(),
                           os.path.join(self.ckpt_dir, f"{self.model_name}.pt"))
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    print(f"  Early stopping at epoch {epoch} (patience={self.config.patience})")
                    break

        history.train_time_sec = time.perf_counter() - t0
        print(f"  Best epoch: {history.best_epoch}  val_f1={best_val_f1:.4f}  "
              f"time={history.train_time_sec:.1f}s")
        return history

    def evaluate_latency(self, texts: list[str], vocab: dict, max_len: int) -> float:
        from ..deep_learning.dataset import TextDataset
        cpu_model = self.model.cpu().eval()
        sample = texts[:100]
        ds = TextDataset(sample, [0] * len(sample), vocab, max_len)
        times = []
        with torch.no_grad():
            for item in ds:
                ids = item["input_ids"].unsqueeze(0)
                t = time.perf_counter()
                cpu_model(ids)
                times.append((time.perf_counter() - t) * 1000)
        self.model.to(self.config.device)
        import numpy as np
        return float(np.mean(times))
