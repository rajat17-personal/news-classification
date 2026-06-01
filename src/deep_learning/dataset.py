import json
import os
from collections import Counter

import torch
from torch.utils.data import DataLoader, Dataset

from ..utils.config import CACHE_PATH, MAX_SEQ_LEN_DL

PAD_IDX = 0
UNK_IDX = 1


class VocabBuilder:
    def build(self, texts: list[str], max_vocab: int = 30_000) -> dict[str, int]:
        counter = Counter(token for text in texts for token in str(text).split())
        vocab = {"<PAD>": PAD_IDX, "<UNK>": UNK_IDX}
        for word, _ in counter.most_common(max_vocab - 2):
            vocab[word] = len(vocab)
        return vocab

    def save(self, vocab: dict[str, int], dataset: str) -> None:
        os.makedirs(CACHE_PATH, exist_ok=True)
        with open(os.path.join(CACHE_PATH, f"vocab_{dataset}.json"), "w") as f:
            json.dump(vocab, f)

    @classmethod
    def load(cls, dataset: str) -> dict[str, int]:
        with open(os.path.join(CACHE_PATH, f"vocab_{dataset}.json")) as f:
            return json.load(f)


class TextDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], vocab: dict[str, int], max_len: int):
        self.labels = labels
        self.max_len = max_len
        self.encoded = [self._encode(str(t), vocab) for t in texts]

    def _encode(self, text: str, vocab: dict[str, int]) -> list[int]:
        tokens = text.split()[:self.max_len]
        ids = [vocab.get(t, UNK_IDX) for t in tokens]
        ids += [PAD_IDX] * (self.max_len - len(ids))
        return ids

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx) -> dict:
        return {
            "input_ids": torch.tensor(self.encoded[idx], dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def build_dataloaders(
    train_df, val_df, test_df,
    vocab: dict[str, int],
    max_len: int = MAX_SEQ_LEN_DL,
    batch_size: int = 64,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    def _make(df, shuffle):
        ds = TextDataset(df["text"].tolist(), df["label"].tolist(), vocab, max_len)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=2, pin_memory=True)

    return _make(train_df, True), _make(val_df, False), _make(test_df, False)
