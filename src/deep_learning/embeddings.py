import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..utils.config import GLOVE_DIM

GLOVE_URL = "https://nlp.stanford.edu/data/glove.840B.300d.zip"


class GloVeLoader:
    def __init__(self, glove_path: str | Path, dim: int = GLOVE_DIM):
        self.dim = dim
        self.glove_path = Path(glove_path)
        self._vectors: dict[str, np.ndarray] | None = None

    def _load(self) -> None:
        if self._vectors is not None:
            return
        if not self.glove_path.exists():
            raise FileNotFoundError(
                f"GloVe file not found at {self.glove_path}.\n")
        print(f"Loading GloVe from {self.glove_path} ...")
        vectors = {}
        with open(self.glove_path, encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip().split(" ")
                word = parts[0]
                try:
                    vectors[word] = np.array(parts[1:], dtype=np.float32)
                except ValueError:
                    continue
        self._vectors = vectors
        print(f"  Loaded {len(vectors):,} GloVe vectors (dim={self.dim})")

    def build_matrix(self, vocab: dict[str, int], freeze: bool) -> nn.Embedding:
        self._load()
        vocab_size = len(vocab)
        matrix = np.zeros((vocab_size, self.dim), dtype=np.float32)

        # PAD stays all-zeros; UNK gets mean vector as fallback
        found = 0
        for word, idx in vocab.items():
            if word in ("<PAD>", "<UNK>"):
                continue
            if word in self._vectors:
                matrix[idx] = self._vectors[word]
                found += 1
            else:
                matrix[idx] = np.random.normal(0, 0.1, self.dim).astype(np.float32)

        # UNK = mean of all found vectors
        matrix[1] = matrix[2:].mean(axis=0)

        print(f"  GloVe coverage: {found}/{vocab_size - 2} ({100*found/(vocab_size-2):.1f}%)")

        embedding = nn.Embedding.from_pretrained(
            torch.from_numpy(matrix),
            freeze=freeze,
            padding_idx=0,
        )
        return embedding
