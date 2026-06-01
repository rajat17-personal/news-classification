import torch
import torch.nn as nn
from torch import Tensor


class BiLSTM(nn.Module):
    def __init__(
        self,
        embedding: nn.Embedding,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_classes: int = 2,
    ):
        super().__init__()
        self.embedding = embedding
        self.lstm = nn.LSTM(
            input_size=embedding.embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, input_ids: Tensor) -> Tensor:
        x = self.embedding(input_ids)               # [B, L, E]
        x = self.dropout(x)
        _, (hidden, _) = self.lstm(x)               # hidden: [2*layers, B, H]
        # take last layer's forward + backward hidden states
        fwd = hidden[-2]                            # [B, H]
        bwd = hidden[-1]                            # [B, H]
        out = torch.cat([fwd, bwd], dim=-1)         # [B, 2H]
        out = self.dropout(out)
        return self.classifier(out)                 # [B, num_classes]


class TextCNN(nn.Module):
    def __init__(
        self,
        embedding: nn.Embedding,
        num_filters: int = 128,
        kernel_sizes: list[int] = None,
        dropout: float = 0.3,
        num_classes: int = 2,
    ):
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [2, 3, 4, 5]
        self.embedding = embedding
        embed_dim = embedding.embedding_dim
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, kernel_size=k)
            for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, input_ids: Tensor) -> Tensor:
        x = self.embedding(input_ids)               # [B, L, E]
        x = self.dropout(x)
        x = x.permute(0, 2, 1)                      # [B, E, L] for Conv1d
        pooled = []
        for conv in self.convs:
            c = torch.relu(conv(x))                 # [B, F, L-k+1]
            c = c.max(dim=-1).values                # [B, F] global max pool
            pooled.append(c)
        out = torch.cat(pooled, dim=-1)             # [B, F*len(kernels)]
        out = self.dropout(out)
        return self.classifier(out)                 # [B, num_classes]
