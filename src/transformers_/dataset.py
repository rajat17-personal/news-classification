import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


class TransformerDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer: AutoTokenizer, max_len: int):
        texts = [t if isinstance(t, str) else "" for t in texts]
        # return_tensors omitted — encode as plain lists, convert per-item in __getitem__
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_len,
        )
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx], dtype=torch.long)
                for k, v in self.encodings.items()}
        item["label"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def build_dataloaders(
    train_df,
    val_df,
    test_df,
    tokenizer: AutoTokenizer,
    max_len: int,
    batch_size: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    def _make(df, shuffle):
        texts = df["text"].tolist()
        labels = df["label"].tolist()
        ds = TransformerDataset(texts, labels, tokenizer, max_len)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=0, pin_memory=True)

    return _make(train_df, True), _make(val_df, False), _make(test_df, False)
