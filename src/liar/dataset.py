"""
P9.1 — LIAR dataset loader and preprocessor.

LIAR TSV columns (no header):
  0: id, 1: label, 2: statement, 3: subject, 4: speaker,
  5: job_title, 6: state_info, 7: party_affiliation,
  8: barely_true_counts, 9: false_counts, 10: half_true_counts,
  11: mostly_true_counts, 12: pants_on_fire_counts, 13: context

6-class label set (kept as-is for multi-class):
  pants-fire=0, false=1, barely-true=2, half-true=3, mostly-true=4, true=5

Only the statement (col 2) is used as text — no full article body.
"""
import os

import pandas as pd
from sklearn.model_selection import train_test_split

from ..utils.config import PROCESSED_DATA_PATH, RAW_DATA_PATH, SEED
from ..utils.preprocessing import clean_text

LIAR_LABELS = {
    "pants-fire": 0,
    "false":      1,
    "barely-true": 2,
    "half-true":  3,
    "mostly-true": 4,
    "true":       5,
}

LIAR_LABEL_NAMES = {v: k for k, v in LIAR_LABELS.items()}

_COLS = [
    "id", "label", "statement", "subject", "speaker",
    "job_title", "state_info", "party_affiliation",
    "barely_true_counts", "false_counts", "half_true_counts",
    "mostly_true_counts", "pants_on_fire_counts", "context",
]


def _load_tsv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", header=None, names=_COLS)
    df = df[["statement", "label"]].copy()
    df = df[df["label"].isin(LIAR_LABELS)]
    df["label"] = df["label"].map(LIAR_LABELS)
    df["text"] = df["statement"].apply(clean_text)
    df = df[df["text"].notna() & (df["text"].str.strip() != "")]
    return df[["text", "label"]]


def load_liar_splits(force: bool = False):
    """
    Returns (train_df, val_df, test_df) using the official LIAR splits.
    Preprocesses and caches on first call.
    """
    out_dir = os.path.join(PROCESSED_DATA_PATH, "liar")
    paths = {s: os.path.join(out_dir, f"{s}.csv") for s in ("train", "val", "test")}

    if not force and all(os.path.exists(p) for p in paths.values()):
        print("LIAR processed splits already exist. Loading from disk.")
        return (
            pd.read_csv(paths["train"]),
            pd.read_csv(paths["val"]),
            pd.read_csv(paths["test"]),
        )

    raw_dir = os.path.join(RAW_DATA_PATH, "liar")
    print("Preprocessing LIAR dataset...")
    train_df = _load_tsv(os.path.join(raw_dir, "train.tsv"))
    val_df   = _load_tsv(os.path.join(raw_dir, "valid.tsv"))
    test_df  = _load_tsv(os.path.join(raw_dir, "test.tsv"))

    os.makedirs(out_dir, exist_ok=True)
    train_df.to_csv(paths["train"], index=False)
    val_df.to_csv(paths["val"],   index=False)
    test_df.to_csv(paths["test"],  index=False)

    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        dist = df["label"].value_counts().sort_index()
        dist_str = "  ".join(f"{LIAR_LABEL_NAMES[i]}={n}" for i, n in dist.items())
        print(f"  {name}: {len(df)} rows — {dist_str}")

    return train_df, val_df, test_df
