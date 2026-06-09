import os

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import PROCESSED_DATA_PATH, SEED, DatasetConfig


COMBINED_CONFIG = DatasetConfig(
    name="combined",
    label_map={0: 0, 1: 1},
    binary=True,
)

def load_combined_splits(force: bool = False):
    out_dir = os.path.join(PROCESSED_DATA_PATH, "combined")
    train_path = os.path.join(out_dir, "train.csv")
    val_path   = os.path.join(out_dir, "val.csv")
    test_path  = os.path.join(out_dir, "test.csv")

    if not force and all(os.path.exists(p) for p in [train_path, val_path, test_path]):
        print("Combined dataset already exists. Loading from disk.")
        return (
            pd.read_csv(train_path),
            pd.read_csv(val_path),
            pd.read_csv(test_path),
        )

    print("Building combined ISOT + WELFake dataset...")
    frames = []
    for name in ("isot", "welfake"):
        src_dir = os.path.join(PROCESSED_DATA_PATH, name)
        for split in ("train", "val", "test"):
            df = pd.read_csv(os.path.join(src_dir, f"{split}.csv"))
            df = df[["text", "label"]].copy()
            df["source"] = name
            frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    # drop rows where text is null or empty
    combined = combined[combined["text"].notna() & (combined["text"].str.strip() != "")]
    # deduplicate by text hash to remove ISOT∩WELFake overlap
    combined = combined.drop_duplicates(subset=["text"])
    combined = combined.sample(frac=1, random_state=SEED).reset_index(drop=True)

    train_df, temp_df = train_test_split(
        combined, test_size=0.2, random_state=SEED, stratify=combined["label"]
    )
    val_df   = temp_df.sample(frac=0.5, random_state=SEED)
    test_df  = temp_df.drop(val_df.index)

    os.makedirs(out_dir, exist_ok=True)
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path,   index=False)
    test_df.to_csv(test_path,  index=False)

    print(f"  Combined: train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}")
    label_dist = combined["label"].value_counts(normalize=True)
    print(f"  Label balance: fake={label_dist.get(0,0):.1%}  real={label_dist.get(1,0):.1%}")
    return train_df, val_df, test_df
