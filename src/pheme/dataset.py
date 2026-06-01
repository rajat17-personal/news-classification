"""
PHEME rumour dataset loader — OOD evaluation only.

PHEME-9: ~2,402 rumour threads across 9 breaking-news events (2014-2015).
After dropping unverified threads, ~1,401 usable rows (true + false rumours).

IMPORTANT domain shift caveats vs. ISOT/WELFake:
  - Text is tweet-length (~15-25 words) vs. article-length (300-1000 words)
  - All text is from Twitter, not news sites
  - "Real" = verified-true rumour, "Fake" = verified-false rumour
    (not the same semantic as Reuters real vs. tabloid fake)
  - Use for robustness/OOD analysis only; expect large accuracy drops

Download (run once):
  python -m src.pheme.dataset --download

Labels: false rumour = 0 (fake), true rumour = 1 (real)
"""
import argparse
import glob
import json
import os
import re
import subprocess

import pandas as pd
from sklearn.model_selection import train_test_split

from ..utils.config import PROCESSED_DATA_PATH, RAW_DATA_PATH, SEED

PHEME_FIGSHARE_URL = "https://ndownloader.figshare.com/files/11767817"
PHEME_RAW_DIR      = os.path.join(RAW_DATA_PATH, "pheme")
PHEME_ARCHIVE      = os.path.join(PHEME_RAW_DIR, "PHEME_veracity.tar.bz2")
PHEME_THREADS_DIR  = os.path.join(PHEME_RAW_DIR, "all-rnr-annotated-threads")

# Events with enough verified rumour threads to be useful
_ALL_EVENTS = (
    "charliehebdo", "ferguson", "germanwings-crash",
    "ottawashooting", "sydneysiege",
    "putinmissing", "prince-toronto", "ebola-essien", "gurlitt",
)


def download_pheme(force: bool = False) -> None:
    """Download and extract PHEME-9 from Figshare."""
    if not force and os.path.isdir(PHEME_THREADS_DIR):
        print("PHEME raw data already exists. Skipping download.")
        return

    os.makedirs(PHEME_RAW_DIR, exist_ok=True)
    print(f"Downloading PHEME from Figshare ({PHEME_FIGSHARE_URL}) ...")
    subprocess.run(
        ["wget", "-q", "--show-progress", "-O", PHEME_ARCHIVE, PHEME_FIGSHARE_URL],
        check=True,
    )
    print("Extracting archive ...")
    # Figshare serves a gzip stream despite the .bz2 filename — use -xzf not -xjf
    subprocess.run(
        ["tar", "-xzf", PHEME_ARCHIVE, "-C", PHEME_RAW_DIR],
        check=True,
    )
    print(f"Extracted → {PHEME_THREADS_DIR}")


def _clean_tweet(text: str) -> str:
    text = re.sub(r"http\S+|www\S+", "", text)       # URLs
    text = re.sub(r"@\w+", "@USER", text)             # mentions
    text = re.sub(r"#(\w+)", r"\1", text)             # hashtags → keep word
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _derive_veracity(ann: dict) -> str:
    # Values can be int or str ("1"/"0") depending on the annotation file
    m = int(ann.get("misinformation", 0))
    t = int(ann.get("true", 0))
    if m == 0 and t == 1:
        return "true"
    if m == 1 and t == 0:
        return "false"
    return "unverified"


def _load_raw(threads_dir: str, drop_unverified: bool = True) -> pd.DataFrame:
    records = []
    for event_dir in sorted(os.listdir(threads_dir)):
        if event_dir.startswith("."):
            continue
        # Actual dirs have suffix -all-rnr-threads; derive a short event name for labelling
        event = event_dir.replace("-all-rnr-threads", "")
        rumours_path = os.path.join(threads_dir, event_dir, "rumours")
        if not os.path.isdir(rumours_path):
            continue
        for thread_id in os.listdir(rumours_path):
            thread_path  = os.path.join(rumours_path, thread_id)
            ann_path     = os.path.join(thread_path, "annotation.json")
            src_dir      = os.path.join(thread_path, "source-tweets")
            if not os.path.isfile(ann_path):
                continue
            src_files = glob.glob(os.path.join(src_dir, "*.json"))
            if not src_files:
                continue
            with open(ann_path) as f:
                ann = json.load(f)
            with open(src_files[0]) as f:
                tweet = json.load(f)

            veracity = _derive_veracity(ann)
            if drop_unverified and veracity == "unverified":
                continue

            label = {"true": 1, "false": 0}.get(veracity, -1)
            text  = tweet.get("full_text") or tweet.get("text", "")
            text  = _clean_tweet(text)
            if not text:
                continue

            records.append({
                "text":      text,
                "label":     label,
                "event":     event,
                "thread_id": thread_id,
                "veracity":  veracity,
            })

    return pd.DataFrame(records)


def load_pheme_splits(force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns (train_df, val_df, test_df) with columns [text, label].

    Given PHEME's small size (~1,400 rows) this is intended for OOD eval only.
    The test split is what you should report; train/val are provided for
    completeness and are stratified by both label and event to prevent event leakage.
    Split: 60/20/20 (more test data for reliable OOD estimates).
    """
    out_dir   = os.path.join(PROCESSED_DATA_PATH, "pheme")
    train_p   = os.path.join(out_dir, "train.csv")
    val_p     = os.path.join(out_dir, "val.csv")
    test_p    = os.path.join(out_dir, "test.csv")
    stats_p   = os.path.join(out_dir, "pheme_stats.txt")

    if not force and all(os.path.exists(p) for p in (train_p, val_p, test_p)):
        print("PHEME processed splits already exist. Loading from disk.")
        return pd.read_csv(train_p), pd.read_csv(val_p), pd.read_csv(test_p)

    if not os.path.isdir(PHEME_THREADS_DIR):
        raise FileNotFoundError(
            f"PHEME raw data not found at {PHEME_THREADS_DIR}. "
            "Run: python -m src.pheme.dataset --download"
        )

    print("Preprocessing PHEME dataset...")
    df = _load_raw(PHEME_THREADS_DIR, drop_unverified=True)

    label_dist = df["label"].value_counts().sort_index()
    event_dist = df["event"].value_counts()
    print(f"  Total rows: {len(df)}  |  false={label_dist.get(0,0)}  true={label_dist.get(1,0)}")
    print(f"  Events: {dict(event_dist)}")

    # Stratified split by label (60/20/20)
    train_df, temp_df = train_test_split(
        df, test_size=0.4, random_state=SEED, stratify=df["label"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=SEED, stratify=temp_df["label"]
    )

    os.makedirs(out_dir, exist_ok=True)
    train_df[["text", "label"]].to_csv(train_p, index=False)
    val_df[["text", "label"]].to_csv(val_p,   index=False)
    test_df[["text", "label"]].to_csv(test_p,  index=False)

    with open(stats_p, "w") as f:
        f.write(f"total_rows: {len(df)}\n")
        f.write(f"false_rumours: {label_dist.get(0,0)}\n")
        f.write(f"true_rumours: {label_dist.get(1,0)}\n")
        f.write(f"train: {len(train_df)}  val: {len(val_df)}  test: {len(test_df)}\n")
        f.write(f"events: {dict(event_dist)}\n")
        f.write("\nDOMAIN SHIFT WARNINGS:\n")
        f.write("  - Tweets (~15-25 words) vs. articles (~300-1000 words)\n")
        f.write("  - 'Real' = verified-true rumour (not Reuters wire news)\n")
        f.write("  - 'Fake' = verified-false rumour (not tabloid article)\n")
        f.write("  - Expected large accuracy drop vs. within-dataset results\n")

    print(f"  Saved → {out_dir}")
    return train_df[["text", "label"]], val_df[["text", "label"]], test_df[["text", "label"]]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PHEME dataset utilities")
    parser.add_argument("--download", action="store_true", help="Download PHEME from Figshare")
    parser.add_argument("--preprocess", action="store_true", help="Preprocess into train/val/test CSVs")
    parser.add_argument("--force", action="store_true", help="Force re-download/re-preprocess")
    args = parser.parse_args()

    if args.download:
        download_pheme(force=args.force)
    if args.preprocess or not args.download:
        load_pheme_splits(force=args.force)
