import argparse
import hashlib
import os

import pandas as pd

from .config import PROCESSED_DATA_PATH


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def deduplicate_probe_split(
    reference_csv: str,
    probe_csv: str,
    output_csv: str,
) -> tuple[int, int]:
    ref_df = pd.read_csv(reference_csv)
    probe_df = pd.read_csv(probe_csv)

    ref_hashes = set(
        ref_df["text"].dropna().astype(str).apply(_sha256)
    )

    probe_df = probe_df.copy()
    probe_df["_hash"] = probe_df["text"].fillna("").astype(str).apply(_sha256)
    before = len(probe_df)
    probe_df = probe_df[~probe_df["_hash"].isin(ref_hashes)].drop(columns=["_hash"])
    after = len(probe_df)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    probe_df.to_csv(output_csv, index=False)
    return before, after


def run():
    isot_train = os.path.join(PROCESSED_DATA_PATH, "isot", "train.csv")
    welfake_test = os.path.join(PROCESSED_DATA_PATH, "welfake", "test.csv")
    output = os.path.join(PROCESSED_DATA_PATH, "welfake", "test_deduped.csv")

    before, after = deduplicate_probe_split(isot_train, welfake_test, output)
    removed = before - after
    print(f"WELFake test: {before} rows → {after} rows after dedup "
          f"({removed} duplicates removed, {removed/before*100:.1f}%)")
    print(f"Saved → {output}")


if __name__ == "__main__":
    run()
