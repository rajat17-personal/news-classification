"""
NELA-GT dataset loader (misinfo-general, ioverho/misinfo-general on HuggingFace).

3-class label mapping (publisher-level via MBFC):
  0 = Reliable     — Left, Left-Center, Least Biased, Right-Center, Right, Pro-Science
  1 = Questionable — Questionable Source
  2 = Conspiracy   — Conspiracy-Pseudoscience
  (Satire excluded — intentional fiction, not deceptive misinformation)

Speed strategy: parquet shards are processed in parallel across CPU cores using
ProcessPoolExecutor. Each worker cleans one shard and writes a per-shard temp CSV;
the main process concatenates them. Peak RAM per worker is ~200 MB (one shard).

Usage:
  # Full dataset for classical models
  python -m src.nela.dataset --preprocess

  # 500k sample for DL (BiLSTM/TextCNN)
  python -m src.nela.dataset --preprocess --sample 500000 --output-suffix sampled_500k

  # Specific years only
  python -m src.nela.dataset --preprocess --years 2019 2020 2021

  # Dry-run: print stats without writing final CSVs
  python -m src.nela.dataset --preprocess --dry-run

  # Control parallelism (default: min(cpu_count, 16))
  python -m src.nela.dataset --preprocess --workers 8
"""
import argparse
import gc
import hashlib
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split

from ..utils.config import PROCESSED_DATA_PATH, SEED
from ..utils.preprocessing import clean_text, input_text
from ..utils.seeds import set_seed

# src/nela/dataset.py → src/ → project_root/
NELA_DIR = Path(__file__).resolve().parents[2] / "misinfo-general"
NELA_DATA_DIR = NELA_DIR / "data"
NELA_METADATA_DB = NELA_DIR / "metadata.db"

ALL_YEARS = [2017, 2018, 2019, 2020, 2021, 2022]

_MBFC_TO_LABEL: dict[str, int] = {
    "Left":                      0,
    "Left-Center":               0,
    "Least Biased":              0,
    "Right-Center":              0,
    "Right":                     0,
    "Pro-Science":               0,
    "Questionable Source":       1,
    "Conspiracy-Pseudoscience":  2,
    # Satire deliberately omitted
}

LABEL_NAMES = {0: "Reliable", 1: "Questionable", 2: "Conspiracy"}


def _build_label_map(con: duckdb.DuckDBPyConnection) -> dict[str, int]:
    """Return {source_name: 0/1/2} from metadata.db. Satire sources are excluded."""
    rows = con.execute("SELECT source, label FROM sources").fetchall()
    label_map = {}
    for source, mbfc_label in rows:
        if mbfc_label not in _MBFC_TO_LABEL:
            continue
        label_map[source] = _MBFC_TO_LABEL[mbfc_label]
    return label_map


def _worker_clean_shard(args: tuple[Path, dict[str, int], str]) -> tuple[str, int]:
    """
    Worker function: clean one parquet shard and write to a temp CSV.
    Returns (temp_csv_path, row_count). Runs in a subprocess.
    """
    shard_path, label_map, tmp_dir = args

    # Each subprocess needs NLTK data — re-import triggers the lookup/download check
    import nltk  # noqa: F401

    tbl = pq.read_table(shard_path, columns=["source", "title", "content"])
    df = tbl.to_pandas()
    del tbl
    gc.collect()

    df = df[df["source"].isin(label_map)].copy()
    if df.empty:
        return ("", 0)

    df["label"] = df["source"].map(label_map)
    df = df.rename(columns={"content": "text"})
    df["text"] = df.apply(lambda row: input_text(row, strategy="full_body"), axis=1)
    df["text"] = df["text"].apply(clean_text)
    df = df[df["text"].notna() & (df["text"].str.strip() != "")]
    df = df[["text", "label", "source"]]

    if df.empty:
        return ("", 0)

    # Write to a unique temp file in the provided directory
    fd, tmp_path = tempfile.mkstemp(suffix=".csv", dir=tmp_dir, prefix=f"shard_{shard_path.stem}_")
    os.close(fd)
    df.to_csv(tmp_path, index=False)
    return (tmp_path, len(df))


def _collect_shards(years: list[int]) -> list[Path]:
    """Return all parquet shard paths for the given years, sorted."""
    shards = []
    for yr in years:
        yr_shards = sorted(NELA_DATA_DIR.glob(f"{yr}-*.parquet"))
        if not yr_shards:
            raise FileNotFoundError(f"No parquet files for year {yr} in {NELA_DATA_DIR}")
        shards.extend(yr_shards)
    return shards


def _process_shards_parallel(
    shards: list[Path],
    label_map: dict[str, int],
    tmp_dir: str,
    n_workers: int,
) -> list[str]:
    """
    Process all shards in parallel. Returns list of per-shard temp CSV paths
    in the same order as input shards (empty-shard entries are filtered out).
    """
    args = [(shard, label_map, tmp_dir) for shard in shards]
    results: dict[int, str] = {}  # index → tmp_path
    total_written = 0

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_worker_clean_shard, a): i for i, a in enumerate(args)}
        done = 0
        for future in as_completed(futures):
            idx = futures[future]
            tmp_path, count = future.result()
            done += 1
            if tmp_path:
                results[idx] = tmp_path
                total_written += count
            shard_name = shards[idx].name
            print(f"  [{done:>2}/{len(shards)}] {shard_name}: {count:,} articles", flush=True)

    # Return in original shard order so the CSV is deterministic
    return [results[i] for i in sorted(results)]


def _merge_shard_csvs(shard_csvs: list[str], out_path: str) -> int:
    """Concatenate per-shard CSVs into one file. Returns total row count."""
    total = 0
    header_written = False
    for csv_path in shard_csvs:
        df = pd.read_csv(csv_path)
        df.to_csv(out_path, mode="a", index=False, header=not header_written)
        header_written = True
        total += len(df)
        os.unlink(csv_path)
    return total


def _dedup_and_split(
    merged_csv: str,
    out_dir: str,
    sample: int | None,
    dry_run: bool,
    chunksize: int = 200_000,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Two-pass approach — never loads the full corpus into RAM:
      Pass 1: stream dedup → write survivors to a per-publisher reservoir on disk
              (one small CSV per publisher in a temp dir).
      Pass 2: for each publisher apply quota sampling, assign train/val/test split
              row-by-row, write directly to output CSVs.

    Peak RAM: one chunk (chunksize rows) + publisher-level reservoirs metadata.
    """
    import random
    random.seed(SEED)

    tmp_dir = tempfile.mkdtemp(dir=os.path.dirname(merged_csv), prefix="nela_pubres_")
    try:
        # --- Pass 1: dedup, stream survivors into per-publisher temp CSVs ---
        seen_hashes: set[str] = set()
        pub_files: dict[str, str] = {}       # source → temp csv path
        pub_counts: dict[str, int] = {}      # source → row count
        pub_writers: dict[str, bool] = {}    # source → header written
        total_in = 0
        total_out = 0

        print("[NELA] Deduplicating (streaming)...")
        for chunk in pd.read_csv(merged_csv, chunksize=chunksize,
                                  dtype={"label": "int8", "source": "category"}):
            total_in += len(chunk)
            chunk["_hash"] = chunk["text"].apply(
                lambda t: hashlib.sha256(str(t).encode()).hexdigest()
            )
            mask = ~chunk["_hash"].isin(seen_hashes)
            seen_hashes.update(chunk.loc[mask, "_hash"].tolist())
            survivors = chunk[mask].drop(columns=["_hash"])
            total_out += len(survivors)

            for src, grp in survivors.groupby("source", observed=True):
                src = str(src)
                if src not in pub_files:
                    fd, path = tempfile.mkstemp(suffix=".csv", dir=tmp_dir,
                                                prefix=f"pub_{src[:20]}_")
                    os.close(fd)
                    pub_files[src] = path
                    pub_counts[src] = 0
                    pub_writers[src] = False
                grp.to_csv(pub_files[src], mode="a", index=False,
                           header=not pub_writers[src])
                pub_writers[src] = True
                pub_counts[src] = pub_counts.get(src, 0) + len(grp)

            print(f"  processed {total_in:,} | kept {total_out:,} | "
                  f"dupes {total_in - total_out:,}", end="\r", flush=True)

        del seen_hashes
        print()
        print(f"[NELA] After dedup: {total_out:,} (removed {total_in - total_out:,})")

        # Print class distribution from pub_counts metadata (cheap)
        label_totals: dict[int, int] = {}
        pub_labels: dict[str, int] = {}
        # Re-read first row of each pub file to get label
        for src, path in pub_files.items():
            row = pd.read_csv(path, nrows=1)
            lbl = int(row["label"].iloc[0])
            pub_labels[src] = lbl
            label_totals[lbl] = label_totals.get(lbl, 0) + pub_counts[src]
        vc_parts = [f"{LABEL_NAMES.get(l, l)}={c:,}" for l, c in sorted(label_totals.items())]
        print(f"         {' | '.join(vc_parts)}")

        # --- Compute per-publisher quotas for stratified sampling ---
        if sample is not None and total_out > sample:
            print(f"[NELA] Subsampling to {sample:,} stratified by publisher...")
            quotas = {src: max(1, round(sample * cnt / total_out))
                      for src, cnt in pub_counts.items()}
            # Trim total to exactly sample
            total_q = sum(quotas.values())
            if total_q > sample:
                excess = total_q - sample
                for src in sorted(quotas, key=lambda s: -quotas[s]):
                    trim = min(excess, quotas[src] - 1)
                    quotas[src] -= trim
                    excess -= trim
                    if excess == 0:
                        break
        else:
            quotas = dict(pub_counts)

        final_total = sum(quotas.values())
        print(f"[NELA] Writing {final_total:,} articles across "
              f"{len(quotas)} publishers to {out_dir}...")

        if dry_run:
            print("[NELA] --dry-run: skipping CSV write.")
            # Return empty frames with correct columns as a stub
            stub = pd.DataFrame(columns=["text", "label", "source"])
            return stub, stub, stub

        # --- Pass 2: sample from each publisher, assign split, write directly ---
        os.makedirs(out_dir, exist_ok=True)
        train_path = os.path.join(out_dir, "train.csv")
        val_path   = os.path.join(out_dir, "val.csv")
        test_path  = os.path.join(out_dir, "test.csv")
        split_headers = {"train": False, "val": False, "test": False}
        split_paths   = {"train": train_path, "val": val_path, "test": test_path}

        train_count = val_count = test_count = 0

        for src, path in pub_files.items():
            quota = quotas.get(src, 0)
            if quota == 0:
                continue
            pub_df = pd.read_csv(path, dtype={"label": "int8"})
            if len(pub_df) > quota:
                pub_df = pub_df.sample(n=quota, random_state=SEED)

            # Assign 80/10/10 within this publisher's rows
            n = len(pub_df)
            n_val  = max(1, round(n * 0.1))
            n_test = max(1, round(n * 0.1))
            n_train = n - n_val - n_test

            pub_df = pub_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
            splits = {
                "train": pub_df.iloc[:n_train],
                "val":   pub_df.iloc[n_train:n_train + n_val],
                "test":  pub_df.iloc[n_train + n_val:],
            }
            for split_name, sdf in splits.items():
                if sdf.empty:
                    continue
                sdf.to_csv(split_paths[split_name], mode="a", index=False,
                           header=not split_headers[split_name])
                split_headers[split_name] = True

            train_count += len(splits["train"])
            val_count   += len(splits["val"])
            test_count  += len(splits["test"])

        print(f"[NELA] Split → train={train_count:,}  val={val_count:,}  test={test_count:,}")
        print(f"[NELA] Saved to {out_dir}")

        # Return lightweight head of each split rather than loading full files
        train_df = pd.read_csv(train_path, dtype={"label": "int8"}, nrows=1000)
        val_df   = pd.read_csv(val_path,   dtype={"label": "int8"}, nrows=1000)
        test_df  = pd.read_csv(test_path,  dtype={"label": "int8"}, nrows=1000)
        return train_df, val_df, test_df

    finally:
        # Clean up per-publisher temp files
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)


def subsample_from_processed(
    source_suffix: str = "",
    sample: int = 500000,
    output_suffix: str = "sampled_500k",
    chunksize: int = 100_000,
) -> None:
    """
    Reuse an already-preprocessed NELA corpus to produce a subsampled variant.
    Three-pass approach — peak RAM is O(sample) integers, not O(corpus) text:

      Pass 1: count rows per publisher (reads only 'source' column)
      Pass 2: reservoir-sample row indices per publisher (stores ints only)
      Pass 3: stream source CSVs, write rows whose global index was selected
    """
    import random
    random.seed(SEED)

    src_name = f"nela_{source_suffix}" if source_suffix else "nela"
    src_dir = os.path.join(PROCESSED_DATA_PATH, src_name)
    out_name = f"nela_{output_suffix}"
    out_dir = os.path.join(PROCESSED_DATA_PATH, out_name)

    src_files = [os.path.join(src_dir, s) for s in ("train.csv", "val.csv", "test.csv")]
    for path in src_files:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Source file not found: {path}\n"
                f"Run --preprocess first to generate {src_name}."
            )

    # --- Pass 1: count rows per publisher ---
    print(f"[NELA] Counting publisher totals in {src_dir}...")
    pub_counts: dict[str, int] = {}
    total_rows = 0
    for path in src_files:
        for chunk in pd.read_csv(path, usecols=["source"], chunksize=chunksize):
            for src, cnt in chunk["source"].value_counts().items():
                pub_counts[str(src)] = pub_counts.get(str(src), 0) + int(cnt)
            total_rows += len(chunk)
    print(f"[NELA] Total: {total_rows:,} articles across {len(pub_counts)} publishers")

    # Compute per-publisher quotas
    quotas: dict[str, int] = {}
    for src, cnt in pub_counts.items():
        quotas[src] = max(1, round(sample * cnt / total_rows))
    total_q = sum(quotas.values())
    if total_q > sample:
        excess = total_q - sample
        for src in sorted(quotas, key=lambda s: -quotas[s]):
            trim = min(excess, quotas[src] - 1)
            quotas[src] -= trim
            excess -= trim
            if excess == 0:
                break
    print(f"[NELA] Quota: {sum(quotas.values()):,} articles across {len(quotas)} publishers")

    # --- Pass 2: reservoir-sample global row indices per publisher (ints only) ---
    # Global index = position across all concatenated source files
    print("[NELA] Selecting row indices (streaming)...")
    # reservoir[src] = list of selected global indices (at most quota[src] ints)
    reservoirs: dict[str, list[int]] = {src: [] for src in quotas}
    seen_counts: dict[str, int] = {src: 0 for src in quotas}
    global_idx = 0

    for path in src_files:
        for chunk in pd.read_csv(path, usecols=["source"], chunksize=chunksize):
            for local_i, src in enumerate(chunk["source"]):
                src = str(src)
                if src not in quotas:
                    global_idx += 1
                    continue
                quota = quotas[src]
                seen_counts[src] += 1
                k = seen_counts[src]
                gidx = global_idx + local_i
                if k <= quota:
                    reservoirs[src].append(gidx)
                else:
                    j = random.randint(1, k)
                    if j <= quota:
                        reservoirs[src][j - 1] = gidx
            global_idx += len(chunk)

    # Flatten to a set of selected global indices for O(1) lookup in pass 3
    selected: set[int] = set()
    for indices in reservoirs.values():
        selected.update(indices)

    # Map each selected index to its split assignment (80/10/10 per publisher)
    idx_to_split: dict[int, str] = {}
    for src, indices in reservoirs.items():
        random.shuffle(indices)
        n = len(indices)
        n_val  = max(1, round(n * 0.1))
        n_test = max(1, round(n * 0.1))
        for i, gidx in enumerate(indices):
            if i < n - n_val - n_test:
                idx_to_split[gidx] = "train"
            elif i < n - n_test:
                idx_to_split[gidx] = "val"
            else:
                idx_to_split[gidx] = "test"

    del reservoirs
    print(f"[NELA] Selected {len(selected):,} rows — writing splits...")

    # --- Pass 3: stream source CSVs, write selected rows to output ---
    os.makedirs(out_dir, exist_ok=True)
    paths   = {s: os.path.join(out_dir, f"{s}.csv") for s in ("train", "val", "test")}
    headers = {"train": False, "val": False, "test": False}
    counts  = {"train": 0, "val": 0, "test": 0}
    global_idx = 0

    for path in src_files:
        for chunk in pd.read_csv(path, chunksize=chunksize,
                                  dtype={"label": "int8"}):
            chunk = chunk[["text", "label", "source"]].reset_index(drop=True)
            # Find which local rows are selected and which split they belong to
            local_indices = range(len(chunk))
            global_indices = [global_idx + i for i in local_indices]
            sel_mask = [gidx in selected for gidx in global_indices]

            if any(sel_mask):
                sel_chunk = chunk[sel_mask].copy()
                sel_chunk["_split"] = [idx_to_split[global_idx + i]
                                       for i, keep in enumerate(sel_mask) if keep]
                for split_name, sdf in sel_chunk.groupby("_split"):
                    sdf = sdf.drop(columns=["_split"])
                    sdf.to_csv(paths[split_name], mode="a", index=False,
                               header=not headers[split_name])
                    headers[split_name] = True
                    counts[split_name] += len(sdf)

            global_idx += len(chunk)
            print(f"  written {sum(counts.values()):,} / {len(selected):,}",
                  end="\r", flush=True)

    print()
    print(f"[NELA] Split → train={counts['train']:,}  val={counts['val']:,}  test={counts['test']:,}")
    print(f"[NELA] Saved to {out_dir}")


def _print_class_dist(df: pd.DataFrame) -> None:
    vc = df["label"].value_counts().sort_index()
    parts = [f"{LABEL_NAMES.get(lbl, lbl)}={cnt:,}" for lbl, cnt in vc.items()]
    print(f"         {' | '.join(parts)}")


def _sample_stratified_by_publisher(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """
    Sample n articles proportionally by publisher so no single outlet dominates.
    Publishers with fewer articles than their quota contribute all their articles.
    """
    pub_counts = df["source"].value_counts()
    total = len(df)
    parts = []
    for source, count in pub_counts.items():
        quota = max(1, round(n * count / total))
        pub_df = df[df["source"] == source]
        parts.append(pub_df if len(pub_df) <= quota else pub_df.sample(n=quota, random_state=seed))

    result = pd.concat(parts, ignore_index=True)
    if len(result) > n:
        result = result.sample(n=n, random_state=seed)
    return result.sample(frac=1, random_state=seed).reset_index(drop=True)


def preprocess(
    years: list[int] | None = None,
    sample: int | None = None,
    output_suffix: str = "",
    dry_run: bool = False,
    n_workers: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Full preprocessing pipeline:
      1. Collect all parquet shards for requested years
      2. Clean shards in parallel (ProcessPoolExecutor) → per-shard temp CSVs
      3. Merge temp CSVs → single CSV; SHA-256 dedup
      4. Optional publisher-stratified subsampling
      5. Stratified 80/10/10 train/val/test split → save to data/processed/nela{suffix}/

    Returns (train_df, val_df, test_df).
    """
    set_seed(SEED)
    years = years or ALL_YEARS
    suffix = f"_{output_suffix}" if output_suffix else ""
    dataset_name = f"nela{suffix}"
    out_dir = os.path.join(PROCESSED_DATA_PATH, dataset_name)

    # Each worker holds ~1.5 GB RAM during NLTK tokenization of one shard.
    # Cap based on available RAM (leave 4 GB headroom for main process + OS).
    if n_workers is not None:
        workers = n_workers
    else:
        import psutil
        available_gb = psutil.virtual_memory().available / (1024 ** 3)
        ram_based = max(1, int((available_gb - 4) / 1.5))
        cpu_count = os.cpu_count() or 4
        workers = min(ram_based, cpu_count, 12)
        print(f"[NELA] Available RAM: {available_gb:.1f} GB → using {workers} workers")

    print(f"[NELA] Loading metadata from {NELA_METADATA_DB}")
    con = duckdb.connect(str(NELA_METADATA_DB), read_only=True)
    label_map = _build_label_map(con)
    con.close()

    n_per_class = {0: 0, 1: 0, 2: 0}
    for v in label_map.values():
        n_per_class[v] += 1
    print(f"[NELA] Publishers: {n_per_class[0]} reliable, {n_per_class[1]} questionable, "
          f"{n_per_class[2]} conspiracy ({len(label_map)} total, Satire excluded)")

    shards = _collect_shards(years)
    print(f"[NELA] {len(shards)} shards across years {years} — using {workers} workers")

    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

    # Use a temp dir for per-shard CSVs; cleaned up in the finally block
    with tempfile.TemporaryDirectory(dir=PROCESSED_DATA_PATH, prefix="nela_shards_") as shard_tmp:
        print("[NELA] Cleaning shards in parallel...")
        shard_csvs = _process_shards_parallel(shards, label_map, shard_tmp, workers)

        # Merge into one temp CSV
        fd, merged_tmp = tempfile.mkstemp(suffix=".csv", dir=PROCESSED_DATA_PATH, prefix="nela_merged_")
        os.close(fd)
        try:
            print("[NELA] Merging shard CSVs...")
            total_raw = _merge_shard_csvs(shard_csvs, merged_tmp)
            print(f"[NELA] Total cleaned articles: {total_raw:,}")
            return _dedup_and_split(merged_tmp, out_dir, sample, dry_run)
        finally:
            if os.path.exists(merged_tmp):
                os.unlink(merged_tmp)


def main():
    parser = argparse.ArgumentParser(description="NELA-GT dataset preprocessor")
    parser.add_argument("--preprocess", action="store_true")
    parser.add_argument("--years", nargs="+", type=int, default=None, metavar="YEAR",
                        help="Years to include (default: all 2017-2022)")
    parser.add_argument("--sample", type=int, default=None,
                        help="Subsample N articles stratified by publisher")
    parser.add_argument("--output-suffix", type=str, default=None,
                        help="e.g. 'sampled_500k' → data/processed/nela_sampled_500k/ "
                             "(use '' for data/processed/nela/)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Parallel workers for shard cleaning (default: min(cpu_count, 16))")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print stats without writing final CSVs")
    parser.add_argument("--from-merged", type=str, default=None, metavar="CSV_PATH",
                        help="Resume from an existing merged CSV (skips shard cleaning). "
                             "Requires --output-suffix.")
    parser.add_argument("--from-processed", action="store_true",
                        help="Subsample from an already-preprocessed nela/ corpus "
                             "instead of re-running shard cleaning")
    parser.add_argument("--source-suffix", type=str, default="",
                        help="Suffix of the source processed dir (default: nela/ i.e. no suffix)")
    args = parser.parse_args()

    if args.from_merged:
        if args.output_suffix is None:
            parser.error("--from-merged requires --output-suffix (use '' for data/processed/nela/)")
        out_dir = os.path.join(PROCESSED_DATA_PATH,
                               f"nela_{args.output_suffix}" if args.output_suffix else "nela")
        _dedup_and_split(args.from_merged, out_dir, args.sample, args.dry_run)
    elif args.from_processed:
        if not args.sample:
            parser.error("--from-processed requires --sample N")
        if not args.output_suffix:
            parser.error("--from-processed requires --output-suffix")
        subsample_from_processed(
            source_suffix=args.source_suffix or "",
            sample=args.sample,
            output_suffix=args.output_suffix,
        )
    elif args.preprocess:
        preprocess(
            years=args.years,
            sample=args.sample,
            output_suffix=args.output_suffix or "",
            dry_run=args.dry_run,
            n_workers=args.workers,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
