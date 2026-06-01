# Load raw dataset (ISOT and WELFAKE), preprocess it for training (train test split: 80/20, with second split 50/50 for validation and test), and save the processed dataset to disk.
# Preprocessing steps: apply clean_text to text column, apply input_text() to get unified ttext column, encode labels using label_map, drop exact duplicate "text" values
# get_stats() fun to get class counts + %, text length stats (mean, median, p95, max), vocab size (unique tokens) and save to data/processed/<dataset_name>_stats.txt

from .config import RAW_DATA_PATH, PROCESSED_DATA_PATH, DatasetConfig, SEED
from .preprocessing import clean_text, input_text
from .seeds import set_seed
import os
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

# Define dataloader for ISOt, WLFAKE and LIAR datasets that loads the processed dataset from disk and returns train, val and test dataframes.
# ISOT (merge true and fake csv, add label column)
# WELFAKE (single file with coliumns: title, text, label (0 for fake, 1 for real))
# LIAR (sepearate test, train and val files with columns: id, label, statement, subject, speaker, job_title, state_info, party_affiliation, barely_true_counts, false_counts, half_true_counts, mostly_true_counts, pants_on_fire_counts)

class DatasetLoader:
    def __init__(self, config: DatasetConfig):
        self.config = config
    
    def load_data(self):
        processed_dir = os.path.join(PROCESSED_DATA_PATH, self.config.name)
        train_df = pd.read_csv(os.path.join(processed_dir, "train.csv"))
        val_df = pd.read_csv(os.path.join(processed_dir, "val.csv"))
        test_df = pd.read_csv(os.path.join(processed_dir, "test.csv"))
        return train_df, val_df, test_df

def data_loader(config: DatasetConfig):
    loader = DatasetLoader(config)
    return loader.load_data()

def get_stats(df: pd.DataFrame, config: DatasetConfig):
    stats = {}
    stats['class_counts'] = df["label"].value_counts().to_dict()
    stats['class_percentages'] = df["label"].value_counts(normalize=True).to_dict()
    stats['text_length_mean'] = df["text"].apply(lambda x: len(str(x).split())).mean()
    stats['text_length_median'] = df["text"].apply(lambda x: len(str(x).split())).median()
    stats['text_length_p95'] = df["text"].apply(lambda x: len(str(x).split())).quantile(0.95)
    stats['text_length_max'] = df["text"].apply(lambda x: len(str(x).split())).max()
    stats['vocab_size'] = len(set(' '.join(df["text"].astype(str)).split()))
    # save stats to disk
    stats_dir = os.path.join(PROCESSED_DATA_PATH, config.name)
    os.makedirs(stats_dir, exist_ok=True)
    with open(os.path.join(stats_dir, f"{config.name}_stats.txt"), "w") as f:
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")
    return stats

def load_and_preprocess_dataset(config: DatasetConfig):
    set_seed(SEED)
    # Load raw dataset (ISOT has two files: true.csv and fake.csv, WELFAKE has one file: data.csv with label column (0 for fake, 1 for real))
    raw_dir = os.path.join(RAW_DATA_PATH, config.name)
    if config.name == "isot":
        df_true = pd.read_csv(os.path.join(raw_dir, "True.csv"))
        df_fake = pd.read_csv(os.path.join(raw_dir, "Fake.csv"))
        df_true['label'] = "Real"
        df_fake['label'] = "Fake"
        df = pd.concat([df_true, df_fake], ignore_index=True)
    elif config.name == "welfake":
        df = pd.read_csv(os.path.join(raw_dir, "WELFake_Dataset.csv"))
    # apply input_text() to get unified text column
    df['text'] = df.apply(lambda row: input_text(row, strategy="full_body"), axis=1)
    # apply clean_text to text column
    df['text'] = df['text'].apply(clean_text)
    # encode labels using label_map
    df['label'] = df['label'].map(config.label_map)
    # drop empty/NaN text rows produced by clean_text on NaN-only rows
    df = df[df['text'].notna() & (df['text'].str.strip() != '')]
    # drop exact duplicate "text" values
    df = df.drop_duplicates(subset=['text'])
    # shuffle the dataset
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    # stratified split into train (80%) and temp (20%), then split temp into validation (50%) and test (50%)
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df['label'])
    val_df = temp_df.sample(frac=0.5, random_state=SEED)
    test_df = temp_df.drop(val_df.index)
    # save processed dataset to disk
    processed_dir = os.path.join(PROCESSED_DATA_PATH, config.name)
    os.makedirs(processed_dir, exist_ok=True)
    train_df.to_csv(os.path.join(processed_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(processed_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(processed_dir, "test.csv"), index=False)
    # get and save statistics
    get_stats(train_df, config)

def load_splits(config: DatasetConfig):
    if config.name == "combined":
        from .combined_loader import load_combined_splits
        return load_combined_splits()
    if config.name.startswith("nela"):
        # NELA must be preprocessed explicitly via src.nela.dataset --preprocess
        processed_dir = os.path.join(PROCESSED_DATA_PATH, config.name)
        if not (os.path.exists(os.path.join(processed_dir, "train.csv")) and
                os.path.exists(os.path.join(processed_dir, "val.csv")) and
                os.path.exists(os.path.join(processed_dir, "test.csv"))):
            raise FileNotFoundError(
                f"NELA processed splits not found at {processed_dir}. "
                f"Run: python -m src.nela.dataset --preprocess "
                f"[--sample 100000 --output-suffix sampled_100k]"
            )
        print(f"Processed dataset for {config.name} already exists. Loading from disk.")
        return data_loader(config)
    # load existing CSVs; skip reprocessing if they already exist
    processed_dir = os.path.join(PROCESSED_DATA_PATH, config.name)
    if os.path.exists(os.path.join(processed_dir, "train.csv")) and \
       os.path.exists(os.path.join(processed_dir, "val.csv")) and \
       os.path.exists(os.path.join(processed_dir, "test.csv")):
        print(f"Processed dataset for {config.name} already exists. Loading from disk.")
        return data_loader(config)
    else:
        print(f"Processed dataset for {config.name} not found. Loading raw data and preprocessing.")
        load_and_preprocess_dataset(config)
        return data_loader(config)