from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SEED = 42
RAW_DATA_PATH       = str(PROJECT_ROOT / "data" / "raw")
PROCESSED_DATA_PATH = str(PROJECT_ROOT / "data" / "processed")
CACHE_PATH          = str(PROJECT_ROOT / "data" / "cache")
CHECKPOINTS_PATH    = str(PROJECT_ROOT / "checkpoints")
RESULTS_PATH        = str(PROJECT_ROOT / "results")
LOGS_PATH           = str(PROJECT_ROOT / "logs")

@dataclass
class DatasetConfig:
    name: str
    label_map: dict
    binary: bool
    title_col: str = "title"
    text_col: str = "text"
    label_col: str = "label"

ISOT_CONFIG = DatasetConfig(
    name="isot",
    label_map={"Fake": 0, "Real": 1},
    binary=True,
)

WELFAKE_CONFIG = DatasetConfig(
    name="welfake",
    label_map={0: 0, 1: 1},
    binary=True,
)

LIAR_CONFIG = DatasetConfig(
    name="liar",
    label_map={"Fake": 0, "Real": 1},
    binary=True,
)

NELA_DL_CONFIG = DatasetConfig(
    name="nela_sampled_500k",
    label_map={0: 0, 1: 1, 2: 2},
    binary=False,
)

# Scalar hyperparameters constants
MAX_TFIDF_FEATURES = 50000
MAX_TFIDF_XGB = 10000
MAX_SEQ_LEN_DL = 512
MAX_SEQ_LEN_TRANSFORMER = 512
GLOVE_DIM = 300