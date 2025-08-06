from pathlib import Path
from dotenv import load_dotenv
from loguru import logger
import os

# Load environment variables
load_dotenv()

# Root paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJ_ROOT / "data"
MODELS_DIR = PROJ_ROOT / "models"
REPORTS_DIR = PROJ_ROOT / "reports"
LOGS_DIR = PROJ_ROOT / "logs"

# Data subdirectories
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
TRAINED_MODELS_DIR = MODELS_DIR / "trained"
PREDICTIONS_DIR = MODELS_DIR / "predictions"
FIGURES_DIR = REPORTS_DIR / "figures"

# Create directories
for path in [RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR, EXTERNAL_DATA_DIR,
             TRAINED_MODELS_DIR, PREDICTIONS_DIR, FIGURES_DIR, LOGS_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# Loguru logging configuration
try:
    from tqdm import tqdm
    logger.remove()
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ImportError:
    logger.warning("tqdm not found, falling back to default logging output")

logger.add(LOGS_DIR / "fake_news_detection.log", level="INFO", rotation="1 MB")

# URLs and file paths
DATA_URLS = {
    "fake_news_train": "https://raw.githubusercontent.com/nishitpatel01/Fake_News_Detection/master/train.csv",
    "fake_news_test": "https://raw.githubusercontent.com/nishitpatel01/Fake_News_Detection/master/test.csv"
}

DATA_FILES = {
    "train_raw": RAW_DATA_DIR / "train.csv",
    "test_raw": RAW_DATA_DIR / "test.csv",
    "train_processed": PROCESSED_DATA_DIR / "train_processed.csv",
    "test_processed": PROCESSED_DATA_DIR / "test_processed.csv",
    "validation_processed": PROCESSED_DATA_DIR / "validation_processed.csv"
}

# Model and preprocessing config
MODEL_PARAMS = {
    "tfidf": {
        "max_features": 5000,
        "min_df": 2,
        "max_df": 0.95,
        "ngram_range": (1, 2),
        "stop_words": "english",
        "sublinear_tf": True
    },
    "logistic_regression": {
        "C": 1.0,
        "max_iter": 1000,
        "random_state": 42,
        "solver": "liblinear"
    },
    "train_test_split": {
        "test_size": 0.2,
        "random_state": 42,
        "stratify": True
    }
}

TEXT_PREPROCESSING = {
    "remove_punctuation": True,
    "convert_to_lowercase": True,
    "remove_numbers": True,
    "remove_extra_whitespace": True,
    "min_text_length": 10
}

RANDOM_SEEDS = {
    "general": 42,
    "train_split": 42,
    "model_training": 42
}

EVALUATION_METRICS = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]

PLOT_SETTINGS = {
    "figure_size": (10, 6),
    "dpi": 300,
    "style": "seaborn-v0_8",
    "color_palette": "husl"
}
