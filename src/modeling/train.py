from loguru import logger
import pandas as pd

from src.config import DATA_FILES, TRAINED_MODELS_DIR
from src.modeling.modeling import train_model


def main():
    logger.info("Loading training and validation datasets...")
    train_df = pd.read_csv(DATA_FILES["train_processed"])
    val_df = pd.read_csv(DATA_FILES["test_processed"])

    X_train = train_df["cleaned_text"]
    y_train = train_df["label"]
    X_val = val_df["cleaned_text"]
    y_val = val_df["label"]

    model_path = TRAINED_MODELS_DIR / "fake_news_detector.pkl"
    _, results = train_model(X_train, y_train, X_val, y_val, save_path=model_path)

    logger.info("Training complete:")
    for k, v in results.items():
        logger.info(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
