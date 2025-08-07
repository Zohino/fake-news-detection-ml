from config import DATA_FILES, PREDICTIONS_DIR, TRAINED_MODELS_DIR
from loguru import logger
from modeling import predict
import pandas as pd


def main():
    logger.info("Loading test data...")
    test_df = pd.read_csv(DATA_FILES["test_processed"])
    X_test = test_df["cleaned_text"]

    model_path = TRAINED_MODELS_DIR / "fake_news_detector.pkl"
    results_df = predict(model_path, X_test)
    results_df["id"] = test_df["id"]

    output_path = PREDICTIONS_DIR / "test_predictions.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)

    logger.info(f"Predictions saved to {output_path}")
    logger.info(f"Class distribution:\n{results_df['label'].value_counts()}")


if __name__ == "__main__":
    main()
