"""
Preprocessing for fake news detection
"""

import re

from loguru import logger
import pandas as pd
from sklearn.model_selection import train_test_split

# Import existing config
from src.config import DATA_FILES, MODEL_PARAMS, PROCESSED_DATA_DIR, TEXT_PREPROCESSING


class EnhancedTextProcessor:
    """Enhanced text preprocessing with better cleaning and feature engineering."""

    def __init__(self):
        self.min_text_length = TEXT_PREPROCESSING.get("min_text_length", 20)

    def clean_text_advanced(self, text: str) -> str:
        """Advanced text cleaning pipeline."""
        if pd.isna(text) or not isinstance(text, str):
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)

        # Remove email addresses
        text = re.sub(r"\S+@\S+", "", text)

        # Remove HTML tags
        text = re.sub(r"<[^<]+?>", "", text)

        # Remove special characters but keep basic punctuation for feature extraction
        text = re.sub(r"[^\w\s!?.,;:]", " ", text)

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def create_text_features(self, df: pd.DataFrame, text_col: str = "Statement") -> pd.DataFrame:
        """Create additional text-based features."""
        df_enhanced = df.copy()

        # Basic length features
        df_enhanced["text_length"] = df[text_col].astype(str).str.len()
        df_enhanced["word_count"] = df[text_col].astype(str).str.split().str.len()
        df_enhanced["avg_word_length"] = df_enhanced["text_length"] / (
            df_enhanced["word_count"] + 1
        )

        # Punctuation features
        df_enhanced["exclamation_count"] = df[text_col].astype(str).str.count(re.escape("!"))
        df_enhanced["question_count"] = df[text_col].astype(str).str.count(re.escape("?"))
        df_enhanced["period_count"] = df[text_col].astype(str).str.count(re.escape("."))

        # Capitalization features
        df_enhanced["caps_count"] = (
            df[text_col].astype(str).apply(lambda x: sum(1 for c in str(x) if c.isupper()))
        )
        df_enhanced["caps_ratio"] = df_enhanced["caps_count"] / (df_enhanced["text_length"] + 1)

        # Sentence-level features
        df_enhanced["sentence_count"] = df[text_col].astype(str).str.count(r"[.!?]+")
        df_enhanced["avg_sentence_length"] = df_enhanced["word_count"] / (
            df_enhanced["sentence_count"] + 1
        )

        # Readability approximations
        df_enhanced["digit_count"] = df[text_col].astype(str).str.count(r"\d")
        df_enhanced["digit_ratio"] = df_enhanced["digit_count"] / (df_enhanced["text_length"] + 1)

        return df_enhanced

    def clean_for_modeling(self, text: str) -> str:
        """Final cleaning for model input - removes most punctuation."""
        if pd.isna(text) or not isinstance(text, str):
            return ""

        # Remove punctuation and numbers for TF-IDF
        text = re.sub(r"[^\w\s]", "", text)
        text = re.sub(r"\d+", "", text)

        # Remove extra whitespace and short words
        words = text.split()
        words = [word for word in words if len(word) >= 3]

        return " ".join(words)


def create_processed_datasets(force_recreate: bool = False) -> None:
    """Create enhanced processed datasets with better preprocessing."""

    # Check if already processed
    if (
        DATA_FILES["train_processed"].exists()
        and DATA_FILES["test_processed"].exists()
        and not force_recreate
    ):
        logger.warning("Processed files already exist. Use force_recreate=True to regenerate.")
        return

    processor = EnhancedTextProcessor()

    # Load raw data
    logger.info("Loading raw datasets...")
    train_df = pd.read_csv(DATA_FILES["train_raw"])
    test_df = pd.read_csv(DATA_FILES["test_raw"])

    print(f"Raw train shape: {train_df.shape}")
    print(f"Raw test shape: {test_df.shape}")

    # Process training data
    logger.info("Processing training data...")

    # Remove duplicates
    initial_train_size = len(train_df)
    train_df = train_df.drop_duplicates(subset=["Statement"])
    logger.info(f"Removed {initial_train_size - len(train_df)} duplicate statements")

    # Create enhanced features first (before cleaning text)
    train_df = processor.create_text_features(train_df)

    # Clean text for analysis
    train_df["statement_cleaned"] = train_df["Statement"].apply(processor.clean_text_advanced)

    # Filter out very short texts
    min_length = processor.min_text_length
    initial_size = len(train_df)
    train_df = train_df[train_df["statement_cleaned"].str.len() >= min_length]
    logger.info(
        f"Removed {initial_size - len(train_df)} texts shorter than {min_length} characters"
    )

    # Create final cleaned text for modeling
    train_df["cleaned_text"] = train_df["statement_cleaned"].apply(processor.clean_for_modeling)

    # Remove empty texts after final cleaning
    train_df = train_df[train_df["cleaned_text"].str.len() > 0]

    # Process test data similarly
    logger.info("Processing test data...")
    test_df = processor.create_text_features(test_df)
    test_df["statement_cleaned"] = test_df["Statement"].apply(processor.clean_text_advanced)
    test_df = test_df[test_df["statement_cleaned"].str.len() >= min_length]
    test_df["cleaned_text"] = test_df["statement_cleaned"].apply(processor.clean_for_modeling)
    test_df = test_df[test_df["cleaned_text"].str.len() > 0]

    # Split training data into train/validation
    logger.info("Creating train/validation split...")

    # Features for modeling
    feature_columns = [
        "cleaned_text",
        "text_length",
        "word_count",
        "avg_word_length",
        "exclamation_count",
        "question_count",
        "period_count",
        "caps_count",
        "caps_ratio",
        "sentence_count",
        "avg_sentence_length",
        "digit_count",
        "digit_ratio",
    ]

    X = train_df[feature_columns + ["Label"]]
    y = train_df["Label"]

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=MODEL_PARAMS["train_test_split"]["test_size"],
        random_state=MODEL_PARAMS["train_test_split"]["random_state"],
        stratify=y,
    )

    # Save processed datasets
    logger.info("Saving processed datasets...")

    # Ensure directory exists
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Save train/validation splits
    X_train.to_csv(DATA_FILES["train_processed"], index=False)
    X_val.to_csv(DATA_FILES["validation_processed"], index=False)

    # Save full test set
    test_features = test_df[feature_columns].copy()
    if "Label" in test_df.columns:
        test_features["label"] = test_df["Label"]

    test_features.to_csv(DATA_FILES["test_processed"], index=False)

    # Log final statistics
    print(f"Final train set: {len(X_train)} samples")
    print(f"Final validation set: {len(X_val)} samples")
    print(f"Final test set: {len(test_features)} samples")

    if "Label" in X_train.columns:
        print(f"Validation class distribution: {X_val['Label'].value_counts().to_dict()}")
        print(f"Train class distribution: {X_train['Label'].value_counts().to_dict()}")


if __name__ == "__main__":
    # Run preprocessing
    create_processed_datasets(force_recreate=True)
    logger.success("Enhanced preprocessing complete!")
