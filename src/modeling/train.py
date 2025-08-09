"""
Enhanced training script
"""

from loguru import logger
import pandas as pd

from src.config import DATA_FILES, TRAINED_MODELS_DIR, MODEL_PARAMS
from src.modeling.modeling import train_model
from src.modeling.optuna_optimization import find_best_params, get_optimized_config


def check_and_create_enhanced_data():
    """Check if enhanced preprocessing was applied, if not run basic enhancement."""

    if not DATA_FILES["train_processed"].exists():
        logger.warning("No processed data found. Please run preprocessing first.")
        logger.info("Run: make preprocess")
        return False

    # Load and check if enhanced features exist
    train_df = pd.read_csv(DATA_FILES["train_processed"])

    enhanced_features = ["text_length", "word_count", "exclamation_count"]
    has_enhanced_features = all(col in train_df.columns for col in enhanced_features)

    if not has_enhanced_features:
        logger.info("Basic processed data found, but no enhanced features.")
        logger.info("Running quick feature enhancement...")

        # Quick feature enhancement
        if "cleaned_text" not in train_df.columns and "statement" in train_df.columns:
            # Basic text cleaning
            import re

            def quick_clean(text):
                if pd.isna(text):
                    return ""
                text = str(text).lower()
                text = re.sub(r"[^\w\s]", "", text)
                text = re.sub(r"\d+", "", text)
                text = re.sub(r"\s+", " ", text).strip()
                return text

            train_df["cleaned_text"] = train_df["statement"].apply(quick_clean)

        # Add basic features if missing
        if "text_length" not in train_df.columns:
            train_df["text_length"] = train_df["statement"].astype(str).str.len()
            train_df["word_count"] = train_df["statement"].astype(str).str.split().str.len()
            train_df["exclamation_count"] = train_df["statement"].astype(str).str.count("!")
            train_df["question_count"] = train_df["statement"].astype(str).str.count(r"\?")
            train_df["caps_ratio"] = (
                train_df["statement"]
                .astype(str)
                .apply(lambda x: sum(1 for c in str(x) if c.isupper()) / (len(str(x)) + 1))
            )

        # Save enhanced version
        train_df.to_csv(DATA_FILES["train_processed"], index=False)
        logger.info("✅ Quick feature enhancement complete")

    return True


def main():
    """Enhanced training pipeline with better performance."""

    # Check and prepare data
    if not check_and_create_enhanced_data():
        return

    logger.info("Loading training and validation datasets...")

    # Load processed datasets
    train_df = pd.read_csv(DATA_FILES["train_processed"])

    # Check if we have separate validation set
    if DATA_FILES["validation_processed"].exists():
        val_df = pd.read_csv(DATA_FILES["validation_processed"])
        logger.info("Using separate validation set")
    else:
        logger.info("Using test set as validation (not ideal but compatible)")
        val_df = pd.read_csv(DATA_FILES["test_processed"])

    # Prepare training data
    X_train = train_df.drop("Label", axis=1) if "Label" in train_df.columns else train_df
    y_train = train_df["Label"] if "Label" in train_df.columns else None

    if y_train is None:
        logger.error("No 'Label' column found in training data!")
        return

    # Prepare validation data
    X_val = val_df.drop("Label", axis=1) if "Label" in val_df.columns else val_df
    y_val = val_df["Label"] if "Label" in val_df.columns else None

    # Log data info
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")

    if y_train is not None:
        train_dist = y_train.value_counts().to_dict()
        logger.info(f"Training class distribution: {train_dist}")

    if y_val is not None:
        val_dist = y_val.value_counts().to_dict()
        logger.info(f"Validation class distribution: {val_dist}")

    # Optuna optimization
    logger.info("Running hyperparameter optimization...")
    best_params = find_best_params()
    optimized_config = get_optimized_config(best_params)
    MODEL_PARAMS.update(optimized_config)

    # Train with optimized parameters
    logger.info("Training with optimized parameters...")
    model_path = TRAINED_MODELS_DIR / "fake_news_detector.pkl"
    model, results = train_model(X_train, y_train, X_val, y_val, save_path=model_path)

    if model is None:
        logger.error("Training failed!")
        return

    # Enhanced results reporting
    logger.info("\n" + "=" * 60)
    logger.info("🎯 ENHANCED TRAINING RESULTS")
    logger.info("=" * 60)

    for metric, value in results.items():
        if isinstance(value, float):
            logger.info(f"{metric:20} : {value:.4f}")

    # Performance analysis
    logger.info("\n" + "=" * 60)
    logger.info("📊 PERFORMANCE ANALYSIS")
    logger.info("=" * 60)

    # Check for overfitting
    if "train_f1" in results and "val_f1" in results:
        train_val_gap = results["train_f1"] - results["val_f1"]
        logger.info(f"Train-Val F1 gap: {train_val_gap:.4f}")

        if train_val_gap > 0.15:
            logger.info("⚠️  High overfitting detected!")
            logger.info("   Recommendations:")
            logger.info("   - Increase regularization (lower C parameter)")
            logger.info("   - Reduce max_features in TF-IDF")
            logger.info("   - Add more training data")
        elif train_val_gap > 0.08:
            logger.info("⚠️  Moderate overfitting")
            logger.info("   Consider tuning regularization parameters")
        else:
            logger.info("✅ Good generalization!")

    # Performance assessment
    val_f1 = results.get("val_f1", 0)
    if val_f1 > 0.80:
        logger.info("🎉 Excellent performance!")
    elif val_f1 > 0.75:
        logger.info("✅ Good performance!")
    elif val_f1 > 0.70:
        logger.info("🔄 Decent performance - room for improvement")
    else:
        logger.info("⚠️  Performance needs improvement")
        logger.info("   Suggestions:")
        logger.info("   - Check data quality")
        logger.info("   - Try ensemble methods")
        logger.info("   - Feature engineering")
        logger.info("   - Hyperparameter tuning")

    # Compare with baseline if available
    baseline_f1 = 0.6778  # Your original result
    if val_f1 > 0:
        improvement = val_f1 - baseline_f1
        improvement_pct = (improvement / baseline_f1) * 100

        logger.info("\n📈 IMPROVEMENT OVER BASELINE:")
        logger.info(f"   Baseline F1: {baseline_f1:.4f}")
        logger.info(f"   Current F1:  {val_f1:.4f}")
        logger.info(f"   Improvement: +{improvement:.4f} ({improvement_pct:+.1f}%)")

        if improvement > 0.05:
            logger.info("🎯 Significant improvement achieved!")
        elif improvement > 0.01:
            logger.info("📈 Moderate improvement")
        else:
            logger.info("🔄 Limited improvement - consider other approaches")

    # Cross-validation results
    if "cv_f1_mean" in results and results["cv_f1_mean"] > 0:
        cv_mean = results["cv_f1_mean"]
        cv_std = results.get("cv_f1_std", 0)
        logger.info(f"\n🔄 Cross-validation F1: {cv_mean:.4f} ± {cv_std:.4f}")

        if cv_std < 0.02:
            logger.info("✅ Stable model performance across folds")
        else:
            logger.info("⚠️  High variance across folds - consider more data")

    logger.info("\n🚀 Training complete! Model saved to:")
    logger.info(f"   {model_path}")
    logger.info("\n📋 Next steps:")
    logger.info("   1. Run predictions: make predict")
    logger.info("   2. Generate report: make publish")


if __name__ == "__main__":
    main()
