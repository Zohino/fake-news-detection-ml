"""
Hyperparameter optimization for fake news detection.
"""

from loguru import logger
import optuna
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import DATA_FILES


def objective(trial):
    """Find best hyperparameters for your overfitting problem."""

    # Load data
    train_df = pd.read_csv(DATA_FILES["train_processed"])
    val_df = pd.read_csv(DATA_FILES["validation_processed"])

    X_train = train_df.drop("Label", axis=1)
    y_train = train_df["Label"]
    X_val = val_df.drop("Label", axis=1)
    y_val = val_df["Label"]

    # Focus on reducing overfitting with stronger regularization
    tfidf_params = {
        "max_features": trial.suggest_int("max_features", 4000, 10000, step=1000),
        "min_df": trial.suggest_int("min_df", 3, 8),
        "max_df": trial.suggest_float("max_df", 0.75, 0.90, step=0.05),
        "ngram_range": (1, trial.suggest_int("ngram_max", 1, 2)),
        "stop_words": "english",
        "sublinear_tf": True,
        "token_pattern": r"\b[a-zA-Z]{3,}\b",
    }

    # Strong regularization to fix overfitting
    lr_params = {
        "C": trial.suggest_float("C", 0.1, 1.0, log=True),  # Lower C = stronger regularization
        "max_iter": 2000,
        "solver": "liblinear",
        "class_weight": "balanced",
        "random_state": 42,
    }

    # Build pipeline
    text_pipeline = Pipeline([("tfidf", TfidfVectorizer(**tfidf_params))])

    numerical_features = [
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

    preprocessor = ColumnTransformer(
        [
            ("text", text_pipeline, "cleaned_text"),
            ("num", Pipeline([("scaler", StandardScaler())]), numerical_features),
        ],
        remainder="drop",
    )

    pipeline = Pipeline(
        [("preprocessor", preprocessor), ("classifier", LogisticRegression(**lr_params))]
    )

    # Evaluate with focus on generalization
    pipeline.fit(X_train, y_train)
    y_val_pred = pipeline.predict(X_val)
    val_f1 = f1_score(y_val, y_val_pred)

    return val_f1


def find_best_params():
    """Run optimization once and return best parameters."""

    logger.info("Finding optimal hyperparameters to fix overfitting...")

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))

    # 30 trials should be enough for good parameters
    study.optimize(objective, n_trials=30, show_progress_bar=True)

    logger.info(f"Best validation F1: {study.best_value:.4f}")
    logger.info(f"Best parameters: {study.best_params}")

    return study.best_params


def get_optimized_config(best_params):
    """Convert best params to config format."""

    return {
        "tfidf": {
            "max_features": best_params["max_features"],
            "min_df": best_params["min_df"],
            "max_df": best_params["max_df"],
            "ngram_range": (1, best_params["ngram_max"]),
            "stop_words": "english",
            "sublinear_tf": True,
            "analyzer": "word",
            "token_pattern": r"\b[a-zA-Z]{3,}\b",
        },
        "logistic_regression": {
            "C": best_params["C"],
            "max_iter": 2000,
            "random_state": 42,
            "solver": "liblinear",
            "class_weight": "balanced",
        },
    }


if __name__ == "__main__":
    # Run optimization once
    best_params = find_best_params()

    # Print config to copy into config.py
    optimized_config = get_optimized_config(best_params)

    print("\n" + "=" * 60)
    print("COPY THIS INTO YOUR config.py MODEL_PARAMS:")
    print("=" * 60)
    print("MODEL_PARAMS = {")
    print(f'    "tfidf": {optimized_config["tfidf"]},')
    print(f'    "logistic_regression": {optimized_config["logistic_regression"]},')
    print("    # ... keep other existing parameters")
    print("}")
    print("=" * 60)
