from pathlib import Path

from joblib import dump, load
from loguru import logger
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import MODEL_PARAMS


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(**MODEL_PARAMS.get("tfidf", {}))),
            ("classifier", LogisticRegression(**MODEL_PARAMS.get("logistic_regression", {}))),
        ]
    )


def build_hybrid_pipeline() -> Pipeline:
    """Build pipeline that uses both text and numerical features."""

    # Text preprocessing
    text_pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=8000,
                    min_df=3,
                    max_df=0.85,
                    ngram_range=(1, 2),
                    stop_words="english",
                    sublinear_tf=True,
                    token_pattern=r"\b[a-zA-Z]{3,}\b",
                ),
            )
        ]
    )

    # Numerical features
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

    numerical_pipeline = Pipeline([("scaler", StandardScaler())])

    # Combine features
    preprocessor = ColumnTransformer(
        [("text", text_pipeline, "cleaned_text"), ("num", numerical_pipeline, numerical_features)],
        remainder="drop",
    )

    # Final classifier
    classifier = LogisticRegression(
        C=1.5, max_iter=2000, random_state=42, solver="liblinear", class_weight="balanced"
    )

    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])


def build_ensemble_pipeline() -> VotingClassifier:
    """Build ensemble of different models."""

    # Enhanced TF-IDF for ensemble
    enhanced_tfidf = {
        "max_features": 12000,
        "min_df": 2,
        "max_df": 0.9,
        "ngram_range": (1, 2),
        "stop_words": "english",
        "sublinear_tf": True,
    }

    # Logistic Regression
    lr_pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(**enhanced_tfidf)),
            (
                "classifier",
                LogisticRegression(
                    C=1.0,
                    max_iter=2000,
                    random_state=42,
                    solver="liblinear",
                    class_weight="balanced",
                ),
            ),
        ]
    )

    # Random Forest
    rf_pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(**enhanced_tfidf)),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=100,
                    max_depth=20,
                    min_samples_split=5,
                    random_state=42,
                    class_weight="balanced",
                    n_jobs=-1,
                ),
            ),
        ]
    )

    return VotingClassifier(estimators=[("lr", lr_pipeline), ("rf", rf_pipeline)], voting="soft")


def train_model(X_train, y_train, X_val=None, y_val=None, save_path: Path = None):
    """Enhanced training with model selection and better evaluation."""

    # Determine if we have additional features for hybrid model
    use_hybrid = isinstance(X_train, pd.DataFrame) and "text_length" in X_train.columns

    models_to_try = {}

    if use_hybrid:
        logger.info("Using hybrid model with text and numerical features")
        models_to_try["hybrid"] = build_hybrid_pipeline()
        models_to_try["text_only"] = build_pipeline()
        models_to_try["ensemble"] = build_ensemble_pipeline()
    else:
        logger.info("Using text-only models")
        models_to_try["enhanced_lr"] = build_pipeline()
        models_to_try["ensemble"] = build_ensemble_pipeline()

    best_model = None
    best_score = 0
    all_results = {}

    for name, model in models_to_try.items():
        logger.info(f"Training {name}...")

        try:
            # Prepare data based on model type
            if name == "hybrid":
                X_train_model = X_train
                X_val_model = X_val if X_val is not None else None
            else:
                # Use only text for non-hybrid models
                X_train_model = X_train["cleaned_text"] if use_hybrid else X_train
                X_val_model = (
                    X_val["cleaned_text"] if (X_val is not None and use_hybrid) else X_val
                )

            # Fit the model
            model.fit(X_train_model, y_train)

            # Training predictions
            y_train_pred = model.predict(X_train_model)
            y_train_prob = model.predict_proba(X_train_model)[:, 1]

            results = {
                "train_accuracy": accuracy_score(y_train, y_train_pred),
                "train_precision": precision_score(y_train, y_train_pred, zero_division=0),
                "train_recall": recall_score(y_train, y_train_pred, zero_division=0),
                "train_f1": f1_score(y_train, y_train_pred, zero_division=0),
                "train_auc": roc_auc_score(y_train, y_train_prob),
            }

            # Validation predictions if available
            if X_val_model is not None and y_val is not None:
                y_val_pred = model.predict(X_val_model)
                y_val_prob = model.predict_proba(X_val_model)[:, 1]
                results.update(
                    {
                        "val_accuracy": accuracy_score(y_val, y_val_pred),
                        "val_precision": precision_score(y_val, y_val_pred, zero_division=0),
                        "val_recall": recall_score(y_val, y_val_pred, zero_division=0),
                        "val_f1": f1_score(y_val, y_val_pred, zero_division=0),
                        "val_auc": roc_auc_score(y_val, y_val_prob),
                    }
                )

                current_score = results["val_f1"]
            else:
                current_score = results["train_f1"]

            # Cross-validation for more robust evaluation
            try:
                cv_scores = cross_val_score(
                    model,
                    X_train_model,
                    y_train,
                    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                    scoring="f1",
                    n_jobs=-1,
                )
                results["cv_f1_mean"] = cv_scores.mean()
                results["cv_f1_std"] = cv_scores.std()
                logger.info(f"{name} CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            except Exception as e:
                logger.warning(f"CV failed for {name}: {e}")
                results["cv_f1_mean"] = 0.0
                results["cv_f1_std"] = 0.0

            all_results[name] = results

            # Track best model
            if current_score > best_score:
                best_score = current_score
                best_model = model
                best_name = name

            logger.info(f"{name} - Val F1: {results.get('val_f1', 'N/A'):.4f}")

        except Exception as e:
            logger.error(f"Error training {name}: {e}")
            continue

    if best_model is None:
        logger.error("No models trained successfully!")
        return None, {}

    logger.info(f"Best model: {best_name} with score: {best_score:.4f}")

    # Save best model
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        dump(best_model, save_path)
        logger.info(f"Best model saved to {save_path}")

    # Return results from best model
    return best_model, all_results[best_name]


def predict(model_path: Path, X_test) -> pd.DataFrame:
    """Enhanced prediction with confidence scores."""

    model = load(model_path)

    # Handle different input types
    try:
        # Try full dataframe first (for hybrid models)
        if isinstance(X_test, pd.DataFrame):
            predictions = model.predict(X_test)
            probabilities = model.predict_proba(X_test)[:, 1]
        else:
            predictions = model.predict(X_test)
            probabilities = model.predict_proba(X_test)[:, 1]
    except Exception:
        # Fall back to text-only prediction
        if isinstance(X_test, pd.DataFrame) and "cleaned_text" in X_test.columns:
            predictions = model.predict(X_test["cleaned_text"])
            probabilities = model.predict_proba(X_test["cleaned_text"])[:, 1]
        else:
            raise ValueError("Unable to make predictions with the given input format")

    # Calculate confidence (distance from decision boundary)
    confidence = np.abs(probabilities - 0.5) * 2

    return pd.DataFrame(
        {"label": predictions, "probability_unreliable": probabilities, "confidence": confidence}
    )
