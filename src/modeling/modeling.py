from pathlib import Path

from joblib import dump, load
from loguru import logger
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline

from src.config import MODEL_PARAMS


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(**MODEL_PARAMS.get("tfidf", {}))),
            ("classifier", LogisticRegression(**MODEL_PARAMS.get("logistic_regression", {}))),
        ]
    )


def train_model(X_train, y_train, X_val=None, y_val=None, save_path: Path = None):
    model = build_pipeline()
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_train_prob = model.predict_proba(X_train)[:, 1]

    results = {
        "train_accuracy": accuracy_score(y_train, y_train_pred),
        "train_precision": precision_score(y_train, y_train_pred),
        "train_recall": recall_score(y_train, y_train_pred),
        "train_f1": f1_score(y_train, y_train_pred),
        "train_auc": roc_auc_score(y_train, y_train_prob),
    }

    if X_val is not None and y_val is not None:
        y_val_pred = model.predict(X_val)
        y_val_prob = model.predict_proba(X_val)[:, 1]
        results.update(
            {
                "val_accuracy": accuracy_score(y_val, y_val_pred),
                "val_precision": precision_score(y_val, y_val_pred),
                "val_recall": recall_score(y_val, y_val_pred),
                "val_f1": f1_score(y_val, y_val_pred),
                "val_auc": roc_auc_score(y_val, y_val_prob),
            }
        )

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        dump(model, save_path)
        logger.info(f"Model saved to {save_path}")

    return model, results


def predict(model_path: Path, X_test: pd.Series) -> pd.DataFrame:
    model = load(model_path)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    return pd.DataFrame({"label": predictions, "probability_unreliable": probabilities})
