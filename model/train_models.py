"""
Wine Quality Classifier Pipeline
=================================
Fetches UCI wine data, engineers a three-tier quality target,
trains five classifiers, evaluates each on six metrics, and
persists all artefacts for downstream Streamlit consumption.

Usage
-----
    python model/train_models.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

# --- Resolve project root so config can be imported when run directly ------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config import (  # noqa: E402
    MODEL_DIR,
    QUALITY_TIERS,
    RANDOM_STATE,
    RED_WINE_URL,
    TEST_DATA_PATH,
    TEST_SIZE,
    WHITE_WINE_URL,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Classifier definitions
# ---------------------------------------------------------------------------
CLASSIFIERS: Dict[str, Any] = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000, solver="lbfgs", random_state=RANDOM_STATE
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=10, random_state=RANDOM_STATE
    ),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, random_state=RANDOM_STATE
    ),
}


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def fetch_wine_data() -> pd.DataFrame:
    """Download red and white wine CSVs from UCI and return a merged frame."""
    logger.info("Downloading red wine data from UCI...")
    red = pd.read_csv(RED_WINE_URL, sep=";")
    red["wine_type"] = 0

    logger.info("Downloading white wine data from UCI...")
    white = pd.read_csv(WHITE_WINE_URL, sep=";")
    white["wine_type"] = 1

    merged = pd.concat([red, white], ignore_index=True)
    logger.info("Merged dataset: %d rows, %d columns", *merged.shape)
    return merged


def _quality_to_tier(score: int) -> str:
    """Map a raw quality score (3-9) to Low / Medium / High."""
    if score <= 5:
        return "Low"
    if score == 6:
        return "Medium"
    return "High"


def prepare_features(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, List[str], LabelEncoder]:
    """
    Create feature matrix and encoded target vector.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
    y : ndarray of shape (n_samples,)
    feature_cols : list of column names used as predictors
    encoder : fitted LabelEncoder for the quality tiers
    """
    df = df.copy()
    df["tier"] = df["quality"].apply(_quality_to_tier)

    encoder = LabelEncoder()
    encoder.fit(QUALITY_TIERS)
    df["tier_code"] = encoder.transform(df["tier"])

    feature_cols = [c for c in df.columns if c not in ("quality", "tier", "tier_code")]
    X = df[feature_cols].values
    y = df["tier_code"].values

    logger.info("Features (%d): %s", len(feature_cols), feature_cols)
    logger.info("Tier distribution:\n%s", df["tier"].value_counts().to_string())
    return X, y, feature_cols, encoder


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> Dict[str, float]:
    """Return a dict of six standard classification metrics."""
    return {
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "AUC": round(
            roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted"), 4
        ),
        "Precision": round(
            precision_score(y_true, y_pred, average="weighted", zero_division=0), 4
        ),
        "Recall": round(
            recall_score(y_true, y_pred, average="weighted", zero_division=0), 4
        ),
        "F1": round(
            f1_score(y_true, y_pred, average="weighted", zero_division=0), 4
        ),
        "MCC": round(matthews_corrcoef(y_true, y_pred), 4),
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline() -> Dict[str, Dict[str, float]]:
    """Execute the full train-evaluate-persist pipeline."""

    # 1. Data ingestion & feature engineering
    wine_df = fetch_wine_data()
    X, y, feature_cols, encoder = prepare_features(wine_df)

    # 2. Stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    logger.info("Split: %d train / %d test", X_train.shape[0], X_test.shape[0])

    # 3. Feature scaling
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # 4. Export holdout set for the Streamlit dashboard
    holdout = pd.DataFrame(X_test, columns=feature_cols)
    holdout["quality_label"] = encoder.inverse_transform(y_test)
    holdout.to_csv(TEST_DATA_PATH, index=False)
    logger.info("Holdout CSV saved: %s  (%d rows)", TEST_DATA_PATH, len(holdout))

    # 5. Train, evaluate, and persist each classifier
    os.makedirs(MODEL_DIR, exist_ok=True)
    results: Dict[str, Dict[str, float]] = {}

    for name, clf in CLASSIFIERS.items():
        logger.info("--- Training: %s ---", name)
        clf.fit(X_train_s, y_train)

        y_pred = clf.predict(X_test_s)
        y_proba = clf.predict_proba(X_test_s)

        metrics = compute_metrics(y_test, y_pred, y_proba)
        results[name] = metrics

        # Console report
        for k, v in metrics.items():
            logger.info("  %-10s : %.4f", k, v)

        cm = confusion_matrix(y_test, y_pred)
        logger.info("  Confusion matrix:\n%s", cm)
        logger.info(
            "  Classification report:\n%s",
            classification_report(
                y_test, y_pred, target_names=list(encoder.classes_), zero_division=0
            ),
        )

        # Persist model
        safe_name = name.lower().replace(" ", "_")
        pkl_path = os.path.join(MODEL_DIR, f"{safe_name}.pkl")
        joblib.dump(clf, pkl_path)
        logger.info("  Saved: %s", pkl_path)

    # 6. Save preprocessing artefacts
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(encoder, os.path.join(MODEL_DIR, "label_encoder.pkl"))

    # 7. Write run metadata
    metadata = {
        "feature_columns": feature_cols,
        "class_names": list(encoder.classes_),
        "results": results,
    }
    metadata_path = os.path.join(MODEL_DIR, "metadata.json")
    with open(metadata_path, "w") as fp:
        json.dump(metadata, fp, indent=2)
    logger.info("Metadata saved: %s", metadata_path)

    # 8. Summary
    logger.info("\n=== RESULTS SUMMARY ===")
    header = f"{'Classifier':<25} {'Acc':>8} {'AUC':>8} {'Prec':>8} {'Rec':>8} {'F1':>8} {'MCC':>8}"
    logger.info(header)
    logger.info("-" * len(header))
    for name, m in results.items():
        logger.info(
            "%-25s %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f",
            name, m["Accuracy"], m["AUC"], m["Precision"],
            m["Recall"], m["F1"], m["MCC"],
        )

    best = max(results.items(), key=lambda kv: kv[1]["F1"])
    logger.info("Top classifier (F1): %s  (%.4f)", best[0], best[1]["F1"])

    return results


if __name__ == "__main__":
    run_pipeline()
