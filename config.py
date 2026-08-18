"""
Shared configuration for the Wine Quality Classification project.
Centralises paths, model registry, feature definitions, and display settings.
"""

from __future__ import annotations

import os
from typing import Dict, List

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT: str = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR: str = os.path.join(PROJECT_ROOT, "model")
TEST_DATA_PATH: str = os.path.join(PROJECT_ROOT, "test_data.csv")
METADATA_PATH: str = os.path.join(MODEL_DIR, "metadata.json")

# ---------------------------------------------------------------------------
# Model registry  –  display name  →  pickle filename
# ---------------------------------------------------------------------------
MODEL_REGISTRY: Dict[str, str] = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl",
}

# ---------------------------------------------------------------------------
# Dataset / feature constants
# ---------------------------------------------------------------------------
QUALITY_TIERS: List[str] = ["Low", "Medium", "High"]

RED_WINE_URL: str = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "wine-quality/winequality-red.csv"
)
WHITE_WINE_URL: str = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "wine-quality/winequality-white.csv"
)

TEST_SIZE: float = 0.20
RANDOM_STATE: int = 42

# ---------------------------------------------------------------------------
# Evaluation metric names (preserves column order everywhere)
# ---------------------------------------------------------------------------
METRIC_NAMES: List[str] = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]

# ---------------------------------------------------------------------------
# Streamlit display
# ---------------------------------------------------------------------------
CARD_COLOURS: List[str] = [
    "#667eea", "#764ba2", "#f093fb", "#4facfe", "#43e97b", "#fa709a",
]
BAR_COLOURS: List[str] = [
    "#667eea", "#764ba2", "#f093fb", "#4facfe", "#43e97b",
]
