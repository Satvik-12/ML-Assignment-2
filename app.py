"""
Streamlit Dashboard - Wine Quality Classifier Evaluation
=========================================================
Interactive interface for evaluating and comparing five ML
classifiers trained on physicochemical wine data.

Usage
-----
    streamlit run app.py
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
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
from sklearn.preprocessing import LabelEncoder, StandardScaler

from config import (
    BAR_COLOURS,
    CARD_COLOURS,
    METADATA_PATH,
    METRIC_NAMES,
    MODEL_DIR,
    MODEL_REGISTRY,
    TEST_DATA_PATH,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Wine Quality Predictor",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .app-title  { font-size:2.5rem; font-weight:700; color:#722F37;
                       text-align:center; margin-bottom:.5rem; }
        .app-subtitle { font-size:1.1rem; color:#666; text-align:center;
                         margin-bottom:2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================================
# Helper functions
# =========================================================================

@st.cache_resource(show_spinner="Loading models...")
def load_assets() -> tuple[
    Dict[str, Any], StandardScaler, LabelEncoder, Dict[str, Any]
]:
    """Deserialise classifiers, scaler, encoder, and run metadata."""
    classifiers: Dict[str, Any] = {}
    for display_name, pkl_file in MODEL_REGISTRY.items():
        path = os.path.join(MODEL_DIR, pkl_file)
        if os.path.exists(path):
            classifiers[display_name] = joblib.load(path)
        else:
            logger.warning("Model file not found: %s", path)

    scaler: StandardScaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    encoder: LabelEncoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))

    with open(METADATA_PATH, "r") as fp:
        metadata = json.load(fp)

    return classifiers, scaler, encoder, metadata


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> Dict[str, float]:
    """Compute six standard classification metrics."""
    return {
        "Accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "AUC":       round(roc_auc_score(y_true, y_proba, multi_class="ovr",
                                          average="weighted"), 4),
        "Precision": round(precision_score(y_true, y_pred, average="weighted",
                                            zero_division=0), 4),
        "Recall":    round(recall_score(y_true, y_pred, average="weighted",
                                         zero_division=0), 4),
        "F1":        round(f1_score(y_true, y_pred, average="weighted",
                                     zero_division=0), 4),
        "MCC":       round(matthews_corrcoef(y_true, y_pred), 4),
    }


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str],
    title: str,
) -> plt.Figure:
    """Return a Seaborn heatmap figure for the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm,
        annot=True, fmt="d", cmap="RdPu",
        xticklabels=labels, yticklabels=labels,
        linewidths=1, linecolor="white",
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12, fontweight="bold")
    ax.set_ylabel("Actual",    fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    return fig


def render_metric_cards(metrics: Dict[str, float]) -> None:
    """Display six metric values as coloured cards in a single row."""
    cols = st.columns(len(metrics))
    for idx, (name, value) in enumerate(metrics.items()):
        with cols[idx]:
            st.markdown(
                f"""
                <div style="background:{CARD_COLOURS[idx]};padding:1rem;
                            border-radius:10px;color:white;text-align:center;">
                    <div style="font-size:1.6rem;font-weight:700;">{value:.4f}</div>
                    <div style="font-size:.85rem;opacity:.9;">{name}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def load_test_data(upload: Optional[Any]) -> Optional[pd.DataFrame]:
    """Return a DataFrame from either an uploaded CSV or the bundled holdout set."""
    if upload is not None:
        df = pd.read_csv(upload)
        st.success(f"Loaded **{upload.name}** ({df.shape[0]} rows, {df.shape[1]} cols)")
        return df

    if os.path.exists(TEST_DATA_PATH):
        df = pd.read_csv(TEST_DATA_PATH)
        st.info(
            "Using the bundled test set. "
            "Upload your own CSV from the sidebar to try different data."
        )
        return df

    return None


# =========================================================================
# Load assets
# =========================================================================
try:
    classifiers, scaler, encoder, metadata = load_assets()
    feature_cols: List[str] = metadata["feature_columns"]
    class_names:  List[str] = metadata["class_names"]
    saved_results: Dict     = metadata["results"]
    app_ready = True
except Exception as exc:
    app_ready = False
    st.error(f"Failed to load model assets: {exc}")

# =========================================================================
# Header
# =========================================================================
st.markdown('<div class="app-title">🍷 Wine Quality Predictor</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">'
    "Evaluate and compare five classifiers on the UCI Wine Quality dataset"
    "</div>",
    unsafe_allow_html=True,
)

if not app_ready:
    st.stop()

# =========================================================================
# Sidebar
# =========================================================================
with st.sidebar:
    st.header("Settings")

    st.subheader("Upload Data")
    csv_upload = st.file_uploader(
        "Provide a CSV file with test samples",
        type=["csv"],
        help="CSV should have the same feature columns as the training data, "
             "plus an optional 'quality_label' column.",
    )

    st.divider()

    st.subheader("Pick a Classifier")
    chosen_model = st.selectbox(
        "Which model to evaluate?",
        options=list(classifiers.keys()),
        index=len(classifiers) - 1,  # default to Random Forest
    )

    st.divider()

    st.subheader("Dataset Info")
    st.markdown("""
    - **Origin**: UCI ML Repository
    - **Records**: 6,497 wine samples
    - **Predictors**: 12 physicochemical properties
    - **Target**: Low / Medium / High quality
    """)

    st.divider()
    st.caption("ML Assignment 2 - BITS Pilani M.Tech (AIML)")

# =========================================================================
# Main content
# =========================================================================
test_df = load_test_data(csv_upload)

if test_df is None:
    st.warning("Please upload a CSV file from the sidebar to begin.")
    st.stop()

# Validate columns
missing = [c for c in feature_cols if c not in test_df.columns]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

has_labels = "quality_label" in test_df.columns

# Prepare feature matrix
X_raw = test_df[feature_cols].values
X_scaled = scaler.transform(X_raw)
y_true = encoder.transform(test_df["quality_label"]) if has_labels else None

# --- Tabs ------------------------------------------------------------------
tab_single, tab_compare, tab_data, tab_about = st.tabs(
    ["Single Model", "Compare All", "Data Preview", "About"]
)

# ── Tab 1: Single model evaluation ────────────────────────────────────────
with tab_single:
    st.subheader(f"Results for {chosen_model}")

    clf = classifiers[chosen_model]
    preds  = clf.predict(X_scaled)
    probas = clf.predict_proba(X_scaled)

    if y_true is not None:
        metrics = evaluate(y_true, preds, probas)
        render_metric_cards(metrics)

        st.markdown("---")
        col_cm, col_report = st.columns(2)

        with col_cm:
            st.subheader("Confusion Matrix")
            fig = plot_confusion_matrix(
                y_true, preds, class_names, f"Confusion Matrix - {chosen_model}"
            )
            st.pyplot(fig)
            plt.close(fig)

        with col_report:
            st.subheader("Per-Class Report")
            report = classification_report(
                y_true, preds, target_names=class_names,
                output_dict=True, zero_division=0,
            )
            st.dataframe(
                pd.DataFrame(report).T.style.format("{:.4f}"),
                width="stretch",
            )
    else:
        st.warning("No 'quality_label' column - showing predictions only.")
        out = test_df.copy()
        out["Predicted Quality"] = encoder.inverse_transform(preds)
        st.dataframe(out, width="stretch")

# ── Tab 2: Side-by-side comparison ────────────────────────────────────────
with tab_compare:
    st.subheader("Head-to-Head Comparison")

    if y_true is not None:
        rows = []
        for name, clf_obj in classifiers.items():
            p  = clf_obj.predict(X_scaled)
            pr = clf_obj.predict_proba(X_scaled)
            row = evaluate(y_true, p, pr)
            row["Model"] = name
            rows.append(row)

        summary = pd.DataFrame(rows)[["Model"] + METRIC_NAMES]

        st.dataframe(
            summary.style
            .highlight_max(subset=METRIC_NAMES, props="background-color:#2e7d32;color:white")
            .highlight_min(subset=METRIC_NAMES, props="background-color:#c62828;color:white")
            .format({c: "{:.4f}" for c in METRIC_NAMES}),
            width="stretch",
            hide_index=True,
        )

        st.markdown("---")
        st.subheader("Metric Visualisation")
        metric_choice = st.selectbox("Choose a metric:", METRIC_NAMES)

        fig, ax = plt.subplots(figsize=(10, 5))
        names = summary["Model"].tolist()
        vals  = summary[metric_choice].tolist()
        bars  = ax.bar(names, vals, color=BAR_COLOURS, edgecolor="white", linewidth=1.5)

        for b, v in zip(bars, vals):
            ax.text(
                b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
                f"{v:.4f}", ha="center", va="bottom", fontweight="bold", fontsize=10,
            )

        ax.set_ylabel(metric_choice, fontsize=12, fontweight="bold")
        ax.set_title(f"{metric_choice} - All Models", fontsize=14, fontweight="bold")
        ax.set_ylim(0, max(vals) * 1.15)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.xticks(rotation=15, ha="right")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        winner  = summary.loc[summary["F1"].idxmax(), "Model"]
        best_f1 = summary["F1"].max()
        st.success(f"**Top Performer (F1): {winner}** with F1 = {best_f1:.4f}")
    else:
        st.warning("Ground-truth labels missing - comparison unavailable.")

# ── Tab 3: Data preview ───────────────────────────────────────────────────
with tab_data:
    st.subheader("Uploaded Data Preview")
    st.dataframe(test_df.head(50), width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Rows", test_df.shape[0])
    with c2:
        st.metric("Columns", test_df.shape[1])

    if has_labels:
        st.subheader("Label Distribution")
        counts = test_df["quality_label"].value_counts()
        fig, ax = plt.subplots(figsize=(6, 4))
        counts.plot(kind="bar", color=["#43e97b", "#667eea", "#fa709a"],
                    edgecolor="white", ax=ax)
        ax.set_ylabel("Count", fontweight="bold")
        ax.set_title("Quality Class Counts in Test Set", fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.xticks(rotation=0)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

# ── Tab 4: About ──────────────────────────────────────────────────────────
with tab_about:
    st.subheader("About This App")
    st.markdown("""
    ### Wine Quality Prediction

    This dashboard showcases **five supervised classifiers** trained on the
    [UCI Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality).

    #### Data Overview
    - **Combined samples**: 6,497 wines (1,599 red + 4,898 white)
    - **Input dimensions**: 12 (11 chemical properties + wine colour)
    - **Output classes**: Low (scores 3-5), Medium (score 6), High (scores 7-9)

    #### Classifiers
    1. **Logistic Regression** - Linear boundary classifier
    2. **Decision Tree** - Rule-based partitioning
    3. **K-Nearest Neighbors** - Distance-based voting
    4. **Gaussian Naive Bayes** - Probabilistic with independence assumption
    5. **Random Forest** - Bagged ensemble of trees

    #### Reported Metrics
    Accuracy, AUC (OVR), Precision, Recall, F1, Matthews Correlation Coefficient

    ---
    *ML Assignment 2 - BITS Pilani M.Tech (AIML)*
    """)
