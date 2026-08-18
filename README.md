# Wine Quality Classification - ML Assignment 2

## a. Problem Statement

This project tackles multi-class wine quality prediction. Using 12 physicochemical measurements, we categorise wines into **Low**, **Medium**, or **High** quality tiers. Five supervised classifiers are trained, evaluated on six metrics, and presented through a live Streamlit dashboard.

Raw quality scores (3 through 9) from the UCI repository are grouped as follows:
- **Low**: scores 3-5
- **Medium**: score 6
- **High**: scores 7-9

## b. Dataset Description

| Attribute | Value |
|---|---|
| **Name** | UCI Wine Quality Dataset |
| **Source** | [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/wine+quality) |
| **Records** | 6,497 (1,599 red + 4,898 white) |
| **Predictors** | 12 |
| **Target** | Quality tier (Low / Medium / High) |
| **Split** | 80% training, 20% test (stratified) |

### Predictor Variables (12):
1. Fixed Acidity
2. Volatile Acidity
3. Citric Acid
4. Residual Sugar
5. Chlorides
6. Free Sulfur Dioxide
7. Total Sulfur Dioxide
8. Density
9. pH
10. Sulphates
11. Alcohol
12. Wine Type (0 = Red, 1 = White)

## c. GitHub Repository Link

> **Repository**: [https://github.com/Satvik-12/ML-Assignment-2](https://github.com/Satvik-12/ML-Assignment-2)
>
> **Streamlit App**: [https://satvik-12-ml-assignment-2.streamlit.app/](https://satvik-12-ml-assignment-2.streamlit.app/)

## d. Models Used

### Evaluation Metrics - All Models

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.5823 | 0.7324 | 0.5873 | 0.5823 | 0.5763 | 0.3268 |
| Decision Tree | 0.6062 | 0.7324 | 0.6086 | 0.6062 | 0.6069 | 0.3836 |
| kNN | 0.5823 | 0.7523 | 0.5855 | 0.5823 | 0.5836 | 0.3474 |
| Naive Bayes | 0.4815 | 0.6554 | 0.4966 | 0.4815 | 0.4816 | 0.2172 |
| Random Forest (Ensemble) | 0.7277 | 0.8805 | 0.7348 | 0.7277 | 0.7276 | 0.5675 |

### Performance Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Delivers 58.23% accuracy, which is reasonable for a linear classifier on multi-class data with overlapping feature distributions. Its AUC of 0.7324 reflects decent probabilistic ranking, though the relatively low MCC (0.3268) reveals difficulty distinguishing between the three quality tiers. Balanced precision and recall suggest the model does not favour any particular class. |
| Decision Tree | Improves on Logistic Regression, reaching 60.62% accuracy. Tree-based splits pick up several non-linear decision boundaries that a linear model misses. Despite sharing an identical AUC of 0.7324, the higher MCC (0.3836) indicates better overall balance across correct and incorrect predictions. Capping depth at 10 helps curb overfitting, yet some variance remains. |
| kNN | Records 58.23% accuracy - on par with Logistic Regression - but edges ahead in AUC (0.7523), pointing to stronger probability estimates for class ranking. Because the algorithm relies on local distances, it benefits from the applied feature standardisation yet still suffers from the moderate dimensionality of 12 predictors. |
| Naive Bayes | Registers the lowest scores across the board: 48.15% accuracy, 0.6554 AUC, and 0.2172 MCC. The Gaussian independence assumption is a poor fit here because many wine properties are correlated (alcohol with density, residual sugar with total SO2). This mismatch degrades both precision and recall, especially for the Medium class. |
| Random Forest (Ensemble) | Clearly the strongest performer with 72.77% accuracy, an AUC of 0.8805 and MCC of 0.5675. Aggregating predictions from 100 individual trees smooths out noise and captures complex feature interactions that single-tree or linear models cannot. Precision and recall remain well-balanced across all three quality tiers. |
| **Overall Winner** | **Random Forest (Ensemble)** - leads every metric by a clear margin. Its ensemble strategy is ideally suited for this dataset, where physicochemical properties interact in non-obvious ways to determine wine quality. |

## Project Structure

```
ML-Assignment-2/
|-- app.py                    # Streamlit dashboard
|-- requirements.txt          # Dependencies
|-- README.md                 # This file
|-- test_data.csv             # Holdout test set (1,300 samples)
|-- model/
    |-- train_models.py       # Training pipeline
    |-- logistic_regression.pkl
    |-- decision_tree.pkl
    |-- knn.pkl
    |-- naive_bayes.pkl
    |-- random_forest.pkl
    |-- scaler.pkl
    |-- label_encoder.pkl
    |-- metadata.json
```

## Running Locally

```bash
pip install -r requirements.txt
python model/train_models.py
streamlit run app.py
```

## Deployment

Hosted on **Streamlit Community Cloud**. Use the app link above to access the live dashboard.

---

*ML Assignment 2 - BITS Pilani M.Tech (AIML) - Machine Learning*
