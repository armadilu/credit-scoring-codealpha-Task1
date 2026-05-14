# Credit Scoring Model

**CodeAlpha Machine Learning Internship — Task 1**

A machine learning project that predicts an individual's creditworthiness using classification algorithms trained on historical financial and credit-related data.

---

## Project Overview

This project builds and evaluates three classification models to predict whether a customer is a credit risk (likely to default) based on their application data and historical credit behavior.

**Target definition:** A customer is labeled as a bad credit risk (1) if they were ever 60 or more days late on a payment (STATUS = 2, 3, 4, or 5 in the credit records). Otherwise, they are labeled as good (0).

---

## Dataset

Source: [Credit Card Approval Prediction — Kaggle](https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction)

| File | Description | Rows × Columns |
|---|---|---|
| `application_record.csv` | Customer demographic and financial info | 438,557 × 18 |
| `credit_record.csv` | Monthly credit/payment status history | 1,048,575 × 3 |

After merging on customer ID: **36,457 records × 19 columns**.

---

## Pipeline

1. Load and clean both datasets
2. Build the TARGET variable from credit history
3. Merge application data with the target label
4. Handle missing values (filled `OCCUPATION_TYPE` with "Unknown")
5. Label-encode categorical features
6. Standardize features with `StandardScaler`
7. Stratified 80/20 train-test split
8. Apply SMOTE to balance the training set (~1.7% bad → 50/50)
9. Train three classification models
10. Evaluate using Accuracy, Precision, Recall, F1, ROC-AUC, and Confusion Matrix
11. Tune Random Forest to address majority-class bias
12. Compare ROC curves across all models

---

## Models Used

- **Logistic Regression** — linear baseline
- **Decision Tree** — single-tree non-linear classifier
- **Random Forest** — ensemble of 200 decision trees with `class_weight='balanced'`

---

## Results

| Model | Accuracy | ROC-AUC | Recall (Bad) | Bad Customers Caught |
|---|---|---|---|---|
| Logistic Regression | 0.36 | 0.52 | 0.69 | 85 / 123 |
| Decision Tree | 0.38 | 0.47 | 0.57 | 70 / 123 |
| Random Forest | 0.98 | 0.49 | 0.00 | 0 / 123 |

---

## Key Insight

**Accuracy is misleading on imbalanced datasets.** Random Forest scored 98% accuracy by predicting "Good" for every customer — useless in practice. The correct metric for credit risk is **recall on the minority class**, because missing a bad customer (false negative) costs far more than incorrectly flagging a good one.

**Best model: Logistic Regression.** Despite having the lowest accuracy, it caught 69% of bad customers — the metric that actually matters for a lender.

---

## Dataset Limitation

All three models produced ROC-AUC values near 0.5, indicating the application features (income, education, occupation, etc.) have weak predictive signal for actual payment behavior in this dataset. Performance would significantly improve with richer features such as debt-to-income ratio, prior loan defaults, or credit utilization history.

---

## How to Run

1. Open `CodeAlpha_Credit_Scoring_Model.ipynb` in Google Colab.
2. Upload `application_record.csv` and `credit_record.csv` to your Google Drive (or to the Colab session directly).
3. Run all cells in order via `Runtime → Run all`.

**Required libraries:** pandas, numpy, matplotlib, seaborn, scikit-learn, imbalanced-learn.

---

## Tech Stack

- Python 3.12
- pandas, NumPy
- Matplotlib, Seaborn
- scikit-learn
- imbalanced-learn (SMOTE)
- Google Colab

---

## Author

**Aamina Khan**
CodeAlpha Machine Learning Internship — Task 1

---
