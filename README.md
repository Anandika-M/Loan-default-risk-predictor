# Loan Default Risk Predictor

Baseline credit risk modeling project to predict the probability of loan default using interpretable machine learning and risk-focused evaluation.

---

## Overview
This project implements a structured end-to-end workflow for credit default prediction, closely aligned with real-world risk management practices.  
The focus is on data quality, interpretability, and robust evaluation rather than model complexity.

---

## Workflow
- Data understanding and schema correction
- Data quality checks and preprocessing
- Exploratory data analysis (EDA) to identify key risk drivers
- Baseline logistic regression model for default prediction
- Model evaluation using ROC-AUC, confusion matrix, and threshold analysis
- Benchmark comparison with a Random Forest classifier

---

## Dataset
The project uses a public credit card default dataset containing borrower demographics, credit limits, billing amounts, and repayment history.

Target variable:
- **default** — indicates whether a borrower defaulted in the following month

---

## Models
- **Logistic Regression**  
  Used as the primary model due to its interpretability and suitability for regulated credit risk environments.

- **Random Forest (Benchmark)**  
  Used as a challenger model to evaluate potential performance gains from non-linear relationships.

---

## Evaluation Metrics
- ROC-AUC (primary metric)
- Confusion Matrix
- Precision, Recall, F1-Score
- Threshold sensitivity analysis

Accuracy is reported but not emphasized due to class imbalance in default prediction.

---

## Tech Stack
- Python
- pandas, numpy
- scikit-learn
- matplotlib, seaborn

---

## How to Run

```bash
pip install -r requirements.txt
python credit_risk_model.py
