import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    accuracy_score,
    precision_recall_curve,
    average_precision_score,
    ConfusionMatrixDisplay
)

import warnings
warnings.filterwarnings('ignore')

# ---- Style 
plt.rcParams.update({
    'figure.dpi': 130,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'font.family': 'DejaVu Sans',
    'axes.titlesize': 13,
    'axes.labelsize': 11,
})
PALETTE = {'Non-Default': '#4C9BE8', 'Default': '#E8704C'}

# 1. LOAD & CLEAN DATA

df = pd.read_csv("credit_data.csv")
print("Original shape:", df.shape)

df.columns = df.iloc[0]
df = df.drop(index=0).reset_index(drop=True)

if 'ID' in df.columns:
    df = df.drop(columns=['ID'])

df = df.rename(columns={'default payment next month': 'default'})

for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

print(f"\nMissing values before imputation: {df.isnull().sum().sum()}")
df = df.fillna(df.median())

# ── Feature engineering 
bill_cols = ['BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6']
pay_cols  = ['PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6']

df['avg_bill']        = df[bill_cols].mean(axis=1)
df['avg_payment']     = df[pay_cols].mean(axis=1)
df['bill_payment_ratio'] = (df['avg_bill'] / (df['avg_payment'] + 1)).clip(upper=50)
df['util_rate']       = (df['avg_bill'] / (df['LIMIT_BAL'] + 1)).clip(upper=2)
df['missed_payments'] = (df[['PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6']] > 0).sum(axis=1)

print("\nDefault distribution (%):")
print(df['default'].value_counts(normalize=True).mul(100).round(2))

# 2. CHART 1 – CLASS BALANCE

fig, ax = plt.subplots(figsize=(5, 4))
counts = df['default'].value_counts()
bars = ax.bar(['Non-Default (0)', 'Default (1)'], counts.values,
              color=[PALETTE['Non-Default'], PALETTE['Default']], edgecolor='white', width=0.5)
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 150,
            f'{val:,}\n({val/len(df)*100:.1f}%)', ha='center', va='bottom', fontsize=10)
ax.set_title("Class Distribution – Loan Default")
ax.set_ylabel("Count")
plt.tight_layout()
plt.savefig("chart_01_class_balance.png")
plt.show()
print("Saved: chart_01_class_balance.png")

# 3. CHART 2 – CREDIT LIMIT & AGE KDE

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, col, title in zip(axes,
                           ['LIMIT_BAL', 'AGE'],
                           ['Credit Limit Distribution', 'Age Distribution']):
    for label, val in [('Non-Default', 0), ('Default', 1)]:
        sns.kdeplot(df[df['default'] == val][col], ax=ax,
                    label=label, fill=True, alpha=0.4,
                    color=PALETTE[label])
    ax.set_title(title)
    ax.legend()
plt.suptitle("Numeric Feature Distributions by Default Status", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig("chart_02_kde_limit_age.png")
plt.show()
print("Saved: chart_02_kde_limit_age.png")

# 4. CHART 3 – DEFAULT RATE BY PAY_0 (repayment status)

fig, ax = plt.subplots(figsize=(8, 4))
dr = df.groupby('PAY_0')['default'].mean().mul(100)
bars = ax.bar(dr.index.astype(str), dr.values,
              color='#E8704C', edgecolor='white')
for bar, val in zip(bars, dr.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
ax.set_title("Default Rate by PAY_0 (Most Recent Repayment Status)")
ax.set_xlabel("PAY_0 Status  (−1=on time, 1–8=months delayed)")
ax.set_ylabel("Default Rate (%)")
plt.tight_layout()
plt.savefig("chart_03_default_by_pay0.png")
plt.show()
print("Saved: chart_03_default_by_pay0.png")

# 5. CHART 4 – MISSED PAYMENTS vs DEFAULT RATE

fig, ax = plt.subplots(figsize=(7, 4))
mp_rate = df.groupby('missed_payments')['default'].mean().mul(100)
ax.plot(mp_rate.index, mp_rate.values, marker='o', color='#E8704C', linewidth=2)
ax.fill_between(mp_rate.index, mp_rate.values, alpha=0.15, color='#E8704C')
ax.set_title("Default Rate by Number of Missed Payments (Last 6 Months)")
ax.set_xlabel("Missed Payment Count")
ax.set_ylabel("Default Rate (%)")
ax.set_xticks(mp_rate.index)
plt.tight_layout()
plt.savefig("chart_04_missed_payments.png")
plt.show()
print("Saved: chart_04_missed_payments.png")

# 6. CHART 5 – CORRELATION HEATMAP (top features only)

top_feats = ['default','LIMIT_BAL','AGE','PAY_0','PAY_2','PAY_3',
             'BILL_AMT1','PAY_AMT1','avg_bill','avg_payment',
             'util_rate','missed_payments','bill_payment_ratio']
corr_matrix = df[top_feats].corr()

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
            cmap='RdBu_r', center=0, linewidths=0.5,
            ax=ax, annot_kws={'size': 8})
ax.set_title("Feature Correlation Heatmap (Selected Features)")
plt.tight_layout()
plt.savefig("chart_05_correlation_heatmap.png")
plt.show()
print("Saved: chart_05_correlation_heatmap.png")

# 7. TRAIN / TEST SPLIT


X = df.drop(columns=['default'])
y = df['default']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)


# 8. MODELS

#  Logistic Regression 
log_model = LogisticRegression(max_iter=1000, solver='liblinear', class_weight='balanced')
log_model.fit(X_train_sc, y_train)
y_pred_log  = log_model.predict(X_test_sc)
y_prob_log  = log_model.predict_proba(X_test_sc)[:, 1]
log_auc     = roc_auc_score(y_test, y_prob_log)
log_ap      = average_precision_score(y_test, y_prob_log)

#  Random Forest 
rf_model = RandomForestClassifier(n_estimators=200, max_depth=6,
                                   random_state=42, class_weight='balanced')
rf_model.fit(X_train, y_train)
rf_pred     = rf_model.predict(X_test)
rf_prob     = rf_model.predict_proba(X_test)[:, 1]
rf_auc      = roc_auc_score(y_test, rf_prob)
rf_ap       = average_precision_score(y_test, rf_prob)

#  Gradient Boosting 
gb_model = GradientBoostingClassifier(n_estimators=150, max_depth=4,
                                       learning_rate=0.05, random_state=42)
gb_model.fit(X_train, y_train)
gb_pred     = gb_model.predict(X_test)
gb_prob     = gb_model.predict_proba(X_test)[:, 1]
gb_auc      = roc_auc_score(y_test, gb_prob)
gb_ap       = average_precision_score(y_test, gb_prob)

print("\n=== Model Summary ===")
for name, auc, ap in [("Logistic Regression", log_auc, log_ap),
                       ("Random Forest",       rf_auc,  rf_ap),
                       ("Gradient Boosting",   gb_auc,  gb_ap)]:
    print(f"{name:22s}  AUC={auc:.4f}  Avg-Precision={ap:.4f}")

# 9. THRESHOLD TUNING (best F1 for defaulters)

thresholds = np.arange(0.10, 0.90, 0.01)
best_f1, best_t = 0, 0
for t in thresholds:
    rpt = classification_report(y_test, (rf_prob >= t).astype(int), output_dict=True)
    f1 = rpt.get('1', {}).get('f1-score', 0)
    if f1 > best_f1:
        best_f1, best_t = f1, t

print(f"\nBest threshold (RF, F1-default): {best_t:.2f}  →  F1={best_f1:.4f}")
rf_pred_tuned = (rf_prob >= best_t).astype(int)

# 10. CHART 6 – ROC CURVES (all 3 models)

fig, ax = plt.subplots(figsize=(7, 5))
for name, fpr_arr, tpr_arr, auc_val, color in [
    ("Logistic Regression", *roc_curve(y_test, y_prob_log)[:2], log_auc, '#4C9BE8'),
    ("Random Forest",       *roc_curve(y_test, rf_prob)[:2],    rf_auc,  '#E8704C'),
    ("Gradient Boosting",   *roc_curve(y_test, gb_prob)[:2],    gb_auc,  '#4CAF50'),
]:
    ax.plot(fpr_arr, tpr_arr, label=f"{name} (AUC={auc_val:.3f})", color=color, linewidth=2)
ax.plot([0,1],[0,1], linestyle='--', color='gray')
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve – All Models")
ax.legend()
plt.tight_layout()
plt.savefig("chart_06_roc_curves.png")
plt.show()
print("Saved: chart_06_roc_curves.png")


# 11. CHART 7 – PRECISION-RECALL CURVES

fig, ax = plt.subplots(figsize=(7, 5))
for name, prob, ap, color in [
    ("Logistic Regression", y_prob_log, log_ap, '#4C9BE8'),
    ("Random Forest",       rf_prob,    rf_ap,  '#E8704C'),
    ("Gradient Boosting",   gb_prob,    gb_ap,  '#4CAF50'),
]:
    prec, rec, _ = precision_recall_curve(y_test, prob)
    ax.plot(rec, prec, label=f"{name} (AP={ap:.3f})", color=color, linewidth=2)
baseline = y_test.mean()
ax.axhline(baseline, linestyle='--', color='gray', label=f'Baseline ({baseline:.2f})')
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curve – All Models")
ax.legend()
plt.tight_layout()
plt.savefig("chart_07_precision_recall.png")
plt.show()
print("Saved: chart_07_precision_recall.png")

# 12. CHART 8 – CONFUSION MATRICES 

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, name, pred in zip(axes,
    ["Logistic Regression", "Random Forest (default)", f"Random Forest (t={best_t:.2f})"],
    [y_pred_log, rf_pred, rf_pred_tuned]):
    cm = confusion_matrix(y_test, pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=['No Default','Default'])
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(name)
plt.suptitle("Confusion Matrices", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig("chart_08_confusion_matrices.png")
plt.show()
print("Saved: chart_08_confusion_matrices.png")

# 13. CHART 9 – FEATURE IMPORTANCE (RF)

importances = pd.Series(
    rf_model.feature_importances_, index=X.columns
).sort_values(ascending=True).tail(15)

fig, ax = plt.subplots(figsize=(8, 6))
engineered = ['missed_payments', 'util_rate', 'bill_payment_ratio', 'avg_bill', 'avg_payment']
colors = ['#E8704C' if f in engineered else '#4C9BE8' for f in importances.index]
importances.plot(kind='barh', ax=ax, color=colors, edgecolor='white')
ax.set_title("Top 15 Feature Importances (Random Forest)\nOrange = engineered features")
ax.set_xlabel("Importance Score")
plt.tight_layout()
plt.savefig("chart_09_feature_importance.png")
plt.show()
print("Saved: chart_09_feature_importance.png")

# 14. CHART 10 – THRESHOLD SENSITIVITY (RF)

results = []
for t in np.arange(0.1, 0.9, 0.02):
    pred_t = (rf_prob >= t).astype(int)
    rpt = classification_report(y_test, pred_t, output_dict=True, zero_division=0)
    results.append({
        'threshold': t,
        'precision_1': rpt.get('1', {}).get('precision', 0),
        'recall_1':    rpt.get('1', {}).get('recall', 0),
        'f1_1':        rpt.get('1', {}).get('f1-score', 0),
    })
thresh_df = pd.DataFrame(results)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(thresh_df['threshold'], thresh_df['precision_1'], label='Precision (Default)', color='#4C9BE8')
ax.plot(thresh_df['threshold'], thresh_df['recall_1'],    label='Recall (Default)',    color='#E8704C')
ax.plot(thresh_df['threshold'], thresh_df['f1_1'],        label='F1 (Default)',        color='#4CAF50', linewidth=2)
ax.axvline(best_t, linestyle='--', color='gray', label=f'Best t={best_t:.2f}')
ax.set_xlabel("Classification Threshold")
ax.set_ylabel("Score")
ax.set_title("Threshold Sensitivity Analysis (Random Forest – Default Class)")
ax.legend()
plt.tight_layout()
plt.savefig("chart_10_threshold_sensitivity.png")
plt.show()
print("Saved: chart_10_threshold_sensitivity.png")

# 15. CHART 11 – CALIBRATION CURVE

fig, ax = plt.subplots(figsize=(6, 5))
for name, prob, color in [
    ("Logistic Regression", y_prob_log, '#4C9BE8'),
    ("Random Forest",       rf_prob,    '#E8704C'),
    ("Gradient Boosting",   gb_prob,    '#4CAF50'),
]:
    frac_pos, mean_pred = calibration_curve(y_test, prob, n_bins=10)
    ax.plot(mean_pred, frac_pos, marker='o', label=name, color=color)
ax.plot([0,1],[0,1], linestyle='--', color='gray', label='Perfect Calibration')
ax.set_xlabel("Mean Predicted Probability")
ax.set_ylabel("Fraction of Positives")
ax.set_title("Calibration Curves – How Well-Calibrated Are the Models?")
ax.legend()
plt.tight_layout()
plt.savefig("chart_11_calibration.png")
plt.show()
print("Saved: chart_11_calibration.png")

# 16. CHART 12 – CROSS-VALIDATED AUC COMPARISON


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = {}
for name, model, Xd in [
    ("Logistic Regression", log_model,  X_train_sc),
    ("Random Forest",       rf_model,   X_train.values),
    ("Gradient Boosting",   gb_model,   X_train.values),
]:
    scores = cross_val_score(model, Xd, y_train, cv=cv, scoring='roc_auc')
    cv_scores[name] = scores
    print(f"{name:22s}  CV AUC: {scores.mean():.4f} ± {scores.std():.4f}")

fig, ax = plt.subplots(figsize=(7, 4))
names  = list(cv_scores.keys())
means  = [cv_scores[n].mean() for n in names]
stds   = [cv_scores[n].std()  for n in names]
colors = ['#4C9BE8','#E8704C','#4CAF50']
bars   = ax.bar(names, means, yerr=stds, capsize=5,
                color=colors, edgecolor='white', width=0.5)
for bar, m in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f'{m:.4f}', ha='center', va='bottom', fontsize=10)
ax.set_ylim(0.6, 0.85)
ax.set_title("5-Fold Cross-Validated AUC (± 1 std)")
ax.set_ylabel("ROC-AUC")
plt.tight_layout()
plt.savefig("chart_12_cv_auc.png")
plt.show()
print("Saved: chart_12_cv_auc.png")

# 17. FINAL SUMMARY

print("\n" + "="*55)
print("FINAL MODEL SUMMARY")
print("="*55)
print(f"{'Model':<25} {'AUC':>6}  {'Avg Prec':>9}")
print("-"*55)
for n, a, p in [("Logistic Regression", log_auc, log_ap),
                 ("Random Forest",       rf_auc,  rf_ap),
                 ("Gradient Boosting",   gb_auc,  gb_ap)]:
    print(f"{n:<25} {a:>6.4f}  {p:>9.4f}")
print("-"*55)
print(f"\nRecommended model : Random Forest (tuned threshold = {best_t:.2f})")
print(f"Best F1 on Default class : {best_f1:.4f}")
print("\nAll charts saved to working directory.")
