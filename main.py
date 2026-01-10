import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report

df = pd.read_csv("credit_data.csv")


# DATA LOAD AND UNDERSTANDING
print("Original shape:", df.shape)

#0th row --> no features 
# Drop the 0th row and take first row for features

df.columns = df.iloc[0]          
df = df.drop(index=0)            
df = df.reset_index(drop=True)

print("After fixing headers:", df.shape)
print("\nNew column names:")
print(df.columns.tolist())

#Drop ID column
if 'ID' in df.columns:
    df = df.drop(columns=['ID'])

#Rename target column

df = df.rename(columns={
    'default payment next month': 'default'
})

print("\nTarget column renamed to 'default'")

#Convert numeric columns
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

print("\nData types after conversion:")
print(df.dtypes)

# Target variable check
if 'default' in df.columns:
    print("\nTarget column found.")
    print("\nDefault distribution (%):")
    print(df['default'].value_counts(normalize=True) * 100)
 
#Feature type identification

numerical_features = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

print("\nNumerical feature count:", len(numerical_features))
print("Sample numerical features:", numerical_features[:10])


df.to_csv("credit_data_phase1_clean.csv", index=False)
print("\nPhase 1 clean dataset saved as 'credit_data_phase1_clean.csv'")

numerical_features = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

print("\nNumerical feature count:", len(numerical_features))
print("Sample numerical features:", numerical_features[:10])


df.to_csv("credit_data_phase1_clean.csv", index=False)
print("\ncleaned dataset saved as 'credit_data_phase1_clean.csv'")


#DATA PREPROCESSING 

#Missing value analysis
missing_summary = df.isnull().sum()
missing_cols = missing_summary[missing_summary > 0]

print("\n--- Missing Values Summary ---")
if missing_cols.empty:
    print("No missing values found.")
else:
    print(missing_cols)

# Simple imputation (safe for this dataset)
df = df.fillna(0)

print("\nMissing values handled (filled with 0).")

#duplicate value analysis
duplicate_count = df.duplicated().sum()
print("\nDuplicate rows found:", duplicate_count)

if duplicate_count > 0:
    df = df.drop_duplicates()
    print("Duplicates removed.")

#general data quality checks 

sanity_checks = {
    "Negative Credit Limit": (df['LIMIT_BAL'] < 0).sum(),
    "Negative Age": (df['AGE'] < 0).sum(),
    "Negative Bill Amounts": (df.filter(like='BILL_AMT') < 0).sum().sum(),
    "Negative Payment Amounts": (df.filter(like='PAY_AMT') < 0).sum().sum()
}

print("\n--- Sanity Check Results ---")
for check, count in sanity_checks.items():
    print(f"{check}: {count}")


#outlier detection 

outlier_summary = {}

for col in ['LIMIT_BAL', 'AGE']:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    
    outliers = ((df[col] < lower) | (df[col] > upper)).sum()
    outlier_summary[col] = outliers

print("\n--- Outlier Summary (Detected, Not Removed) ---")
for col, count in outlier_summary.items():
    print(f"{col}: {count}")

# target variables
print("\n--- Target Variable Validation ---")
print(df['default'].value_counts())
print("\nDefault Rate (%):")
print(df['default'].value_counts(normalize=True) * 100)


#EDA

# Separate defaulters and non-defaulters
default_0 = df[df['default'] == 0]
default_1 = df[df['default'] == 1]

#Credit Limit Analysis

print("\n--- Credit Limit Summary ---")
print("Non-defaulters mean LIMIT_BAL:", default_0['LIMIT_BAL'].mean())
print("Defaulters mean LIMIT_BAL:", default_1['LIMIT_BAL'].mean())

plt.figure(figsize=(6,4))
sns.kdeplot(default_0['LIMIT_BAL'], label='Non-Default', fill=True)
sns.kdeplot(default_1['LIMIT_BAL'], label='Default', fill=True)
plt.title("Credit Limit Distribution by Default Status")
plt.legend()
plt.show()

#Age distribute

plt.figure(figsize=(6,4))
sns.kdeplot(default_0['AGE'], label='Non-Default', fill=True)
sns.kdeplot(default_1['AGE'], label='Default', fill=True)
plt.title("Age Distribution by Default Status")
plt.legend()
plt.show()

#Bill amount 
bill_cols = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3']

bill_summary = df.groupby('default')[bill_cols].mean()
print("\n--- Average Bill Amounts by Default ---")
print(bill_summary)

#correlation analysis
corr = df.corr()['default'].sort_values(ascending=False)

print("\n--- Top Correlated Features with Default ---")
print(corr.head(10))

print("\n--- Least Correlated Features with Default ---")
print(corr.tail(10))

#Feature determination 

X = df.drop(columns=['default'])
y = df['default']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

print("\nTrain default rate:", y_train.mean())
print("Test default rate:", y_test.mean())

#defining log regression model
log_model = LogisticRegression(
    max_iter=1000,
    solver='liblinear'
)

log_model.fit(X_train, y_train)

#predict model 
y_pred = log_model.predict(X_test)
y_prob = log_model.predict_proba(X_test)[:, 1]

#model evaluation 
auc = roc_auc_score(y_test, y_prob)
print("\nROC-AUC Score:", round(auc, 4))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

fpr, tpr, _ = roc_curve(y_test, y_prob)

#The confusion matrix initially gives 
# [[6998    3]
#  [1988    1]]

thresholds = [0.05, 0.1, 0.2, 0.3]

for t in thresholds:
    y_pred_t = (y_prob >= t).astype(int)
    cm = confusion_matrix(y_test, y_pred_t)
    print(f"\nThreshold = {t}")
    print(cm)



plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, label=f"Logistic Regression (AUC = {auc:.3f})")
plt.plot([0,1], [0,1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – Credit Default Model")
plt.legend()
plt.show()


rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=6,
    random_state=42,
    class_weight='balanced'
)

rf_model.fit(X_train, y_train)

# Predictions
rf_pred = rf_model.predict(X_test)
rf_prob = rf_model.predict_proba(X_test)[:, 1]

# Evaluation
rf_auc = roc_auc_score(y_test, rf_prob)

print("\n--- Random Forest Results ---")
print("ROC-AUC Score:", round(rf_auc, 4))

print("\nClassification Report:")
print(classification_report(y_test, rf_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, rf_pred))

#comparison
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_prob)

plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, label=f"Logistic Regression (AUC = {auc:.3f})")
plt.plot(rf_fpr, rf_tpr, label=f"Random Forest (AUC = {rf_auc:.3f})")


#accuracy comparison 

rf_acc = accuracy_score(y_test, rf_pred)
print("Random Forest Accuracy:", round(rf_acc, 4))

acc = accuracy_score(y_test, y_pred)
print("Logistic regression Accuracy:", round(acc, 4))


plt.plot([0,1], [0,1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.show()