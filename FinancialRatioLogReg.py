import pandas as pd
import numpy as np

from mrmr import mrmr_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# -----------------------------
# File paths
# -----------------------------

paths = {
    "IMXI": r"C:\Users\tate2\OneDrive\Desktop\Financial Data\imxi-ratios-quarterly.csv",
    "BIG": r"C:\Users\tate2\OneDrive\Desktop\Financial Data\BIGGQ-ratios-quarterly.csv",
    "FVRR": r"C:\Users\tate2\OneDrive\Desktop\Financial Data\fvrr-ratios-quarterly.csv",
}


# -----------------------------
# Helper function to clean columns
# -----------------------------

def clean_numeric_col(col):
    col = col.astype(str).str.strip()
    col = col.str.replace(",", "", regex=False)

    is_percent = col.str.endswith("%")

    col = col.str.replace("%", "", regex=False)

    out = pd.to_numeric(col, errors="coerce")

    # Convert values like 9.60% to 0.096
    out = out.where(~is_percent, out / 100)

    return out


# -----------------------------
# Prepare one company's dataframe
# -----------------------------

def prepDF(path, ticker):
    df = (
        pd.read_csv(path)
        .rename(columns={"Date": "metric"})
        .set_index("metric")
        .T
    )

    # Convert index to dates
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"

    # Sort oldest to newest
    df = df.sort_index()

    # Convert financial values to numeric
    df = df.apply(clean_numeric_col)

    if "Market Capitalization" not in df.columns:
        raise ValueError(f"{ticker}: Market Capitalization column not found.")

    # Replace zero market caps with NaN to avoid divide-by-zero problems
    market_cap = df["Market Capitalization"].replace(0, np.nan)

    # Target:
    # Current quarter financial data -> next quarter market cap change
    df["MC Change"] = market_cap.shift(-1) / market_cap - 1

    # Optional checking column
    df["Next Q Market Cap"] = market_cap.shift(-1)

    # Drop rows where the future target cannot be calculated
    df = df.dropna(subset=["MC Change"])

    # Replace infinities with NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    # Add company ticker for tracking
    df.insert(0, "Ticker", ticker)

    # Fill missing feature values only after target is created
    df = df.fillna(0)

    return df


# -----------------------------
# Combine all companies
# -----------------------------

def combineDFs(dfs):
    # Keep only columns that all companies have
    common_cols = dfs[0].columns

    for df in dfs[1:]:
        common_cols = common_cols.intersection(df.columns)

    combined = pd.concat(
        [df[common_cols].copy() for df in dfs],
        axis=0
    )

    # Sort combined data by date
    combined = combined.sort_index()

    # Binary target: 1 if next quarter market cap increased, else 0
    combined["MC Binary"] = (combined["MC Change"] > 0).astype(int)

    return combined


# -----------------------------
# Load and combine data
# -----------------------------

dfs = [
    prepDF(path, ticker)
    for ticker, path in paths.items()
]

df = combineDFs(dfs)




from mrmr import mrmr_classif

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

import numpy as np
import pandas as pd


# -----------------------------
# Build classification dataframe
# -----------------------------

df_model = df.reset_index().sort_values("Date").reset_index(drop=True)

# Make sure binary target exists
df_model["MC Binary"] = (df_model["MC Change"] > 0).astype(int)

target_col = "MC Binary"

# Do not use these as predictors
drop_cols = [
    "Date",
    "Ticker",
    "MC Change",
    "MC Binary",
    "Market Capitalization",
    "Next Q Market Cap",
]

X = df_model.drop(columns=drop_cols, errors="ignore")
y = df_model[target_col]

# Force X to numeric
X = X.apply(pd.to_numeric, errors="coerce")
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(0)

# Drop constant columns
constant_cols = X.columns[X.nunique(dropna=False) <= 1]
X = X.drop(columns=constant_cols)

print("Dropped constant columns:")
print(list(constant_cols))

print()
print("Class balance:")
print(y.value_counts())
print()
print(y.value_counts(normalize=True))


# -----------------------------
# Chronological train/test split
# -----------------------------

unique_dates = np.array(sorted(df_model["Date"].unique()))
split_date = unique_dates[int(len(unique_dates) * 0.8)]

train_mask = df_model["Date"] < split_date
test_mask = df_model["Date"] >= split_date

X_train = X.loc[train_mask].copy()
X_test = X.loc[test_mask].copy()

y_train = y.loc[train_mask].copy()
y_test = y.loc[test_mask].copy()

print()
print("Train dates:")
print(df_model.loc[train_mask, "Date"].min(), "to", df_model.loc[train_mask, "Date"].max())

print()
print("Test dates:")
print(df_model.loc[test_mask, "Date"].min(), "to", df_model.loc[test_mask, "Date"].max())

print()
print("Train class balance:")
print(y_train.value_counts(normalize=True))

print()
print("Test class balance:")
print(y_test.value_counts(normalize=True))


# -----------------------------
# mRMR classification feature selection
# -----------------------------

K = min(10, X_train.shape[1])

selected_features = mrmr_classif(
    X=X_train,
    y=y_train,
    K=K
)

print()
print("Selected features:")
print(selected_features)


# -----------------------------
# Logistic regression classifier
# -----------------------------

clf = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        max_iter=5000,
        class_weight="balanced"
    ))
])

clf.fit(
    X_train[selected_features],
    y_train
)

preds = clf.predict(
    X_test[selected_features]
)

probs = clf.predict_proba(
    X_test[selected_features]
)[:, 1]


# -----------------------------
# Baseline classifier
# -----------------------------

# Baseline: always predict the most common class in training set
majority_class = y_train.mode()[0]

baseline_preds = np.repeat(
    majority_class,
    len(y_test)
)

# -----------------------------
# Evaluation
# -----------------------------

print()
print("Model performance:")
print("Accuracy:", accuracy_score(y_test, preds))
print("Balanced accuracy:", balanced_accuracy_score(y_test, preds))

if y_test.nunique() == 2:
    print("ROC AUC:", roc_auc_score(y_test, probs))
else:
    print("ROC AUC: cannot calculate because test set has only one class.")

print()
print("Baseline performance:")
print("Baseline accuracy:", accuracy_score(y_test, baseline_preds))
print("Baseline balanced accuracy:", balanced_accuracy_score(y_test, baseline_preds))

print()
print("Confusion matrix:")
print(confusion_matrix(y_test, preds))

print()
print("Classification report:")
print(classification_report(y_test, preds))


# -----------------------------
# Coefficients
# -----------------------------

log_reg = clf.named_steps["model"]

coef_table = pd.DataFrame({
    "Feature": selected_features,
    "Coefficient": log_reg.coef_[0]
}).sort_values("Coefficient", ascending=False)

print()
print("Coefficient table:")
print(coef_table)

# -----------------------------
# Inspecting other thresholds
# -----------------------------

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

threshold_results = []

for threshold in np.arange(0.05, 0.96, 0.05):
    threshold_preds = (probs >= threshold).astype(int)

    threshold_results.append({
        "Threshold": threshold,
        "Accuracy": accuracy_score(y_test, threshold_preds),
        "Balanced Accuracy": balanced_accuracy_score(y_test, threshold_preds),
        "Precision 1": precision_score(y_test, threshold_preds, zero_division=0),
        "Recall 1": recall_score(y_test, threshold_preds, zero_division=0),
        "F1 1": f1_score(y_test, threshold_preds, zero_division=0),
        "Predicted Ups": threshold_preds.sum()
    })

threshold_df = pd.DataFrame(threshold_results)

print(threshold_df.sort_values("Balanced Accuracy", ascending=False))

print(pd.Series(probs).describe())

print()
print("Predicted probabilities:")
print(pd.Series(probs).sort_values())
