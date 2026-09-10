# ============================================================
# HDFC BANK LTD.
# LOAN APPROVAL CLASSIFICATION SYSTEM
# FINAL MODEL: RANDOM FOREST CLASSIFIER
# ============================================================

import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# 1. FIND DATASET
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET = os.path.join(
    BASE_DIR,
    "loan_approval_dataset.csv"
)

print("=" * 70)
print("HDFC BANK - LOAN APPROVAL CLASSIFICATION")
print("=" * 70)

print("\nDataset location:")
print(DATASET)


# ============================================================
# 2. LOAD DATASET
# ============================================================

if not os.path.exists(DATASET):
    print("\nERROR: Dataset not found!")
    print("\nPlease keep these two files in the SAME folder:")
    print("1. hdfc_loan_random_forest_model.py")
    print("2. loan_approval_dataset.csv")
    input("\nPress Enter to exit...")
    exit()

df = pd.read_csv(DATASET)

print("\nDataset loaded successfully!")
print("Number of records:", len(df))
print("Number of columns:", len(df.columns))


# ============================================================
# 3. CLEAN COLUMN NAMES
# ============================================================

df.columns = df.columns.str.strip()


# ============================================================
# 4. CLEAN TEXT DATA
# ============================================================

# This avoids the Pandas select_dtypes warning
string_columns = df.select_dtypes(
    include=["object", "string"]
).columns

for col in string_columns:
    df[col] = df[col].astype(str).str.strip()


# ============================================================
# 5. REMOVE LOAN ID
# ============================================================

if "loan_id" in df.columns:
    df = df.drop(columns=["loan_id"])


# ============================================================
# 6. ENCODE CATEGORICAL VARIABLES
# ============================================================

df["education"] = df["education"].map({
    "Graduate": 1,
    "Not Graduate": 0
})

df["self_employed"] = df["self_employed"].map({
    "Yes": 1,
    "No": 0
})

df["loan_status"] = df["loan_status"].map({
    "Approved": 1,
    "Rejected": 0
})


# ============================================================
# 7. FEATURE ENGINEERING
# ============================================================

# Total assets
df["total_assets_value"] = (
    df["residential_assets_value"]
    + df["commercial_assets_value"]
    + df["luxury_assets_value"]
    + df["bank_asset_value"]
)


# Asset-to-loan ratio
df["asset_to_loan_ratio"] = (
    (df["total_assets_value"] + 1)
    / (df["loan_amount"] + 1)
)


# Loan-to-income ratio
df["loan_to_income_ratio"] = (
    (df["loan_amount"] + 1)
    / (df["income_annum"] + 1)
)


# ============================================================
# 8. SEPARATE INPUT AND TARGET
# ============================================================

X = df.drop(columns=["loan_status"])

y = df["loan_status"]


print("\nInput features:", X.shape[1])
print("Target variable: loan_status")


# ============================================================
# 9. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))


# ============================================================
# 10. RANDOM FOREST MODEL
# ============================================================

model = RandomForestClassifier(
    n_estimators=150,
    max_depth=20,
    criterion="gini",
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)


# ============================================================
# 11. TRAIN MODEL
# ============================================================

print("\nTraining Random Forest...")

model.fit(
    X_train,
    y_train
)

print("Training completed!")


# ============================================================
# 12. MAKE PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)

y_probability = model.predict_proba(X_test)[:, 1]


# ============================================================
# 13. MODEL PERFORMANCE
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)


print("\n")
print("=" * 70)
print("MODEL PERFORMANCE")
print("=" * 70)

print(f"\nAccuracy  : {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Precision : {precision:.4f} ({precision * 100:.2f}%)")
print(f"Recall    : {recall:.4f} ({recall * 100:.2f}%)")
print(f"F1-score  : {f1:.4f} ({f1 * 100:.2f}%)")
print(f"ROC-AUC   : {roc_auc:.4f}")


# ============================================================
# 14. CLASSIFICATION REPORT
# ============================================================

print("\n")
print("=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Rejected",
            "Approved"
        ]
    )
)


# ============================================================
# 15. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\n")
print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print("\n                Predicted")
print("              Rejected  Approved")
print(
    f"Actual Rejected   {cm[0][0]:4d}      {cm[0][1]:4d}"
)
print(
    f"       Approved   {cm[1][0]:4d}      {cm[1][1]:4d}"
)


# ============================================================
# 16. FEATURE IMPORTANCE
# ============================================================

feature_importance = pd.Series(
    model.feature_importances_,
    index=X.columns
)

feature_importance = feature_importance.sort_values(
    ascending=False
)

print("\n")
print("=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)

for feature, importance in feature_importance.items():

    print(
        f"{feature:<30} "
        f"{importance * 100:.2f}%"
    )


# ============================================================
# 17. NEW APPLICANT PREDICTION FUNCTION
# ============================================================

def predict_loan_application(
    no_of_dependents,
    education,
    self_employed,
    income_annum,
    loan_amount,
    loan_term,
    cibil_score,
    residential_assets_value,
    commercial_assets_value,
    luxury_assets_value,
    bank_asset_value
):

    # Convert categorical values
    education_value = (
        1 if education == "Graduate"
        else 0
    )

    self_employed_value = (
        1 if self_employed == "Yes"
        else 0
    )


    # Calculate total assets
    total_assets = (
        residential_assets_value
        + commercial_assets_value
        + luxury_assets_value
        + bank_asset_value
    )


    # Calculate ratios
    asset_to_loan_ratio = (
        (total_assets + 1)
        / (loan_amount + 1)
    )

    loan_to_income_ratio = (
        (loan_amount + 1)
        / (income_annum + 1)
    )


    # Create applicant dataframe
    applicant = pd.DataFrame([{

        "no_of_dependents":
            no_of_dependents,

        "education":
            education_value,

        "self_employed":
            self_employed_value,

        "income_annum":
            income_annum,

        "loan_amount":
            loan_amount,

        "loan_term":
            loan_term,

        "cibil_score":
            cibil_score,

        "residential_assets_value":
            residential_assets_value,

        "commercial_assets_value":
            commercial_assets_value,

        "luxury_assets_value":
            luxury_assets_value,

        "bank_asset_value":
            bank_asset_value,

        "total_assets_value":
            total_assets,

        "asset_to_loan_ratio":
            asset_to_loan_ratio,

        "loan_to_income_ratio":
            loan_to_income_ratio

    }])


    # Ensure exact feature order
    applicant = applicant[
        X.columns
    ]


    # Prediction
    prediction = model.predict(
        applicant
    )[0]


    # Probability
    probability = model.predict_proba(
        applicant
    )[0]


    # Result
    if prediction == 1:

        result = "APPROVED"

    else:

        result = "REJECTED"


    print("\n")
    print("=" * 70)
    print("NEW LOAN APPLICATION PREDICTION")
    print("=" * 70)

    print("\nPrediction:", result)

    print(
        f"\nProbability of Rejection: "
        f"{probability[0] * 100:.2f}%"
    )

    print(
        f"Probability of Approval: "
        f"{probability[1] * 100:.2f}%"
    )

    print("=" * 70)

    return result


# ============================================================
# 18. SAMPLE APPLICANT
# ============================================================

predict_loan_application(

    no_of_dependents=2,

    education="Graduate",

    self_employed="No",

    income_annum=9500000,

    loan_amount=25000000,

    loan_term=10,

    cibil_score=820,

    residential_assets_value=18000000,

    commercial_assets_value=9000000,

    luxury_assets_value=24000000,

    bank_asset_value=8500000
)


# ============================================================
# END
# ============================================================

print("\nModel execution completed successfully!")