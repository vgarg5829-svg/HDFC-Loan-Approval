
import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

st.set_page_config(
    page_title="HDFC Bank - Loan Approval Predictor",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 HDFC Bank — Loan Approval Predictor")
st.write("Supervised Machine Learning system using Random Forest Classification.")

DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "loan_approval_dataset.csv")

@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    # Clean text columns
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype("string").str.strip()

    # Remove ID if present
    if "loan_id" in df.columns:
        df = df.drop(columns=["loan_id"])

    # Encode categorical columns
    if "education" in df.columns:
        df["education"] = df["education"].replace({
            "Graduate": 1,
            "Not Graduate": 0
        })

    if "self_employed" in df.columns:
        df["self_employed"] = df["self_employed"].replace({
            "Yes": 1,
            "No": 0
        })

    if "loan_status" in df.columns:
        df["loan_status"] = df["loan_status"].replace({
            "Approved": 1,
            "Rejected": 0
        })

    # Convert numeric columns
    for col in df.columns:
        if col != "loan_status":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Feature engineering
    if all(c in df.columns for c in [
        "residential_assets_value",
        "commercial_assets_value",
        "luxury_assets_value",
        "bank_asset_value"
    ]):
        df["total_assets_value"] = (
            df["residential_assets_value"]
            + df["commercial_assets_value"]
            + df["luxury_assets_value"]
            + df["bank_asset_value"]
        )

    if all(c in df.columns for c in ["total_assets_value", "loan_amount"]):
        df["asset_to_loan_ratio"] = (
            df["total_assets_value"] / df["loan_amount"].replace(0, np.nan)
        )

    if all(c in df.columns for c in ["loan_amount", "income_annum"]):
        df["loan_to_income_ratio"] = (
            df["loan_amount"] / df["income_annum"].replace(0, np.nan)
        )

    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df


@st.cache_resource
def train_model(df):
    target = "loan_status"

    X = df.drop(columns=[target])
    y = df[target].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=20,
        criterion="gini",
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_prob)
    }

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(
        y_test, y_pred,
        target_names=["Rejected", "Approved"],
        zero_division=0
    )

    return model, X.columns.tolist(), metrics, cm, report


# ---------------- MAIN APP ----------------

if not os.path.exists(DATA_FILE):
    st.error("loan_approval_dataset.csv was not found.")
    st.info(
        "Place loan_approval_dataset.csv in the same folder as app.py and restart Streamlit."
    )
    st.stop()

try:
    df = load_data(DATA_FILE)
    model, feature_names, metrics, cm, report = train_model(df)
except Exception as e:
    st.error("The application could not load/train the model.")
    st.exception(e)
    st.stop()

st.success("Model loaded successfully!")

# Sidebar
st.sidebar.header("Model Information")
st.sidebar.write("**Algorithm:** Random Forest Classifier")
st.sidebar.write("**Train/Test Split:** 80% / 20%")
st.sidebar.write("**Random State:** 42")
st.sidebar.write("**Trees:** 150")
st.sidebar.write("**Maximum Depth:** 20")

# Dataset summary
st.header("📊 Dataset & Model Performance")

c1, c2, c3 = st.columns(3)
c1.metric("Applications", f"{len(df):,}")
c2.metric("Input Features", len(feature_names))
c3.metric("Test Samples", f"{int(len(df) * 0.20):,}")

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Accuracy", f"{metrics['Accuracy']:.2%}")
m2.metric("Precision", f"{metrics['Precision']:.2%}")
m3.metric("Recall", f"{metrics['Recall']:.2%}")
m4.metric("F1 Score", f"{metrics['F1 Score']:.2%}")
m5.metric("ROC-AUC", f"{metrics['ROC-AUC']:.3f}")

# Tabs
tab1, tab2, tab3 = st.tabs([
    "🔮 Predict Loan",
    "📈 Model Analysis",
    "📋 Dataset"
])

with tab1:
    st.header("Enter Applicant Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        no_of_dependents = st.number_input(
            "Number of Dependents", min_value=0, max_value=20, value=2
        )
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self Employed", ["Yes", "No"])
        income_annum = st.number_input(
            "Annual Income", min_value=0.0, value=5000000.0, step=100000.0
        )

    with col2:
        loan_amount = st.number_input(
            "Loan Amount", min_value=0.0, value=20000000.0, step=500000.0
        )
        loan_term = st.number_input(
            "Loan Term (years)", min_value=1, max_value=50, value=10
        )
        cibil_score = st.number_input(
            "CIBIL Score", min_value=300, max_value=900, value=750
        )

    with col3:
        residential_assets_value = st.number_input(
            "Residential Assets Value", min_value=0.0, value=5000000.0, step=500000.0
        )
        commercial_assets_value = st.number_input(
            "Commercial Assets Value", min_value=0.0, value=3000000.0, step=500000.0
        )
        luxury_assets_value = st.number_input(
            "Luxury Assets Value", min_value=0.0, value=2000000.0, step=500000.0
        )
        bank_asset_value = st.number_input(
            "Bank Asset Value", min_value=0.0, value=3000000.0, step=500000.0
        )

    if st.button("🔍 Predict Loan Status", type="primary", use_container_width=True):
        total_assets = (
            residential_assets_value
            + commercial_assets_value
            + luxury_assets_value
            + bank_asset_value
        )

        asset_to_loan = total_assets / loan_amount if loan_amount != 0 else 0
        loan_to_income = loan_amount / income_annum if income_annum != 0 else 0

        applicant = pd.DataFrame([{
            "no_of_dependents": no_of_dependents,
            "education": 1 if education == "Graduate" else 0,
            "self_employed": 1 if self_employed == "Yes" else 0,
            "income_annum": income_annum,
            "loan_amount": loan_amount,
            "loan_term": loan_term,
            "cibil_score": cibil_score,
            "residential_assets_value": residential_assets_value,
            "commercial_assets_value": commercial_assets_value,
            "luxury_assets_value": luxury_assets_value,
            "bank_asset_value": bank_asset_value,
            "total_assets_value": total_assets,
            "asset_to_loan_ratio": asset_to_loan,
            "loan_to_income_ratio": loan_to_income
        }])

        applicant = applicant.reindex(columns=feature_names, fill_value=0)

        prediction = int(model.predict(applicant)[0])
        probability = model.predict_proba(applicant)[0]

        if prediction == 1:
            st.success("### ✅ LOAN APPROVED")
        else:
            st.error("### ❌ LOAN REJECTED")

        p1, p2 = st.columns(2)
        p1.metric("Approval Probability", f"{probability[1]:.2%}")
        p2.metric("Rejection Probability", f"{probability[0]:.2%}")

with tab2:
    st.header("Confusion Matrix")

    fig, ax = plt.subplots()
    ax.imshow(cm)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Rejected", "Approved"])
    ax.set_yticklabels(["Rejected", "Approved"])

    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    st.pyplot(fig)
    plt.close(fig)

    st.header("Feature Importance")

    importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)

    st.bar_chart(
        importance.set_index("Feature")["Importance"].head(10)
    )

    st.dataframe(
        importance.style.format({"Importance": "{:.4f}"}),
        use_container_width=True
    )

    st.header("Classification Report")
    st.code(report)

with tab3:
    st.header("Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)

    st.write("### Dataset Shape")
    st.write(f"Rows: **{df.shape[0]:,}**")
    st.write(f"Columns: **{df.shape[1]}**")

st.divider()
st.caption(
    "Educational ML project — predictions are model outputs and should not be used "
    "as the sole basis for real-world lending decisions."
)
