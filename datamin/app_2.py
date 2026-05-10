import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve, auc
)

# ── Page configuration ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="assets/heart_icon.png" if False else None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Minimal custom styling ───────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 16px 20px;
        text-align: center;
    }
    .metric-label { font-size: 13px; color: #6c757d; margin-bottom: 4px; }
    .metric-value { font-size: 26px; font-weight: 700; color: #212529; }
    .section-divider { border-top: 1px solid #dee2e6; margin: 24px 0; }
    h1, h2, h3 { font-family: 'Segoe UI', sans-serif; }
</style>
""", unsafe_allow_html=True)


# ── Data loading & preprocessing ────────────────────────────────────────────
@st.cache_data
def load_and_preprocess(path: str):
    df = pd.read_csv(path)

    df["Sex"] = df["Sex"].map({"M": 1, "F": 0})
    df["ExerciseAngina"] = df["ExerciseAngina"].map({"Y": 1, "N": 0})
    df = pd.get_dummies(
        df,
        columns=["ChestPainType", "RestingECG", "ST_Slope"],
        drop_first=True,
    )

    numerical_cols = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
    for col in numerical_cols:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        df[col] = df[col].clip(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)

    return df, numerical_cols


@st.cache_data
def train_models(df: pd.DataFrame, numerical_cols: list):
    X = df.drop("HeartDisease", axis=1)
    y = df["HeartDisease"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=100, stratify=y
    )

    scaler = StandardScaler()
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Support Vector Machine": SVC(probability=True),
    }

    results = []
    trained = {}
    roc_data = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)

        results.append({
            "Model": name,
            "Accuracy": round(accuracy_score(y_test, y_pred), 4),
            "Precision": round(precision_score(y_test, y_pred), 4),
            "Recall": round(recall_score(y_test, y_pred), 4),
            "F1 Score": round(f1_score(y_test, y_pred), 4),
            "AUC": round(roc_auc, 4),
        })
        trained[name] = (model, y_pred, y_prob)
        roc_data[name] = (fpr, tpr, roc_auc)

    df_results = pd.DataFrame(results).sort_values("F1 Score", ascending=False).reset_index(drop=True)
    return df_results, trained, roc_data, X_test, y_test, scaler, X.columns.tolist()


# ── Load data ────────────────────────────────────────────────────────────────
DATA_PATH = "heart.csv"

try:
    raw_df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    st.error("heart.csv not found. Please place it in the same directory as app.py.")
    st.stop()

df, numerical_cols = load_and_preprocess(DATA_PATH)
df_results, trained_models, roc_data, X_test, y_test, scaler, feature_names = train_models(df, numerical_cols)

best_model_name = df_results.iloc[0]["Model"]


# ── Sidebar navigation ───────────────────────────────────────────────────────
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Overview", "Exploratory Analysis", "Model Evaluation", "Prediction"],
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset:** Heart Failure Prediction")
st.sidebar.markdown(f"**Samples:** {len(raw_df):,} | **Features:** {raw_df.shape[1] - 1}")
st.sidebar.markdown(f"**Best Model:** {best_model_name}")


# ════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.title("Heart Disease Prediction")
    st.markdown(
        "This application trains and evaluates several classification models on the "
        "**Heart Failure Prediction Dataset** (918 patients). Use the sidebar to navigate "
        "between sections."
    )

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.subheader("Dataset Summary")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Total Patients</div><div class="metric-value">{len(raw_df):,}</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Features</div><div class="metric-value">{raw_df.shape[1] - 1}</div></div>', unsafe_allow_html=True)
    with col3:
        positive = int(raw_df["HeartDisease"].sum())
        st.markdown(f'<div class="metric-card"><div class="metric-label">Heart Disease Cases</div><div class="metric-value">{positive}</div></div>', unsafe_allow_html=True)
    with col4:
        negative = len(raw_df) - positive
        st.markdown(f'<div class="metric-card"><div class="metric-label">Healthy Cases</div><div class="metric-value">{negative}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.subheader("Raw Data Preview")
    st.dataframe(raw_df.head(10), use_container_width=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.subheader("Descriptive Statistics")
    st.dataframe(raw_df.describe().round(2), use_container_width=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.subheader("Pipeline Overview")
    steps = [
        ("1. Data Loading", "Read heart.csv with 918 rows and 12 columns."),
        ("2. Encoding", "Binary mapping for Sex and ExerciseAngina; one-hot encoding for ChestPainType, RestingECG, and ST_Slope."),
        ("3. Outlier Clipping", "IQR-based clipping on 5 numerical features."),
        ("4. Train / Test Split", "80 / 20 stratified split (random_state=100)."),
        ("5. Scaling", "StandardScaler applied to numerical columns."),
        ("6. Model Training", "5 classifiers trained and evaluated."),
    ]
    for title, desc in steps:
        st.markdown(f"**{title}** — {desc}")


# ════════════════════════════════════════════════════════════════════════════
# PAGE 2 — EXPLORATORY ANALYSIS
# ════════════════════════════════════════════════════════════════════════════
elif page == "Exploratory Analysis":
    st.title("Exploratory Data Analysis")

    # Target distribution
    st.subheader("Target Distribution")
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = raw_df["HeartDisease"].value_counts()
    bars = ax.bar(["No Disease", "Heart Disease"], counts.values, color=["#4C72B0", "#DD8452"], width=0.5)
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 5,
            str(int(bar.get_height())),
            ha="center", va="bottom", fontsize=11,
        )
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Heart Disease")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    st.pyplot(fig)
    plt.close()

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Sex breakdown
    st.subheader("Heart Disease by Sex")
    fig, ax = plt.subplots(figsize=(6, 4))
    sex_counts = raw_df.groupby(["Sex", "HeartDisease"]).size().unstack(fill_value=0)
    sex_counts.index = ["Female", "Male"]
    sex_counts.columns = ["No Disease", "Heart Disease"]
    sex_counts.plot(kind="bar", ax=ax, color=["#4C72B0", "#DD8452"], width=0.5)
    ax.set_xlabel("")
    ax.set_ylabel("Count")
    ax.set_title("Heart Disease by Sex")
    ax.legend(title="")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xticks(rotation=0)
    st.pyplot(fig)
    plt.close()

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Numerical feature distribution
    st.subheader("Numerical Feature Distributions")
    num_feature = st.selectbox(
        "Select a feature to visualize",
        ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"],
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    for val, label, color in [(0, "No Disease", "#4C72B0"), (1, "Heart Disease", "#DD8452")]:
        subset = raw_df[raw_df["HeartDisease"] == val][num_feature]
        ax.hist(subset, bins=25, alpha=0.6, label=label, color=color)
    ax.set_xlabel(num_feature)
    ax.set_ylabel("Frequency")
    ax.set_title(f"Distribution of {num_feature} by Heart Disease Status")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    st.pyplot(fig)
    plt.close()

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(df.corr().round(2), annot=True, fmt=".2f", cmap="coolwarm", ax=ax, linewidths=0.5)
    ax.set_title("Feature Correlation Matrix")
    st.pyplot(fig)
    plt.close()

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Feature correlation with target
    st.subheader("Feature Correlation with Heart Disease")
    corr_target = df.corr()["HeartDisease"].drop("HeartDisease").sort_values()
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#DD8452" if v > 0 else "#4C72B0" for v in corr_target.values]
    ax.barh(corr_target.index, corr_target.values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Correlation Coefficient")
    ax.set_title("Feature Correlation with Heart Disease")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    st.pyplot(fig)
    plt.close()


# ════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL EVALUATION
# ════════════════════════════════════════════════════════════════════════════
elif page == "Model Evaluation":
    st.title("Model Evaluation")

    # Leaderboard
    st.subheader("Performance Leaderboard")
    st.dataframe(
        df_results.style.highlight_max(
            subset=["Accuracy", "Precision", "Recall", "F1 Score", "AUC"],
            color="#d4edda",
        ).format("{:.4f}", subset=["Accuracy", "Precision", "Recall", "F1 Score", "AUC"]),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ROC Curves
    st.subheader("ROC Curves")
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, (fpr, tpr, roc_auc) in roc_data.items():
        ax.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Models")
    ax.legend(loc="lower right", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    st.pyplot(fig)
    plt.close()

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Confusion matrix per model
    st.subheader("Confusion Matrix")
    model_choice = st.selectbox("Select a model", list(trained_models.keys()))
    _, y_pred, _ = trained_models[model_choice]

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=["No Disease", "Heart Disease"],
        yticklabels=["No Disease", "Heart Disease"],
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_choice}")
    st.pyplot(fig)
    plt.close()

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Metric bar chart
    st.subheader("Metric Comparison Across Models")
    metric = st.selectbox("Select a metric", ["Accuracy", "Precision", "Recall", "F1 Score", "AUC"])
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(df_results["Model"], df_results[metric], color="#4C72B0", width=0.5)
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{bar.get_height():.3f}",
            ha="center", va="bottom", fontsize=9,
        )
    ax.set_ylim(0, 1.1)
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} by Model")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xticks(rotation=15, ha="right")
    st.pyplot(fig)
    plt.close()


# ════════════════════════════════════════════════════════════════════════════
# PAGE 4 — PREDICTION
# ════════════════════════════════════════════════════════════════════════════
elif page == "Prediction":
    st.title("Patient Prediction")
    st.markdown(
        "Enter patient details below. The selected model will predict whether the patient "
        "is likely to have heart disease."
    )

    selected_model_name = st.selectbox("Select Model", list(trained_models.keys()), index=0)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.subheader("Patient Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider("Age", 20, 80, 50)
        sex = st.selectbox("Sex", ["Male", "Female"])
        chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
        resting_bp = st.slider("Resting Blood Pressure (mm Hg)", 80, 200, 130)

    with col2:
        cholesterol = st.slider("Cholesterol (mg/dL)", 100, 600, 200)
        fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", ["No", "Yes"])
        resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
        max_hr = st.slider("Maximum Heart Rate", 60, 210, 150)

    with col3:
        exercise_angina = st.selectbox("Exercise-Induced Angina", ["No", "Yes"])
        oldpeak = st.slider("Oldpeak (ST depression)", -3.0, 7.0, 0.0, step=0.1)
        st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

    # Build input row
    input_data = {
        "Age": age,
        "Sex": 1 if sex == "Male" else 0,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": 1 if fasting_bs == "Yes" else 0,
        "MaxHR": max_hr,
        "ExerciseAngina": 1 if exercise_angina == "Yes" else 0,
        "Oldpeak": oldpeak,
        "ChestPainType_NAP": 1 if chest_pain == "NAP" else 0,
        "ChestPainType_ASY": 1 if chest_pain == "ASY" else 0,
        "ChestPainType_TA":  1 if chest_pain == "TA" else 0,
        "RestingECG_Normal": 1 if resting_ecg == "Normal" else 0,
        "RestingECG_ST":     1 if resting_ecg == "ST" else 0,
        "ST_Slope_Flat":     1 if st_slope == "Flat" else 0,
        "ST_Slope_Up":       1 if st_slope == "Up" else 0,
    }

    # Align to training feature order
    input_df = pd.DataFrame([input_data])
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[feature_names]

    # Scale numerical columns
    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

    if st.button("Run Prediction", type="primary"):
        model_obj, _, _ = trained_models[selected_model_name]
        prediction = model_obj.predict(input_df)[0]
        probability = model_obj.predict_proba(input_df)[0][1]

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        with col_a:
            if prediction == 1:
                st.error("Prediction: Heart Disease Detected")
            else:
                st.success("Prediction: No Heart Disease Detected")

        with col_b:
            st.metric("Probability of Heart Disease", f"{probability:.1%}")

        st.markdown(
            f"*Model used: **{selected_model_name}** — "
            f"F1 Score on test set: "
            f"{df_results.loc[df_results['Model'] == selected_model_name, 'F1 Score'].values[0]:.4f}*"
        )
