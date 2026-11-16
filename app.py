import os
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ---------------------------------------------------------
# HEART DISEASE / BLOOD PRESSURE CATEGORY PREDICTION APP
# Keep this file in the same folder as cardio_data_processed.csv
# ---------------------------------------------------------

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main {background-color: #f7f9fc;}
    .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
    .main-title {font-size: 2.2rem; font-weight: 800; color: #16324f;}
    .sub-title {color: #526477; font-size: 1rem;}
    div[data-testid="stMetric"] {
        background: white; padding: 14px; border-radius: 12px;
        border: 1px solid #e7edf4;
    }
    .stTabs [data-baseweb="tab-list"] {gap: 8px;}
</style>
""", unsafe_allow_html=True)

DATA_FILE = "cardio_data_processed.csv"

FEATURES = ["gender", "weight", "cholesterol", "smoke", "alco", "age_years"]
TARGET = "bp_category"
EXPECTED_CATEGORIES = [
    "Normal", "Elevated", "Hypertension Stage 1", "Hypertension Stage 2"
]

@st.cache_data(show_spinner=False)
def load_data():
    # Resolve CSV relative to this app.py file, not the terminal's current folder.
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, DATA_FILE)
    data = pd.read_csv(path)

    required = FEATURES + [TARGET]
    missing = [col for col in required if col not in data.columns]
    if missing:
        raise ValueError("CSV mein required columns missing hain: " + ", ".join(missing))

    data = data.copy()
    for col in FEATURES:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data[TARGET] = data[TARGET].astype(str).str.strip()
    data = data.dropna(subset=FEATURES + [TARGET])
    data = data[data[TARGET] != ""]
    return data

@st.cache_resource(show_spinner="Model train ho raha hai...")
def train_model(data):
    X = data[FEATURES]
    y = data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # The target is a category, so DecisionTreeClassifier is suitable here.
    model = DecisionTreeClassifier(
        max_depth=8,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    labels = sorted(y.unique().tolist())
    cm = confusion_matrix(y_test, predictions, labels=labels)
    report = classification_report(
        y_test, predictions, labels=labels, output_dict=True, zero_division=0
    )
    return model, accuracy, labels, cm, report, len(X_train), len(X_test)

try:
    data = load_data()
    model, accuracy, labels, cm, report, train_count, test_count = train_model(data)
except FileNotFoundError:
    st.error(
        "CSV file nahi mili. 'app.py' aur 'cardio_data_processed.csv' "
        "ko ek hi folder mein rakho, phir app dobara run karo."
    )
    st.stop()
except Exception as error:
    st.error("Project load karte waqt problem aayi: " + str(error))
    st.stop()

# Sidebar navigation
st.sidebar.title("❤️ HeartCare")
st.sidebar.caption("Heart Disease & BP Category Prediction")
page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Data Analysis", "Prediction", "Model Performance", "Dataset"]
)
st.sidebar.divider()
st.sidebar.caption("Dataset: cardio_data_processed.csv")
st.sidebar.caption("Model: Decision Tree Classifier")

# Header
st.markdown('<div class="main-title">Heart Disease Prediction System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Explore the dataset and estimate a blood pressure category '
    'using the trained machine learning model.</div>',
    unsafe_allow_html=True
)
st.write("")

if page == "Dashboard":
    st.subheader("Dashboard Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", f"{len(data):,}")
    c2.metric("Total Features", len(data.columns))
    c3.metric("BP Categories", data[TARGET].nunique())
    c4.metric("Model Accuracy", f"{accuracy * 100:.2f}%")

    st.divider()
    left, right = st.columns(2)
    with left:
        st.markdown("#### Blood Pressure Category Distribution")
        counts = data[TARGET].value_counts().reindex(
            [x for x in EXPECTED_CATEGORIES if x in data[TARGET].unique()]
        ).dropna()
        st.bar_chart(counts.to_frame(name="Count"))
    with right:
        st.markdown("#### Cardio Outcome Distribution")
        if "cardio" in data.columns:
            cardio_counts = data["cardio"].value_counts().sort_index()
            cardio_counts.index = ["No (0)" if x == 0 else "Yes (1)" for x in cardio_counts.index]
            st.bar_chart(cardio_counts)
        else:
            st.info("CSV mein cardio column available nahi hai.")

    st.markdown("#### Dataset Information")
    st.write(
        "The dataset contains patient attributes such as age, gender, weight, "
        "cholesterol, smoking and alcohol intake. The prediction section estimates "
        "the blood pressure category."
    )

elif page == "Data Analysis":
    st.subheader("Exploratory Data Analysis")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Age Distribution")
        st.bar_chart(data["age_years"].value_counts().sort_index())
    with col2:
        st.markdown("#### Cholesterol Level Distribution")
        st.bar_chart(data["cholesterol"].value_counts().sort_index())

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("#### Gender Distribution")
        gender_counts = data["gender"].value_counts().sort_index()
        gender_counts.index = [
            "Female (1)" if i == 1 else "Male (2)" if i == 2 else str(i)
            for i in gender_counts.index
        ]
        st.bar_chart(gender_counts)
    with col4:
        st.markdown("#### Blood Pressure Categories")
        st.bar_chart(data[TARGET].value_counts())

    st.markdown("#### Descriptive Statistics")
    st.dataframe(data[FEATURES + [TARGET]].describe(include="all").transpose(),
                 use_container_width=True)

    st.markdown("#### Average Weight by BP Category")
    st.bar_chart(data.groupby(TARGET)["weight"].mean().sort_values(ascending=False))

elif page == "Prediction":
    st.subheader("Blood Pressure Category Prediction")
    st.write("Patient details enter karo aur **Predict Category** par click karo.")
    with st.form("prediction_form"):
        a, b = st.columns(2)
        with a:
            gender_label = st.selectbox("Gender", ["Female", "Male"])
            weight = st.number_input("Weight (kg)", min_value=20.0, max_value=250.0,
                                     value=65.0, step=0.5)
            cholesterol = st.selectbox(
                "Cholesterol Level", [1, 2, 3],
                format_func=lambda x: {
                    1: "1 - Normal", 2: "2 - Above Normal", 3: "3 - Well Above Normal"
                }[x]
            )
        with b:
            age_years = st.number_input("Age (years)", min_value=18, max_value=100,
                                        value=45, step=1)
            smoke = st.selectbox("Smoking", [0, 1],
                                 format_func=lambda x: "No" if x == 0 else "Yes")
            alco = st.selectbox("Alcohol Intake", [0, 1],
                                format_func=lambda x: "No" if x == 0 else "Yes")

        submitted = st.form_submit_button("Predict Category", use_container_width=True)

    if submitted:
        gender_value = 1 if gender_label == "Female" else 2
        input_df = pd.DataFrame([{
            "gender": gender_value,
            "weight": float(weight),
            "cholesterol": int(cholesterol),
            "smoke": int(smoke),
            "alco": int(alco),
            "age_years": int(age_years)
        }], columns=FEATURES)

        predicted_category = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]
        confidence = float(np.max(probabilities)) * 100

        st.divider()
        st.markdown("### Prediction Result")
        st.success(f"Predicted Blood Pressure Category: **{predicted_category}**")
        st.metric("Model Confidence", f"{confidence:.1f}%")
        st.caption(
            "This is an educational machine-learning estimate, not a medical diagnosis. "
            "For health concerns, consult a qualified healthcare professional."
        )

elif page == "Model Performance":
    st.subheader("Model Performance")
    m1, m2, m3 = st.columns(3)
    m1.metric("Algorithm", "Decision Tree")
    m2.metric("Training Records", f"{train_count:,}")
    m3.metric("Testing Records", f"{test_count:,}")
    st.metric("Test Accuracy", f"{accuracy * 100:.2f}%")

    st.markdown("#### Confusion Matrix")
    cm_df = pd.DataFrame(
        cm,
        index=[f"Actual: {x}" for x in labels],
        columns=[f"Predicted: {x}" for x in labels]
    )
    st.dataframe(cm_df, use_container_width=True)

    st.markdown("#### Classification Report")
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.round(3), use_container_width=True)

    st.markdown("#### Feature Importance")
    importance = pd.Series(model.feature_importances_, index=FEATURES)
    importance = importance.sort_values(ascending=False)
    st.bar_chart(importance)

elif page == "Dataset":
    st.subheader("Dataset Preview")
    st.write(f"Showing {min(100, len(data))} records out of {len(data):,}.")
    st.dataframe(data.head(100), use_container_width=True)

    st.markdown("#### Dataset Summary")
    st.write({
        "Rows": int(data.shape[0]),
        "Columns": int(data.shape[1]),
        "Missing Values": int(data.isna().sum().sum()),
        "Duplicate Rows": int(data.duplicated().sum())
    })

    csv_bytes = data.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Processed Dataset",
        data=csv_bytes,
        file_name="cardio_data_processed.csv",
        mime="text/csv"
    )

st.sidebar.divider()
st.sidebar.caption("Academic project demonstration • Not for clinical use")
