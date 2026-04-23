import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline

# ── Page config ──────────────────────────────────────────
st.set_page_config(
    page_title="Sleep Health Dashboard",
    page_icon="😴",
    layout="wide"
)

# ── Load data ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("Sleep_health_and_lifestyle_dataset.csv")
    df.columns = df.columns.str.replace(" ", "_")
    df["Sleep_Disorder"] = df["Sleep_Disorder"].fillna("None")
    df[["BP_Systolic", "BP_Diastolic"]] = df["Blood_Pressure"].str.split("/", expand=True).astype(int)
    df = df.drop(columns=["Person_ID", "Blood_Pressure"])
    return df

df = load_data()

# ── Train model ───────────────────────────────────────────
@st.cache_resource
def train_model(df):
    features = ["Stress_Level", "Heart_Rate", 
                "Sleep_Duration", "Age", "Daily_Steps"]
    le = LabelEncoder()
    X = df[features]
    y = le.fit_transform(df["Sleep_Disorder"])
    
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=5000, random_state=42,
                                      class_weight="balanced"))
    ])
    pipeline.fit(X, y)
    return pipeline, le, features

model, le, features = train_model(df)

# ── Navigation ────────────────────────────────────────────
st.sidebar.title("😴 Sleep Health")
page = st.sidebar.radio("Navigate", ["Overview", "Predict", "About"])

if page == "Overview":
    st.title("Sleep Health & Lifestyle Analysis")
    st.write("Exploring patterns in sleep disorders across 374 individuals.")

    # ── Sidebar filters ───────────────────────────────────
    st.sidebar.divider()
    st.sidebar.subheader("Filters")
    selected_occupation = st.sidebar.multiselect(
        "Occupation",
        options=df["Occupation"].unique(),
        default=df["Occupation"].unique()
    )
    selected_gender = st.sidebar.multiselect(
        "Gender",
        options=df["Gender"].unique(),
        default=df["Gender"].unique()
    )

    filtered_df = df[
        (df["Occupation"].isin(selected_occupation)) &
        (df["Gender"].isin(selected_gender))
    ]

    if len(filtered_df) == 0:
        st.warning("No data matches your filters. Please adjust the sidebar.")
        st.stop()

    # ── KPI cards ─────────────────────────────────────────
    st.subheader("Dataset Overview")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Records", len(filtered_df))
    k2.metric("Avg Sleep Duration", f"{filtered_df['Sleep_Duration'].mean():.1f}h")
    k3.metric("Avg Stress Level", f"{filtered_df['Stress_Level'].mean():.1f}")
    k4.metric("Disorder Rate", f"{(filtered_df['Sleep_Disorder'] != 'None').mean()*100:.1f}%")

    st.divider()

    # ── Charts ────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Stress Level vs Sleep Duration")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=filtered_df, x="Stress_Level", 
                    y="Sleep_Duration", ax=ax)
        ax.set_xlabel("Stress Level")
        ax.set_ylabel("Sleep Duration (hours)")
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Sleep Disorders by Occupation")
        fig, ax = plt.subplots(figsize=(6, 4))
        disorder_counts = filtered_df.groupby(
            ["Occupation", "Sleep_Disorder"]).size().unstack(fill_value=0)
        disorder_counts.plot(kind="bar", stacked=True, ax=ax,
                             color=["#2ecc71", "#e74c3c", "#3498db"])
        ax.set_xlabel("Occupation")
        ax.set_ylabel("Count")
        plt.xticks(rotation=45, ha="right")
        ax.legend(title="Sleep Disorder", bbox_to_anchor=(1.05, 1))
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.divider()

    # ── Correlation heatmap ───────────────────────────────
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    corr = filtered_df.select_dtypes(include="number").corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn",
                center=0, vmin=-1, vmax=1, ax=ax)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    
elif page == "Predict":
    st.title("Predict Your Sleep Disorder Risk")
    st.write("Enter your lifestyle details below and the model will predict your sleep disorder risk.")

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 18, 60, 30)
        stress = st.slider("Stress Level (1–10)", 1, 10, 5)
        sleep_duration = st.slider("Sleep Duration (hours)", 4.0, 9.0, 7.0, step=0.1)

    with col2:
        heart_rate = st.slider("Resting Heart Rate (bpm)", 60, 100, 72)
        daily_steps = st.slider("Daily Steps", 1000, 15000, 7000, step=500)

    # Predict button
    if st.button("Predict"):
        input_data = pd.DataFrame([[stress, heart_rate, sleep_duration, 
                                     age, daily_steps]],
                                   columns=features)
        
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        predicted_label = le.inverse_transform([prediction])[0]

        st.divider()

        if predicted_label == "None":
            st.success(f"Good news — no sleep disorder detected.")
        elif predicted_label == "Insomnia":
            st.warning(f"Risk detected: Insomnia")
        else:
            st.warning(f"Risk detected: Sleep Apnea")

        st.subheader("Prediction Probabilities")
        prob_df = pd.DataFrame({
            "Disorder": le.classes_,
            "Probability": probabilities
        }).sort_values("Probability", ascending=False)

        fig, ax = plt.subplots(figsize=(6, 3))
        colors = ["#2ecc71" if d == "None" else "#e74c3c" 
                  for d in prob_df["Disorder"]]
        ax.barh(prob_df["Disorder"], prob_df["Probability"], color=colors)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Probability")
        for i, (val, name) in enumerate(zip(prob_df["Probability"], 
                                             prob_df["Disorder"])):
            ax.text(val + 0.01, i, f"{val:.1%}", va="center")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

elif page == "About":
    st.title("About This Project")

    st.markdown("""
    ## Sleep Health & Lifestyle Dashboard
    
    This dashboard analyses the relationship between lifestyle factors 
    and sleep disorders across 374 individuals.
    
    ### Dataset
    - **Source:** Sleep Health and Lifestyle Dataset (Kaggle)
    - **Records:** 374 individuals
    - **Features:** Sleep duration, stress level, BMI, occupation, 
      heart rate, daily steps, blood pressure
    - **Target:** Sleep disorder classification (None / Insomnia / Sleep Apnea)
    
    ### Methodology
    - **Exploratory Data Analysis** — distribution plots, correlation heatmap, 
      occupation breakdown
    - **Linear Regression** — predicting sleep quality score from lifestyle features
    - **Logistic Regression** — classifying sleep disorder type
    - **Class imbalance** handled with class_weight="balanced"
    - **Model validation** via 5-fold stratified cross-validation (F1 macro = 0.81)
    - **Pipeline** used to prevent data leakage during cross-validation
    
    ### Key Findings
    - Stress level is the strongest predictor of sleep quality (r = -0.90)
    - Nurses show highest Sleep Apnea rates despite moderate stress
    - Salespersons and Teachers are most affected by Insomnia
    - Heart rate is the second strongest sleep predictor after stress
    
    ### Tech Stack
    - Python, pandas, numpy
    - scikit-learn, imbalanced-learn
    - matplotlib, seaborn
    - Streamlit
    
    ### Limitations
    - Dataset is relatively small (374 rows)
    - Cross-validation F1 = 0.81 on synthetic data — 
      real-world performance may differ
    - Predictions are indicative only, not medical advice
    """)

    st.info("Built as a portfolio project to demonstrate end-to-end data analysis and ML deployment skills.")
    
