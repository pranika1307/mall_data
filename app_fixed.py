import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="🛍️ Mall Feedback Sentiment", page_icon="🛍️", layout="centered")

st.title("🛍️ Mall Feedback Sentiment Analysis")
st.caption("Loads a single scikit-learn Pipeline artifact (vectorizer + scaler + model).")

PIPELINE_PATH = "sentiment_pipeline.pkl"
FEEDBACK_FILE = "user_feedback.csv"

@st.cache_resource(show_spinner=False)
def load_pipeline(path):
    return joblib.load(path)

try:
    pipe = load_pipeline(PIPELINE_PATH)
    st.success("Model pipeline loaded.")
except Exception as e:
    st.error(f"Failed to load pipeline: {e}")
    st.stop()

st.subheader("Enter Customer Details")
name = st.text_input("Name")
email = st.text_input("Email")
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=18, max_value=100, step=1)
rating = st.slider("Rating", 1, 5, 3)
feedback = st.text_area("Write your feedback here...")

if st.button("Analyze Feedback"):
    if not feedback.strip():
        st.warning("Please write some feedback text first.")
        st.stop()

    gender_val = 0 if gender == "Male" else 1

    X = pd.DataFrame([{
        "feedback": feedback,
        "age": age,
        "gender": gender_val,
        "rating": rating
    }])

    try:
        pred = pipe.predict(X)[0]
        proba = getattr(pipe, "predict_proba", lambda X: [[None, None]])(X)[0]
        sentiment = "Positive ✅" if int(pred) == 1 else "Negative ❌"
        st.subheader("Prediction Result")
        st.success(f"Sentiment: {sentiment}")
        if proba[0] is not None:
            st.caption(f"Confidence (class 1): {proba[1]:.2f}")
    except Exception as e:
        st.error(f"Inference failed: {e}")
        st.stop()

    # Save feedback to CSV
    new_row = {
        "Name": name,
        "Email": email,
        "Gender": gender,
        "Age": age,
        "Rating": rating,
        "Feedback": feedback,
        "Predicted_Sentiment": sentiment
    }
    try:
        old = pd.read_csv(FEEDBACK_FILE)
        df = pd.concat([old, pd.DataFrame([new_row])], ignore_index=True)
    except FileNotFoundError:
        df = pd.DataFrame([new_row])

    df.to_csv(FEEDBACK_FILE, index=False)
    st.info("✅ Feedback saved successfully!")

if st.checkbox("Show Reports"):
    try:
        data = pd.read_csv(FEEDBACK_FILE)
        st.write("### Feedback Data")
        st.dataframe(data.tail())

        st.write("### Sentiment Distribution")
        fig, ax = plt.subplots()
        data["Predicted_Sentiment"].value_counts().plot(kind="bar", ax=ax)
        st.pyplot(fig)

        st.write("### Ratings Distribution")
        fig, ax = plt.subplots()
        data["Rating"].value_counts().sort_index().plot(kind="bar", ax=ax)
        st.pyplot(fig)
    except FileNotFoundError:
        st.warning("No feedback data available yet. Submit some feedback first.")
