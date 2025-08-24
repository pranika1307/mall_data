import streamlit as st
import pickle, joblib
import pandas as pd
import numpy as np

# Load model & vectorizer
model = joblib.load("logistic_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
scaler = joblib.load("scaler.pkl")

FILE = "user_feedback.csv"

st.title("🛍️ Mall Feedback Sentiment Analyzer")

# --- User Form ---
name = st.text_input("Name")
gender = st.radio("Gender", ["Male", "Female"])
age = st.number_input("Age", 18, 100, 25)
feedback = st.text_area("Your Feedback")

# --- Predict Button ---
if st.button("Analyze Feedback"):
    gender_val = 0 if gender == "Male" else 1
    text_features = vectorizer.transform([feedback]).toarray()
    num_features = np.array([[age, gender_val]])
    features = np.hstack((text_features, num_features))
    features = scaler.transform(features)

    pred = model.predict(features)[0]
    sentiment = "Positive ✅" if pred == 1 else "Negative ❌"
    st.success(f"Sentiment: {sentiment}")

    # Save feedback
    entry = pd.DataFrame([[name, gender, age, feedback, sentiment]],
                         columns=["Name","Gender","Age","Feedback","Sentiment"])
    try:
        old = pd.read_csv(FILE)
        df = pd.concat([old, entry], ignore_index=True)
    except FileNotFoundError:
        df = entry
    df.to_csv(FILE, index=False)

# --- Reports ---
if st.checkbox("Show Reports"):
    try:
        data = pd.read_csv(FILE)
        st.write("Recent Feedbacks", data.tail())
        st.bar_chart(data["Sentiment"].value_counts())
    except FileNotFoundError:
        st.info("No feedback yet. Submit one above!")
