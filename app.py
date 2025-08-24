import streamlit as st
import pickle
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
st.title("🛍️ Mall Feedback Sentiment Analysis")
st.subheader("Enter Customer Details")
name = st.text_input("Name")
email = st.text_input("Email")
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=18, max_value=100, step=1)
rating = st.slider("Rating", 1, 5, 3)
feedback = st.text_area("Write your feedback here...")
log_reg = joblib.load("logistic_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
scaler = joblib.load("scaler.pkl")# Storage for feedback (CSV)
FEEDBACK_FILE = "user_feedback.csv"
if st.button("Analyze Feedback"):
    # Encode Gender
    gender_val = 0 if gender == "Male" else 1

    # TF-IDF for feedback
    text_features = vectorizer.transform([feedback]).toarray()

    # Combine with numeric
    numeric_features = np.array([[age, gender_val, rating]])
    final_features = np.hstack((text_features, numeric_features))

    # Scale numeric + text
    final_scaled = scaler.transform(final_features)

    # Predict sentiment
    prediction = log_reg.predict(final_scaled)[0]
    sentiment = "Positive ✅" if prediction == 1 else "Negative ❌"

    st.subheader("Prediction Result")
    st.success(f"Sentiment: {sentiment}")

    # Save feedback to CSV
    new_data = pd.DataFrame([{
        "Name": name,
        "Email": email,
        "Gender": gender,
        "Age": age,
        "Rating": rating,
        "Feedback": feedback,
        "Predicted_Sentiment": sentiment
    }])

    try:
        old_data = pd.read_csv(FEEDBACK_FILE)
        updated_data = pd.concat([old_data, new_data], ignore_index=True)
    except FileNotFoundError:
        updated_data = new_data

    updated_data.to_csv(FEEDBACK_FILE, index=False)
    st.info("✅ Feedback saved successfully!")
    st.subheader("📊 Feedback Reports")

if st.checkbox("Show Reports"):
    try:
        data = pd.read_csv(FEEDBACK_FILE)

        st.write("### Feedback Data")
        st.dataframe(data.tail())

        st.write("### Sentiment Distribution")
        fig, ax = plt.subplots()
        data["Predicted_Sentiment"].value_counts().plot(kind="bar", color=["green","red"], ax=ax)
        st.pyplot(fig)

        st.write("### Ratings Distribution")
        fig, ax = plt.subplots()
        sns.countplot(x="Rating", data=data, palette="viridis", ax=ax)
        st.pyplot(fig)

    except FileNotFoundError:
        st.warning("No feedback data available yet. Submit some feedback first.")