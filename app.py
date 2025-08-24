{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "55e2b3f3-b3ff-4582-93aa-c759bf66fbd8",
   "metadata": {},
   "source": [
    "### PART-1: Set Up Streamlit App"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "c1cb4ce8-b09e-4dfb-968e-ce5afe3b0edb",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "import pickle\n",
    "import joblib\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f15fd954-9bfc-4b21-a70e-0a7b3c746405",
   "metadata": {},
   "source": [
    "### PART-2: Build Input Form"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "9814f8e8-0af9-4263-9dd2-f4be7bdfeb96",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "DeltaGenerator()"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "st.title(\"🛍️ Mall Feedback Sentiment Analysis\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "c4a9d483-8e94-4a18-b870-2e450f24b7a4",
   "metadata": {},
   "outputs": [],
   "source": [
    "st.subheader(\"Enter Customer Details\")\n",
    "name = st.text_input(\"Name\")\n",
    "email = st.text_input(\"Email\")\n",
    "gender = st.selectbox(\"Gender\", [\"Male\", \"Female\"])\n",
    "age = st.number_input(\"Age\", min_value=18, max_value=100, step=1)\n",
    "rating = st.slider(\"Rating\", 1, 5, 3)\n",
    "feedback = st.text_area(\"Write your feedback here...\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "bfd0c8fc-31af-4809-b6d4-7ecdb685ff1f",
   "metadata": {},
   "source": [
    "### PART-3: Load Trained Model & Vectorizer"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "60f9f5db-a0cb-4a7e-aea9-3583253d1a5a",
   "metadata": {},
   "outputs": [],
   "source": [
    "log_reg = joblib.load(\"logistic_model.pkl\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "f1dba7d4-6679-40c2-876f-00a90cf9a3d5",
   "metadata": {},
   "outputs": [],
   "source": [
    "vectorizer = joblib.load(\"tfidf_vectorizer.pkl\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "3ba5a6aa-2c47-4604-8724-6e2528d1aec9",
   "metadata": {},
   "outputs": [],
   "source": [
    "scaler = joblib.load(\"scaler.pkl\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "4725d29e-627f-4c9d-9de2-6b0354bd885a",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Storage for feedback (CSV)\n",
    "FEEDBACK_FILE = \"user_feedback.csv\""
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e61e9748-7e78-4c3c-8311-89f99af1ac63",
   "metadata": {},
   "source": [
    "### PART-4: Make Predictions"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "96d2a03d-536f-4500-b2b7-979c2b466bf2",
   "metadata": {},
   "outputs": [],
   "source": [
    "if st.button(\"Analyze Feedback\"):\n",
    "    # Encode Gender\n",
    "    gender_val = 0 if gender == \"Male\" else 1\n",
    "\n",
    "    # TF-IDF for feedback\n",
    "    text_features = vectorizer.transform([feedback]).toarray()\n",
    "\n",
    "    # Combine with numeric\n",
    "    numeric_features = np.array([[age, gender_val, rating]])\n",
    "    final_features = np.hstack((text_features, numeric_features))\n",
    "\n",
    "    # Scale numeric + text\n",
    "    final_scaled = scaler.transform(final_features)\n",
    "\n",
    "    # Predict sentiment\n",
    "    prediction = log_reg.predict(final_scaled)[0]\n",
    "    sentiment = \"Positive ✅\" if prediction == 1 else \"Negative ❌\"\n",
    "\n",
    "    st.subheader(\"Prediction Result\")\n",
    "    st.success(f\"Sentiment: {sentiment}\")\n",
    "\n",
    "    # Save feedback to CSV\n",
    "    new_data = pd.DataFrame([{\n",
    "        \"Name\": name,\n",
    "        \"Email\": email,\n",
    "        \"Gender\": gender,\n",
    "        \"Age\": age,\n",
    "        \"Rating\": rating,\n",
    "        \"Feedback\": feedback,\n",
    "        \"Predicted_Sentiment\": sentiment\n",
    "    }])\n",
    "\n",
    "    try:\n",
    "        old_data = pd.read_csv(FEEDBACK_FILE)\n",
    "        updated_data = pd.concat([old_data, new_data], ignore_index=True)\n",
    "    except FileNotFoundError:\n",
    "        updated_data = new_data\n",
    "\n",
    "    updated_data.to_csv(FEEDBACK_FILE, index=False)\n",
    "    st.info(\"✅ Feedback saved successfully!\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8c3fc210-58c2-45d5-91ed-806b768046e4",
   "metadata": {},
   "source": [
    "### PART-5: Generate Reports"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "2d907550-f0a7-4315-91df-c45793831e1c",
   "metadata": {},
   "outputs": [],
   "source": [
    "st.subheader(\"📊 Feedback Reports\")\n",
    "\n",
    "if st.checkbox(\"Show Reports\"):\n",
    "    try:\n",
    "        data = pd.read_csv(FEEDBACK_FILE)\n",
    "\n",
    "        st.write(\"### Feedback Data\")\n",
    "        st.dataframe(data.tail())\n",
    "\n",
    "        st.write(\"### Sentiment Distribution\")\n",
    "        fig, ax = plt.subplots()\n",
    "        data[\"Predicted_Sentiment\"].value_counts().plot(kind=\"bar\", color=[\"green\",\"red\"], ax=ax)\n",
    "        st.pyplot(fig)\n",
    "\n",
    "        st.write(\"### Ratings Distribution\")\n",
    "        fig, ax = plt.subplots()\n",
    "        sns.countplot(x=\"Rating\", data=data, palette=\"viridis\", ax=ax)\n",
    "        st.pyplot(fig)\n",
    "\n",
    "    except FileNotFoundError:\n",
    "        st.warning(\"No feedback data available yet. Submit some feedback first.\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "79d28082-d8b3-414a-ba49-942993591e4c",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "06bbd5cf-a185-4c6e-96ab-501f02198fc9",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3c900a89-08fc-4954-9317-7b1ef2a2a443",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "df244115-bbe9-4e50-b810-fc6c5b688988",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "99a938cf-2829-46b1-b49e-7c6617921df6",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9b6fdba5-7eb4-4f08-a09a-3672596032ed",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5bd00271-cc82-4698-a06d-fc62ea326330",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d1967c68-9d2f-4f1f-935e-b68300c1f053",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5bbb657d-ae69-43f0-a13d-c2b82fe04074",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d919b6c4-279f-47ef-84dd-73eaa39e54c6",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8f2d3113-113e-4d0c-ad53-120bcb986eda",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c04556a2-3918-4daa-9ad1-a5379dc1f142",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "717b5f02-9d38-417f-af65-3c19ff25746a",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "64d81774-687f-4d88-93a0-8be9ea33f2d5",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "56a02f91-5001-43fa-ac8f-3b08cdeec72b",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "87be144c-87bf-46d6-bf48-5f89fa8c670e",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "60c735d1-dff4-4704-86d6-a3b1e1e1c472",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a753704d-ff5e-4f0c-9037-54d0fe92fcea",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9e38341e-4004-4ee5-9a8d-428c7d7062f3",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "33e907f8-469d-4eb2-a1dc-ebe29d660c6e",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "549cc80a-e40d-40ee-aa78-be48be9786bb",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "63bd180f-621e-4f66-8535-d0eb391c9fb5",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3ccde9ad-05f8-455e-a784-d2dada6e1433",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a7f9e7ad-3968-4792-bb2f-dc59ba20cba5",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "45c8e816-0de6-42ae-b398-923764dd4394",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a2603af9-b944-41d1-bda7-33ef4e102869",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "fce8fde4-4ec4-448c-90b6-33bd25f74159",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "52654308-7312-4b1f-8cd4-d0f46e2c03e0",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "fd2b4203-ce95-4aac-afbf-6dbf92ff755a",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a617c4f6-e7c8-4880-b650-78835925b9c8",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "76f6fa18-a0d6-4c05-8cc7-3c036eabe3b7",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "60ad94d8-558f-47a1-9bbb-ab85d3f06329",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1f1e4246-ca23-4bca-b279-24e89e29c24e",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d5a44572-fbcf-4dbb-b53f-a4897fc8624a",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "813f8771-7d91-4396-841c-ca2bfce606f9",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8d592647-931d-4540-8949-8a2d13067f15",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [conda env:base] *",
   "language": "python",
   "name": "conda-base-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
