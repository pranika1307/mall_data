"""
Training script for sentiment analysis pipeline.

It builds a single Pipeline:
- TF-IDF for "feedback" text
- StandardScaler for numeric features
- Logistic Regression classifier

Saves: sentiment_pipeline.pkl
"""

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# === Load your dataset ===
# Replace 'data.csv' with the actual filename
df = pd.read_csv("data.csv")

# Map gender text to numeric (if needed)
if df["gender"].dtype == object:
    df["gender"] = df["gender"].map({"Male": 0, "Female": 1})

# Features and label
X = df.drop(columns=["label"])
y = df["label"]

text_col = "feedback"
num_cols = ["age", "gender", "rating"]

# Preprocessing
preproc = ColumnTransformer(
    transformers=[
        ("text", TfidfVectorizer(min_df=1), text_col),
        ("num", StandardScaler(with_mean=False), num_cols),
    ]
)

# Build pipeline
pipe = Pipeline([
    ("pre", preproc),
    ("clf", LogisticRegression(max_iter=1000))
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Fit pipeline
pipe.fit(X_train, y_train)

# Evaluate
print("Classification report on test set:")
print(classification_report(y_test, pipe.predict(X_test)))

# Save pipeline
joblib.dump(pipe, "sentiment_pipeline.pkl")
print("✅ Saved sentiment_pipeline.pkl")
