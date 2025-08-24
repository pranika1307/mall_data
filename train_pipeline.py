"""
Sample training script that builds a single Pipeline handling:
- TF-IDF for "feedback" text
- StandardScaler (without mean) for numeric features
- LogisticRegression classifier

It saves 'sentiment_pipeline.pkl'. Replace the toy data with your labeled dataset.
"""
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# === Replace this with your real dataset ===
toy = pd.DataFrame({
    "feedback": [
        "Great experience, helpful staff and clean stores",
        "Terrible service, long lines and rude personnel",
        "Loved the discounts and the ambience",
        "Worst visit ever, dirty restrooms and no assistance"
    ],
    "age": [30, 45, 22, 38],
    "gender": [0, 1, 0, 1],  # 0=Male, 1=Female (match the app encoding)
    "rating": [5, 1, 4, 1],
    "label": [1, 0, 1, 0],   # 1=Positive, 0=Negative
})

X = toy.drop(columns=["label"])
y = toy["label"]

text_col = "feedback"
num_cols = ["age", "gender", "rating"]

preproc = ColumnTransformer(
    transformers=[
        ("text", TfidfVectorizer(min_df=1), text_col),
        ("num", StandardScaler(with_mean=False), num_cols),
    ]
)

pipe = Pipeline([
    ("pre", preproc),
    ("clf", LogisticRegression(max_iter=1000))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
pipe.fit(X_train, y_train)

print(classification_report(y_test, pipe.predict(X_test)))

joblib.dump(pipe, "sentiment_pipeline.pkl")
print("Saved sentiment_pipeline.pkl")
