import pandas as pd
import joblib
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


print("\n========================================")
print("       🛡️ JOBGUARD AI TRAINING")
print("========================================\n")


# Load dataset
df = pd.read_csv("fake_job_postings.csv")

print("Dataset loaded successfully!")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())


# Find columns
title_col = None
description_col = None
target_col = None


for col in df.columns:

    name = col.lower().strip()

    if "title" in name:
        title_col = col

    if "description" in name:
        description_col = col

    if name in ["fraudulent", "fake", "label", "target", "is_fake"]:
        target_col = col


print("\nDetected columns:")
print("Title:", title_col)
print("Description:", description_col)
print("Target:", target_col)


# Check required columns
if description_col is None:
    raise Exception("❌ Job description column not found!")

if target_col is None:
    raise Exception("❌ Target column not found!")


# Combine title + description
if title_col:

    df["text"] = (
        df[title_col].fillna("").astype(str)
        + " "
        + df[description_col].fillna("").astype(str)
    )

else:

    df["text"] = (
        df[description_col]
        .fillna("")
        .astype(str)
    )


# Clean text
def clean_text(text):

    text = str(text).lower()

    text = re.sub(
        r"http\S+|www\S+",
        " ",
        text
    )

    text = re.sub(
        r"[^a-zA-Z\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


df["text"] = df["text"].apply(clean_text)


# Remove missing data
df = df[
    df["text"].str.len() > 0
]

df = df.dropna(
    subset=[target_col]
)


# Prepare X and y
X = df["text"]

y = pd.to_numeric(
    df[target_col],
    errors="coerce"
)

valid = y.notna()

X = X[valid]

y = y[valid].astype(int)


print("\nTraining samples:", len(X))


# TF-IDF
print("\nCreating TF-IDF features...")

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=20000,
    ngram_range=(1, 2)
)

X_tfidf = vectorizer.fit_transform(X)

print(
    "TF-IDF feature count:",
    X_tfidf.shape[1]
)


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# Logistic Regression
print("\nTraining Logistic Regression...")

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

model.fit(
    X_train,
    y_train
)


# Test
predictions = model.predict(
    X_test
)

accuracy = accuracy_score(
    y_test,
    predictions
)


print("\n========================================")
print("           TRAINING COMPLETE")
print("========================================")

print(
    f"Accuracy: {accuracy * 100:.2f}%"
)

print(
    "Model: Logistic Regression"
)

print(
    "Features:",
    X_tfidf.shape[1]
)


# Save model
joblib.dump(
    model,
    "fake_job_model.pkl"
)


# Save vectorizer
joblib.dump(
    vectorizer,
    "tfidf_vectorizer.pkl"
)


print("\n✅ fake_job_model.pkl created")
print("✅ tfidf_vectorizer.pkl created")

print("\n========================================")
print("       🛡️ JOBGUARD AI READY")
print("========================================")