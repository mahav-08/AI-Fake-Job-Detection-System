from flask import Flask, render_template, request
import joblib
import re

app = Flask(__name__)

# Load AI model
model = joblib.load("fake_job_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

print("✅ AI model loaded successfully!")


# Same text cleaning used during training
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


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    title = request.form.get("title", "")
    company = request.form.get("company", "")
    description = request.form.get("description", "")

    if not title or not description:
        return render_template(
            "index.html",
            error="Please enter Job Title and Job Description."
        )

    # Use the same text fields as training
    text = title + " " + description

    # Apply the same preprocessing used during training
    cleaned_text = clean_text(text)

    # Convert text to TF-IDF
    features = vectorizer.transform([cleaned_text])

    # AI prediction
    prediction = model.predict(features)[0]

    # Probability
    probability = model.predict_proba(features)[0]
    confidence = max(probability) * 100

    if prediction == 1:
        result = "FAKE JOB"
        message = "⚠️ This job posting may be fraudulent."
    else:
        result = "REAL JOB"
        message = "✅ This job posting appears to be genuine."

    return render_template(
        "index.html",
        analyzed=True,
        prediction=prediction,
        result=result,
        message=message,
        confidence=round(confidence, 2),
        title=title,
        company=company,
        description=description
    )


if __name__ == "__main__":
    app.run(debug=True)