import tkinter as tk
from tkinter import messagebox
import joblib
import os
from model import SentimentClassifier  # Import from your model.py

# Paths to saved model and vectorizer
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "sentiment_classifier_logistic_regression.joblib")
VECTORIZER_PATH = os.path.join(BASE_DIR, "models", "tfidf_vectorizer.joblib")

# Load trained classifier & vectorizer
classifier = SentimentClassifier.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

def predict_sentiment():
    text = input_text.get("1.0", tk.END).strip()
    if not text:
        messagebox.showwarning("Input Error", "Please enter some text.")
        return

    features = vectorizer.transform([text])
    prediction = classifier.predict(features)[0]
    sentiment = "Positive 😀" if prediction == 1 else "Negative 😞"
    result_label.config(text=f"Sentiment: {sentiment}")

# GUI setup
root = tk.Tk()
root.title("Movie Review Sentiment Detector")
root.geometry("400x300")

tk.Label(root, text="Enter your sentence:", font=("Arial", 12)).pack(pady=5)
input_text = tk.Text(root, height=5, width=40, font=("Arial", 10))
input_text.pack(pady=5)
tk.Button(root, text="Detect Sentiment", command=predict_sentiment, font=("Arial", 12), bg="lightblue").pack(pady=10)
result_label = tk.Label(root, text="", font=("Arial", 14, "bold"))
result_label.pack(pady=10)

root.mainloop()
