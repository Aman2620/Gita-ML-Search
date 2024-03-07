# app.py
from flask import Flask, render_template, request, jsonify
import json
import re
import numpy as np
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from keras.models import load_model
import joblib
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


# Load pre-trained model
model = load_model("model.h5")

# Load pre-trained vectorizer
vectorizer = joblib.load("vectorizer.joblib")

# Load and preprocess data
with open("MLData.json", "r", encoding="utf-8") as file:
    data = json.load(file)

verses = []
verse_numbers = []

for key, value in data.items():
    verse_numbers.append(value["verse_number"])
    verse_text = value["translation"] + " ".join(value["purport"])
    cleaned_text = re.sub(r"[^a-zA-Z0-9\s]", "", verse_text.lower())
    verses.append(cleaned_text)

# Tokenization and Embedding using TF-IDF
X = vectorizer.transform(verses)
X.sort_indices()

# Search Implementation
def search(query, model, vectorizer, top_k=5):
    cleaned_query = re.sub(r"[^a-zA-Z0-9\s]", "", query.lower())
    query_vector = vectorizer.transform([cleaned_query])

    # Compute similarity scores using cosine similarity
    similarity_scores = cosine_similarity(query_vector, X)

    # Find the indices of the top-k most similar verses
    top_k_indices = np.argsort(similarity_scores[0])[-top_k:][::-1]

    # Display the top-k results with highlighted text
    results = []
    for idx in top_k_indices:
        result_verse = verses[idx]
        highlighted_text = highlight_matched_words(result_verse, cleaned_query)
        results.append({"verse_number": verse_numbers[idx], "text": highlighted_text})

    return results

# Helper function to highlight matched words
def highlight_matched_words(text, query):
    words = text.split()
    highlighted_words = [f"<span style='color:red'>{word}</span>" if word in query.split() else word for word in words]
    return ' '.join(highlighted_words)

# Define base URL for verse links
base_url = "https://gita-learn.vercel.app/VerseDetail?chapterVerse="

# Routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/search", methods=["POST"])
def search_results():
    user_query = request.form.get("query")
    results = search(user_query, model, vectorizer)
    highlighted_content = results[0]["text"] if results else None
    verse_number = results[0]["verse_number"] if results else None
    verse_link = f"{base_url}{verse_number}"

    response_data = {
        'user_query': user_query,
        'verse_number': verse_number,
        'highlighted_content': highlighted_content,
        'verse_link': verse_link
    }
    return jsonify(response_data)

if __name__ == "__main__":
    app.run(debug=True)
