from flask import Flask, render_template, request,jsonify
import json
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load and preprocess data
with open("MLData.json", "r", encoding="utf-8") as file:
    data = json.load(file)

verses = []
verse_numbers = []
original_texts = []  # New list to store original text

for key, value in data.items():
    verse_numbers.append(value["verse_number"])
    verse_text = value["translation"] + " ".join(value["purport"])
    cleaned_text = re.sub(r"[^a-zA-Z0-9\s]", "", verse_text.lower())
    verses.append(cleaned_text)
    original_texts.append(verse_text)  # Store the original text

# Tokenization and Embedding using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(verses)
X.sort_indices()

# Search Implementation
def search(query, vectorizer, top_k=5):
    cleaned_query = re.sub(r"[^a-zA-Z0-9\s]", "", query.lower())
    query_vector = vectorizer.transform([cleaned_query])
    
    # Compute similarity scores using cosine similarity
    similarity_scores = cosine_similarity(query_vector, X)
    
    # Find the indices of the top-k most similar verses
    top_k_indices = np.argsort(similarity_scores[0])[-top_k:][::-1]
    
    # Display the top-k results with highlighted text
    results = []
    for i, idx in enumerate(top_k_indices):
        result_verse = original_texts[idx]  # Use original text
        highlighted_text = highlight_matched_words(result_verse, cleaned_query)
        results.append({"verse_number": verse_numbers[idx], "text": highlighted_text})
    
    return results

# Helper function to highlight matched words
def highlight_matched_words(original_text, query):
    words = original_text.split()
    highlighted_words = [f"<span style='color:red'>{word}</span>" if word.lower() in query.lower().split() else word for word in words]
    return ' '.join(highlighted_words)

base_url = "https://gita-learn.vercel.app/VerseDetail?chapterVerse="

# Routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/search", methods=["POST"])
def search_results():
    user_query = request.form.get("query")
    results = search(user_query, vectorizer)
    verse_number = results[0]["verse_number"] if results else None
    verse_link = f"{base_url}{verse_number}"

    # Assuming you want to display the highlighted text in the results
    highlighted_content = results[0]["text"] if results else None

    response_data = {
        'user_query': user_query,
        'verse_number': verse_number,
        'highlighted_content': highlighted_content,
        'verse_link': verse_link
    }
    return jsonify(response_data)


if __name__ == "__main__":
    app.run(debug=True)
