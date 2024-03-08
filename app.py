import nltk
nltk.download('stopwords')
nltk.download('punkt')

from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import json
import joblib
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the Bhagavad Gita JSON file with explicit encoding
with open('MLData.json', 'r', encoding='utf-8') as file:
    bhagavad_gita_data = json.load(file)

# Combine translation and purport text for each verse
combined_text = {key: f"{value['translation']} {' '.join(value['purport'])}" for key, value in bhagavad_gita_data.items()}

# Preprocess text
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

def preprocess_text(text):
    words = word_tokenize(text)
    stemmed_words = [ps.stem(word.lower()) for word in words if word.isalnum() and word.lower() not in stop_words]
    return ' '.join(stemmed_words), ' '.join(words)  # Return both stemmed and original text

# Update combined_text with preprocessed text
combined_text = {key: preprocess_text(value) for key, value in combined_text.items()}

# Load pre-processed data and vectorizer using joblib
vectorizer = joblib.load('vectorizer.pkl')
tfidf_matrix = joblib.load('tfidf_matrix.pkl')

# Define base URL for verse links
base_url = "https://gita-learn.vercel.app/VerseDetail?chapterVerse="

@app.route('/')
def home():
    return render_template('index.html')

# Define route for handling search query
@app.route('/search', methods=['POST'])
def search():
    user_query = request.json['user_query']

    # Preprocess user query
    user_query_stemmed, user_query_original = preprocess_text(user_query)

    # Vectorize the user query
    query_vector = vectorizer.transform([user_query_stemmed])

    # Calculate cosine similarity between the query and Bhagavad Gita verses
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

    # Get the indices of the top 5 most similar verses
    top_k_indices = cosine_similarities.argsort()[-5:][::-1]

    # Get the verse numbers and contents of the top 5 most similar verses
    top_results = []
    for idx in top_k_indices:
        verse_number = list(combined_text.keys())[idx]
        verse_content_stemmed, verse_content_original = combined_text[verse_number]

        # Highlight individual words in the content
        highlighted_words = [f"<span class='text-red-500'>{word}</span>" if word.lower() in user_query_original.split() else word for word in verse_content_original.split()]
        highlighted_content = ' '.join(highlighted_words)

        # Generate the link for the entire verse
        verse_link = f"{base_url}{verse_number}"

        # Prepare response data for each result
        result_data = {
            'user_query': user_query_original,
            'verse_number': verse_number,
            'highlighted_content': highlighted_content,
            'verse_link': verse_link
        }
        top_results.append(result_data)

    return jsonify(top_results)

if __name__ == '__main__':
    app.run(debug=True)
