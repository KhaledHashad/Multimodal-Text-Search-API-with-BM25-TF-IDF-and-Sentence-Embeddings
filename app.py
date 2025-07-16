import nltk
nltk.download('punkt')

from flask import Flask, request, jsonify
from nltk.tokenize import word_tokenize as orig_word_tokenize
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktLanguageVars
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, util
from pyngrok import ngrok, conf

# 🔐 Set your ngrok Authtoken
conf.get_default().auth_token = "2zp8rTsJ1Gg6Fgf8GKNO7dY9YJ8_6vxC1e4TqKgPVee2XMe1a"

# --- Safe word_tokenize wrapper to handle 'punkt_tab' errors ----
class CustomLanguageVars(PunktLanguageVars):
    sent_end_chars = ('.', '!', '?')

tokenizer = PunktSentenceTokenizer(lang_vars=CustomLanguageVars())

def safe_word_tokenize(text):
    try:
        return orig_word_tokenize(text)
    except LookupError as e:
        print("⚠️ Fallback tokenizer used due to missing punkt model:", e)
        return text.split()

# --- Create Flask App ---
app = Flask(__name__)

# --- Sample Documents ---
docs = [
    "Neural networks are trained using backpropagation.",
    "Support vector machines are used for classification.",
    "Regularization helps prevent overfitting.",
    "Cats are active during the night and sleep all day.",
]

# --- BM25 ---
def build_bm25_index(docs):
    return BM25Okapi([safe_word_tokenize(doc.lower()) for doc in docs])

bm25 = build_bm25_index(docs)

# --- TF-IDF ---
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(docs)

# --- Semantic Embeddings ---
model = SentenceTransformer('all-MiniLM-L6-v2')
doc_embeddings = model.encode(docs, convert_to_tensor=True)

# --- Search Endpoint ---
@app.route("/search", methods=["POST"])
def search():
    data = request.get_json()
    query = data.get("query", "")
    if not query:
        return jsonify({"error": "Missing query"}), 400

    # BM25
    tokenized_query = safe_word_tokenize(query.lower())
    bm25_scores = bm25.get_scores(tokenized_query)

    # TF-IDF
    query_vec = vectorizer.transform([query])
    tfidf_scores = (query_vec @ tfidf_matrix.T).toarray()[0]

    # Semantic Search
    query_embedding = model.encode(query, convert_to_tensor=True)
    semantic_scores = util.cos_sim(query_embedding, doc_embeddings)[0].tolist()

    # Combine Results
    results = []
    for i in range(len(docs)):
        results.append({
            "doc": docs[i],
            "bm25_score": round(float(bm25_scores[i]), 4),
            "tfidf_score": round(float(tfidf_scores[i]), 4),
            "semantic_score": round(float(semantic_scores[i]), 4)
        })

    return jsonify({
        "query": query,
        "results": results
    })

# --- Start ngrok and Flask ---
if __name__ == "__main__":
    public_url = ngrok.connect(5000)
    print("🚀 Public URL:", public_url)
    app.run(port=5000)
