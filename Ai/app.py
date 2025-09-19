from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
import random
import os
import re
import threading
import time
import requests
from dotenv import load_dotenv

# -------------------- setup --------------------

load_dotenv()  # load variables from .env
HF_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
HF_MODEL = os.getenv("HUGGINGFACE_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")

app = Flask(__name__)

MODEL_PATH = "chat_model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"

# -------------------- utils --------------------

def clean_text(text):
    return re.sub(r"[^a-zA-Z0-9\s]", "", text.lower().strip())

# -------------------- dataset --------------------

training_data = {
    "hello": {
        "examples": ["hello", "hi", "hey", "yo", "hiya"],
        "responses": ["Hey!", "Hello 👋", "Hi there!", "Yo!", "What’s up?"]
    },
    "bye": {
        "examples": ["bye", "goodbye", "see you"],
        "responses": ["Goodbye 👋", "Catch you later!", "See ya!"]
    },
    "thanks": {
        "examples": ["thanks", "thank you", "thx"],
        "responses": ["You’re welcome 🙌", "No problem!", "Anytime!"]
    }
}

# -------------------- model --------------------

def train_model(training_data):
    X, y = [], []
    for label, data in training_data.items():
        for phrase in data["examples"]:
            X.append(clean_text(phrase))
            y.append(label)

    vectorizer = TfidfVectorizer()
    X_vectors = vectorizer.fit_transform(X)

    model = MultinomialNB()
    model.fit(X_vectors, y)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    return model, vectorizer

if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
else:
    model, vectorizer = train_model(training_data)

fallbacks = [
    "Hmm 🤔 I didn’t get that.",
    "Can you say that another way?",
    "I’m still learning!",
    "Interesting… tell me more!"
]

# -------------------- api --------------------

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Welcome to Baby GPT 🤖 API!",
        "endpoints": {
            "chat": "POST /chat (send JSON: { 'message': 'your text' })"
        }
    })

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message", "").strip()
    if not user_message:
        return jsonify({"reply": "Please send a message!"})

    reply = None

    # --- try Hugging Face first ---
    if HF_API_KEY:
        try:
            headers = {"Authorization": f"Bearer {HF_API_KEY}"}
            payload = {
                "inputs": f"User: {user_message}\nAssistant:",
                "parameters": {"max_new_tokens": 200, "temperature": 0.8}
            }
            response = requests.post(
                f"https://api-inference.huggingface.co/models/{HF_MODEL}",
                headers=headers,
                json=payload,
                timeout=15
            )
            result = response.json()

            if isinstance(result, list) and "generated_text" in result[0]:
                reply = result[0]["generated_text"].replace(f"User: {user_message}\nAssistant:", "").strip()
        except Exception as e:
            print(f"Hugging Face API error: {e}")

    # --- fallback to local model ---
    if not reply:
        user_vector = vectorizer.transform([clean_text(user_message)])
        proba = model.predict_proba(user_vector)[0]
        max_prob = max(proba)
        predicted_label = model.classes_[proba.argmax()]
        if max_prob < 0.2:
            reply = random.choice(fallbacks)
        else:
            reply = random.choice(training_data[predicted_label]["responses"])

    return jsonify({"reply": reply})

# -------------------- keep alive --------------------

def start_pinger():
    """Background thread that pings the app every 5 minutes"""
    while True:
        try:
            url = os.environ.get("PING_URL")  # set your Render URL as env var
            if url:
                requests.get(url)
        except Exception as e:
            print(f"Pinger error: {e}")
        time.sleep(300)

if __name__ == "__main__":
    threading.Thread(target=start_pinger, daemon=True).start()
    app.run(host="0.0.0.0", port=5000)
