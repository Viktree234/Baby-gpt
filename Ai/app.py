from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
import random
import os
import re

app = Flask(__name__)

MODEL_PATH = "chat_model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"

# -------------------- utils --------------------

def clean_text(text):
    return re.sub(r"[^a-zA-Z0-9\s]", "", text.lower().strip())

# -------------------- dataset --------------------

training_data = {
    "hello": {
        "examples": [
            "hello", "hi", "hey", "yo", "hiya", "good morning", "good afternoon", "good evening",
            "hey there", "sup", "what’s up", "howdy", "hi friend", "heya", "morning", "evening"
        ],
        "responses": ["Hey!", "Hello 👋", "Hi there!", "Yo!", "What’s up?", "Howdy partner 🤠", "Good to see you!"]
    },
    "how are you": {
        "examples": [
            "how are you", "how’s it going", "you good?", "what’s up with you", "how are things",
            "how are you doing", "everything good?", "how have you been", "are you ok", "you alright"
        ],
        "responses": [
            "I’m good, thanks for asking!", "Doing great 🤖", "Chilling, you?", 
            "All systems green ✅", "Feeling chatty today!", "I’m alive and running 🚀"
        ]
    },
    "bye": {
        "examples": [
            "bye", "goodbye", "see you", "catch you later", "laters", "talk to you later",
            "peace out", "take care", "adios", "see ya soon"
        ],
        "responses": ["Goodbye 👋", "Catch you later!", "See ya!", "Take care!", "Peace out ✌️"]
    },
    "thanks": {
        "examples": [
            "thanks", "thank you", "thx", "appreciate it", "thanks a lot", "cheers", "ty",
            "many thanks", "much obliged", "thank you so much"
        ],
        "responses": ["You’re welcome 🙌", "No problem!", "Anytime!", "Glad to help 🤖", "My pleasure!"]
    },
    "name": {
        "examples": [
            "what’s your name", "who are you", "tell me your name", "do you have a name",
            "what should I call you", "introduce yourself", "your name please"
        ],
        "responses": ["I’m Baby GPT 🤖", "You can call me Baby GPT!", "I’m your mini AI bot.", "Baby GPT at your service!"]
    },
    "joke": {
        "examples": [
            "tell me a joke", "make me laugh", "say something funny", "give me a joke", "funny please",
            "another joke", "joke time", "make me giggle", "say a funny one"
        ],
        "responses": [
            "Why don’t robots get tired? Because they recharge! ⚡",
            "I tried to catch fog yesterday… I mist 😂",
            "Why was the math book sad? Too many problems.",
            "Parallel lines have so much in common… it’s a shame they’ll never meet.",
            "Why can’t your nose be 12 inches long? Because then it’d be a foot!"
        ]
    },
    "weather": {
        "examples": [
            "what’s the weather", "weather today", "is it raining", "tell me the forecast",
            "how’s the weather", "today’s forecast", "is it sunny", "temperature now"
        ],
        "responses": [
            "I don’t have live weather data 🌦, but I hope it’s nice where you are!",
            "Looks like perfect chat weather 😎",
            "Weather? Always sunny in Baby GPT land ☀️",
            "No idea about the rain, but my vibes are clear skies 🌤️"
        ]
    },
    "time": {
        "examples": [
            "what time is it", "current time", "can you tell me the time", "time please",
            "do you know the time", "check the clock", "now time"
        ],
        "responses": [
            "I don’t have a clock ⏰, but it’s always chat o’clock!",
            "Time to vibe and chat 😎",
            "My system says: time to have fun 🤖"
        ]
    },
    "date": {
        "examples": [
            "what’s the date", "today’s date", "date please", "do you know the date",
            "which day is it", "day today"
        ],
        "responses": [
            "I don’t track dates 📅, but today is a good day!",
            "It’s today… obviously 😂",
            "Every day is Baby GPT day 🚀"
        ]
    },
    "hobbies": {
        "examples": [
            "what’s your hobby", "do you have hobbies", "what do you do for fun",
            "tell me your hobbies", "what do you enjoy", "favorite activity"
        ],
        "responses": [
            "I like chatting 💬", "Training my brain is my hobby 🧠",
            "I enjoy making humans laugh 😂", "Talking to you is my favorite activity!"
        ]
    },
    "motivation": {
        "examples": [
            "motivate me", "say something inspiring", "give me motivation", "inspire me",
            "uplift me", "say something positive", "encouragement please"
        ],
        "responses": [
            "You got this 💪", "Believe in yourself 🚀", "Keep pushing, you’re doing great!",
            "Small steps lead to big changes 🌱", "Stay strong, stay positive ✨"
        ]
    },
    "food": {
        "examples": [
            "what’s your favorite food", "do you eat", "what do you like to eat",
            "tell me your favorite meal", "are you hungry", "food?"
        ],
        "responses": [
            "I don’t eat, but if I did… probably electricity ⚡",
            "I’d snack on data packets all day 📡",
            "Pizza sounds good 🍕",
            "Do energy drinks count as food? 😅"
        ]
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

# -------------------- init --------------------

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

@app.route("/chat", methods=["POST"])
def chat():
    user_message = clean_text(request.json.get("message", ""))
    if not user_message:
        return jsonify({"reply": "Please send a message!"})

    user_vector = vectorizer.transform([user_message])
    proba = model.predict_proba(user_vector)[0]
    max_prob = max(proba)
    predicted_label = model.classes_[proba.argmax()]

    if max_prob < 0.4:  # low confidence
        reply = random.choice(fallbacks)
    else:
        reply = random.choice(training_data[predicted_label]["responses"])

    return jsonify({"reply": reply})

# -------------------- run --------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
