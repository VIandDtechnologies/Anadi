import json
import random
import sqlite3
import os
import numpy as np
import faiss
import uuid
from sentence_transformers import SentenceTransformer
from flask import Flask, request, jsonify, render_template

# ------------------- Database setup -------------------
DB_NAME = "dealer.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS customers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            phone TEXT,
            email TEXT,
            interest TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def save_customer(name, phone, email, interest):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        INSERT INTO customers (name, phone, email, interest)
        VALUES (?, ?, ?, ?)
    ''', (name, phone, email, interest))
    conn.commit()
    conn.close()

# ------------------- Load intents & build FAISS index -------------------
with open("dealer_intents.json", "r", encoding="utf-8-sig") as f:
    intents = json.load(f)

model = SentenceTransformer('all-MiniLM-L6-v2')

all_patterns = []
pattern_to_tag = []
for intent in intents:
    for pattern in intent["patterns"]:
        all_patterns.append(pattern)
        pattern_to_tag.append(intent["tag"])

print(f"Loaded {len(all_patterns)} patterns from {len(intents)} intents.")

pattern_embeddings = model.encode(all_patterns, convert_to_numpy=True, normalize_embeddings=True)
dimension = pattern_embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(pattern_embeddings.astype('float32'))

SIMILARITY_THRESHOLD = 0.6

# In-memory storage for lead-capture state per user
user_sessions = {}

def start_lead_capture():
    return "Great! Let me get your details. What is your full name?"

def process_message(user_id, message):
    # If user is in the middle of lead capture
    if user_id in user_sessions:
        state, data = user_sessions[user_id]
        if state == "awaiting_name":
            data["name"] = message.strip()
            user_sessions[user_id] = ("awaiting_phone", data)
            return "Thanks! What's your phone number?"
        elif state == "awaiting_phone":
            data["phone"] = message.strip()
            user_sessions[user_id] = ("awaiting_email", data)
            return "Got it. What's your email address? (Type 'skip' if you don't have one)"
        elif state == "awaiting_email":
            email = message.strip()
            if email.lower() == "skip":
                email = ""
            data["email"] = email
            user_sessions[user_id] = ("awaiting_interest", data)
            return "Almost done! Which vehicle / model are you interested in?"
        elif state == "awaiting_interest":
            data["interest"] = message.strip()
            save_customer(data["name"], data["phone"], data.get("email", ""), data["interest"])
            del user_sessions[user_id]
            return f"Thank you, {data['name']}! Our team will contact you soon. Feel free to ask anything else."
        else:
            del user_sessions[user_id]   # reset unexpected state

    # Normal intent matching
    user_vec = model.encode([message], normalize_embeddings=True).astype('float32')
    scores, indices = index.search(user_vec, 1)
    best_score = scores[0][0]
    best_idx = indices[0][0]

    if best_score >= SIMILARITY_THRESHOLD:
        matched_tag = pattern_to_tag[best_idx]
        if matched_tag == "new_customer":
            # Start lead capture
            user_sessions[user_id] = ("awaiting_name", {})
            for intent in intents:
                if intent["tag"] == "new_customer":
                    return random.choice(intent["responses"]) + " " + start_lead_capture()
        else:
            for intent in intents:
                if intent["tag"] == matched_tag:
                    return random.choice(intent["responses"])
    return "I'm not sure about that. You can ask about our vehicles, financing, or location."

# ------------------- Flask app -------------------
app = Flask(__name__)
app.secret_key = "a-random-secret-key-12345"

@app.route('/')
def home():
    return render_template('index.html')

# ✅ This is the critical line – must include methods=['POST']
@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message', '').strip()
    user_id = data.get('user_id', str(uuid.uuid4()))
    response = process_message(user_id, user_message)
    return jsonify({'response': response, 'user_id': user_id})

if __name__ == '__main__':
    init_db()
    # Use environment variable PORT for cloud deployment, default to 5000 locally
    port = int(os.environ.get('PORT', 7860))
    app.run(debug=False, host='0.0.0.0', port=port)
