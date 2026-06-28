import json
import random
import sqlite3
from datetime import datetime
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ------------------- Database Setup -------------------
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
    print("[DB] Customer saved successfully.")

# ------------------- Load Intents -------------------
with open("dealer_intents.json", "r", encoding="utf-8-sig") as f:
    intents = json.load(f)


model = None

def get_model():
    global model
    if model is None:
        model = SentenceTransformer("all-MiniLM-L6-v2")
    return model

all_patterns = []
pattern_to_tag = []

for intent in intents:
    for pattern in intent["patterns"]:
        all_patterns.append(pattern)
        pattern_to_tag.append(intent["tag"])

print(f"Loaded {len(all_patterns)} patterns from {len(intents)} intents.")

# Encode patterns and build FAISS index
pattern_embeddings = model.encode(all_patterns, convert_to_numpy=True, normalize_embeddings=True)
dimension = pattern_embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)  # inner product = cosine similarity for normalized vectors
index.add(pattern_embeddings.astype('float32'))

SIMILARITY_THRESHOLD = 0.6

# ------------------- Lead Capture State Machine -------------------
# States: None (no active capture), "awaiting_name", "awaiting_phone", "awaiting_email", "awaiting_interest"
capture_state = None
capture_data = {}

def start_lead_capture():
    global capture_state, capture_data
    capture_state = "awaiting_name"
    capture_data = {}
    return "Bot: Great! Let me get your details.\nWhat is your full name?"

def process_lead_input(user_input):
    global capture_state, capture_data
    if capture_state == "awaiting_name":
        capture_data["name"] = user_input.strip()
        capture_state = "awaiting_phone"
        return "Bot: Thanks! What's your phone number?"
    elif capture_state == "awaiting_phone":
        capture_data["phone"] = user_input.strip()
        capture_state = "awaiting_email"
        return "Bot: Got it. What's your email address? (Type 'skip' if you don't have one)"
    elif capture_state == "awaiting_email":
        email = user_input.strip()
        if email.lower() == "skip":
            email = ""
        capture_data["email"] = email
        capture_state = "awaiting_interest"
        return "Bot: Almost done! Which vehicle / model are you interested in?"
    elif capture_state == "awaiting_interest":
        capture_data["interest"] = user_input.strip()
        # Save to database
        save_customer(
            capture_data["name"],
            capture_data["phone"],
            capture_data.get("email", ""),
            capture_data["interest"]
        )
        # Reset state
        capture_state = None
        return f"Bot: Thank you, {capture_data['name']}! Our team will contact you soon. In the meantime, feel free to ask any other questions."
    return None  # Shouldn't happen

# ------------------- Main Loop -------------------
init_db()
print("AI Dealer Chatbot Ready")
print("Type 'exit' to quit")
print("=" * 50)

while True:
    user_input = input("You: ").strip()
    if user_input.lower() == "exit":
        print("Bot: Goodbye! Visit us again.")
        break

    # If we are in the middle of capturing customer details, handle it
    if capture_state is not None:
        reply = process_lead_input(user_input)
        print(reply)
        continue

    # Otherwise, semantic intent matching
    user_vec = model.encode([user_input], normalize_embeddings=True).astype('float32')
    scores, indices = index.search(user_vec, 1)
    best_score = scores[0][0]
    best_idx = indices[0][0]

    if best_score >= SIMILARITY_THRESHOLD:
        matched_tag = pattern_to_tag[best_idx]

        # ---------- Lead capture intent ----------
        if matched_tag == "new_customer":
            # Give the initial response and start capture flow
            for intent in intents:
                if intent["tag"] == "new_customer":
                    print("Bot:", random.choice(intent["responses"]))
                    break
            print(start_lead_capture())
        else:
            # Regular FAQ response
            for intent in intents:
                if intent["tag"] == matched_tag:
                    print("Bot:", random.choice(intent["responses"]))
                    break
    else:
        print("Bot: I'm not sure about that. You can ask about our vehicles, financing, or location. Type 'exit' to leave.")
