from flask import Flask, request, jsonify, render_template
import openai
import json
import pymongo
import os
from datetime import datetime, timezone
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Load knowledge base
KB_FILE = "knowledge_base.json"

def load_knowledge_base():
    try:
        with open(KB_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

knowledge_base = load_knowledge_base()

# MongoDB connection for logging
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["faq_assistant"]
logs_collection = db["query_logs"]

# OpenAI API Key (Set using .env file)
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("❌ OpenAI API Key not found. Check .env file or set manually.")

def query_llm(user_query):
    """Query the OpenAI GPT model using the latest API format."""
    context = "\n".join([f"{k}: {v}" for k, v in knowledge_base.items()])
    messages = [
        {"role": "system", "content": "You are a helpful FAQ assistant."},
        {"role": "user", "content": f"{context}\nUser: {user_query}\nAssistant:"}
    ]

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",  # Use OpenAI's latest model
            messages=messages
        )
        return response["choices"][0]["message"]["content"].strip()
    except openai.error.OpenAIError as e:
        print(f"❌ OpenAI API Error: {e}")
        return "I'm sorry, I couldn't process your request at the moment."

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    user_query = data.get("query", "")
    if not user_query:
        return jsonify({"error": "Query cannot be empty"}), 400

    response = query_llm(user_query)

    logs_collection.insert_one({
        "query": user_query,
        "response": response,
        "timestamp": datetime.now(timezone.utc)
    })

    return jsonify({"response": response})

@app.route("/admin/update_kb", methods=["POST"])
def update_kb():
    data = request.json
    global knowledge_base
    knowledge_base = data.get("knowledge_base", {})
    with open(KB_FILE, "w") as f:
        json.dump(knowledge_base, f, indent=4)
    return jsonify({"message": "Knowledge base updated successfully"})

@app.route("/admin/logs", methods=["GET"])
def get_logs():
    logs = list(logs_collection.find({}, {"_id": 0}))
    return jsonify({"logs": logs})

if __name__ == "__main__":
    app.run(debug=True)
