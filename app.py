
from flask import Flask, request, jsonify
from model import improved_chat
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return jsonify({"status": "ok", "message": "Chatbot API lista 🚀"})

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    print("DEBUG: request.data =", request.data)
    print("DEBUG: request.json =", request.get_json(silent=True))
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "Falta el campo 'query'"}), 400

    answer = improved_chat(query)
    return jsonify({"query": query, "answer": answer})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
