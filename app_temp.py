from flask import Flask, request, jsonify
import fasttext
import os
import psutil
import gzip

app = Flask(__name__)

# Lade das FastText-Modell beim Start des Servers
MODEL_PATH = "models/cc.de.300.bin"
print(f"Loading FastText model from {MODEL_PATH}...")
model = fasttext.load_model(MODEL_PATH)
print("FastText model loaded successfully!")

# API-Endpoint: Vektor für ein Wort abrufen
@app.route("/get_vector", methods=["POST"])
def get_vector():
    data = request.json
    word = data.get("word")
    
    if not word:
        return jsonify({"error": "No word provided"}), 400
    
    # Vektor für das Wort abrufen
    vector = model.get_word_vector(word).tolist()
    return jsonify({"word": word, "vector": vector})

# API-Endpoint: Ähnlichkeit zwischen zwei Wörtern berechnen
@app.route("/similarity", methods=["POST"])
def similarity():
    data = request.json
    word1 = data.get("word1")
    word2 = data.get("word2")
    
    if not word1 or not word2:
        return jsonify({"error": "Both word1 and word2 are required"}), 400
    
    vector1 = model.get_word_vector(word1)
    vector2 = model.get_word_vector(word2)
    
    # Kosinus-Distanz berechnen
    similarity_score = sum(vector1 * vector2) / (
        (sum(vector1 ** 2) ** 0.5) * (sum(vector2 ** 2) ** 0.5)
    )
    return jsonify({"word1": word1, "word2": word2, "similarity": similarity_score})

# API-Status überprüfen
@app.route("/", methods=["GET"])
def health_check():
    process = psutil.Process()
    memory_usage = process.memory_info().rss / (1024 ** 3)  # In GB
    return jsonify({"status": "API is running", "memory_usage_gb": memory_usage})

if __name__ == "__main__":
    app.run(debug=True, port=8503)
