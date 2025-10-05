from flask import Flask, request, jsonify
from flask_cors import CORS
import random

app = Flask(__name__)
CORS(app)

# Dummy labels
labels = ["battery", "biological", "cardboard", "clothes", "glass", "metal", "paper", "plastic", "shoes", "trash"]

@app.route("/")
def home():
    return jsonify({"message": "Backend is alive!"})

@app.route("/classify", methods=["POST"])
def classify():
    # Get the uploaded image
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image = request.files["image"]

    # For now: just return a random classification
    classification = random.choice(labels)

    # Pretend AI + Gemini explanation
    explanation = f"This item looks like a {classification}. Remember to rinse before recycling!"

    return jsonify({
        "classification": classification,
        "explanation": explanation
    })

if __name__ == "__main__":
    app.run(port=5000, debug=True)