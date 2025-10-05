# -------------------------
# app.py
# -------------------------
from flask import Flask, request, jsonify
import joblib
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from flask_cors import CORS


# -------------------------
# 1️⃣ Load your model + label encoder
# -------------------------
clf = joblib.load("AiModel/random_forest_classifier.pkl")
label_encoder = joblib.load("AiModel/label_encoder.pkl")

# -------------------------
# 2️⃣ Load your ResNet50 model for embeddings
# -------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# ✅ Use modern weights parameter instead of deprecated "pretrained"
from torchvision.models import resnet50, ResNet50_Weights

weights = ResNet50_Weights.DEFAULT
resnet = resnet50(weights=weights)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])  # Remove last FC layer
resnet.eval()
resnet.to(device)

# Image preprocessing (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def get_embedding(image):
    """Convert an image into a 2048-dimensional embedding using ResNet50."""
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = resnet(img_tensor).squeeze().cpu().numpy()
    return embedding

# -------------------------
# 3️⃣ Create Flask app
# -------------------------
app = Flask(__name__)
CORS(app)


@app.route("/predict", methods=["POST"])
def predict():
    print("📩 /predict endpoint hit")

    if "file" not in request.files:
        print("❌ No file field in request.files")
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    print(f"✅ Received file: {file.filename}, Content-Type: {file.content_type}")

    try:
        # Try to open the image
        image = Image.open(file).convert("RGB")
        print("🖼️ Image successfully opened and converted to RGB")
    except Exception as e:
        print(f"❌ Error opening image: {e}")
        return jsonify({"error": "Invalid image format"}), 400

    # Get embedding
    try:
        embedding = get_embedding(image).reshape(1, -1)
        print(f"🔍 Embedding shape: {embedding.shape}")
    except Exception as e:
        print(f"❌ Error getting embedding: {e}")
        return jsonify({"error": "Embedding failed"}), 500

    # Predict class
    try:
        pred_id = clf.predict(embedding)[0]
        pred_class = label_encoder.inverse_transform([pred_id])[0]
        print(f"✅ Prediction complete — class ID: {pred_id}, label: {pred_class}")
    except Exception as e:
        print(f"❌ Error predicting: {e}")
        return jsonify({"error": "Prediction failed"}), 500

    print("🚀 Returning response to frontend\n")
    return jsonify({"predicted_class": pred_class})

# -------------------------
# 4️⃣ Run the Flask server
# -------------------------
if __name__ == "__main__":
    app.run(debug=True)
