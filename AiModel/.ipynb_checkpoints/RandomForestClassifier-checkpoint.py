# -------------------------
# random_forest_image_classifier.py
# -------------------------

import os
import torch
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision import transforms
from PIL import Image
import joblib
import numpy as np

# -------------------------
# 1️⃣ Paths to model files
# -------------------------
MODEL_FOLDER = "AiModel"
RF_PATH = os.path.join(MODEL_FOLDER, "random_forest_classifier.pkl")
LE_PATH = os.path.join(MODEL_FOLDER, "label_encoder.pkl")

# -------------------------
# 2️⃣ Load classifier and label encoder
# -------------------------
clf = joblib.load(RF_PATH)
label_encoder = joblib.load(LE_PATH)

# -------------------------
# 3️⃣ Prepare EfficientNet-B0 for embeddings
# -------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

weights = EfficientNet_B0_Weights.DEFAULT
model = efficientnet_b0(weights=weights)
model.classifier = torch.nn.Identity()  # remove final classifier layer
model.eval()
model.to(device)

# -------------------------
# 4️⃣ Image preprocessing (same as training)
# -------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# -------------------------
# 5️⃣ Function to get embeddings
# -------------------------
def get_embedding(image_path):
    """Load an image and convert it to 2048-D embedding."""
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(img_tensor).squeeze().cpu().numpy()
    return embedding

# -------------------------
# 6️⃣ Predict new image
# -------------------------
def predict_new_image(image_path):
    """Predict the class of a new image using the trained RandomForest."""
    embedding = get_embedding(image_path).reshape(1, -1)  # 1 sample, 2048 features
    pred_id = clf.predict(embedding)[0]
    pred_class = label_encoder.inverse_transform([pred_id])[0]
    
    # Optional: get probabilities for all classes
    probs = clf.predict_proba(embedding)[0]
    class_probs = dict(zip(label_encoder.classes_, probs))
    
    return pred_class, class_probs

# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    test_image_path = os.path.join("AiModel", "garbage-dataset", "test_images", "testimg1.jpg")
    
    if not os.path.exists(test_image_path):
        print(f"❌ File not found: {test_image_path}")
    else:
        pred_class, class_probs = predict_new_image(test_image_path)
        print("✅ Predicted class:", pred_class)
        print("Class probabilities:")
        for cls, prob in class_probs.items():
            print(f"  {cls}: {prob:.2f}")
