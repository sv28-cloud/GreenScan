# -------------------------
# random_forest_image_classifier.py
# -------------------------

import pandas as pd
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# -------------------------
# 1️⃣ Load embeddings CSV
# -------------------------
df = pd.read_csv("AiModel\embeddings.csv")  # Make sure this is the CSV you generated

X = df.drop(columns=["label"]).values
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["label"])

# -------------------------
# 2️⃣ Split for training/testing
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------
# 3️⃣ Train RandomForest classifier
# -------------------------
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)

# -------------------------
# 4️⃣ Evaluate classifier
# -------------------------
y_pred = clf.predict(X_test)
print("Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("RandomForest Confusion Matrix")
plt.show()

# -------------------------
# 5️⃣ Save classifier and label encoder
# -------------------------
joblib.dump(clf, r"AiModel\random_forest_classifier.pkl")
joblib.dump(label_encoder, r"AiModel\label_encoder.pkl")
print("✅ Model and label encoder saved.")

# -------------------------
# 6️⃣ Prepare CNN for embedding new images
# -------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model = models.resnet50(pretrained=True)
model = torch.nn.Sequential(*(list(model.children())[:-1]))  # Remove final FC layer
model.eval()
model.to(device)

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

def get_embedding(image_path):
    """Load an image and convert it to 512-D embedding."""
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(img_tensor).squeeze().cpu().numpy()
    return embedding

# -------------------------
# 7️⃣ Predict new image
# -------------------------
def predict_new_image(image_path):
    """Predict the class of a new image using the trained RandomForest classifier."""
    embedding = get_embedding(image_path).reshape(1, -1)
    clf_loaded = joblib.load("random_forest_classifier.pkl")
    label_encoder_loaded = joblib.load("label_encoder.pkl")
    pred_id = clf_loaded.predict(embedding)[0]
    pred_class = label_encoder_loaded.inverse_transform([pred_id])[0]
    return pred_class

# Example usage:
# print(predict_new_image("new_image.jpg"))
test_image_path = r"AiModel\garbage-dataset\test_images\testimg1.jpg"
test_embedding = get_embedding(test_image_path)
pred_id = clf.predict(test_embedding.reshape(1, -1))[0]
pred_class = label_encoder.inverse_transform([pred_id])[0]

print("Predicted class:", pred_class)
