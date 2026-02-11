import os
import pickle
import cv2
import torch
from PIL import Image
from transformers import ViTImageProcessor, ViTModel


# =====================================================
# 1. LOAD embeddings.pkl
# =====================================================
def load_embeddings(model_path="models/embeddings.pkl"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"{model_path} not found")

    with open(model_path, "rb") as f:
        data = pickle.load(f)

    embeddings = data["embeddings"]
    labels = data["labels"]

    print(f"[INFO] Loaded {len(embeddings)} embeddings")
    return embeddings, labels


# =====================================================
# 2. LOAD A TEST IMAGE
# =====================================================
def load_test_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Failed to load image")

    print("[INFO] Test image loaded")
    return image


# =====================================================
# 3. DETECT FACE IN IMAGE
# =====================================================
def detect_face(image):
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    if len(faces) == 0:
        raise ValueError("No face detected")

    x, y, w, h = faces[0]
    face = image[y:y+h, x:x+w]

    print("[INFO] Face detected and cropped")
    return face


# =====================================================
# 4. CONVERT FACE → ViT EMBEDDING
# =====================================================
def face_to_embedding(face_image):
    processor = ViTImageProcessor.from_pretrained(
        "google/vit-base-patch16-224-in21k"
    )
    model = ViTModel.from_pretrained(
        "google/vit-base-patch16-224-in21k"
    )
    model.eval()

    # Convert OpenCV BGR → RGB PIL image
    face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(face_rgb)

    inputs = processor(images=pil_image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    embedding = outputs.last_hidden_state[:, 0, :].numpy().flatten()

    print("[INFO] Face embedding generated")
    return embedding

import numpy as np

def recognize_face(test_embedding, known_embeddings, known_labels, threshold=0.5):
    """
    Compare test embedding with known embeddings using cosine distance.

    Returns:
        recognized_name (str)
        min_distance (float)
    """

    def cosine_distance(a, b):
        return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    min_distance = float("inf")
    recognized_name = "Unknown Face"

    for emb, label in zip(known_embeddings, known_labels):
        dist = cosine_distance(test_embedding, emb)

        if dist < min_distance:
            min_distance = dist
            recognized_name = label

    print(f"[INFO] Minimum distance: {min_distance:.4f}")

    if min_distance < threshold:
        return recognized_name, min_distance
    else:
        return "Unknown Face", min_distance
