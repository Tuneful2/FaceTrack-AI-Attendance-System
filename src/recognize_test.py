import os
import pickle
import cv2
import numpy as np
import torch
from PIL import Image
from transformers import ViTImageProcessor, ViTModel
from recognize import append_attendance



# =============================================================
# PATH SETUP (PROJECT ROOT SAFE)
# =============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "embeddings.pkl")


# =============================================================
# 1. LOAD SAVED EMBEDDINGS
# =============================================================
def load_saved_embeddings(model_path=MODEL_PATH):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            "embeddings.pkl not found. Run embedding generation first."
        )

    with open(model_path, "rb") as f:
        data = pickle.load(f)

    embeddings = data["embeddings"]
    labels = data["labels"]

    print(f"[INFO] Loaded {len(embeddings)} embeddings")
    return embeddings, labels


# =============================================================
# 2. LOAD ViT MODEL
# =============================================================
def load_vit_model():
    processor = ViTImageProcessor.from_pretrained(
        "google/vit-base-patch16-224-in21k"
    )
    model = ViTModel.from_pretrained(
        "google/vit-base-patch16-224-in21k"
    )
    model.eval()
    return processor, model


# =============================================================
# 3. DETECT FACE FROM IMAGE PATH
# =============================================================
def detect_face(image_path):
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not load test image")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    if len(faces) == 0:
        raise ValueError("No face detected")

    x, y, w, h = faces[0]
    face = img[y:y+h, x:x+w]

    return face


# =============================================================
# 4. CONVERT FACE → EMBEDDING
# =============================================================
def face_to_embedding(face_img):
    processor, model = load_vit_model()

    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(face_rgb)

    inputs = processor(images=pil_image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    embedding = outputs.last_hidden_state[:, 0, :].numpy().flatten()
    return embedding


# =============================================================
# 5. COSINE DISTANCE RECOGNITION
# =============================================================
def recognize_face(test_embedding, known_embeddings, known_labels, threshold=0.5):

    def cosine_distance(a, b):
        return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    min_dist = float("inf")
    identity = "Unknown Face"

    for emb, label in zip(known_embeddings, known_labels):
        dist = cosine_distance(test_embedding, emb)
        if dist < min_dist:
            min_dist = dist
            identity = label

    if min_dist < threshold:
        return identity, min_dist
    else:
        return "Unknown Face", min_dist


# =============================================================
# 6. MAIN SYSTEM FUNCTION (IMPORTANT)
# =============================================================
def identify_face(image_path, threshold=0.5):

    print("\n[INFO] Starting face identification pipeline...")

    known_embeddings, known_labels = load_saved_embeddings()
    if len(known_embeddings) == 0:
        return "No Known Faces", None

    try:
        face = detect_face(image_path)
    except Exception as e:
        print("[ERROR] Face detection failed:", e)
        return "No Face Detected", None

    test_embedding = face_to_embedding(face)

    name, distance = recognize_face(
        test_embedding,
        known_embeddings,
        known_labels,
        threshold=threshold
    )

    print("[INFO] Identification complete.")
    return name, distance



# =============================================================
# MAIN EXECUTION (TEST / DEMO)
# =============================================================
if __name__ == "__main__":

    # Put test image in project root (ASEP/)
    test_image_path = os.path.join(BASE_DIR, "soham.jpg")

    name, distance = identify_face(test_image_path, threshold=0.5)

    print("\n===== FINAL RESULT =====")
    print("Recognized as:", name)
    print("Distance:", distance)
