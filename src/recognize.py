import os
import csv
import pickle
from datetime import datetime

import torch
import numpy as np
from PIL import Image
from transformers import ViTImageProcessor, ViTModel


# ============================================================
# PATH SETUP
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

KNOWN_FACES_DIR = os.path.join(BASE_DIR, "known_faces")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models", "embeddings.pkl")
ATTENDANCE_FILE = os.path.join(BASE_DIR, "attendance_log.csv")


# ============================================================
# ATTENDANCE CSV FUNCTIONS
# ============================================================
def initialize_attendance_file():
    """Create attendance_log.csv with header if not present"""
    if not os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Name", "Date", "Time"])
        print("[INFO] attendance_log.csv created with header")


def get_current_timestamp():
    """Return current date and time"""
    now = datetime.now()
    return now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")


def append_attendance(name):
    """
    Append attendance ONLY for known users.
    Does NOT write anything for:
    - Unknown Face
    - None / empty names
    - Errors
    """

    if name is None:
        return

    if name == "Unknown Face":
        return

    if not isinstance(name, str) or name.strip() == "":
        return

    try:
        date, time = get_current_timestamp()

        with open(ATTENDANCE_FILE, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([name, date, time])

        print(f"[INFO] Attendance appended for {name}")

    except Exception as e:
        # Important: do NOT append anything on error
        print("[ERROR] Attendance not recorded:", e)
    print("[DEBUG] append_attendance called with:", name)
    print("[DEBUG] Saving to:", ATTENDANCE_FILE)




def load_vit_model():
    print("[INFO] Loading ViT model...")

    processor = ViTImageProcessor.from_pretrained(
        "google/vit-base-patch16-224-in21k"
    )
    model = ViTModel.from_pretrained(
        "google/vit-base-patch16-224-in21k"
    )

    model.eval()
    print("[INFO] ViT loaded successfully.")
    return processor, model



def get_embedding_from_image(image_path, processor, model):
    print(f"[INFO] Generating embedding for: {image_path}")

    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    embedding = outputs.last_hidden_state[:, 0, :].numpy().flatten()
    return embedding



def load_known_faces():
    image_paths = []
    labels = []

    print(f"[INFO] Scanning known faces directory: {KNOWN_FACES_DIR}")

    for root, _, files in os.walk(KNOWN_FACES_DIR):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                full_path = os.path.join(root, file)
                folder = os.path.basename(root)

                if root == KNOWN_FACES_DIR:
                    label = os.path.splitext(file)[0]
                else:
                    label = folder

                image_paths.append(full_path)
                labels.append(label)

    print(f"[INFO] Found {len(image_paths)} images belonging to {len(set(labels))} users")
    return image_paths, labels



def generate_embeddings_for_known_faces():
    processor, model = load_vit_model()
    image_paths, labels = load_known_faces()

    embeddings = []
    for img_path in image_paths:
        emb = get_embedding_from_image(img_path, processor, model)
        embeddings.append(emb)

    print("\n================ COMPLETE ================")
    print("Total images:", len(image_paths))
    print("Total embeddings:", len(embeddings))
    print("Labels:", labels)
    print("=========================================\n")

    return embeddings, labels



def save_embeddings(embeddings, labels):
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    with open(MODEL_SAVE_PATH, "wb") as f:
        pickle.dump(
            {"embeddings": embeddings, "labels": labels},
            f
        )

    print(f"[INFO] Embeddings saved → {MODEL_SAVE_PATH}")



if __name__ == "__main__":

    # Initialize attendance CSV
    initialize_attendance_file()

    # Generate embeddings
    embeddings, labels = generate_embeddings_for_known_faces()

    if len(embeddings) == 0:
        print("[ERROR] No embeddings generated")
    else:
        save_embeddings(embeddings, labels)
        print("First embedding length:", len(embeddings[0]))
