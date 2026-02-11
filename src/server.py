from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from io import BytesIO
import base64
import os
import csv
import pickle


from recognize_test import identify_face
from recognize import (
    append_attendance,
    generate_embeddings_for_known_faces,
    save_embeddings
)

app = Flask(__name__)
CORS(app)


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KNOWN_FACES_DIR = os.path.join(BASE_DIR, "known_faces")
MODEL_PATH = os.path.join(BASE_DIR, "models", "embeddings.pkl")
CSV_PATH = os.path.join(BASE_DIR, "attendance_log.csv")

os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)


@app.route("/recognize", methods=["POST"])
def recognize():
    try:
        data = request.get_json()
        image_data = data["image"]

        print("\n[INFO] 🔵 Image received for recognition")

        image_bytes = image_data.split(",")[1]
        img = Image.open(BytesIO(base64.b64decode(image_bytes))).convert("RGB")

        temp_path = os.path.join(BASE_DIR, "temp_webcam.jpg")
        img.save(temp_path)

        name, distance = identify_face(temp_path)

        if name not in ["Unknown Face", "No Face Detected", "Error"]:
            append_attendance(name)
            print(f"[INFO] 🟢 Attendance marked for {name}")
        else:
            print("[INFO] ⚪ Attendance not logged")

        return jsonify({
            "name": name,
            "distance": float(distance) if distance is not None else None
        })

    except Exception as e:
        print("[ERROR] Recognition failed:", e)
        return jsonify({"name": "Error", "distance": None})



@app.route("/attendance_data", methods=["GET"])
def attendance_data():
    rows = []

    try:
        with open(CSV_PATH, "r") as file:
            reader = csv.reader(file)
            next(reader)  # skip header
            for row in reader:
                rows.append(row)

    except Exception as e:
        print("[ERROR] Could not load attendance:", e)
        return jsonify([])

    return jsonify(rows)


# =====================================================
# STUDENT ENROLLMENT (WEBCAM – FIXED)
# =====================================================
@app.route("/enroll", methods=["POST"])
def enroll_student():
    try:
        data = request.get_json()
        name = data.get("name", "").strip()
        image_data = data.get("image")

        if not name or not image_data:
            return jsonify({"message": "Invalid name or image"}), 400

        print(f"\n[INFO] ➕ Enrolling student: {name}")

        # Decode image
        image_bytes = image_data.split(",")[1]
        img = Image.open(BytesIO(base64.b64decode(image_bytes))).convert("RGB")

        # ✅ CREATE PER-STUDENT FOLDER (CRITICAL FIX)
        student_dir = os.path.join(KNOWN_FACES_DIR, name)
        os.makedirs(student_dir, exist_ok=True)

        image_path = os.path.join(student_dir, f"{name}_1.jpg")
        img.save(image_path)

        print(f"[INFO] 📸 Image saved at {image_path}")

        # 🔁 REGENERATE EMBEDDINGS SAFELY
        embeddings, labels = generate_embeddings_for_known_faces()
        save_embeddings(embeddings, labels)

        print(f"[INFO] 🟢 {name} enrolled successfully & embeddings updated")

        return jsonify({"message": f"✅ {name} enrolled successfully!"})

    except Exception as e:
        print("[ERROR] Enrollment failed:", e)
        return jsonify({"message": "Enrollment failed"}), 500


# =====================================================
# SERVER START
# =====================================================
if __name__ == "__main__":
    print("[INFO] 🚀 Starting Flask server...")
    app.run(debug=True)
