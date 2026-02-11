import os

# Import identify_face from recognize.py
from recognize import identify_face


def main():
    print("====== FACE RECOGNITION TEST INPUT ======")

    # Ask user for image path
    image_path = input("Enter image path: ").strip()

    # Convert to absolute path (safe)
    image_path = os.path.abspath(image_path)

    # Check if file exists
    if not os.path.exists(image_path):
        print("[ERROR] Image file does not exist.")
        return

    # Run face recognition
    name, distance = identify_face(image_path, threshold=0.5)

    # Print result
    print("\n===== RESULT =====")
    print("Recognized as:", name)
    print("Distance:", distance)

    if name == "Unknown Face":
        print("[INFO] No attendance logged (unknown face).")
    elif name == "No Face Detected":
        print("[INFO] No attendance logged (no face detected).")
    else:
        print("[INFO] Attendance logged successfully.")


if __name__ == "__main__":
    main()
