import cv2

# Load the Haar Cascade file for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Start webcam (0 = default camera)
cap = cv2.VideoCapture(0)

while True:
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale (Haar cascades need grayscale)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the video
    cv2.imshow("Webcam Face Detection", frame)

    key = cv2.waitKey(1) & 0xFF

    # Save face when 's' is pressed
    if key == ord('s'):
        if len(faces) > 0:  
            x, y, w, h = faces[0]  # First detected face

            # Crop the face
            face_roi = frame[y:y+h, x:x+w]

            # Resize to 200x200
            resized_face = cv2.resize(face_roi, (200, 200))

            # Save as face.jpg
            cv2.imwrite("face.jpg", resized_face)
            print("Face saved as face.jpg!")
        else:
            print("No face detected. Try again.")

    # Quit when 'q' is pressed
    if key == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
