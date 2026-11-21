"""
Stage 1: confirm your webcam works and OpenCV can detect a face in it.

Run this before installing face_recognition — it uses OpenCV's built-in
Haar Cascade detector, so there's nothing extra to install beyond
opencv-python. This is the same frame-capture pattern you'll reuse on the
Pi camera later; only the source changes.

Press 'q' in the video window to quit.
"""

import cv2

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)  # 0 = default webcam; try 1 or 2 if this fails

if not cap.isOpened():
    raise RuntimeError("Could not open webcam. Try changing the index (0, 1, 2...).")

print("Webcam opened. Press 'q' in the video window to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.putText(
        frame,
        f"Faces detected: {len(faces)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )

    cv2.imshow("Stage 1: face detection test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
