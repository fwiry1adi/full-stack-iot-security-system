# facial_recognition/

Face detection, encoding, and authorized-employee matching (Objective 3).

## Stage 1 — `test_webcam_face_detection.py`
Confirms your camera works and OpenCV can detect a face in the frame. Uses
OpenCV's built-in Haar Cascade detector, so there's nothing to install beyond
`opencv-python`.

## Stage 2 — coming next
Face encoding + comparison against `data/known_faces/`, using the
`face_recognition` library. Will add `enroll.py` (register an employee) and
`recognize.py` (the real-time matching loop).
