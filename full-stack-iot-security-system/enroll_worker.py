#!/usr/bin/env python3
"""
enroll_worker.py -- capture face samples for one worker.

Writes SAMPLE_COUNT grayscale 200x200 face crops to
data/known_faces/<worker_id>/ and registers the worker in workers.json.
"""

import os
import sys
import time

import cv2

import worker_db

SAMPLE_COUNT = 30
SAMPLE_SIZE = (200, 200)
FACES_DIR = os.path.join("data", "known_faces")

# The ARM64 OpenCV package ships code but not the cascade XML, so this path
# must be explicit. CascadeClassifier() silently returns an empty classifier
# on a bad path, and detectMultiScale then dies with an !empty() assertion.
CASCADE_PATH = os.path.expanduser(
    "~/haarcascades/haarcascade_frontalface_default.xml")


def load_cascade():
    cascade = cv2.CascadeClassifier(CASCADE_PATH)
    if cascade.empty():
        sys.exit(f"Cascade failed to load from {CASCADE_PATH}")
    return cascade


def open_camera():
    from picamera2 import Picamera2
    cam = Picamera2()
    cam.configure(cam.create_preview_configuration(
        main={"size": (640, 480), "format": "RGB888"}))
    cam.start()
    time.sleep(2)
    return cam


def main() -> None:
    print("Enrol a worker\n")
    name = input("Name: ").strip()
    if not name:
        sys.exit("Name cannot be empty.")

    roles = worker_db.list_roles()
    print("\nRoles:")
    for r, items in roles.items():
        print(f"  {r:20s} {', '.join(items)}")
    role = input("\nRole: ").strip()
    if role not in roles:
        sys.exit(f"Unknown role {role!r}")

    worker_id = worker_db.add_worker(name, role)
    out_dir = os.path.join(FACES_DIR, str(worker_id))
    os.makedirs(out_dir, exist_ok=True)
    print(f"\nRegistered {name} as worker {worker_id} ({role})")
    print(f"Samples -> {out_dir}\n")

    print("=" * 62)
    print("VARY YOUR POSE AS IT CAPTURES. This matters more than it sounds.")
    print()
    print("The first enrolment used 30 near-identical shots and the")
    print("recognizer could not separate faces afterwards. Across the 30:")
    print()
    print("  - turn left and right, ~30 degrees each way")
    print("  - tilt up and down")
    print("  - move closer and further (0.5m to 1.5m)")
    print("  - change lighting -- face a window, then away")
    print("  - neutral, smiling, talking")
    print("=" * 62)
    input("\nPress Enter to start...")

    cascade = load_cascade()
    cam = open_camera()

    captured = 0
    skipped_none = 0
    skipped_many = 0

    try:
        while captured < SAMPLE_COUNT:
            frame = cam.capture_array()
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            faces = cascade.detectMultiScale(gray, scaleFactor=1.3,
                                             minNeighbors=5, minSize=(80, 80))

            if len(faces) == 0:
                skipped_none += 1
                time.sleep(0.1)
                continue

            if len(faces) > 1:
                # Two faces means we cannot tell which is the worker.
                # Labelling the wrong one would poison the model with a
                # second identity under one ID, so skip the frame.
                skipped_many += 1
                print(f"  {len(faces)} faces visible -- skipping")
                time.sleep(0.4)
                continue

            x, y, w, h = faces[0]
            crop = cv2.resize(gray[y:y + h, x:x + w], SAMPLE_SIZE)
            cv2.imwrite(os.path.join(out_dir, f"{captured:03d}.jpg"), crop)
            captured += 1
            print(f"  [{captured:2d}/{SAMPLE_COUNT}] saved  (move now)")
            time.sleep(0.6)

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        cam.stop()

    print(f"\nCaptured {captured} samples for {name} (id {worker_id})")
    if skipped_none:
        print(f"  {skipped_none} frames had no detectable face")
    if skipped_many:
        print(f"  {skipped_many} frames had more than one face")
    print("\nNext:  python3 train_recognizer.py")


if __name__ == "__main__":
    main()
