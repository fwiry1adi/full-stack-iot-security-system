#!/usr/bin/env python3
"""
recognize_test.py -- live face recognition check, with distance reporting.

    python3 recognize_test.py              # live identification
    python3 recognize_test.py --calibrate  # measure the threshold

LBPH returns a DISTANCE: lower means a closer match. The threshold below
decides who counts as recognised, and it was never measured -- it was
picked. That is why strangers have matched as enrolled workers.
--calibrate exists to replace the guess with data.
"""

import argparse
import os
import statistics
import sys
import time

import cv2

import worker_db

CONFIDENCE_THRESHOLD = 70.0          # provisional -- see --calibrate

MODEL_PATH = os.path.join("data", "lbph_model.yml")
CASCADE_PATH = os.path.expanduser(
    "~/haarcascades/haarcascade_frontalface_default.xml")
SAMPLE_SIZE = (200, 200)


def load_all():
    cascade = cv2.CascadeClassifier(CASCADE_PATH)
    if cascade.empty():
        sys.exit(f"Cascade failed to load from {CASCADE_PATH}")

    if not os.path.exists(MODEL_PATH):
        sys.exit(f"No model at {MODEL_PATH}. Run train_recognizer.py first.")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_PATH)

    from picamera2 import Picamera2
    cam = Picamera2()
    cam.configure(cam.create_preview_configuration(
        main={"size": (640, 480), "format": "RGB888"}))
    cam.start()
    time.sleep(2)
    return cascade, recognizer, cam


def largest_face(cascade, gray):
    faces = cascade.detectMultiScale(gray, 1.3, 5, minSize=(80, 80))
    if len(faces) == 0:
        return None
    return max(faces, key=lambda f: f[2] * f[3])


def run_live(cascade, recognizer, cam):
    print(f"Threshold: {CONFIDENCE_THRESHOLD} (lower distance = better)")
    print("Ctrl-C to stop.\n")
    try:
        while True:
            gray = cv2.cvtColor(cam.capture_array(), cv2.COLOR_RGB2GRAY)
            face = largest_face(cascade, gray)
            if face is None:
                time.sleep(0.3)
                continue

            x, y, w, h = face
            crop = cv2.resize(gray[y:y + h, x:x + w], SAMPLE_SIZE)
            worker_id, distance = recognizer.predict(crop)

            if distance <= CONFIDENCE_THRESHOLD:
                worker = worker_db.get_worker(worker_id)
                if worker:
                    req = worker_db.required_ppe(worker_id)
                    print(f"  {worker['name']:18s} ({worker['role']})  "
                          f"distance {distance:6.1f}   needs: {', '.join(req)}")
                else:
                    print(f"  id {worker_id} not in workers.json  "
                          f"distance {distance:6.1f}")
            else:
                print(f"  unrecognised                        "
                      f"distance {distance:6.1f}  (> {CONFIDENCE_THRESHOLD})")
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopped.")


def run_calibrate(cascade, recognizer, cam):
    """Collect distances for one person so the threshold can be set.

    Run once for yourself, then once per person who is NOT enrolled.
    A threshold is only defensible if it separates the two distributions.
    """
    label = input("Who is in front of the camera? ").strip()
    enrolled = input("Are they enrolled? [y/N] ").strip().lower() == "y"
    n = int(input("How many samples? [40] ").strip() or "40")

    print(f"\nCollecting {n} readings for {label}. Move around.\n")

    distances = []
    try:
        while len(distances) < n:
            gray = cv2.cvtColor(cam.capture_array(), cv2.COLOR_RGB2GRAY)
            face = largest_face(cascade, gray)
            if face is None:
                time.sleep(0.2)
                continue
            x, y, w, h = face
            crop = cv2.resize(gray[y:y + h, x:x + w], SAMPLE_SIZE)
            wid, dist = recognizer.predict(crop)
            distances.append(dist)
            print(f"  {len(distances):3d}/{n}  distance {dist:6.1f} -> id {wid}")
            time.sleep(0.3)
    except KeyboardInterrupt:
        pass

    if not distances:
        print("No readings collected.")
        return

    distances.sort()
    print(f"\n{label}  ({'enrolled' if enrolled else 'NOT enrolled'})")
    print(f"  n       {len(distances)}")
    print(f"  min     {distances[0]:.1f}")
    print(f"  median  {statistics.median(distances):.1f}")
    print(f"  mean    {statistics.mean(distances):.1f}")
    print(f"  max     {distances[-1]:.1f}")
    if len(distances) > 1:
        print(f"  stdev   {statistics.stdev(distances):.1f}")

    at_70 = sum(1 for d in distances if d <= 70.0)
    pct = at_70 / len(distances) * 100
    print(f"\n  At threshold 70.0: {at_70}/{len(distances)} accepted ({pct:.0f}%)")
    if enrolled:
        print("  For an enrolled person, that is the TRUE ACCEPT rate.")
    else:
        print("  For a non-enrolled person, that is the FALSE ACCEPT rate.")
        print("  More than a few percent means the threshold is too loose.")

    print("\nRun this for yourself and at least three people who are NOT")
    print("enrolled. Set the threshold between the two distributions. If")
    print("they overlap, LBPH cannot separate these faces and the")
    print("recognizer needs replacing with a learned embedding model.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--calibrate", action="store_true",
                    help="measure the distance distribution for one person")
    args = ap.parse_args()

    cascade, recognizer, cam = load_all()
    try:
        if args.calibrate:
            run_calibrate(cascade, recognizer, cam)
        else:
            run_live(cascade, recognizer, cam)
    finally:
        cam.stop()


if __name__ == "__main__":
    main()
