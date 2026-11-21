#!/usr/bin/env python3
"""
train_recognizer.py -- build the LBPH face model from enrolled samples.

Reads data/known_faces/<worker_id>/*.jpg, writes data/lbph_model.yml.

LBPH is not a learned model the way a CNN is. There is no loss function and
nothing is optimised: for each sample it computes a histogram of local binary
patterns and files it with its label. Recognition is nearest-neighbour
matching, and the output is a DISTANCE -- lower is a better match.

That matters downstream. A model holding one person cannot answer "nobody I
know"; it can only answer "the closest person I have". The distance threshold
is doing all the work of rejection, which is why it must be measured.
"""

import os
import sys

import cv2
import numpy as np

import worker_db

FACES_DIR = os.path.join("data", "known_faces")
MODEL_PATH = os.path.join("data", "lbph_model.yml")


def load_samples():
    images, labels = [], []
    if not os.path.isdir(FACES_DIR):
        sys.exit(f"No samples at {FACES_DIR}. Run enroll_worker.py first.")

    for entry in sorted(os.listdir(FACES_DIR)):
        sample_dir = os.path.join(FACES_DIR, entry)
        if not os.path.isdir(sample_dir) or not entry.isdigit():
            continue
        worker_id = int(entry)
        count = 0
        for fname in sorted(os.listdir(sample_dir)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img = cv2.imread(os.path.join(sample_dir, fname),
                             cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"  ! could not read {fname}")
                continue
            images.append(img)
            labels.append(worker_id)
            count += 1
        worker = worker_db.get_worker(worker_id)
        who = worker["name"] if worker else "(not in workers.json)"
        print(f"  id {worker_id:>3}  {who:20s} {count:3d} samples")

    return images, np.array(labels, dtype=np.int32)


def main() -> None:
    print("Loading samples...\n")
    images, labels = load_samples()
    if not images:
        sys.exit("\nNo samples found. Run enroll_worker.py first.")

    n_people = len(set(labels.tolist()))
    print(f"\n{len(images)} samples across {n_people} "
          f"{'person' if n_people == 1 else 'people'}")

    if n_people == 1:
        print()
        print("  Only one person is enrolled. The recognizer will match")
        print("  EVERY face to them -- it has no alternative to offer. The")
        print("  distance threshold is the only thing that can reject a")
        print("  stranger, and it has never been measured.")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(images, labels)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    recognizer.write(MODEL_PATH)

    print(f"\nWrote {MODEL_PATH} "
          f"({os.path.getsize(MODEL_PATH) / 1024:.0f} KB)")
    print("\nNext:  python3 recognize_test.py")


if __name__ == "__main__":
    main()
