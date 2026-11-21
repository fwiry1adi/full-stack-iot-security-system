# IoT Security System

Full-stack IoT security system that detects unauthorized storage-room access
and item interactions in real time, using a Raspberry Pi, dual AI detection
(facial recognition + object detection), cloud logging, and an iOS alert app.

Independent study — Purdue University CIT, Summer 2026
Student: Fiona Wiryadi · Supervising professor: Dr. Eric T. Matson

## How it works

PIR motion sensor -> Pi camera wakes up -> facial recognition + object
detection run in parallel -> alert fusion logic -> cloud (incident log +
push notification) -> iOS app

## Repository structure

| Folder | Purpose |
|---|---|
| `facial_recognition/` | Face detection, encoding, and authorized-employee matching |
| `object_detection/` | Hand/item interaction detection |
| `sensors/` | Motion sensor + GPIO interfacing |
| `cloud/` | Incident logging + push notification integration |
| `ios_app/` | Swift/SwiftUI mobile application |
| `docs/` | Architecture notes, benchmark results, TPS reports |
| `data/known_faces/` | Enrolled employee face data (gitignored, see privacy note) |

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Status

Week 1-2: environment setup + facial recognition prototyping on a laptop
webcam (Raspberry Pi arrives 7/9).

## Privacy note

Employee face photos and encodings are excluded from version control (see
`.gitignore`). This repo should never contain real biometric data in its
commit history, even if the repo is private.
