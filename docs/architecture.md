# System architecture (draft — for TPS Report #1)

## Pipeline

1. PIR motion sensor detects movement at the storage room entrance
2. Pi wakes the camera and the dual AI pipeline
3. Facial recognition classifies the person as authorized / unauthorized
4. Object detection watches for hand-to-item interaction
5. Alert fusion logic combines both signals into two alert types:
   - Person-level alert (unauthorized entry)
   - Interaction-level alert (item handling)
6. Alerts, photo, timestamp, and location get logged to the cloud
7. Cloud sends a push notification to the iOS app

## Hardware

- Raspberry Pi (arriving 7/9)
- Pi Camera Module
- PIR motion sensor (HC-SR501)

## Open decisions

- [ ] Cloud platform (Firebase / AWS / custom backend)
- [ ] Object detection model (MediaPipe Hands / YOLO)
- [ ] Whether an AI accelerator (Hailo) is needed for real-time performance

## Early challenges

(fill this in as you go — this section becomes your TPS Report #1 material)
