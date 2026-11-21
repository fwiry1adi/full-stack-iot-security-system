# sensors/

PIR motion sensor + GPIO interfacing (Objective 2).

- Sensor: HC-SR501 (PIR)
- Wiring: VCC -> Pi pin 2 (5V), GND -> Pi pin 6, OUT -> Pi pin 7 (GPIO4)
- Library: `gpiozero` (`MotionSensor` class)

This is the trigger that wakes the camera + AI pipeline — see
`docs/architecture.md` for how it fits into the full pipeline.
