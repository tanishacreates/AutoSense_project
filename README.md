# VigiDrive — Integrated Driver Assistance System

## Overview

This is the **fully integrated** combination of:
- **Person 1** — Vision & AI Core (face detection, EAR/MAR, head pose, CNN)
- **Person 2** — System Integration (sensors, ML fusion, alerts, dashboard, logging)

---

## Project tructure

```
vigidrive_integrated/
│
├── main.py               ← ENTRY POINT — run this
├── bridge.py             ← Integration glue (DriverState → VisionFeatures)
├── requirements.txt
│
├── core/                 ── Person 1's modules (unchanged)
│   ├── detector.py       ← DriverMonitor, DriverState, EAR/MAR/head pose
│   └── __init__.py
│
├── utils/                ── Person 1's modules (unchanged)
│   ├── display.py        ← Dashboard HUD overlay
│   └── __init__.py
│
├── train_model.py        ← Person 1's CNN training script
│
├── sensor_module.py      ── Person 2's modules (unchanged)
├── fusion_engine.py
├── risk_classifier.py
├── alert_system.py
├── dashboard.py
├── logger.py
│
├── models/               ← Trained ML model saved here
├── logs/                 ← Session CSV files + timeline plots
└── tests/
    └── test_integration.py
```

---

## How Integration Works

Only **one file** was written to connect both sides: `bridge.py`

```
Person 1                   bridge.py                   Person 2
─────────────────────────────────────────────────────────────────
DriverMonitor              driver_state_to_            FusionEngine
  .analyze(frame)    →     vision_features(state)  →    .fuse(vf, sensor)
    → DriverState             → VisionFeatures           → FusionResult
                                                              ↓
                                                     RiskClassifier.predict()
                                                              ↓
                                                     AlertSystem.trigger()
```

**Nothing in either person's original code was modified.**

### Field mapping (bridge.py)

| Person 1 `DriverState` | Person 2 `VisionFeatures` |
|------------------------|---------------------------|
| `ear`                  | `ear`                     |
| `mar`                  | `mar`                     |
| `pitch`                | `head_pitch`              |
| `yaw`                  | `head_yaw`                |
| `roll`                 | `head_roll`               |
| `blink_rate`           | `blink_rate_pm`           |
| `eyes_closed`          | `microsleep`              |
| `yawning`              | `yawn_detected`           |
| `face_detected`        | `face_visible`            |
| `drowsy` + `drowsiness_level` | `cnn_state` + `cnn_confidence` |

---

## Setup & Run

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run integration tests (no camera needed)
python tests/test_integration.py

# 4. Run the full system
python main.py

# No camera? Headless mode?
python main.py --no-display --no-dashboard

# Change sensitivity
python main.py --sensitivity high

# Simulate drowsy driving sensor data
python main.py --sensor-pattern drowsy

# Save Person 1's original event CSV
python main.py --save-log

# Evaluate a saved session
python main.py --evaluate logs/session_YYYYMMDD_HHMMSS.csv
```

---

## What You See on Screen

The integrated display has **3 layers**:

```
┌──────────────────────────────────────────────────────┐
│ [P1] VigiDrive metrics panel (top-left)              │
│  EAR  0.285     PERCLOS  8.2%                        │
│  MAR  0.220     BLINK/m  14.3                        │
│  PITCH -4.2°    YAW  3.1°                            │
│                                                      │
│      [Face mesh landmarks drawn on face]             │
│                                                      │
│                          [P2 Risk HUD - bottom-right]│
│                           P2: WARNING                │
│                           Risk  : 0.412              │
│                           Speed : 73 km/h            │
│                           Lane  : +4 cm              │
│                           ████░░░░░░░░ (risk bar)    │
│                                                      │
│ YAWN  EYES     [P1 status bar]    DROWSY    00:45    │
└──────────────────────────────────────────────────────┘
  [P2 coloured border — green/yellow/red by alert level]
```

---

## Alert Levels

Both alert systems run in parallel:

| Level    | Person 1 triggers when…         | Person 2 triggers when…          |
|----------|----------------------------------|----------------------------------|
| INFO     | Mild drowsiness (level 1)        | —                                |
| WARNING  | Moderate drowsiness / yawn       | Risk score 0.35–0.65             |
| ALERT    | Severe drowsiness / distraction  | Risk score 0.65–0.85             |
| CRITICAL | —                                | Risk > 0.85 or drowsiness level 3|

---

## Retraining the CNN (Person 1)

```bash
# Download CEW dataset → data/eye_dataset/open/ and data/eye_dataset/closed/
python train_model.py --data_dir data/eye_dataset --epochs 30
```

---

## Running Tests

```bash
# Integration tests
python tests/test_integration.py

# Person 2's original unit tests
python -m pytest tests/ -v
```
